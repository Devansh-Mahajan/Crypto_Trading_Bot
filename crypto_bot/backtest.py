"""
Industrial-grade backtesting engine for crypto strategies.

Architecture (VectorBT-inspired):
  1. Data      — CoinGecko OHLCV via CoinGeckoData
  2. Signals   — computed vectorially (numpy/pandas) by each Strategy
  3. Execution — fast sequential simulation (position state is inherently sequential)
  4. Metrics   — full suite: CAGR, Sharpe, Sortino, Calmar, max-DD, trade stats
  5. Reporting — console table + multi-strategy comparison

Execution model (prevents look-ahead bias):
  signal fires at close[i]  →  order fills at open[i+1]
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any

import numpy as np
import pandas as pd

from .data import CoinGeckoData
from .strategies import REGISTRY
from .strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# ─── Domain objects ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    entry_price: float
    exit_price:  float
    size:        float   # units held
    pnl:         float   # net profit/loss in USD
    pnl_pct:     float   # % return on capital deployed
    fee_paid:    float   # total fees (entry + exit)
    exit_reason: str     # 'signal' | 'stop_loss' | 'take_profit' | 'end_of_data'


@dataclass
class BacktestResult:
    strategy_name: str
    symbol:        str
    days:          int
    candles:       int
    metrics:       dict[str, Any]
    trades:        list[Trade]
    equity:        pd.Series      # equity curve (same index as data)
    drawdown:      pd.Series      # drawdown series (%)
    signals:       pd.DataFrame   # raw signal DataFrame from strategy
    data:          pd.DataFrame   # OHLCV DataFrame used


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Vectorized-signal, sequential-execution backtesting engine.

    Signals are computed over the full dataset in one vectorised pass by the
    strategy (fast).  Trade simulation is a tight O(n) Python loop — the only
    sequential part — because position state is inherently causal.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value in USD.
    fees_pct : float
        Trading fee as a percentage per side  (0.1 = 0.1 %).
    slippage_pct : float
        Estimated slippage per side as a percentage  (0.05 = 0.05 %).
    position_pct : float
        Fraction of available capital invested per trade  (95 = 95 %).
    stop_loss_pct : float | None
        Per-trade stop-loss below entry price  (5 = 5 %).  None to disable.
    take_profit_pct : float | None
        Per-trade take-profit above entry price  (10 = 10 %).  None to disable.
    """

    def __init__(
        self,
        initial_capital:  float = 10_000.0,
        fees_pct:         float = 0.1,
        slippage_pct:     float = 0.05,
        position_pct:     float = 95.0,
        stop_loss_pct:    float | None = 5.0,
        take_profit_pct:  float | None = 10.0,
    ) -> None:
        self.initial_capital  = initial_capital
        self.fees_pct         = fees_pct
        self.slippage_pct     = slippage_pct
        self.position_pct     = position_pct
        self.stop_loss_pct    = stop_loss_pct
        self.take_profit_pct  = take_profit_pct
        self._cost_pct        = (fees_pct + slippage_pct) / 100.0

    def run(self, data: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """
        Run `strategy` against `data` and return a fully populated BacktestResult.
        `data` must have a DatetimeIndex (UTC) and columns: open, high, low, close, volume.
        """
        signals = strategy.generate_signals(data)
        equity_series, trades = self._simulate(data, signals)
        roll_max = equity_series.cummax()
        drawdown = (equity_series - roll_max) / roll_max * 100.0
        metrics = _compute_metrics(equity_series, drawdown, trades, data, self.initial_capital)
        return BacktestResult(
            strategy_name=strategy.name,
            symbol="",
            days=0,
            candles=len(data),
            metrics=metrics,
            trades=trades,
            equity=equity_series,
            drawdown=drawdown,
            signals=signals,
            data=data,
        )

    # ------------------------------------------------------------------

    def _simulate(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> tuple[pd.Series, list[Trade]]:
        n          = len(data)
        opens      = data["open"].to_numpy(dtype=float)
        highs      = data["high"].to_numpy(dtype=float)
        lows       = data["low"].to_numpy(dtype=float)
        closes     = data["close"].to_numpy(dtype=float)
        entries    = signals["entry"].to_numpy(dtype=bool)
        exits      = signals["exit"].to_numpy(dtype=bool)
        idx        = data.index

        capital      = float(self.initial_capital)
        pos_amount   = 0.0
        pos_entry    = 0.0
        pos_entry_ts: pd.Timestamp | None = None
        in_position  = False
        equity_arr   = np.empty(n, dtype=float)
        trades: list[Trade] = []

        for i in range(n):
            price = closes[i]
            equity_arr[i] = capital + (pos_amount * price if in_position else 0.0)

            # ── Intra-bar SL/TP check (uses high/low range, no look-ahead) ──
            if in_position:
                reason, fill = self._check_sl_tp(pos_entry, lows[i], highs[i])
                if reason:
                    t, capital = _close(capital, pos_amount, pos_entry, fill,
                                        pos_entry_ts, idx[i], self._cost_pct, reason)
                    trades.append(t)
                    pos_amount = pos_entry = 0.0
                    pos_entry_ts = None
                    in_position = False
                    equity_arr[i] = capital
                    continue

            # ── Signal execution at next bar's open ──────────────────────────
            if i + 1 >= n:
                continue

            fill_open = opens[i + 1]

            if in_position and exits[i]:
                fill = fill_open * (1.0 - self.slippage_pct / 100.0)
                t, capital = _close(capital, pos_amount, pos_entry, fill,
                                    pos_entry_ts, idx[i + 1], self._cost_pct, "signal")
                trades.append(t)
                pos_amount = pos_entry = 0.0
                pos_entry_ts = None
                in_position = False

            elif not in_position and entries[i]:
                invest    = capital * self.position_pct / 100.0
                fee       = invest * self.fees_pct / 100.0
                slip_fill = fill_open * (1.0 + self.slippage_pct / 100.0)
                pos_amount   = (invest - fee) / slip_fill
                capital     -= invest
                pos_entry    = slip_fill
                pos_entry_ts = idx[i + 1]
                in_position  = True

        # ── Force-close at last bar ───────────────────────────────────────────
        if in_position and pos_amount > 0:
            t, capital = _close(capital, pos_amount, pos_entry, closes[-1],
                                pos_entry_ts, idx[-1], self._cost_pct, "end_of_data")
            trades.append(t)

        return pd.Series(equity_arr, index=idx, name="equity"), trades

    def _check_sl_tp(
        self, entry: float, low: float, high: float
    ) -> tuple[str, float]:
        if self.stop_loss_pct is not None:
            sl = entry * (1.0 - self.stop_loss_pct / 100.0)
            if low <= sl:
                return "stop_loss", sl
        if self.take_profit_pct is not None:
            tp = entry * (1.0 + self.take_profit_pct / 100.0)
            if high >= tp:
                return "take_profit", tp
        return "", 0.0


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _compute_metrics(
    equity:  pd.Series,
    drawdown: pd.Series,
    trades:  list[Trade],
    data:    pd.DataFrame,
    initial: float,
) -> dict[str, Any]:
    final        = float(equity.iloc[-1])
    total_return = (final - initial) / initial * 100.0

    # CAGR
    years = _years(equity.index)
    cagr  = ((final / initial) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0

    # Buy-and-hold comparison on close prices
    bh_return = (data["close"].iloc[-1] / data["close"].iloc[0] - 1.0) * 100.0

    # Per-candle returns for risk metrics
    rets     = equity.pct_change().dropna()
    mean_r   = float(rets.mean())
    std_r    = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    af       = _annual_factor(equity.index)   # candles per year

    sharpe   = mean_r / std_r * math.sqrt(af) if std_r > 0 else 0.0

    down_r   = rets[rets < 0]
    down_std = float(down_r.std(ddof=1)) if len(down_r) > 1 else 0.0
    sortino  = mean_r / down_std * math.sqrt(af) if down_std > 0 else 0.0

    ann_vol  = std_r * math.sqrt(af) * 100.0

    max_dd   = float(drawdown.min())

    # Max drawdown duration (consecutive bars in drawdown)
    in_dd         = drawdown < 0
    dd_groups     = (in_dd != in_dd.shift()).cumsum()
    max_dd_dur    = int(in_dd.groupby(dd_groups).sum().max()) if in_dd.any() else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Trade statistics
    wins         = [t for t in trades if t.pnl > 0]
    losses       = [t for t in trades if t.pnl <= 0]
    win_rate     = len(wins) / len(trades) * 100.0 if trades else 0.0
    avg_win      = sum(t.pnl_pct for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss     = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
    best_trade   = max((t.pnl_pct for t in trades), default=0.0)
    worst_trade  = min((t.pnl_pct for t in trades), default=0.0)
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_fees   = sum(t.fee_paid for t in trades)

    # Average trade duration
    if trades:
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        avg_dur_h = sum(durations) / len(durations)
    else:
        avg_dur_h = 0.0

    return {
        "total_return_pct":   total_return,
        "cagr_pct":           cagr,
        "buy_hold_pct":       bh_return,
        "sharpe":             sharpe,
        "sortino":            sortino,
        "calmar":             calmar,
        "max_drawdown_pct":   max_dd,
        "max_dd_duration_bars": max_dd_dur,
        "ann_volatility_pct": ann_vol,
        "total_trades":       len(trades),
        "win_rate_pct":       win_rate,
        "winning_trades":     len(wins),
        "losing_trades":      len(losses),
        "avg_win_pct":        avg_win,
        "avg_loss_pct":       avg_loss,
        "best_trade_pct":     best_trade,
        "worst_trade_pct":    worst_trade,
        "profit_factor":      profit_factor,
        "total_fees_usd":     total_fees,
        "avg_trade_dur_h":    avg_dur_h,
        "initial_balance":    initial,
        "final_balance":      final,
    }


def _years(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    return (index[-1] - index[0]).total_seconds() / (365.25 * 86_400)


def _annual_factor(index: pd.DatetimeIndex) -> float:
    """Candles per year, used to annualise Sharpe / Sortino."""
    if len(index) < 2:
        return 365.0
    avg_secs = (index[-1] - index[0]).total_seconds() / (len(index) - 1)
    return 365.25 * 86_400 / avg_secs


# ─── Trade helper ─────────────────────────────────────────────────────────────

def _close(
    capital:    float,
    amount:     float,
    entry:      float,
    exit_price: float,
    entry_ts:   pd.Timestamp | None,
    exit_ts:    pd.Timestamp,
    cost_pct:   float,
    reason:     str,
) -> tuple[Trade, float]:
    proceeds   = amount * exit_price
    fee        = proceeds * cost_pct
    cost_basis = amount * entry
    net        = proceeds - fee
    pnl        = net - cost_basis
    pnl_pct    = pnl / cost_basis * 100.0 if cost_basis else 0.0
    new_cap    = capital + net
    t = Trade(
        entry_time  = entry_ts or exit_ts,
        exit_time   = exit_ts,
        entry_price = entry,
        exit_price  = exit_price,
        size        = amount,
        pnl         = pnl,
        pnl_pct     = pnl_pct,
        fee_paid    = fee,
        exit_reason = reason,
    )
    return t, new_cap


# ─── Reporting ────────────────────────────────────────────────────────────────

def _pct(v: float, plus: bool = False) -> str:
    s = f"+{v:.2f}%" if (plus and v >= 0) else f"{v:.2f}%"
    return s


def print_report(r: BacktestResult) -> None:
    m   = r.metrics
    SEP = "═" * 62
    HR  = "─" * 62

    # Header
    print(f"\n{SEP}")
    print(f"  {r.strategy_name}")
    print(f"  {r.symbol}  ·  {r.days}d  ·  {r.candles} candles")
    print(SEP)

    # Returns
    _row("Total Return",           _pct(m["total_return_pct"], plus=True))
    _row("CAGR",                   _pct(m["cagr_pct"],         plus=True))
    _row("Buy & Hold (close-to-close)", _pct(m["buy_hold_pct"], plus=True))
    print(f"  {HR}")

    # Risk
    _row("Max Drawdown",           _pct(m["max_drawdown_pct"]))
    _row("Max DD Duration (bars)", str(m["max_dd_duration_bars"]))
    _row("Ann. Volatility",        _pct(m["ann_volatility_pct"]))
    _row("Sharpe Ratio (ann.)",    f"{m['sharpe']:.3f}")
    _row("Sortino Ratio (ann.)",   f"{m['sortino']:.3f}")
    _row("Calmar Ratio",           f"{m['calmar']:.3f}")
    print(f"  {HR}")

    # Trades
    _row("Total Trades",           str(m["total_trades"]))
    _row("Win / Loss",             f"{m['winning_trades']} / {m['losing_trades']}")
    _row("Win Rate",               _pct(m["win_rate_pct"]))
    _row("Avg Win",                _pct(m["avg_win_pct"],  plus=True))
    _row("Avg Loss",               _pct(m["avg_loss_pct"]))
    _row("Best Trade",             _pct(m["best_trade_pct"], plus=True))
    _row("Worst Trade",            _pct(m["worst_trade_pct"]))
    pf = m["profit_factor"]
    _row("Profit Factor",          "∞" if pf == float("inf") else f"{pf:.2f}")
    _row("Avg Trade Duration",     f"{m['avg_trade_dur_h']:.1f} h")
    _row("Total Fees Paid",        f"${m['total_fees_usd']:.2f}")
    print(f"  {HR}")
    _row("Initial Balance",        f"${m['initial_balance']:,.2f}")
    _row("Final Balance",          f"${m['final_balance']:,.2f}")

    # Trade log
    if r.trades:
        print(f"\n  Last 5 trades:")
        print(f"  {'Entry Date':<19} {'Entry $':>9} {'Exit $':>9} {'PnL $':>9} {'PnL %':>8}  Reason")
        print(f"  {'-'*19} {'-'*9} {'-'*9} {'-'*9} {'-'*8}  {'-'*13}")
        for t in r.trades[-5:]:
            s = "+" if t.pnl >= 0 else ""
            dt = t.entry_time.strftime("%Y-%m-%d %H:%M")
            print(
                f"  {dt:<19} {t.entry_price:>9.2f} {t.exit_price:>9.2f}"
                f" {s}{t.pnl:>8.2f} {s}{t.pnl_pct:>7.2f}%  {t.exit_reason}"
            )

    print(f"\n{SEP}\n")


def _row(label: str, value: str, label_w: int = 30) -> None:
    print(f"  {label:<{label_w}} {value:>10}")


def print_comparison(results: list[BacktestResult]) -> None:
    """Render a side-by-side comparison table for multiple strategies."""
    ROWS: list[tuple[str, str, str]] = [
        ("total_return_pct",     "Total Return %",    "{:+.2f}%"),
        ("cagr_pct",             "CAGR %",            "{:+.2f}%"),
        ("buy_hold_pct",         "Buy & Hold %",      "{:+.2f}%"),
        ("sharpe",               "Sharpe (ann.)",     "{:.3f}"),
        ("sortino",              "Sortino (ann.)",    "{:.3f}"),
        ("calmar",               "Calmar",            "{:.3f}"),
        ("max_drawdown_pct",     "Max Drawdown %",    "{:.2f}%"),
        ("ann_volatility_pct",   "Ann. Volatility %", "{:.2f}%"),
        ("win_rate_pct",         "Win Rate %",        "{:.1f}%"),
        ("profit_factor",        "Profit Factor",     "{:.2f}"),
        ("total_trades",         "Trades",            "{:d}"),
        ("avg_win_pct",          "Avg Win %",         "{:+.2f}%"),
        ("avg_loss_pct",         "Avg Loss %",        "{:.2f}%"),
        ("total_fees_usd",       "Fees Paid ($)",     "${:.2f}"),
        ("final_balance",        "Final Balance ($)", "${:,.0f}"),
    ]

    names  = [r.strategy_name for r in results]
    col_w  = max(max(len(n) for n in names), 28) + 2
    lbl_w  = 22
    total  = lbl_w + col_w * len(results) + 4

    sep  = "─" * total
    dsep = "═" * total

    print(f"\n{dsep}")
    print(f"  {'STRATEGY COMPARISON':^{total - 4}}")
    print(dsep)
    header = f"  {'Metric':<{lbl_w}}" + "".join(f"{n:^{col_w}}" for n in names)
    print(header)
    print(sep)

    for key, label, fmt in ROWS:
        row = f"  {label:<{lbl_w}}"
        for r in results:
            val = r.metrics.get(key, 0)
            if val == float("inf"):
                cell = "∞"
            else:
                try:
                    cell = fmt.format(int(val) if fmt.endswith("d}") else val)
                except (ValueError, TypeError):
                    cell = str(val)
            row += f"{cell:^{col_w}}"
        print(row)

    print(dsep + "\n")


# ─── Convenience entry point ──────────────────────────────────────────────────

def run_backtest(
    symbol:          str   = "BTC/USDT",
    strategy_name:   str   = "ema_rsi",
    # ── CoinGecko source (default) ──────────────────────────────────────
    days:            int   = 90,
    vs_currency:     str   = "usd",
    api_key:         str | None = None,
    # ── CCXT source (set exchange to activate) ──────────────────────────
    exchange:        str | None = None,   # e.g. 'binance'
    timeframe:       str   = "4h",
    bars:            int   = 500,
    # ── Engine params ────────────────────────────────────────────────────
    initial_capital: float = 10_000.0,
    fees_pct:        float = 0.1,
    slippage_pct:    float = 0.05,
    position_pct:    float = 95.0,
    stop_loss_pct:   float | None = 5.0,
    take_profit_pct: float | None = 10.0,
) -> BacktestResult | list[BacktestResult]:
    """
    Fetch OHLCV and run the requested strategy (or all four).

    Data source:
      exchange=None  →  CoinGecko (free, no key, limited timeframes)
      exchange='binance' etc.  →  CCXT (any timeframe, longer history)

    strategy_name: 'ema_rsi' | 'macd' | 'bollinger' | 'supertrend' | 'all'
    """
    if exchange:
        from .data_ccxt import CCXTData
        data = CCXTData(exchange).fetch_ohlcv(symbol, timeframe=timeframe, bars=bars)
    else:
        client = CoinGeckoData(api_key=api_key)
        data   = client.fetch_ohlcv(symbol, days=days, vs_currency=vs_currency)

    if len(data) < 30:
        raise ValueError(
            f"Only {len(data)} candles returned — too few to backtest. "
            f"Try a larger --days value."
        )

    engine = BacktestEngine(
        initial_capital  = initial_capital,
        fees_pct         = fees_pct,
        slippage_pct     = slippage_pct,
        position_pct     = position_pct,
        stop_loss_pct    = stop_loss_pct,
        take_profit_pct  = take_profit_pct,
    )

    strategy_keys = list(REGISTRY) if strategy_name == "all" else [strategy_name]

    if strategy_name != "all" and strategy_name not in REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {', '.join(REGISTRY)} or 'all'."
        )

    results: list[BacktestResult] = []
    for key in strategy_keys:
        strat  = REGISTRY[key]()
        result = engine.run(data, strat)
        result.symbol = symbol
        result.days   = days
        results.append(result)
        logger.info("Finished: %s", strat.name)

    if strategy_name == "all":
        print_comparison(results)
        for r in results:
            print_report(r)
        return results

    print_report(results[0])
    return results[0]
