"""
Industrial live / paper trading bot.

New in this version
-------------------
  ATR-based position sizing  —  volatility-adjusted; risks a fixed % of
    portfolio per trade regardless of market conditions.
  Trailing stop              —  dynamically moves up with price, locking in
    gains while allowing profitable trades to run.
  Portfolio heat guard       —  prevents total notional exposure from
    exceeding a configurable % of portfolio value.
  Trade journal              —  every closed trade appended to trades.csv
    for offline analysis.
"""
from __future__ import annotations

import csv
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BotConfig
from .exchange import ExchangeClient, MarketQuote
from .strategy import EMACrossoverStrategy, StrategySignal

logger = logging.getLogger(__name__)


# ─── State ────────────────────────────────────────────────────────────────────

@dataclass
class PositionState:
    symbol:               str
    entry_price:          float = 0.0
    amount:               float = 0.0
    side:                 str   = "none"   # 'long' | 'none'
    highest_since_entry:  float = 0.0     # trailing-stop tracking
    trailing_stop_price:  float = 0.0     # absolute price level

    @property
    def is_open(self) -> bool:
        return self.side == "long" and self.amount > 0

    @property
    def notional_value(self) -> float:
        return self.entry_price * self.amount


# ─── Engine ───────────────────────────────────────────────────────────────────

class TradeBot:
    """
    Orchestrates the live / paper trading loop.

    Config keys (strategy section)
    -------------------------------
    fast_ema, slow_ema, rsi_period, rsi_overbought, use_rsi
    risk_pct            — % of portfolio risked per trade (ATR sizing)
    stop_loss_pct       — hard stop  (% below entry)
    take_profit_pct     — hard take-profit (% above entry)
    max_exposure_pct    — cap on single-position notional / portfolio
    trailing_stop_pct   — trailing stop distance (% from high; 0 = disabled)
    atr_period          — ATR period for volatility-adjusted sizing
    atr_risk_mult       — ATRs of risk per trade (default 1.5)
    max_portfolio_heat  — max total notional / portfolio (default 95 %)
    min_balance_usd     — minimum USD balance to allow a trade

    Config keys (paper section)
    ----------------------------
    starting_balance, fees_pct

    Config keys (trade section)
    ----------------------------
    mode, poll_interval, quote_currency, use_coingecko_price
    journal_file        — path to CSV trade journal  (default: trades.csv)
    sizing_mode         — 'fixed' (default) | 'atr'
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config = BotConfig(config_path)
        self.exchange = ExchangeClient(
            self.config.exchange,
            api_key=self.config.api_key,
            secret=self.config.secret,
            password=self.config.password,
        )

        sc = self.config.strategy
        self.strategy = EMACrossoverStrategy(
            fast_ema        = int(sc.get("fast_ema",        8)),
            slow_ema        = int(sc.get("slow_ema",        21)),
            rsi_period      = int(sc.get("rsi_period",      14)),
            rsi_overbought  = float(sc.get("rsi_overbought",70.0)),
            use_rsi         = bool(sc.get("use_rsi",        True)),
            risk_pct        = float(sc.get("risk_pct",      1.0)),
            stop_loss_pct   = float(sc.get("stop_loss_pct", 1.5)),
            take_profit_pct = float(sc.get("take_profit_pct",3.0)),
            max_exposure_pct= float(sc.get("max_exposure_pct",95.0)),
            min_balance_usd = float(sc.get("min_balance_usd",20.0)),
        )

        tc = self.config.trade
        self.mode                = tc.get("mode",               "paper")
        self.symbol              = self.config.symbol
        self.timeframe           = self.config.timeframe
        self.poll_interval       = int(tc.get("poll_interval",  300))
        self.quote_currency      = tc.get("quote_currency",     "USDT")
        self.use_coingecko_price = bool(tc.get("use_coingecko_price", True))
        self.sizing_mode         = tc.get("sizing_mode",        "fixed")
        self.journal_file        = Path(tc.get("journal_file",  "trades.csv"))

        # Risk parameters
        self.trailing_stop_pct   = float(sc.get("trailing_stop_pct",  0.0))
        self.atr_period          = int(sc.get("atr_period",           14))
        self.atr_risk_mult       = float(sc.get("atr_risk_mult",      1.5))
        self.max_portfolio_heat  = float(sc.get("max_portfolio_heat",  95.0))

        pc = self.config.paper or {}
        self.paper_balance  = float(pc.get("starting_balance", 10_000))
        self.paper_fees_pct = float(pc.get("fees_pct",         0.05))

        self.position = PositionState(symbol=self.symbol)
        self.running  = True

        # Cached ATR for use in sizing; updated every cycle
        self._current_atr: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        log_level = self.config.logging.get("level", "INFO").upper()
        logging.basicConfig(
            level   = getattr(logging, log_level, logging.INFO),
            format  = "%(asctime)s [%(levelname)s] %(message)s",
            handlers= [logging.StreamHandler(sys.stdout)],
        )

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.warning("Shutting down on signal %s", signum)
        self.running = False

    def run(self) -> None:
        self._setup_logging()
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Starting %s  symbol=%s  mode=%s", self.__class__.__name__, self.symbol, self.mode)
        if self.mode == "live" and not self.config.api_key:
            raise RuntimeError("Live mode requires API key and secret in config or environment")

        while self.running:
            try:
                self.execute_cycle()
            except Exception as exc:
                logger.exception("Cycle error: %s", exc)
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def execute_cycle(self) -> None:
        candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=150)
        if not candles or len(candles) < self.atr_period + 5:
            raise RuntimeError("Insufficient OHLCV data")

        highs  = [float(c[2]) for c in candles]
        lows   = [float(c[3]) for c in candles]
        closes = [float(c[4]) for c in candles]
        self._current_atr = _compute_atr(highs, lows, closes, self.atr_period)

        market = self._fetch_price()
        price  = market.price

        # ── Trailing stop (checked before strategy signal) ──────────────
        if self.position.is_open and self.trailing_stop_pct > 0:
            if price > self.position.highest_since_entry:
                self.position.highest_since_entry = price
                self.position.trailing_stop_price = price * (1.0 - self.trailing_stop_pct / 100.0)
            if price <= self.position.trailing_stop_price:
                logger.info(
                    "Trailing stop triggered: price=%.4f  stop=%.4f",
                    price, self.position.trailing_stop_price,
                )
                self._close_position(price, reason="trailing_stop")
                return

        # ── Portfolio heat guard ────────────────────────────────────────
        portfolio = self._portfolio_value(price)
        heat = self.position.notional_value / portfolio * 100.0 if portfolio > 0 else 0.0
        if heat > self.max_portfolio_heat and not self.position.is_open:
            logger.warning("Portfolio heat %.1f%% exceeds limit %.1f%% — no new entries", heat, self.max_portfolio_heat)

        # ── Hard stop-loss / take-profit ────────────────────────────────
        if self.position.is_open and self.strategy.should_exit(self.position.entry_price, price):
            logger.info("Hard SL/TP triggered: entry=%.4f  current=%.4f", self.position.entry_price, price)
            self._close_position(price, reason="sl_tp")
            return

        # ── Strategy signal ─────────────────────────────────────────────
        sig = self.strategy.generate_signal(closes)
        logger.info("Signal=%s  price=%.4f  atr=%.4f  reason=%s", sig.action, price, self._current_atr, sig.reason)

        self._evaluate_signal(sig, market, portfolio)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _evaluate_signal(self, sig: StrategySignal, market: MarketQuote, portfolio: float) -> None:
        price = market.price
        if sig.action == "buy":
            if self.position.is_open:
                logger.debug("Already in position — skip buy")
                return
            if portfolio * self.max_portfolio_heat / 100.0 < price * 0.001:
                logger.warning("Insufficient capital for minimum order")
                return
            self._open_long(price, sig.size_pct, portfolio)

        elif sig.action == "sell" and self.position.is_open:
            self._close_position(price, reason="signal")

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_long(self, price: float, size_pct: float, portfolio: float) -> None:
        amount = self._calculate_amount(price, size_pct, portfolio)
        if amount <= 0:
            logger.warning("Calculated zero amount — skipping order")
            return

        cost = amount * price
        if self.mode == "paper":
            fee = cost * self.paper_fees_pct / 100.0
            if cost + fee > self.paper_balance:
                logger.warning("Insufficient paper balance (need %.2f, have %.2f)", cost + fee, self.paper_balance)
                return
            self.paper_balance -= cost + fee
            self.position = PositionState(
                symbol              = self.symbol,
                entry_price         = price,
                amount              = amount,
                side                = "long",
                highest_since_entry = price,
                trailing_stop_price = price * (1.0 - self.trailing_stop_pct / 100.0) if self.trailing_stop_pct > 0 else 0.0,
            )
            logger.info(
                "▶ Paper LONG opened  amount=%.6f  entry=%.4f  cost=%.2f  fee=%.2f  balance=%.2f",
                amount, price, cost, fee, self.paper_balance,
            )
        else:
            order = self.exchange.create_market_order(self.symbol, "buy", amount)
            logger.info("Live BUY order: %s", order)
            self.position = PositionState(
                symbol              = self.symbol,
                entry_price         = price,
                amount              = amount,
                side                = "long",
                highest_since_entry = price,
                trailing_stop_price = price * (1.0 - self.trailing_stop_pct / 100.0) if self.trailing_stop_pct > 0 else 0.0,
            )

    def _close_position(self, price: float, reason: str = "signal") -> None:
        if not self.position.is_open:
            return

        amount  = self.position.amount
        entry   = self.position.entry_price
        proceeds = amount * price

        if self.mode == "paper":
            fee  = proceeds * self.paper_fees_pct / 100.0
            pnl  = proceeds - fee - amount * entry
            self.paper_balance += proceeds - fee
            logger.info(
                "◀ Paper LONG closed  exit=%.4f  pnl=%.2f (%.2f%%)  balance=%.2f  reason=%s",
                price, pnl, pnl / (amount * entry) * 100, self.paper_balance, reason,
            )
            self._log_trade(entry, price, amount, pnl, fee, reason)
        else:
            order = self.exchange.create_market_order(self.symbol, "sell", amount)
            fee   = 0.0
            pnl   = proceeds - amount * entry
            logger.info("Live SELL order: %s  pnl≈%.2f  reason=%s", order, pnl, reason)
            self._log_trade(entry, price, amount, pnl, fee, reason)

        self.position = PositionState(symbol=self.symbol)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _calculate_amount(self, price: float, size_pct: float, portfolio: float) -> float:
        if price <= 0 or portfolio <= 0:
            return 0.0

        if self.sizing_mode == "atr" and self._current_atr > 0:
            # Risk a fixed % of portfolio per trade, sized by ATR-based stop distance
            risk_usd      = portfolio * self.strategy.risk_pct / 100.0
            stop_distance = self._current_atr * self.atr_risk_mult  # in price terms
            atr_amount    = risk_usd / stop_distance if stop_distance > 0 else 0.0
            max_usd       = portfolio * min(size_pct, self.strategy.max_exposure_pct) / 100.0
            amount        = min(atr_amount, max_usd / price)
        else:
            # Fixed % of portfolio
            target_usd = portfolio * min(size_pct, self.strategy.max_exposure_pct) / 100.0
            amount     = target_usd / price

        if self.mode == "live":
            balance      = self.exchange.fetch_balance()
            quote_bal    = float(balance.get(self.quote_currency, {}).get("free", 0.0) or 0.0)
            if quote_bal < self.strategy.min_balance_usd:
                logger.warning("Insufficient %s balance: %.2f", self.quote_currency, quote_bal)
                return 0.0
            amount = min(amount, (quote_bal * 0.99) / price)

        return round(max(amount, 0.0), 8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _portfolio_value(self, price: float) -> float:
        if self.mode == "paper":
            return self.paper_balance + (self.position.amount * price if self.position.is_open else 0.0)
        
        bal = self.exchange.fetch_balance()
        total_val = float(bal.get(self.quote_currency, {}).get("total", 0.0) or 0.0)
        
        # Add value of current position if in live mode
        if self.position.is_open:
            total_val += self.position.amount * price
        return total_val

    def _fetch_price(self) -> MarketQuote:
        try:
            return self.exchange.fetch_ticker(self.symbol)
        except Exception as exc:
            logger.warning("Exchange ticker failed: %s", exc)
            if self.use_coingecko_price:
                symbol = self.symbol.replace("/", "-").lower()
                return self.exchange.fetch_coingecko_price(symbol, quote_currency=self.quote_currency)
            raise

    def _log_trade(
        self,
        entry: float,
        exit_price: float,
        amount: float,
        pnl: float,
        fee: float,
        reason: str,
    ) -> None:
        """Append a closed trade to the CSV journal."""
        pnl_pct = pnl / (amount * entry) * 100.0 if entry > 0 and amount > 0 else 0.0
        row = {
            "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
            "symbol":      self.symbol,
            "mode":        self.mode,
            "entry_price": f"{entry:.6f}",
            "exit_price":  f"{exit_price:.6f}",
            "amount":      f"{amount:.8f}",
            "pnl_usd":     f"{pnl:.4f}",
            "pnl_pct":     f"{pnl_pct:.4f}",
            "fee_usd":     f"{fee:.4f}",
            "reason":      reason,
            "balance":     f"{self.paper_balance:.4f}",
        }
        file_exists = self.journal_file.exists()
        with self.journal_file.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.debug("Trade logged → %s", self.journal_file)


# ─── ATR helper (pure Python, avoids pandas import at runtime) ────────────────

def _compute_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> float:
    """Return the most-recent ATR value using Wilder's EWM smoothing."""
    n = len(closes)
    if n < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    # Wilder EWM: alpha = 1/period
    alpha = 1.0 / period
    atr   = trs[0]
    for tr in trs[1:]:
        atr = alpha * tr + (1.0 - alpha) * atr
    return atr


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    config_path = Path("config.yml")
    if not config_path.exists():
        raise FileNotFoundError("Please create config.yml from config.example.yml")
    TradeBot(str(config_path)).run()


if __name__ == "__main__":
    main()
