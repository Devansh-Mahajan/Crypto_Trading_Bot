"""
Root entrypoint.

Subcommands
-----------
  python bot.py backtest   [options]  — backtest with CoinGecko or CCXT data
  python bot.py optimize   [options]  — grid-search or walk-forward optimisation
  python bot.py live       [options]  — live / paper trading bot
  python bot.py            (no args)  — defaults to live bot
"""
from __future__ import annotations

import argparse
import logging
import sys


# ─── backtest ────────────────────────────────────────────────────────────────

def _add_backtest_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "backtest",
        help="Backtest a strategy. Use --exchange for CCXT data, or CoinGecko by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_data_args(p)
    _add_common_engine_args(p)
    p.add_argument("--strategy", default="ema_rsi",
                   choices=["ema_rsi", "macd", "bollinger", "supertrend", "all"],
                   help="Strategy to run. 'all' prints a side-by-side comparison table.")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logs")


def _run_backtest(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    from crypto_bot.backtest import run_backtest
    run_backtest(
        symbol          = args.symbol,
        strategy_name   = args.strategy,
        exchange        = args.exchange or None,
        timeframe       = args.timeframe,
        bars            = args.bars,
        days            = args.days,
        vs_currency     = args.currency,
        api_key         = args.api_key,
        initial_capital = args.capital,
        fees_pct        = args.fees,
        slippage_pct    = args.slippage,
        position_pct    = args.position_pct,
        stop_loss_pct   = args.stop_loss   if args.stop_loss   > 0 else None,
        take_profit_pct = args.take_profit if args.take_profit > 0 else None,
    )


# ─── optimize ────────────────────────────────────────────────────────────────

def _add_optimize_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "optimize",
        help="Grid-search or walk-forward parameter optimisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_data_args(p)
    _add_common_engine_args(p)
    p.add_argument("--strategy", default="ema_rsi",
                   choices=["ema_rsi", "macd", "bollinger", "supertrend"],
                   help="Strategy whose parameters are optimised.")
    p.add_argument("--metric", default="sharpe",
                   choices=["sharpe", "sortino", "calmar", "total_return_pct",
                            "cagr_pct", "win_rate_pct", "profit_factor", "max_drawdown_pct"],
                   help="Optimisation objective (higher is always better internally).")
    p.add_argument("--top-n", type=int, default=10, metavar="N",
                   help="Rows shown in the grid-search ranking table.")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward analysis instead of plain grid search.")
    p.add_argument("--splits", type=int, default=5, metavar="N",
                   help="Number of walk-forward folds.")
    p.add_argument("--param", action="append", default=[], metavar="NAME=v1,v2,v3",
                   help="Override a parameter range, e.g. --param fast_ema=5,8,12. "
                        "Repeat for multiple params.")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logs")


def _run_optimize(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse --param NAME=v1,v2,v3 overrides
    param_overrides: dict[str, list] = {}
    for spec in args.param:
        name, _, values_str = spec.partition("=")
        raw_vals = [v.strip() for v in values_str.split(",") if v.strip()]
        parsed: list = []
        for v in raw_vals:
            try:
                parsed.append(float(v) if "." in v else int(v))
            except ValueError:
                parsed.append(v)
        param_overrides[name.strip()] = parsed

    # Engine kwargs for the optimizer
    engine_kwargs: dict = dict(
        initial_capital  = args.capital,
        fees_pct         = args.fees,
        slippage_pct     = args.slippage,
        position_pct     = args.position_pct,
        stop_loss_pct    = args.stop_loss   if args.stop_loss   > 0 else None,
        take_profit_pct  = args.take_profit if args.take_profit > 0 else None,
    )

    # Fetch data
    if args.exchange:
        from crypto_bot.data_ccxt import CCXTData
        data = CCXTData(args.exchange).fetch_ohlcv(args.symbol, timeframe=args.timeframe, bars=args.bars)
    else:
        from crypto_bot.data import CoinGeckoData
        data = CoinGeckoData(api_key=args.api_key).fetch_ohlcv(
            args.symbol, days=args.days, vs_currency=args.currency
        )

    from crypto_bot.optimizer import run_optimization
    run_optimization(
        data            = data,
        strategy_name   = args.strategy,
        metric          = args.metric,
        top_n           = args.top_n,
        walk_forward    = args.walk_forward,
        n_splits        = args.splits,
        param_overrides = param_overrides or None,
        engine_kwargs   = engine_kwargs,
        symbol          = args.symbol,
    )


# ─── live ────────────────────────────────────────────────────────────────────

def _add_live_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "live",
        help="Run live / paper trading bot (requires config.yml).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yml", metavar="PATH")


def _run_live(_args: argparse.Namespace) -> None:
    from crypto_bot.bot import main
    main()


# ─── Shared argument groups ───────────────────────────────────────────────────

def _add_common_data_args(p: argparse.ArgumentParser) -> None:
    dg = p.add_argument_group("data source")
    # CoinGecko (default)
    dg.add_argument("--symbol",   default="BTC/USDT", metavar="PAIR",
                    help="Trading pair, e.g. ETH/USDT")
    dg.add_argument("--days",     type=int, default=90, metavar="N",
                    help="[CoinGecko] Days of history (≤30d→4h, 31+d→4-day candles on free tier)")
    dg.add_argument("--currency", default="usd", metavar="CCY",
                    help="[CoinGecko] Quote currency (usd, eur, btc …)")
    dg.add_argument("--api-key",  default=None, metavar="KEY",
                    help="[CoinGecko] Optional API key to raise rate limits")
    # CCXT (activated by --exchange)
    dg.add_argument("--exchange",  default=None, metavar="ID",
                    help="[CCXT] Exchange ID (binance, bybit, okx …). Activates CCXT data source.")
    dg.add_argument("--timeframe", default="4h", metavar="TF",
                    help="[CCXT] Candle timeframe: 1m 5m 15m 1h 4h 1d 1w")
    dg.add_argument("--bars",      type=int, default=500, metavar="N",
                    help="[CCXT] Number of candles to fetch")


def _add_common_engine_args(p: argparse.ArgumentParser) -> None:
    eg = p.add_argument_group("engine")
    eg.add_argument("--capital",      type=float, default=10_000.0, metavar="USD")
    eg.add_argument("--fees",         type=float, default=0.1,      metavar="PCT",
                    help="Fee %% per side (0.1 = 0.1 %%)")
    eg.add_argument("--slippage",     type=float, default=0.05,     metavar="PCT")
    eg.add_argument("--position-pct", type=float, default=95.0,     metavar="PCT",
                    help="Capital deployed per trade (%%)")
    eg.add_argument("--stop-loss",    type=float, default=5.0,      metavar="PCT",
                    help="Stop-loss below entry (%%, 0=disabled)")
    eg.add_argument("--take-profit",  type=float, default=10.0,     metavar="PCT",
                    help="Take-profit above entry (%%, 0=disabled)")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bot",
        description=(
            "Crypto Trading Bot  ·  backtesting (CoinGecko / CCXT)  ·  "
            "parameter optimisation  ·  live / paper trading"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    _add_backtest_parser(sub)
    _add_optimize_parser(sub)
    _add_live_parser(sub)

    args = parser.parse_args()

    if args.command == "backtest":
        _run_backtest(args)
    elif args.command == "optimize":
        _run_optimize(args)
    else:
        _run_live(args)


if __name__ == "__main__":
    main()
