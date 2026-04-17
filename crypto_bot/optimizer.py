"""
Parameter optimisation engine for backtesting strategies.

Two modes
---------
Grid Search
  Exhaustive search over every parameter combination in the supplied grid.
  Ranks results by a chosen metric and prints a top-N table.

Walk-Forward Analysis
  Anti-overfitting validation.  Splits the data into expanding train/test
  windows.  On each fold the grid search finds the best in-sample params,
  then those params are evaluated on the out-of-sample test window.
  Reports IS vs OOS degradation so you know whether the strategy is
  curve-fit or genuinely robust.

Usage
-----
python bot.py optimize --symbol BTC/USDT --strategy ema_rsi --metric sharpe
python bot.py optimize --symbol BTC/USDT --strategy ema_rsi --walk-forward --splits 5
python bot.py optimize --symbol BTC/USDT --strategy ema_rsi \\
    --param fast_ema=5,8,12 --param slow_ema=18,21,26
"""
from __future__ import annotations

import itertools
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .backtest import BacktestEngine, _compute_metrics
from .strategies import REGISTRY
from .strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# ─── Default parameter grids (sensible search spaces) ────────────────────────

DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    "ema_rsi": {
        "fast_ema":       [5, 8, 12],
        "slow_ema":       [18, 21, 26, 30],
        "rsi_period":     [10, 14, 21],
        "rsi_overbought": [65, 70, 75],
    },
    "macd": {
        "fast":           [8, 12, 16],
        "slow":           [22, 26, 30],
        "signal_period":  [7, 9, 11],
        "rsi_overbought": [60, 65, 70],
    },
    "bollinger": {
        "bb_period":   [15, 20, 25],
        "bb_std":      [1.5, 2.0, 2.5],
        "rsi_oversold": [35, 40, 45],
        "rsi_exit":    [65, 68, 72],
    },
    "supertrend": {
        "period":        [7, 10, 14],
        "multiplier":    [2.0, 3.0, 4.0],
        "rsi_overbought": [70, 75, 80],
    },
    "momentum": {
        "trend_period":     [30, 50, 100],
        "rsi_oversold":     [30, 35, 40],
        "rsi_overbought":   [60, 65, 70],
        "stoch_oversold":   [20, 25, 30],
        "stoch_overbought": [70, 75, 80],
    },
}

# Metrics where higher is better (others: lower is better)
_HIGHER_IS_BETTER = {
    "sharpe", "sortino", "calmar", "total_return_pct", "cagr_pct",
    "win_rate_pct", "profit_factor", "final_balance",
}

# ─── Result containers ────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    params:  dict[str, Any]
    metrics: dict[str, Any]

    def metric(self, key: str) -> float:
        v = self.metrics.get(key, 0.0)
        return float("inf") if v == float("inf") else float(v or 0.0)


@dataclass
class OptimizationResult:
    strategy_name:  str
    symbol:         str
    metric:         str
    n_combinations: int
    n_valid:        int
    best_params:    dict[str, Any]
    best_metrics:   dict[str, Any]
    ranked:         list[RunRecord]   # all valid results, sorted by metric


@dataclass
class WalkForwardWindow:
    fold:       int
    train_bars: int
    test_bars:  int
    train_from: pd.Timestamp
    train_to:   pd.Timestamp
    test_from:  pd.Timestamp
    test_to:    pd.Timestamp
    best_params: dict[str, Any]
    is_metric:   float
    oos_metric:  float
    oos_return:  float
    oos_trades:  int


@dataclass
class WalkForwardResult:
    strategy_name:       str
    symbol:              str
    metric:              str
    n_splits:            int
    windows:             list[WalkForwardWindow]
    avg_is_metric:       float
    avg_oos_metric:      float
    degradation_pct:     float   # (IS − OOS) / |IS| × 100; lower = more robust
    recommended_params:  dict[str, Any]


# ─── Grid Search ──────────────────────────────────────────────────────────────

class GridSearchOptimizer:
    """
    Exhaustive grid search over all parameter combinations for one strategy.

    Parameters
    ----------
    strategy_name : key in REGISTRY  ('ema_rsi', 'macd', 'bollinger', 'supertrend')
    param_grid    : dict mapping param names to lists of values to try.
                    Defaults to DEFAULT_GRIDS[strategy_name].
    engine_kwargs : extra kwargs forwarded to BacktestEngine (fees, SL/TP, …).
    metric        : optimisation objective  (any key from BacktestResult.metrics).
    top_n         : rows to show in the printed report.
    """

    def __init__(
        self,
        strategy_name:  str,
        param_grid:     dict[str, list] | None = None,
        engine_kwargs:  dict | None = None,
        metric:         str = "sharpe",
        top_n:          int = 10,
    ) -> None:
        if strategy_name not in REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. Choose from: {', '.join(REGISTRY)}"
            )
        self.strategy_name = strategy_name
        self.param_grid    = param_grid or DEFAULT_GRIDS[strategy_name]
        self.engine_kwargs = engine_kwargs or {}
        self.metric        = metric
        self.top_n         = top_n
        self._engine       = BacktestEngine(**self.engine_kwargs)

    def run(self, data: pd.DataFrame) -> OptimizationResult:
        combos      = _expand_grid(self.param_grid)
        n_total     = len(combos)
        valid_combos = [c for c in combos if _is_valid(self.strategy_name, c)]
        n_valid_in  = len(valid_combos)

        logger.info(
            "Grid search: %s  %d combinations (%d valid) → metric=%s",
            self.strategy_name, n_total, n_valid_in, self.metric,
        )

        records: list[RunRecord] = []
        t0 = time.time()

        for i, params in enumerate(valid_combos, 1):
            try:
                strat  = REGISTRY[self.strategy_name](**params)
                result = self._engine.run(data, strat)
                records.append(RunRecord(params=params, metrics=result.metrics))
            except Exception as exc:
                logger.debug("Combo %d failed: %s %s", i, params, exc)
            if i % 50 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / i * (n_valid_in - i)
                logger.info("  %d/%d done  ETA %.0fs", i, n_valid_in, eta)

        if not records:
            raise RuntimeError("No valid parameter combinations produced results.")

        ranked = _sort_records(records, self.metric)
        best   = ranked[0]

        logger.info(
            "Done in %.1fs. Best %s=%.4f  params=%s",
            time.time() - t0, self.metric, best.metric(self.metric), best.params,
        )

        return OptimizationResult(
            strategy_name  = self.strategy_name,
            symbol         = "",
            metric         = self.metric,
            n_combinations = n_total,
            n_valid        = n_valid_in,
            best_params    = best.params,
            best_metrics   = best.metrics,
            ranked         = ranked,
        )

    def print_report(self, result: OptimizationResult, top_n: int | None = None) -> None:
        top_n = top_n or self.top_n
        m     = self.metric
        rows  = result.ranked[:top_n]

        # Collect column headers
        param_keys = list(self.param_grid.keys())
        extra_cols = [
            ("sharpe",           "Sharpe"),
            ("total_return_pct", "Return%"),
            ("max_drawdown_pct", "MaxDD%"),
            ("win_rate_pct",     "WinRate%"),
            ("total_trades",     "Trades"),
        ]
        if m not in {k for k, _ in extra_cols}:
            extra_cols.insert(0, (m, m.replace("_", " ").title()))

        col_w = 9
        lbl_w = max(max(len(k) for k in param_keys), 7) + 1

        SEP  = "═" * 72
        HR   = "─" * 72
        print(f"\n{SEP}")
        print(f"  GRID SEARCH  ·  {result.strategy_name}  ·  {result.symbol}")
        print(f"  Metric: {m}  ·  {result.n_valid} valid combos / {result.n_combinations} total")
        print(SEP)

        # Header
        hdr = f"  {'#':>4}  " + "  ".join(f"{k:<{lbl_w}}" for k in param_keys)
        hdr += "  " + "  ".join(f"{lbl:>{col_w}}" for _, lbl in extra_cols)
        print(hdr)
        print(f"  {HR}")

        for rank, rec in enumerate(rows, 1):
            row = f"  {rank:>4}  "
            row += "  ".join(f"{_fmtv(rec.params.get(k, '')):<{lbl_w}}" for k in param_keys)
            for key, _ in extra_cols:
                v   = rec.metrics.get(key, 0.0)
                row += f"  {_fmtm(key, v):>{col_w}}"
            print(row)

        print(f"\n  ✦ Best params:")
        for k, v in result.best_params.items():
            print(f"      {k} = {v}")

        # Suggest CLI command
        flag_map = {
            "fast_ema": "--fast-ema", "slow_ema": "--slow-ema",
            "rsi_period": "--rsi-period", "rsi_overbought": "--rsi-overbought",
            "fast": "--macd-fast", "slow": "--macd-slow",
            "signal_period": "--macd-signal",
            "bb_period": "--bb-period", "bb_std": "--bb-std",
            "period": "--st-period", "multiplier": "--st-mult",
        }
        flags = " ".join(
            f"{flag_map.get(k, '--' + k.replace('_', '-'))} {v}"
            for k, v in result.best_params.items()
            if k in flag_map
        )
        print(f"\n  Run with: python bot.py backtest --strategy {result.strategy_name} {flags}")
        print(f"{SEP}\n")


# ─── Walk-Forward ─────────────────────────────────────────────────────────────

class WalkForwardOptimizer:
    """
    Walk-forward analysis: anti-overfitting validation.

    Uses an expanding train window so each fold uses all available history.
    The test window is a fixed slice immediately following the train period.

    Parameters
    ----------
    strategy_name : key in REGISTRY
    param_grid    : parameter search space (defaults to DEFAULT_GRIDS[strategy_name])
    engine_kwargs : forwarded to BacktestEngine
    n_splits      : number of train/test windows
    train_pct     : fraction of data used for training in each fold
    metric        : IS optimisation objective
    min_train_bars: minimum bars required for a valid train fold
    """

    def __init__(
        self,
        strategy_name:   str,
        param_grid:      dict[str, list] | None = None,
        engine_kwargs:   dict | None = None,
        n_splits:        int   = 5,
        train_pct:       float = 0.7,
        metric:          str   = "sharpe",
        min_train_bars:  int   = 60,
    ) -> None:
        if strategy_name not in REGISTRY:
            raise ValueError(f"Unknown strategy '{strategy_name}'.")
        self.strategy_name  = strategy_name
        self.param_grid     = param_grid or DEFAULT_GRIDS[strategy_name]
        self.engine_kwargs  = engine_kwargs or {}
        self.n_splits       = n_splits
        self.train_pct      = train_pct
        self.metric         = metric
        self.min_train_bars = min_train_bars
        self._gs            = GridSearchOptimizer(
            strategy_name, self.param_grid, engine_kwargs, metric, top_n=1
        )

    def run(self, data: pd.DataFrame) -> WalkForwardResult:
        n     = len(data)
        # Each fold uses 1/(n_splits+1) of total data as its test window
        chunk = max(n // (self.n_splits + 1), 1)

        windows: list[WalkForwardWindow] = []
        param_votes: list[str] = []

        for fold in range(self.n_splits):
            test_end   = (fold + 2) * chunk
            test_start = test_end - chunk
            train_end  = test_start
            # Train window always starts at 0 (expanding)
            train_data = data.iloc[:train_end]
            test_data  = data.iloc[test_start:test_end]

            if len(train_data) < self.min_train_bars or len(test_data) < 10:
                logger.warning("Fold %d: not enough bars (train=%d, test=%d) — skip",
                               fold + 1, len(train_data), len(test_data))
                continue

            # ── In-sample: grid search on train ──
            try:
                is_result  = self._gs.run(train_data)
                best_p     = is_result.best_params
                is_metric  = is_result.ranked[0].metric(self.metric)
            except Exception as exc:
                logger.warning("Fold %d IS grid search failed: %s", fold + 1, exc)
                continue

            # ── Out-of-sample: apply best params to test ──
            try:
                engine   = BacktestEngine(**self.engine_kwargs)
                strat    = REGISTRY[self.strategy_name](**best_p)
                oos_res  = engine.run(test_data, strat)
                oos_metric = oos_res.metrics.get(self.metric, 0.0) or 0.0
                oos_ret    = oos_res.metrics.get("total_return_pct", 0.0) or 0.0
                oos_trades = oos_res.metrics.get("total_trades", 0) or 0
            except Exception as exc:
                logger.warning("Fold %d OOS evaluation failed: %s", fold + 1, exc)
                continue

            param_votes.append(_params_key(best_p))

            windows.append(WalkForwardWindow(
                fold        = fold + 1,
                train_bars  = len(train_data),
                test_bars   = len(test_data),
                train_from  = train_data.index[0],
                train_to    = train_data.index[-1],
                test_from   = test_data.index[0],
                test_to     = test_data.index[-1],
                best_params = best_p,
                is_metric   = is_metric,
                oos_metric  = float(oos_metric),
                oos_return  = float(oos_ret),
                oos_trades  = int(oos_trades),
            ))
            logger.info(
                "Fold %d  IS %s=%.3f  OOS %s=%.3f  params=%s",
                fold + 1, self.metric, is_metric, self.metric, oos_metric, best_p,
            )

        if not windows:
            raise RuntimeError("Walk-forward produced no valid windows.")

        avg_is  = sum(w.is_metric  for w in windows) / len(windows)
        avg_oos = sum(w.oos_metric for w in windows) / len(windows)
        deg     = (avg_is - avg_oos) / abs(avg_is) * 100 if avg_is != 0 else 0.0

        # Most-voted best params
        most_common_key = Counter(param_votes).most_common(1)[0][0]
        rec_params = _parse_params_key(most_common_key)

        return WalkForwardResult(
            strategy_name       = self.strategy_name,
            symbol              = "",
            metric              = self.metric,
            n_splits            = len(windows),
            windows             = windows,
            avg_is_metric       = avg_is,
            avg_oos_metric      = avg_oos,
            degradation_pct     = deg,
            recommended_params  = rec_params,
        )

    def print_report(self, result: WalkForwardResult) -> None:
        SEP = "═" * 72
        HR  = "─" * 72
        m   = result.metric

        robustness = (
            "✓ Robust"    if result.degradation_pct < 30  else
            "⚠ Moderate"  if result.degradation_pct < 60  else
            "✗ Overfit"
        )

        print(f"\n{SEP}")
        print(f"  WALK-FORWARD ANALYSIS  ·  {result.strategy_name}  ·  {result.symbol}")
        print(f"  Metric: {m}  ·  {result.n_splits} folds")
        print(SEP)

        # Window table
        date_w = 11
        print(
            f"  {'Fold':>4}  {'Train':>{date_w}}→{'':<{date_w}}  "
            f"{'Test':>{date_w}}→{'':<{date_w}}  "
            f"{'IS':>7}  {'OOS':>7}  {'OOSRet%':>8}  {'OOSTrades':>10}  Best Params"
        )
        print(f"  {HR}")

        for w in result.windows:
            tf  = w.train_from.strftime("%Y-%m-%d")
            tt  = w.train_to.strftime("%Y-%m-%d")
            xf  = w.test_from.strftime("%Y-%m-%d")
            xt  = w.test_to.strftime("%Y-%m-%d")
            ps  = "  ".join(f"{k}={v}" for k, v in w.best_params.items())
            s   = "+" if w.oos_return >= 0 else ""
            print(
                f"  {w.fold:>4}  {tf}→{tt}  {xf}→{xt}"
                f"  {w.is_metric:>7.3f}  {w.oos_metric:>7.3f}"
                f"  {s}{w.oos_return:>7.2f}%  {w.oos_trades:>10}  {ps}"
            )

        print(f"\n  {HR}")
        print(f"  Avg IS  {m}: {result.avg_is_metric:.3f}")
        print(f"  Avg OOS {m}: {result.avg_oos_metric:.3f}")
        print(f"  IS→OOS degradation: {result.degradation_pct:.1f}%   {robustness}")
        print(f"\n  Recommended params (most frequent across folds):")
        for k, v in result.recommended_params.items():
            print(f"      {k} = {v}")

        flag_map = {
            "fast_ema": "--fast-ema", "slow_ema": "--slow-ema",
            "rsi_period": "--rsi-period", "rsi_overbought": "--rsi-overbought",
        }
        flags = " ".join(
            f"{flag_map.get(k, '--' + k.replace('_', '-'))} {v}"
            for k, v in result.recommended_params.items()
            if k in flag_map
        )
        print(f"\n  Run with: python bot.py backtest --strategy {result.strategy_name} {flags}")
        print(f"{SEP}\n")


# ─── Convenience entry point ──────────────────────────────────────────────────

def run_optimization(
    data:           pd.DataFrame,
    strategy_name:  str   = "ema_rsi",
    metric:         str   = "sharpe",
    top_n:          int   = 10,
    walk_forward:   bool  = False,
    n_splits:       int   = 5,
    param_overrides: dict[str, list] | None = None,
    engine_kwargs:  dict | None = None,
    symbol:         str   = "",
) -> OptimizationResult | WalkForwardResult:
    """
    Run grid search (default) or walk-forward on `data`.
    Called by the CLI `python bot.py optimize …`.
    """
    grid = {**DEFAULT_GRIDS.get(strategy_name, {}), **(param_overrides or {})}

    if walk_forward:
        opt    = WalkForwardOptimizer(strategy_name, grid, engine_kwargs, n_splits=n_splits, metric=metric)
        result = opt.run(data)
        result.symbol = symbol
        opt.print_report(result)
        return result
    else:
        opt    = GridSearchOptimizer(strategy_name, grid, engine_kwargs, metric=metric, top_n=top_n)
        result = opt.run(data)
        result.symbol = symbol
        opt.print_report(result)
        return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys   = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _is_valid(strategy_name: str, params: dict) -> bool:
    if strategy_name in ("ema_rsi",):
        if params.get("fast_ema", 0) >= params.get("slow_ema", 9999):
            return False
    if strategy_name == "macd":
        if params.get("fast", 0) >= params.get("slow", 9999):
            return False
    return True


def _sort_records(records: list[RunRecord], metric: str) -> list[RunRecord]:
    reverse = metric in _HIGHER_IS_BETTER
    return sorted(records, key=lambda r: r.metric(metric), reverse=reverse)


def _fmtv(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _fmtm(key: str, v: float) -> str:
    if v == float("inf"):
        return "∞"
    if key in ("total_return_pct", "max_drawdown_pct", "win_rate_pct", "cagr_pct"):
        s = "+" if (key != "max_drawdown_pct" and v >= 0) else ""
        return f"{s}{v:.1f}%"
    if key == "total_trades":
        return str(int(v))
    return f"{v:.3f}"


def _params_key(params: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(params.items()))


def _parse_params_key(key: str) -> dict[str, Any]:
    result = {}
    for part in key.split("|"):
        k, _, v = part.partition("=")
        try:
            result[k] = float(v) if "." in v else int(v)
        except ValueError:
            result[k] = v
    return result
