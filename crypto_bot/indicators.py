"""
Vectorized technical indicators.

All functions accept array-like inputs (pd.Series or np.ndarray) and return
pd.Series.  Every computation is backward-looking only — no look-ahead bias.
Results align with TradingView's default settings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[pd.Series, np.ndarray]


def _arr(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(x, dtype=float)


def _s(arr: np.ndarray, index=None, name: str | None = None) -> pd.Series:
    return pd.Series(arr, index=index, name=name, dtype=float)


# ─── Trend ────────────────────────────────────────────────────────────────────

def ema(close: ArrayLike, period: int, index=None) -> pd.Series:
    """Exponential Moving Average (pandas EWM, adjust=False)."""
    return pd.Series(_arr(close), index=index).ewm(span=period, adjust=False).mean()


def sma(close: ArrayLike, period: int, index=None) -> pd.Series:
    """Simple Moving Average."""
    return pd.Series(_arr(close), index=index).rolling(period, min_periods=period).mean()


def dema(close: ArrayLike, period: int, index=None) -> pd.Series:
    """Double EMA (reduces lag)."""
    e = ema(close, period, index=index)
    return 2 * e - ema(e, period, index=index)


# ─── Momentum ─────────────────────────────────────────────────────────────────

def rsi(close: ArrayLike, period: int = 14, index=None) -> pd.Series:
    """
    Wilder's RSI via EWM (matches TradingView).
    Values for the first `period` bars are NaN.
    """
    c = _arr(close)
    delta = np.diff(c, prepend=c[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    avg_up = pd.Series(up).ewm(com=period - 1, adjust=False).mean().to_numpy()
    avg_dn = pd.Series(dn).ewm(com=period - 1, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_dn > 0, avg_up / avg_dn, np.inf)
        vals = np.where(avg_dn > 0, 100.0 - 100.0 / (1.0 + rs), 100.0)
    vals[:period] = np.nan
    return _s(vals, index=index, name="rsi")


def macd(
    close: ArrayLike,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    index=None,
) -> dict[str, pd.Series]:
    """
    MACD line, signal line, and histogram.
    Returns dict with keys: 'macd', 'signal', 'hist'.
    """
    c = pd.Series(_arr(close), index=index)
    macd_line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}


def stochastic(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_period: int = 14,
    d_period: int = 3,
    index=None,
) -> dict[str, pd.Series]:
    """
    Stochastic Oscillator %K and %D.
    Returns dict with keys: 'k', 'd'.
    """
    h = pd.Series(_arr(high), index=index)
    l = pd.Series(_arr(low), index=index)
    c = pd.Series(_arr(close), index=index)
    k = 100.0 * (c - l.rolling(k_period).min()) / (h.rolling(k_period).max() - l.rolling(k_period).min())
    d = k.rolling(d_period).mean()
    return {"k": k, "d": d}


def cci(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 20,
    index=None,
) -> pd.Series:
    """Commodity Channel Index."""
    h = _arr(high)
    l = _arr(low)
    c = _arr(close)
    tp = (h + l + c) / 3.0
    s = pd.Series(tp, index=index)
    mean_dev = s.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return _s((tp - s.rolling(period).mean().to_numpy()) / (0.015 * mean_dev.to_numpy()), index=index, name="cci")


# ─── Volatility ───────────────────────────────────────────────────────────────

def atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
    index=None,
) -> pd.Series:
    """Average True Range using Wilder's EWM smoothing."""
    h = _arr(high)
    l = _arr(low)
    c = _arr(close)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr, index=index).ewm(com=period - 1, adjust=False).mean()


def bollinger_bands(
    close: ArrayLike,
    period: int = 20,
    std_dev: float = 2.0,
    index=None,
) -> dict[str, pd.Series]:
    """
    Bollinger Bands.
    Returns dict with keys: 'upper', 'mid', 'lower', 'pct_b', 'width'.
      pct_b  = (close - lower) / (upper - lower)
      width  = (upper - lower) / mid  (normalized bandwidth)
    """
    c = pd.Series(_arr(close), index=index)
    mid = c.rolling(period, min_periods=period).mean()
    sigma = c.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std_dev * sigma
    lower = mid - std_dev * sigma
    band_width = upper - lower
    pct_b = (c - lower) / band_width
    width = band_width / mid
    return {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b, "width": width}


def supertrend(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 10,
    multiplier: float = 3.0,
    index=None,
) -> dict[str, pd.Series]:
    """
    SuperTrend indicator.

    direction: +1 = price above SuperTrend (uptrend / bullish)
               -1 = price below SuperTrend (downtrend / bearish)

    Algorithm matches TradingView's built-in SuperTrend.
    The sequential update loop is necessary because each bar's final
    band depends on the previous bar's final band.
    """
    h = _arr(high)
    l = _arr(low)
    c = _arr(close)
    n = len(c)

    atr_vals = atr(h, l, c, period).to_numpy()
    hl2 = (h + l) / 2.0

    basic_upper = hl2 + multiplier * atr_vals
    basic_lower = hl2 - multiplier * atr_vals

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int8)

    # Warm-up: initialise at first bar where ATR is stable
    start = max(period, 1)
    if start >= n:
        return {
            "supertrend": _s(st, index=index, name="supertrend"),
            "direction": _s(direction.astype(float), index=index, name="direction"),
        }

    st[start] = final_upper[start]
    direction[start] = -1  # begin bearish

    for i in range(start + 1, n):
        # Final upper band: only tighten if price was above it last bar
        if basic_upper[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band: only raise if price was below it last bar
        if basic_lower[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # SuperTrend value follows the active band
        prev_st = st[i - 1]
        if np.isnan(prev_st) or prev_st >= final_upper[i - 1]:
            st[i] = final_lower[i] if c[i] > final_upper[i] else final_upper[i]
        else:
            st[i] = final_upper[i] if c[i] < final_lower[i] else final_lower[i]

        direction[i] = 1 if c[i] > st[i] else -1

    return {
        "supertrend": _s(st, index=index, name="supertrend"),
        "direction": _s(direction.astype(float), index=index, name="direction"),
    }
