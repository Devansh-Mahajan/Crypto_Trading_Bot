from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from ..indicators import rsi, sma, stochastic, ema


class MomentumStrategy(BaseStrategy):
    """
    Oversold-bounce momentum strategy with trend filter.

    Logic
    -----
    Only trades in the direction of the prevailing trend (price above/below
    the slow SMA). Inside that trend, enters when two independent oscillators
    agree the market is oversold, and exits when both flip overbought.

    Entry (long):
      price > sma(trend_period)        — we are in an uptrend
      RSI  < rsi_oversold              — momentum oversold
      Stochastic %K < stoch_oversold   — stochastic confirms oversold

    Exit:
      RSI  > rsi_overbought            — momentum exhausted
      OR Stochastic %K > stoch_overbought
      OR price < sma(trend_period)     — trend has reversed

    Why it complements the other strategies
    ----------------------------------------
    The EMA/MACD/SuperTrend strategies are trend-following — they enter on
    the start of a trend and ride it.  This strategy is mean-reverting —
    it fades short-term weakness inside an established trend, targeting
    quick recoveries.  Running both together in `--strategy all` gives
    diversified signal exposure.
    """

    def __init__(
        self,
        trend_period:     int   = 50,
        rsi_period:       int   = 14,
        rsi_oversold:     float = 35.0,
        rsi_overbought:   float = 65.0,
        stoch_k:          int   = 14,
        stoch_d:          int   = 3,
        stoch_oversold:   float = 25.0,
        stoch_overbought: float = 75.0,
    ) -> None:
        self.trend_period     = trend_period
        self.rsi_period       = rsi_period
        self.rsi_oversold     = rsi_oversold
        self.rsi_overbought   = rsi_overbought
        self.stoch_k          = stoch_k
        self.stoch_d          = stoch_d
        self.stoch_oversold   = stoch_oversold
        self.stoch_overbought = stoch_overbought

    @property
    def name(self) -> str:
        return (
            f"Momentum(SMA{self.trend_period}·RSI{self.rsi_period}"
            f"·Stoch{self.stoch_k})"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        idx   = data.index
        close = data["close"]
        high  = data["high"]
        low   = data["low"]

        trend     = sma(close, self.trend_period, index=idx)
        rsi_vals  = rsi(close, self.rsi_period, index=idx)
        stoch     = stochastic(high, low, close, self.stoch_k, self.stoch_d, index=idx)
        k         = stoch["k"]

        in_uptrend  = close > trend

        entry = (
            in_uptrend
            & (rsi_vals  < self.rsi_oversold)
            & (k         < self.stoch_oversold)
        )

        exit_ = (
            (rsi_vals > self.rsi_overbought)
            | (k       > self.stoch_overbought)
            | (~in_uptrend)
        )

        return pd.DataFrame({
            "entry":   entry.fillna(False),
            "exit":    exit_.fillna(False),
            "sma":     trend,
            "rsi":     rsi_vals,
            "stoch_k": k,
            "stoch_d": stoch["d"],
        }, index=idx)
