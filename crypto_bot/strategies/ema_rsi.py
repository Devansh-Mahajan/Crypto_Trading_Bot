from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from ..indicators import ema, rsi


class EMARSIStrategy(BaseStrategy):
    """
    EMA Crossover with RSI filter  —  trend-following.

    Entry:  fast EMA crosses above slow EMA  AND  RSI < overbought threshold
    Exit:   fast EMA crosses below slow EMA  OR   RSI > exit threshold

    The RSI filter prevents buying into already-overbought conditions.
    """

    def __init__(
        self,
        fast_ema: int = 8,
        slow_ema: int = 21,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_exit: float = 75.0,
    ) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_exit = rsi_exit

    @property
    def name(self) -> str:
        return f"EMA({self.fast_ema}/{self.slow_ema}) + RSI({self.rsi_period})"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        idx = data.index
        close = data["close"]

        fast = ema(close, self.fast_ema, index=idx)
        slow = ema(close, self.slow_ema, index=idx)
        rsi_vals = rsi(close, self.rsi_period, index=idx)

        bullish_cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        bearish_cross = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        entry = bullish_cross & (rsi_vals < self.rsi_overbought)
        exit_ = bearish_cross | (rsi_vals > self.rsi_exit)

        return pd.DataFrame({
            "entry":    entry.fillna(False),
            "exit":     exit_.fillna(False),
            "fast_ema": fast,
            "slow_ema": slow,
            "rsi":      rsi_vals,
        }, index=idx)
