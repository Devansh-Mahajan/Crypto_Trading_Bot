from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from ..indicators import supertrend as supertrend_indicator, rsi


class SupertrendStrategy(BaseStrategy):
    """
    SuperTrend trend-following with RSI entry filter.

    Entry:  SuperTrend flips bearish → bullish (direction: -1 → +1)
            AND  RSI is not overbought (avoid chasing exhausted moves)
    Exit:   SuperTrend flips bullish → bearish (direction: +1 → -1)

    SuperTrend is ATR-based, so it self-adjusts to market volatility.
    The multiplier controls sensitivity: larger = fewer but more confident signals.
    """

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        rsi_period: int = 14,
        rsi_overbought: float = 75.0,
    ) -> None:
        self.period = period
        self.multiplier = multiplier
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought

    @property
    def name(self) -> str:
        return f"SuperTrend({self.period},{self.multiplier}) + RSI({self.rsi_period})"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        idx = data.index
        close = data["close"]

        st = supertrend_indicator(
            data["high"], data["low"], close,
            period=self.period,
            multiplier=self.multiplier,
            index=idx,
        )
        rsi_vals = rsi(close, self.rsi_period, index=idx)
        direction = st["direction"]

        flip_bullish = (direction == 1) & (direction.shift(1) == -1)
        flip_bearish = (direction == -1) & (direction.shift(1) == 1)

        entry = flip_bullish & (rsi_vals < self.rsi_overbought)
        exit_ = flip_bearish

        return pd.DataFrame({
            "entry":      entry.fillna(False),
            "exit":       exit_.fillna(False),
            "supertrend": st["supertrend"],
            "direction":  direction,
            "rsi":        rsi_vals,
        }, index=idx)
