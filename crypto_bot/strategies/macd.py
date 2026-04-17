from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from ..indicators import macd as macd_indicator, rsi


class MACDStrategy(BaseStrategy):
    """
    MACD Histogram crossover with RSI filter  —  momentum.

    Entry:  MACD histogram crosses zero from below (negative → positive)
            AND  RSI < overbought threshold
    Exit:   MACD histogram crosses zero from above (positive → negative)
            OR   RSI > exit threshold

    The histogram zero-cross is a stronger momentum signal than the MACD/signal
    line crossover because it already incorporates the signal smoothing.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
        rsi_period: int = 14,
        rsi_overbought: float = 65.0,
        rsi_exit: float = 75.0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_exit = rsi_exit

    @property
    def name(self) -> str:
        return f"MACD({self.fast}/{self.slow}/{self.signal_period}) + RSI({self.rsi_period})"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        idx = data.index
        close = data["close"]

        m = macd_indicator(close, self.fast, self.slow, self.signal_period, index=idx)
        hist = m["hist"]
        rsi_vals = rsi(close, self.rsi_period, index=idx)

        hist_cross_up = (hist > 0) & (hist.shift(1) <= 0)
        hist_cross_dn = (hist < 0) & (hist.shift(1) >= 0)

        entry = hist_cross_up & (rsi_vals < self.rsi_overbought)
        exit_ = hist_cross_dn | (rsi_vals > self.rsi_exit)

        return pd.DataFrame({
            "entry":  entry.fillna(False),
            "exit":   exit_.fillna(False),
            "macd":   m["macd"],
            "signal": m["signal"],
            "hist":   hist,
            "rsi":    rsi_vals,
        }, index=idx)
