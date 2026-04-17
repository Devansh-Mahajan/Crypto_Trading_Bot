from __future__ import annotations

import pandas as pd

from .base import BaseStrategy
from ..indicators import bollinger_bands, rsi


class BollingerStrategy(BaseStrategy):
    """
    Bollinger Band mean-reversion with RSI confirmation.

    Entry:  price was below the lower band last bar (oversold touch)
            AND price closes back above the lower band this bar (reversion starts)
            AND RSI was below the oversold threshold (confirms weakness, not breakdown)
    Exit:   price reaches 99 % of the upper band (target reached)
            OR RSI > exit threshold (momentum exhausted)
            OR price falls back below the lower band (trade invalidated)

    Works best on range-bound / mean-reverting markets.
    On 4h candles the 20-period window covers ~3.3 days — appropriate.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 40.0,
        rsi_exit: float = 68.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_exit = rsi_exit

    @property
    def name(self) -> str:
        return f"BB({self.bb_period},{self.bb_std}) + RSI({self.rsi_period})"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        idx = data.index
        close = data["close"]

        bb = bollinger_bands(close, self.bb_period, self.bb_std, index=idx)
        rsi_vals = rsi(close, self.rsi_period, index=idx)

        upper = bb["upper"]
        lower = bb["lower"]

        was_below_lower = close.shift(1) < lower.shift(1)
        now_above_lower = close > lower
        rsi_was_low = rsi_vals.shift(1) < self.rsi_oversold

        entry = was_below_lower & now_above_lower & rsi_was_low

        at_upper = close >= upper * 0.99
        rsi_high = rsi_vals > self.rsi_exit
        fell_back = close < lower  # stop — reversion failed

        exit_ = at_upper | rsi_high | fell_back

        return pd.DataFrame({
            "entry":    entry.fillna(False),
            "exit":     exit_.fillna(False),
            "bb_upper": upper,
            "bb_mid":   bb["mid"],
            "bb_lower": lower,
            "pct_b":    bb["pct_b"],
            "rsi":      rsi_vals,
        }, index=idx)
