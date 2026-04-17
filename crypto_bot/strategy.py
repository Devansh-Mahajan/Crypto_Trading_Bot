from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class StrategySignal:
    action: str
    reason: str
    size_pct: float = 0.0


def calculate_ema(values: Iterable[float], period: int) -> list[float]:
    values_list = list(values)
    if period <= 0 or period > len(values_list):
        raise ValueError("EMA period must be positive and <= number of values")

    multiplier = 2 / (period + 1)
    ema = [sum(values_list[:period]) / period]
    for value in values_list[period:]:
        ema.append((value - ema[-1]) * multiplier + ema[-1])
    return ema


def calculate_rsi(values: list[float], period: int = 14) -> list[float]:
    """Wilder's RSI. Returns -1 for positions without enough history."""
    if len(values) < period + 1:
        return [-1.0] * len(values)

    rsi: list[float] = [-1.0] * period
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi.append(100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi.append(100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    return rsi


class EMACrossoverStrategy:
    """
    EMA crossover strategy with RSI filter.

    Entry:  fast EMA crosses above slow EMA  AND  RSI < rsi_overbought
    Exit:   fast EMA crosses below slow EMA  (stop-loss / take-profit handled separately)
    """

    def __init__(
        self,
        fast_ema: int = 8,
        slow_ema: int = 21,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        risk_pct: float = 1.0,
        stop_loss_pct: float = 1.5,
        take_profit_pct: float = 3.0,
        max_exposure_pct: float = 3.0,
        min_balance_usd: float = 20,
        use_rsi: bool = True,
    ) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.risk_pct = risk_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_exposure_pct = max_exposure_pct
        self.min_balance_usd = min_balance_usd
        self.use_rsi = use_rsi

    def generate_signal(self, closes: list[float]) -> StrategySignal:
        required = max(self.fast_ema, self.slow_ema, self.rsi_period) + 3
        if len(closes) < required:
            return StrategySignal("hold", "Not enough data")

        fast = calculate_ema(closes, self.fast_ema)
        slow = calculate_ema(closes, self.slow_ema)
        if len(fast) < 3 or len(slow) < 3:
            return StrategySignal("hold", "EMA calculation incomplete")

        fast_last, fast_prev = fast[-1], fast[-2]
        slow_last, slow_prev = slow[-1], slow[-2]

        rsi_last = -1.0
        if self.use_rsi:
            rsi_vals = calculate_rsi(closes, self.rsi_period)
            rsi_last = rsi_vals[-1]

        # Bullish EMA crossover
        if fast_prev <= slow_prev and fast_last > slow_last:
            if self.use_rsi and rsi_last >= 0 and rsi_last >= self.rsi_overbought:
                return StrategySignal(
                    "hold",
                    f"EMA bullish crossover blocked — RSI overbought ({rsi_last:.1f} >= {self.rsi_overbought})",
                )
            rsi_note = f", RSI={rsi_last:.1f}" if rsi_last >= 0 else ""
            return StrategySignal(
                "buy",
                f"Fast EMA crossed above slow EMA ({self.fast_ema}/{self.slow_ema}){rsi_note}",
                size_pct=self.risk_pct,
            )

        # Bearish EMA crossover
        if fast_prev >= slow_prev and fast_last < slow_last:
            rsi_note = f", RSI={rsi_last:.1f}" if rsi_last >= 0 else ""
            return StrategySignal(
                "sell",
                f"Fast EMA crossed below slow EMA ({self.fast_ema}/{self.slow_ema}){rsi_note}",
                size_pct=self.risk_pct,
            )

        return StrategySignal("hold", "No crossover signal")

    def should_exit(self, entry_price: float, current_price: float) -> bool:
        if entry_price <= 0:
            return False
        drawdown = (entry_price - current_price) / entry_price if current_price < entry_price else 0
        gain = (current_price - entry_price) / entry_price if current_price > entry_price else 0
        if drawdown >= self.stop_loss_pct / 100:
            return True
        if gain >= self.take_profit_pct / 100:
            return True
        return False
