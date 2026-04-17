from .ema_rsi import EMARSIStrategy
from .macd import MACDStrategy
from .bollinger import BollingerStrategy
from .supertrend import SupertrendStrategy
from .momentum import MomentumStrategy
from .base import BaseStrategy

REGISTRY: dict[str, type[BaseStrategy]] = {
    "ema_rsi":    EMARSIStrategy,
    "macd":       MACDStrategy,
    "bollinger":  BollingerStrategy,
    "supertrend": SupertrendStrategy,
    "momentum":   MomentumStrategy,
}

__all__ = [
    "BaseStrategy",
    "EMARSIStrategy",
    "MACDStrategy",
    "BollingerStrategy",
    "SupertrendStrategy",
    "MomentumStrategy",
    "REGISTRY",
]
