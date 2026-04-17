import os
from pathlib import Path
from typing import Any, Dict

import yaml


class BotConfig:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or "config.yml"
        self.raw = self._load_config()
        self.exchange = self.raw.get("exchange", "binance")
        self.api_key = self._env_or_value("api_key")
        self.secret = self._env_or_value("secret")
        self.password = self._env_or_value("password")
        self.symbol = self.raw.get("symbol", "BTC/USDT")
        self.timeframe = self.raw.get("timeframe", "1h")
        self.trade = self.raw.get("trade", {})
        self.strategy = self.raw.get("strategy", {})
        self.logging = self.raw.get("logging", {})
        self.paper = self.raw.get("paper", {})

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(self.path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data

    def _env_or_value(self, key: str) -> str | None:
        env_key = key.upper()
        if env_key in os.environ:
            return os.environ[env_key]
        return self.raw.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)
