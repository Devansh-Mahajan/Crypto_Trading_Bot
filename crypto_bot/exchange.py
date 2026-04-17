from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import ccxt
import requests

logger = logging.getLogger(__name__)


@dataclass
class MarketQuote:
    symbol: str
    price: float
    timestamp: int


class ExchangeClient:
    def __init__(self, exchange_id: str, api_key: str | None = None, secret: str | None = None, password: str | None = None, rate_limit: bool = True) -> None:
        exchange_cls = getattr(ccxt, exchange_id.lower(), None)
        if exchange_cls is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        self.exchange = exchange_cls({
            "apiKey": api_key,
            "secret": secret,
            "password": password,
            "enableRateLimit": rate_limit,
        })
        self.exchange.load_markets()
        self.symbols = set(self.exchange.symbols)
        self.session = requests.Session()

    def normalize_symbol(self, symbol: str) -> str:
        if symbol in self.symbols:
            return symbol
        normalized = symbol.replace("-", "/").upper()
        if normalized in self.symbols:
            return normalized
        raise ValueError(f"Symbol not available on exchange: {symbol}")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> list[list[float]]:
        symbol = self.normalize_symbol(symbol)
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> MarketQuote:
        symbol = self.normalize_symbol(symbol)
        ticker = self.exchange.fetch_ticker(symbol)
        return MarketQuote(
            symbol=symbol,
            price=float(ticker["last"]),
            timestamp=int(ticker.get("timestamp", time.time() * 1000)),
        )

    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict[str, Any]:
        symbol = self.normalize_symbol(symbol)
        logger.info("Placing market order %s %s %s", side, amount, symbol)
        return self.exchange.create_order(symbol, "market", side, amount)

    def cancel_all_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            symbol = self.normalize_symbol(symbol)
            open_orders = self.exchange.fetch_open_orders(symbol)
        else:
            open_orders = self.exchange.fetch_open_orders()
        canceled = []
        for order in open_orders:
            try:
                canceled.append(self.exchange.cancel_order(order["id"], order["symbol"]))
            except Exception as exc:
                logger.warning("Failed to cancel order %s: %s", order.get("id"), exc)
        return canceled

    def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return self.exchange.fetch_open_orders(self.normalize_symbol(symbol)) if symbol else self.exchange.fetch_open_orders()

    def fetch_coingecko_price(self, base_symbol: str, quote_currency: str = "usd") -> MarketQuote:
        base = base_symbol.split("/")[0].lower()
        quote = quote_currency.lower()
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={base}&vs_currencies={quote}"
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        if base not in data or quote not in data[base]:
            raise ValueError(f"CoinGecko missing price for {base_symbol}")
        price = float(data[base][quote])
        return MarketQuote(symbol=base_symbol, price=price, timestamp=int(time.time() * 1000))
