"""
CCXT-based OHLCV data fetcher for backtesting and live data.

Advantages over CoinGecko:
  - Any timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w …
  - Full native OHLCV including volume
  - Much longer history  (Binance: 3+ years of 1h data)
  - No API key required for public historical endpoints on most exchanges

Supported exchanges (public OHLCV, no key): binance, bybit, okx, kraken,
  kucoin, gate, mexc, bitget, coinbase, bitstamp, and 100+ more.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# Milliseconds per timeframe — used for pagination
_TF_MS: dict[str, int] = {
    "1m":  60_000,       "3m":  180_000,      "5m":  300_000,
    "15m": 900_000,      "30m": 1_800_000,    "1h":  3_600_000,
    "2h":  7_200_000,    "4h":  14_400_000,   "6h":  21_600_000,
    "8h":  28_800_000,   "12h": 43_200_000,
    "1d":  86_400_000,   "3d":  259_200_000,  "1w":  604_800_000,
}


class CCXTData:
    """
    Paginated OHLCV data client built on CCXT.

    No API key is required for fetching public historical candles on
    Binance, Bybit, OKX, Kraken, KuCoin, and most other major venues.

    Example
    -------
    >>> client = CCXTData("binance")
    >>> df = client.fetch_ohlcv("BTC/USDT", timeframe="4h", bars=500)
    """

    def __init__(self, exchange_id: str = "binance") -> None:
        cls = getattr(ccxt, exchange_id.lower(), None)
        if cls is None:
            raise ValueError(
                f"Unknown exchange '{exchange_id}'. "
                f"Run: python -c \"import ccxt; print(ccxt.exchanges)\" for the full list."
            )
        self.exchange = cls({"enableRateLimit": True})
        self.exchange_id = exchange_id
        logger.info("CCXTData client: %s", exchange_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        bars: int = 500,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with DatetimeIndex (UTC) and float columns:
          open, high, low, close, volume

        Paginates automatically until `bars` candles are collected.
        Signal at most one CCXT request per exchange rate-limit window.

        Parameters
        ----------
        symbol    : e.g. "BTC/USDT"
        timeframe : one of 1m 5m 15m 30m 1h 4h 1d 1w (exchange-dependent)
        bars      : target number of candles (most recent)
        """
        if timeframe not in _TF_MS:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. Supported: {', '.join(_TF_MS)}"
            )

        tf_ms       = _TF_MS[timeframe]
        per_call    = min(getattr(self.exchange, "ohlcvLimit", 1_000), 1_000)
        rate_sleep  = self.exchange.rateLimit / 1_000  # seconds

        # Compute start timestamp so we fetch at least `bars` candles back
        since_ms = int(time.time() * 1_000) - (bars + 10) * tf_ms

        logger.info(
            "Fetching %d × %s OHLCV for %s on %s …",
            bars, timeframe, symbol, self.exchange_id,
        )

        raw: list[list] = []
        fetch_since = since_ms

        while len(raw) < bars:
            need  = bars - len(raw)
            limit = min(need + 5, per_call)
            try:
                batch = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=fetch_since, limit=limit
                )
            except ccxt.BaseError as exc:
                if raw:
                    logger.warning(
                        "Fetch error after %d candles — stopping pagination: %s", len(raw), exc
                    )
                    break
                raise
            if not batch:
                break
            raw.extend(batch)
            if len(batch) < limit:
                break
            fetch_since = batch[-1][0] + tf_ms
            time.sleep(rate_sleep)

        if not raw:
            raise ValueError(
                f"No OHLCV data returned for {symbol} on {self.exchange_id} ({timeframe})"
            )

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.tail(bars).astype(float)

        logger.info(
            "  %d candles  %s → %s",
            len(df),
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
        )
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def list_timeframes(self) -> list[str]:
        """Return timeframes supported by the connected exchange."""
        try:
            self.exchange.load_markets()
            return list(self.exchange.timeframes or {})
        except Exception:
            return list(_TF_MS)

    @staticmethod
    def list_exchanges() -> list[str]:
        """Return all CCXT exchange IDs that support public OHLCV."""
        return sorted(
            eid for eid in ccxt.exchanges
            if hasattr(getattr(ccxt, eid), "fetch_ohlcv")
        )
