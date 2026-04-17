from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Maps ticker base symbol → CoinGecko coin ID.
# Free-tier OHLC granularity: ≤2d → 30-min, 3-30d → 4-hour, 31+d → 4-day.
COINGECKO_ID_MAP: dict[str, str] = {
    "BTC":   "bitcoin",
    "ETH":   "ethereum",
    "SOL":   "solana",
    "BNB":   "binancecoin",
    "ADA":   "cardano",
    "XRP":   "ripple",
    "DOGE":  "dogecoin",
    "DOT":   "polkadot",
    "MATIC": "matic-network",
    "POL":   "matic-network",
    "AVAX":  "avalanche-2",
    "LINK":  "chainlink",
    "UNI":   "uniswap",
    "LTC":   "litecoin",
    "ATOM":  "cosmos",
    "NEAR":  "near",
    "ALGO":  "algorand",
    "XLM":   "stellar",
    "VET":   "vechain",
    "SHIB":  "shiba-inu",
    "TRX":   "tron",
    "TON":   "the-open-network",
    "APT":   "aptos",
    "ARB":   "arbitrum",
    "OP":    "optimism",
    "INJ":   "injective-protocol",
    "SUI":   "sui",
    "FET":   "fetch-ai",
    "PEPE":  "pepe",
    "WIF":   "dogwifcoin",
    "BONK":  "bonk",
    "NOT":   "notcoin",
    "RENDER":"render-token",
    "SEI":   "sei-network",
    "TAO":   "bittensor",
}

_GRANULARITY_LABEL = {
    "1": "30-min", "2": "30-min",
    "7": "4-hour", "14": "4-hour", "30": "4-hour",
    "90": "4-hour",
    "180": "4-day", "365": "4-day", "max": "4-day",
}


class CoinGeckoData:
    """
    CoinGecko data client.

    Fetches OHLCV data for any coin via the free public API.
    Merges the OHLC endpoint (high/low needed for ATR, SuperTrend)
    with volume data from the market_chart endpoint, resampled to the
    same candle frequency.

    No API key required for free-tier usage.
    """

    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._session = requests.Session()
        hdrs = {"User-Agent": "crypto-backtest-engine/2.0", "Accept": "application/json"}
        if api_key:
            hdrs["x-cg-demo-api-key"] = api_key
        self._session.headers.update(hdrs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def symbol_to_id(self, symbol: str) -> str:
        """'BTC/USDT' → 'bitcoin'."""
        base = symbol.split("/")[0].upper()
        coin_id = COINGECKO_ID_MAP.get(base)
        if not coin_id:
            supported = ", ".join(sorted(COINGECKO_ID_MAP))
            raise ValueError(
                f"Unknown coin '{base}'. Add it to COINGECKO_ID_MAP in crypto_bot/data.py.\n"
                f"Supported: {supported}"
            )
        return coin_id

    def fetch_ohlcv(
        self,
        symbol: str,
        days: int = 90,
        vs_currency: str = "usd",
    ) -> pd.DataFrame:
        """
        Return a DataFrame with DatetimeIndex (UTC) and float columns:
          open, high, low, close, volume

        Free-tier granularity (days param):
          ≤ 2 days  →  30-minute candles
          3–30 days →  4-hour candles
          31+ days  →  4-day candles
        """
        coin_id = self.symbol_to_id(symbol)
        label = _GRANULARITY_LABEL.get(str(days), "candles")
        logger.info("Fetching %d-day %s OHLCV for %s (%s)…", days, label, symbol, coin_id)

        ohlc_df = self._fetch_ohlc(coin_id, days, vs_currency)
        vol_series = self._fetch_volumes(coin_id, days, vs_currency)

        # Align volume to OHLC candle boundaries via resample
        if vol_series is not None and not vol_series.empty:
            freq = _infer_candle_freq(ohlc_df.index)
            try:
                vol_resampled = vol_series.resample(freq).sum().reindex(ohlc_df.index, method="nearest", tolerance=pd.Timedelta(freq) / 2)
                ohlc_df["volume"] = vol_resampled.fillna(0.0)
            except Exception:
                ohlc_df["volume"] = 0.0
        else:
            ohlc_df["volume"] = 0.0

        ohlc_df = ohlc_df.astype(float)
        logger.info(
            "  %d candles  %s → %s",
            len(ohlc_df),
            ohlc_df.index[0].strftime("%Y-%m-%d"),
            ohlc_df.index[-1].strftime("%Y-%m-%d"),
        )
        return ohlc_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None, retries: int = 3) -> dict | list:
        url = f"{self.BASE}{endpoint}"
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.get(url, params=params or {}, timeout=30)
                if resp.status_code == 429:
                    wait = 65
                    logger.warning(
                        "CoinGecko rate-limit hit — waiting %ds (attempt %d/%d)…",
                        wait, attempt, retries,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == retries:
                    raise
                logger.warning("Request failed (attempt %d/%d): %s — retrying…", attempt, retries, exc)
                time.sleep(5 * attempt)
        raise RuntimeError(f"CoinGecko request failed after {retries} attempts: {url}")

    def _fetch_ohlc(self, coin_id: str, days: int, vs_currency: str) -> pd.DataFrame:
        raw = self._get(f"/coins/{coin_id}/ohlc", {"vs_currency": vs_currency, "days": str(days)})
        if not raw:
            raise ValueError(f"CoinGecko returned empty OHLC for '{coin_id}' ({days} days)")
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def _fetch_volumes(self, coin_id: str, days: int, vs_currency: str) -> pd.Series | None:
        try:
            raw = self._get(
                f"/coins/{coin_id}/market_chart",
                {"vs_currency": vs_currency, "days": str(days)},
            )
            vols = raw.get("total_volumes", [])
            if not vols:
                return None
            df = pd.DataFrame(vols, columns=["ts", "volume"])
            df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
            return df["volume"].sort_index()
        except Exception as exc:
            logger.warning("Could not fetch volumes: %s", exc)
            return None


def _infer_candle_freq(index: pd.DatetimeIndex) -> str:
    """Guess candle frequency string for pd.resample()."""
    if len(index) < 2:
        return "4h"
    secs = (index[1] - index[0]).total_seconds()
    if secs <= 1_800:
        return "30min"
    if secs <= 14_400:
        return "4h"
    return "4D"
