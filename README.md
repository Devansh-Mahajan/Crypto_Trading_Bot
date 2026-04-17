# Crypto Trading Bot

A CCXT-based crypto trading bot for fast exchange portability and CoinGecko price fallback.

## Features

- Uses `ccxt` for exchange abstraction (Binance, Kraken, KuCoin, and 100+ exchanges)
- Supports `paper` trading mode and `live` execution mode
- Fetches historical OHLCV price data from exchanges
- Uses CoinGecko as a fallback price source
- Implements a robust EMA crossover strategy with risk management
- Configurable via `config.yml`

## Setup

1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the example config

```bash
cp config.example.yml config.yml
```

3. Edit `config.yml`

- Set `exchange` to your desired exchange, e.g. `binance`, `kraken`, `kucoin`
- Set `mode` to `paper` for simulated trading or `live` for real orders
- Add API credentials only if using `live`

## Running

```bash
python bot.py
```

## Example strategy

This bot uses an EMA crossover strategy:

- Buy when the fast EMA crosses above the slow EMA
- Sell when the fast EMA crosses below the slow EMA
- Uses stop-loss and take-profit rules
- Keeps risk and exposure under control

## Notes

- `paper` mode simulates execution using `paper.starting_balance`
- `live` mode executes real market orders through the selected exchange
- Use CoinGecko fallback if the exchange quote is unavailable
- Add or replace the strategy in `crypto_bot/strategy.py` for custom logic

## File structure

- `crypto_bot/config.py` — configuration loader
- `crypto_bot/exchange.py` — CCXT exchange wrapper and CoinGecko fallback
- `crypto_bot/strategy.py` — EMA crossover strategy and risk controls
- `crypto_bot/bot.py` — main bot orchestration loop
- `bot.py` — root entrypoint
- `config.example.yml` — example configuration template
- `requirements.txt` — Python dependencies
