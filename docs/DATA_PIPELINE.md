# Data pipeline

Source: the Binance public archive (`https://data.binance.vision`, documented in
`binance/binance-public-data`), USD-M futures (`data/futures/um/`). No API key is used.

## 1. Universe (no survivorship bias)

```powershell
uv run python -m laya_crypto.cli universe            # top 100 (configs/data.yaml: universe.top_n)
uv run python -m laya_crypto.cli universe --top-n 300
```

1. The bucket listing gives **every** symbol that ever had a monthly kline archive (1,056 at the
   time of writing), delisted contracts included.
2. Candidates: `*USDT`, perpetuals only (no `_YYMMDD` deliveries), not in `exclude_symbols`
   (stablecoin pairs).
3. 1d klines are downloaded for all candidates.
4. For each month M, a symbol is eligible only if, **using data before M**, it had at least
   `min_history_days` daily candles, its trailing `rank_window_days` mean daily quote volume is at
   least `min_avg_daily_quote_volume`, it traded on the last day before M, and it ranks in the top N.

Outputs: `data/universe/universe.parquet` (month, symbol, rank, trailing volume, history days) and
`universe_report.json`. Dataset rows are kept only for months in which their symbol was in the
universe, so a coin that collapsed and was delisted is still in the data for the months it was big,
and a coin that is big today is absent from the months before it qualified.

Known limitation: rows whose label horizon runs past the end of a delisted symbol's data have no
label and are dropped, so the very last hours before a delisting are not in the training data.

## 2. Download

```powershell
uv run python -m laya_crypto.cli download                         # all universe symbols, data.yaml range
uv run python -m laya_crypto.cli download --symbols BTCUSDT ETHUSDT --start 2024-01-01 --end 2024-12-31
```

| data type (data.yaml `datasets:`) | archive | granularity | coverage note |
|---|---|---|---|
| `klines` (OHLCV, taker buy volume, trade count) | `klines/<tf>` | monthly + daily for the open month | per symbol listing |
| `mark_price_klines`, `index_price_klines`, `premium_index_klines` | `markPriceKlines`, … | monthly | base timeframe only |
| `funding_rate` | `fundingRate` | monthly | settled rates only (no historical predicted rate) |
| `metrics` (open interest, long/short ratios, taker L/S volume ratio) | `metrics` | daily, 5m snapshots | from 2020-09 for BTCUSDT |
| `agg_trades` (disabled by default, large) | `aggTrades` | monthly | aggregated to 1m buy/sell flow bars |
| `book_depth` (disabled by default) | `bookDepth` | daily | depth % snapshots from 2023-01 only |
| liquidations | — | — | **not available**: there is no USD-M liquidation archive; nothing is synthesised |
| historical order books | — | — | only the `bookDepth` percentage snapshots exist; current order books are never used to infer the past |

Mechanics: listing-driven planning (no requests for files that do not exist), bounded concurrency
(`download.concurrency`), exponential backoff with jitter on 429/418/5xx and `Retry-After`,
SHA-256 verification against Binance's `.CHECKSUM` file, `.part` + rename so interrupted runs resume
(verified files are skipped), and incremental updates: re-running a download fetches only new files.

Processed tables: `data/processed/<kind>/<tf>/<SYMBOL>.parquet`, epoch-millisecond UTC timestamps
(seconds/microseconds archives are normalised by magnitude), sorted, de-duplicated on the time key.
`data/manifest.json` records rows, first/last timestamp, duplicates dropped, parse errors, source
and SHA-256 of every table.

Contract metadata: `uv run python -m laya_crypto.cli exchange-info` saves the **current**
`exchangeInfo` (tick size, step size, min notional, onboard date). It is a snapshot with its fetch
time and is used by the backtest for order filters; it is not historical metadata.

## 3. Validate

```powershell
uv run python -m laya_crypto.cli validate --timeframe 5m
```

Per symbol: rows, expected rows, missing bars and gap runs, misaligned timestamps, duplicate
timestamps, OHLC violations, robust return outliers (> 12 MAD), zero-volume bars, and a quality
score. Written to `reports/data_quality.json` and shown in the dashboard. Problems are reported; no
bar is filled in. The feature pipeline puts gaps back on the time grid as NaN rows, so no rolling
window silently spans missing time, and rows with missing bars in their label path get no label.

## 4. Dataset

```powershell
uv run python -m laya_crypto.cli dataset --name main
```

See [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md), [LABELING.md](LABELING.md) and the split
section of [MODEL_EVALUATION.md](MODEL_EVALUATION.md). Datasets are immutable: rebuilding an existing
name needs `--force`; choose a new name for a changed configuration.
