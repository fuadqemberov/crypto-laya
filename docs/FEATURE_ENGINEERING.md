# Feature engineering

Implementation: `src/laya_crypto/features/`. Parameters: `configs/features.yaml`.

## The causality rule

Row t is a decision at the **close** of base bar t (`decision_time = open_time + timeframe`).
Every feature on row t uses candles that closed at or before that time. External series join *as
of* their own availability:

| input | available at |
|---|---|
| base candle t | its close |
| 1h / 4h candle | its close (`open_time + duration`), built only from complete groups of base candles |
| funding rate | its settlement time (`calc_time`) |
| open interest snapshot | `create_time + metrics_publication_lag_ms` (default one 5m bar, conservative) |
| mark / index / premium candle t | its close (same bar) |
| BTC / ETH context candle t | its close (same bar) |

`tests/test_features.py::test_full_pipeline_is_causal` rebuilds the features from truncated
histories and asserts that no value at row k changes when rows after k are removed.

## Groups

| group | features |
|---|---|
| `trend` | RSI (Wilder), MACD line/hist (÷ price), EMA 9/21/50/200 and SMA 20/50 distance in ATR, ADX, DI spread, ROC 3/12/48, Stochastic RSI K/D, 12-bar momentum, EMA slopes, trend strength |
| `volatility` | ATR %, true range %, Bollinger width and %B, realised vol 12/48/288, volatility percentile (rolling rank, causal), range % |
| `volume` | relative volume, volume z-score and spike flag, taker imbalance (bar, 1h, 4h), CVD over 1 day from kline taker volumes, rolling VWAP distance; aggressive trade imbalance if `agg_trades` was downloaded; depth imbalance within 1% if `book_depth` was downloaded |
| `structure` | swing high/low distance, HH/LH and HL/LL state, structure direction, BOS / CHoCH up/down, bars since last break, bullish/bearish FVG (flag, size, bars since), order-block distance and in-zone flags, liquidity sweeps, rolling support/resistance distance |
| `derivatives` | funding rate and change, OI change 1/12/48 bars, price/OI divergence, top-trader and account long/short ratios (**proxies** for crowding, named `proxy_*`), taker L/S volume ratio, mark-index premium, last/mark basis, premium index |
| `mtf` | per 1h and 4h: trend (EMA21/50 + close), RSI, last return, EMA21 distance in ATR, ADX, ATR % |
| `regime` | rule-based trend regime (bull/bear/sideways), volatility regime (high/normal/low), abnormal flag (|return z| ≥ 6 in last 12 bars), GMM regime (fitted on TRAIN rows only) |
| `context` | BTC and ETH returns 12/48 bars, RSI, realised vol |

Missing data types leave their columns NaN with explicit `has_funding`, `has_open_interest`,
`has_depth`, `has_trade_flow` indicators. Tree baselines use NaN natively; the logistic baseline
imputes TRAIN medians; LAYA sees `NA` in the text.

## Smart Money Concepts definitions

Exact definitions are in the module docstring of `features/structure.py`. In short (L =
`swing_lookback`): a swing high is a strict maximum of the L bars on each side and is **usable only
L bars after the pivot**; BOS is a close through the latest confirmed swing in the direction of the
last break, CHoCH a close through it against that direction; an FVG is `low[t] > high[t-2]` (bullish)
of at least `fvg_min_atr` × ATR; an order block is the last opposite-colour candle within
`order_block_lookback` bars before a break; a liquidity sweep trades through the previous
`sweep_lookback`-bar extreme and closes back inside. All of them are unit-tested.

## What LAYA sees

LAYA gets a compact text serialization of 34 selected features plus the regime
(`datasets/serialize.py::STATE_FIELDS`), about 380 tokens, which fits `max_len 512` together with
the ~100-token question head. Baselines get all ~98 numeric columns. To change what LAYA sees, edit
`STATE_FIELDS` and build a new dataset.

## Ablation

```powershell
uv run python -m laya_crypto.cli ablation --dataset main --model lightgbm
uv run python -m laya_crypto.cli baselines --dataset main --models lightgbm --groups trend volatility
```

`ablation` retrains the model without each group and with each group alone, on the validation split
only, and writes `reports/<dataset>/ablation_lightgbm.json`. Small deltas are within noise: confirm
with walk-forward folds and other seeds before removing a group.
