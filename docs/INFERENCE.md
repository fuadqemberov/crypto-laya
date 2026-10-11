# Inference

```powershell
uv run python -m laya_crypto.cli predict --symbol BTCUSDT
uv run python -m laya_crypto.cli predict --symbol BTCUSDT ETHUSDT SOLUSDT --timeframe 5m
uv run python -m laya_crypto.cli predict --symbol BTCUSDT --offline          # cached archive data only
uv run python -m laya_crypto.cli predict --symbol BTCUSDT --checkpoint models\checkpoints\<run>\final
uv run python -m laya_crypto.cli benchmark --symbol BTCUSDT --n 50
```

## Path

1. Cached archive data (`data/processed/...`) for the symbol and BTC/ETH context.
2. Unless `--offline`: the most recent **closed** candles, funding, open interest, mark / index /
   premium candles from the public Binance USD-M REST API (`/fapi/v1/...`, no key). The candle that
   is still forming is dropped.
3. The same feature pipeline as training, over the last `warmup_bars` (2,400) bars, so rolling
   windows such as the 7-day volatility percentile match training.
4. Feature validation: core features must be finite; missing state fields are listed.
5. The same serialization as training, then `laya.load(checkpoint).predict_batch(...)` with the
   checkpoint's calibrated temperatures.
6. Abstention: if the top probability is below `abstain_below` (0.45), the decision is WAIT and
   `abstained` is true. Tune this threshold on validation, never on test.

## Output

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "5m",
  "decision": "WAIT",
  "model_argmax": "LONG",
  "abstained": true,
  "probabilities": {"LONG": 0.41, "SHORT": 0.27, "WAIT": 0.32},
  "model_version": {"checkpoint": "...", "fingerprint": "...", "profile": "research", "seed": 0},
  "data_timestamp_utc": "...",
  "data_age_minutes": 3.2,
  "stale": false,
  "data_source": "archive+live",
  "missing_state_fields": [],
  "notes": [],
  "latency_ms": {"data": 0, "features": 0, "model": 0, "total": 0}
}
```

The numbers above only show the shape of the output. Nothing in the inference path is hard-coded.

- If live data cannot be fetched (network, or HTTP 451 where Binance refuses a location), the result
  is computed on cached data, `data_source` is `archive`, the reason is in `notes`, and `stale` is
  true once the newest candle is older than `max_data_age_minutes`.
- If there is no usable data or the warm-up is incomplete, the response is an `error`, never a guess.
- `benchmark` measures model-only latency (single state, p50/p95), batched cost per state (32), and
  end-to-end latency including data and features, separately, on the current machine.

The dashboard's Inference tab calls the same `Predictor`.
