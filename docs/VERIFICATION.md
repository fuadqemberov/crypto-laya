# Verification log

Everything below was **executed** while building the project, on 2026-10-11, in the development
container. The container is **not** the target PC: Linux 6.18, Intel Xeon @ 2.80 GHz, 4 vCPU,
15.7 GB RAM, **no GPU** (PyTorch 2.14.1+cpu). No GPU benchmark exists yet. Run `hardware` on your
own machine. Nothing here is a claim of predictive skill.

## Environment and LAYA

| check | result |
|---|---|
| `uv lock` / `uv sync --extra dev` | 70 packages resolved; Python 3.12.3 |
| `laya` version | 0.4.2 (PyPI); upstream repo inspected at `68804629e8ccd9d616d48a40e87de9fabbeae069` |
| base checkpoint `english` | `convaiinnovations/laya@55cf4c4e…`, 842,609,210-byte `model.safetensors`, SHA-256 verified against the Hub LFS hash |
| model | ModernBERT-large encoder, 421,293,827 parameters |
| `laya.load(...)` | loaded in 27.5 s on CPU; one zero-shot prediction in 0.5 s |
| official `laya-train --dry-run` on our files | `train.jsonl`: 36,585 rows → 36,585 valid items, 0 skipped; `val.csv` (`--text-column state --label-column label`): 9,105/9,105 valid |
| `cli hardware` | report written; CPU tier selected (no CUDA: "PyTorch was built without CUDA") |

## Data (real Binance archive)

| check | result |
|---|---|
| bucket listing | 1,056 USD-M symbols with kline archives (delisted included) |
| `download` BTC/ETH/SOL, 2025-01-15 → 2025-07-31, 5m + 1h + mark/index/premium + funding + metrics | 720 files, 720 ok, 0 missing, 0 errors, all SHA-256 verified |
| `universe --max-candidates 30` | 58 months, 27 unique symbols selected from 30 candidates |
| incremental download (Colab simulation) | 1st run 100 files; raw cache deleted → 0 files planned; range +7 days → 7 files |
| `pytest -m network` | real 1d BTCUSDT files listed, downloaded, verified, re-run skipped, parsed: 60 rows, 0 gaps |
| Binance live API (`fapi.binance.com`) from this container | HTTP 451 (location refused). Inference used cached data and marked it `stale` |

## Dataset `probe3` (BTCUSDT, ETHUSDT, SOLUSDT; Mar–Jul 2025; `sample_every: 3`)

- 60,873 rows: train 36,585 / val 9,105 / calib 6,054 / test 9,129; 98 features + GMM regime.
- Labels: train WAIT 35.1%, SHORT 32.7%, LONG 32.3% (test: 35.6 / 31.4 / 32.9).
- Leakage report: all hard checks passed. Each split's last label ends before the next split's first
  decision. 0 duplicate (symbol, time). 0 exact state duplicates across splits. 0 near-duplicate
  feature vectors between train and test.

## Results on `probe3` (too small to mean anything; shown because they were run)

Validation / test log loss for three classes at chance is ln 3 = 1.0986.

| model | val log loss | test log loss | test acc | test expectancy / signal (after costs) |
|---|---|---|---|---|
| XGBoost | 1.0963 | 1.0979 | 0.350 | −0.163% |
| LightGBM | 1.0967 | 1.0972 | 0.351 | −0.164% |
| logistic regression | 1.0967 | 1.0979 | 0.345 | −0.155% |
| trend following | 1.0984 | 1.0986 | 0.350 | −0.186% |
| LAYA smoke (48 rows only) | 1.1161 | 1.0940 | 0.375 | −0.149% |

- LightGBM walk-forward (4 purged folds, test excluded): log loss 1.0960–1.0963, expectancy
  −0.13% to −0.17% per signal in every fold.
- Ablation (LightGBM, val): removing any group except `structure` *lowers* log loss slightly
  (−0.0002 to −0.0006). On this sample the features add noise, not signal.
- Backtest, test split, 2,000 USDT paper equity, default risk rules: LightGBM −24.98% (151 trades,
  377 USDT fees, drawdown halt hit). LAYA smoke −7.37% (48 trades, only 48 test decisions scored).
- Verdict written by `evaluate`: LAYA smoke is **not better** than the best baseline on validation.
  With 5 months of 3 symbols, nothing beats chance after costs. This is a pipeline check, not a
  research result.

## LAYA smoke fine-tune (CPU)

`train --profile smoke --dataset probe3 --device cpu`: real full fine-tune of the 421 M-parameter
model, 64 train items, 2 epochs, 32 optimizer updates, effective batch 4.

| | |
|---|---|
| measured throughput | 0.03–0.11 items/s (CPU shared with tests) |
| epoch mean train loss | 1.2248, 1.2201 |
| val (48 rows) after epoch 1 / 2 | loss 1.1552 / 1.3180, acc 0.208 / 0.313 → best epoch 1 kept |
| calibration (48 chronological calib items) | choice temperature 2.84 ("fitted on only 48 calibration items") |
| collapse warning (upstream) | "predictions collapsed to the class prior": expected with 32 updates |
| `select-best` → `models/best`, then `predict --offline` | valid schema, `abstained: true`, `stale: true`, source `archive` |
| `benchmark --offline` (CPU) | model-only p50 2,030 ms, p95 2,337 ms; end-to-end p50 2,389 ms |
| resume state | 4.8 GB `state.pt` (+ 4.8 GB `.prev`) |

## Automated tests

`uv run pytest -q`: **58 passed** (1 network test deselected by default, passes with `-m network`).
`uv run ruff check src tests`: clean. Covered: parsing/units/dedup/gaps/quality, indicator
correctness, full-pipeline causality, HTF alignment, SMC definitions, as-of lag, every labeling
rule, splits/purge/embargo/walk-forward, leakage detection, LAYA item schema (real tokenizer),
universe point-in-time rules, backtest fees/slippage/ordering/funding/vetoes, metrics/ECE/bootstrap,
temperature calibration, hardware tiering, tiny-LAYA training → save → `laya.load`, interrupted +
resumed run = uninterrupted run (max weight diff < 1e-3), config-change refusal, corrupt-state
fallback, out-of-step synced state, auto-resume, CPU refusal, inference schema, dashboard
endpoints and the no-concurrent-jobs rule. Bugs found by these tests and fixed: a refused resume
overwrote the run fingerprint; the backtest skipped the first decision bar; the dashboard sent NaN
as invalid JSON.

## Dashboard

Launched with `cli dashboard` (127.0.0.1:8765). All tabs rendered in headless Chromium with no JS
errors, showing the real `probe3` runs, metrics, comparison tables and backtests.

## Paper-trading bot

`tests/test_paper_bot.py` (7 tests, fake market): exact fill/fee/slippage arithmetic, stop wins a
same-candle tie, timeout, every veto, daily-loss halt, restart catching up on missed candles,
graceful stop, dashboard endpoints. The Paper tab rendered (headless Chromium, no JS errors) on a
300-candle simulated account with *random* decisions: −7.2%, 75 trades, PF 0.69, daily-loss halt
triggered. That is what costs do to a model without an edge. It was not run against live
Binance data here (HTTP 451 from this container).

## Not executed (ready to run)

- `research` and `deep` profiles, multi-seed runs, LAYA HPO: these need a GPU.
- Full top-100 universe over the full history (~57k 1d files plus full data for 100 symbols).
- `notebooks/colab_train.ipynb` was not run on Colab. Its storage, incremental and auto-resume logic
  was tested locally with the same environment variables.
- Live inference with fresh Binance data (blocked here by HTTP 451).
