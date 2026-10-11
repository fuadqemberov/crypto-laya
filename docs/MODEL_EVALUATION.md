# Model evaluation

## Splits (built once per dataset, immutable)

`configs/dataset.yaml` splits the decision-time span into **train 60% → val 15% → calib 10% →
test 15%**, in that order, with the same calendar boundaries for every symbol.

- **Purge**: a row whose label horizon (`label_end_time`) reaches past the end of its split is dropped.
- **Embargo**: rows within `embargo_bars` (default = label horizon) after a boundary are dropped.
- `train` fits models, scalers and the GMM regime detector; `val` selects (early stopping, seeds,
  hyper-parameters, thresholds); `calib` fits temperatures only; `test` is the latest period,
  scored and reported, and never used for any choice.
- Leakage checks (`datasets/leakage.py`, written to `leakage_report.json`, hard failures stop the
  build): time-ordered splits with resolved labels, unique (symbol, time) rows, decision time =
  bar close, exact duplicate state texts across splits, near-duplicate feature vectors between
  train and test, and no target column among the features.

Walk-forward: `walk-forward` uses purged expanding-window folds over the non-test period only.

## Metrics (identical for every model)

Classification (`evaluation/metrics.py`): accuracy, balanced accuracy, macro F1, per-class
precision/recall/F1/support, confusion matrix, multi-class Brier score, top-1 ECE with a reliability
diagram, log loss, coverage-vs-accuracy (abstention) curve, predicted class distribution.

Signal level: acting on every LONG/SHORT under the label's execution model (next-open entry,
barriers, costs, funding): number of trades and frequency, net return sum, expectancy per trade,
win rate, average win/loss, profit factor, long/short counts, and a **moving-block bootstrap**
confidence interval of the per-decision mean return (block = 1 day of bars, for serial dependence).
Breakdowns by symbol and by regime.

Portfolio level: see [BACKTESTING.md](BACKTESTING.md).

## Commands

```powershell
uv run python -m laya_crypto.cli baselines --dataset main
uv run python -m laya_crypto.cli evaluate --dataset main                       # writes reports\main\comparison.md
uv run python -m laya_crypto.cli evaluate --dataset main --checkpoint models\best --name laya-best
uv run python -m laya_crypto.cli walk-forward --dataset main --model lightgbm
uv run python -m laya_crypto.cli ablation --dataset main
uv run python -m laya_crypto.cli hpo --dataset main --target lightgbm --trials 30
```

Per-model reports: `reports/<dataset>/<model>.json` with `train` (in-sample), `val` and `test`
sections, clearly labelled.

## Calibration

LAYA: one temperature per question type fitted by `laya.calibrate.fit_temperature_map` on the
chronological `calib` split, stored in the checkpoint and applied by `laya.load`. Baselines: one
temperature on the same `calib` rows (`TemperatureCalibrator`). Temperature scaling does not change
the argmax, only confidences. Fewer than 10 calibration items leave the temperature at 1.0 and the
report says so (the smoke profile hits this on purpose).

A calibrated 0.9 means that, on held-out rows of this dataset, about 90% of 0.9-confidence answers
matched the label. It is **not** a 90% chance the trade makes money: the label is a barrier event
under assumed costs, and the future distribution can differ from the past.

## Reading the comparison

The question the project must answer is whether fine-tuned LAYA beats the cheaper models on the
same rows. `comparison.md` ends with that verdict on validation log loss. Look at the test table,
the bootstrap interval and the seed spread before concluding anything. A simpler model winning is a
valid result and is reported as such; thresholds are never tuned on test to change the verdict.
