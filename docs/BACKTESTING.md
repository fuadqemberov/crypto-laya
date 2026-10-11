# Backtesting

Implementation: `src/laya_crypto/backtesting/engine.py` (event-driven, causally ordered),
`backtesting/run.py`. Parameters: `configs/backtest.yaml` and the costs in `configs/labeling.yaml`.

```powershell
uv run python -m laya_crypto.cli backtest --dataset main --model lightgbm --split test
uv run python -m laya_crypto.cli backtest --dataset main --model laya-research-main-s0-xxxxxx --split test
uv run python -m laya_crypto.cli backtest --dataset main --model xgboost --split val --min-confidence 0.5
```

`--model` is a prediction file in `artifacts/predictions/<dataset>/`; every model (baselines and
LAYA) is backtested with the same engine, costs, sizing and risk rules.

## Order of events per bar

1. **Fills** of orders decided at the previous close, at this bar's open ± (half spread + slippage),
   paying the entry fee.
2. **Exits**: liquidation (isolated margin, `maintenance_margin_rate`) before stop; gaps through the
   stop / liquidation / target exit at the open; a bar touching both stop and target exits at the
   **stop**; otherwise target; the vertical barrier exits at the close of the last holding bar
   (same geometry as the labels).
3. **Funding** settled at its timestamps on open positions (longs pay positive rates).
4. **Decisions** at the close: the model's prediction goes to the risk engine.

## Risk engine (may veto the model)

| rule | default |
|---|---|
| initial paper equity | 2,000 USDT (research default) |
| risk per trade (loss at the stop incl. costs) | 0.5% of realised equity |
| leverage cap | 3× (margin = notional / 3) |
| max position notional | 50% × equity × leverage |
| max open positions | 5, one per symbol |
| free margin | margin of all positions ≤ realised equity |
| drawdown halt | no new entries after a 25% portfolio drawdown |
| confidence floor | `min_confidence` (0 = take every LONG/SHORT) |
| abnormal regime | veto when `regime_abnormal = 1` |
| exchange filters | step size / min notional from the `exchange-info` snapshot, if present |

Sizing never counts open profits. Leverage is a cap, not a lever for returns: raising it does not
change the stop-based position size, it only frees margin.

## Outputs

`reports/<dataset>/backtests/<model>_<split>.json` (+ `_trades.parquet`, `_equity.parquet`):
net PnL, return %, max drawdown, Sharpe and Sortino from daily equity (annualised with √365),
win rate, average win/loss, profit factor, expectancy, fees, funding, average holding time,
exposure (share of bars with an open position), turnover, exit reasons, risk vetoes, and PnL by
symbol, regime and side.

## Assumptions to keep in mind

- Fills at the next open with fixed spread + slippage; no partial fills, queue position or market
  impact beyond that.
- Liquidation is modelled from candle extremes with isolated margin; real liquidation uses the mark
  price and tiered maintenance margin.
- Exchange filters use the **current** metadata snapshot; historical tick/step sizes are not
  published in the archive.
- Probe result (BTC/ETH/SOL, test period Jun–Jul 2025, LightGBM, defaults): −25% with 377 USDT of
  fees on 151 trades and the drawdown halt triggered. On 5m bars with 1–2 ATR barriers, round-trip
  costs of ~0.16% dominate; this is a finding to investigate (longer horizons, maker execution,
  confidence floors chosen on validation), not something to hide by changing test-period rules.
