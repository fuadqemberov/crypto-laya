# Paper-trading bot

The bot runs your trained model in real time. At every candle close it asks the model for
LONG / SHORT / WAIT on each configured symbol, passes the answer through a risk manager, and
executes it on a **simulated account** with the same execution model as the backtest. It uses the
public Binance market-data API only. **It never sends an order and needs no API key.**

Why it exists: a backtest is scored on history the researcher has already seen. Paper trading
scores the model on candles that did not exist when it was trained or selected. It is the honest
test of whether there is an edge before any real money is involved.

## One command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_bot.ps1
```

It downloads the data (only new files on re-runs), builds the dataset once, trains the baselines
once, picks the model with the lowest **validation** log loss (`paper setup`), writes it and the
dataset's symbols into `configs\bot.yaml`, opens the dashboard and starts the paper bot. Options:
`-Symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT -Start 2023-01-01 -Dataset laptop4` (a new dataset name
for a new symbol set). After training LAYA (e.g. on Colab) and `select-best`, `paper setup` also
considers LAYA if it was scored on the same dataset.

## Start

```powershell
# configs\bot.yaml: model, symbols, min_confidence, risk limits
uv run python -m laya_crypto.cli paper run                  # runs until stopped
uv run python -m laya_crypto.cli paper run --name lgbm      # a second, independent account
uv run python -m laya_crypto.cli paper status
uv run python -m laya_crypto.cli paper report               # return, drawdown, PF, win rate, days running
uv run python -m laya_crypto.cli paper stop                 # graceful stop; state is kept
```

Or `scripts\paper_bot.ps1`. The dashboard's **Paper** tab shows equity, open positions, decisions
with their vetoes, and a stop button.

Run it on your PC, not on Colab: Binance refuses live market data to US-hosted machines (HTTP 451).
To use a model trained on Colab, set `$env:LAYA_CRYPTO_STORAGE` to the synced Drive folder.

## Which model

- `model: laya` uses `models/best` (picked with `select-best`, on validation).
- `model: lightgbm` with `model_dataset: main` uses that baseline and its calibration. Use this if
  the baseline beat LAYA on validation: the bot should trade whichever model earned it. Both can
  run side by side under different `--name`s.

## What happens at each candle close

1. **Fills**: orders decided at the previous close fill at this candle's open ± (half spread +
   slippage), paying the taker fee. Size = `risk_per_trade` × equity ÷ stop distance (costs
   included), capped by `max_position_notional_frac` × equity × `max_leverage` and by free margin.
2. **Exits**: a gap through stop or target exits at the open; a candle touching both counts as the
   **stop**; then target; timeout after `max_hold_bars` (from `labeling.yaml`) at the close.
3. **Funding** settled while a position is open is charged (longs pay positive rates).
4. **Risk state**: `max_daily_loss` stops new trades for the rest of the UTC day; `max_drawdown`
   from the peak stops new trades until you start a fresh account (`--name`).
5. **Decisions**: the model is asked for every symbol. Vetoes: no or stale prediction, prediction
   not for the latest candle, halted, below `min_confidence`, abnormal regime, already in that
   symbol, `max_positions`, missing ATR. Stop and target distances are the labeling barrier widths:
   the same trade geometry the model was trained to predict.

Everything is journaled to `artifacts/paper/<name>/journal.jsonl`, equity per candle goes to
`equity.jsonl`, and the state to `state.json`, written atomically after every step. After a restart
the bot processes the candles it missed for open positions (a stop hit while the PC was off is
recorded), discards decisions it was too late to act on, and continues.

## How to read the result

- Run it for weeks, not days: at least a month and dozens to hundreds of trades before
  concluding anything.
- Compare the paper results with the backtest of the same model on the test split. Paper much worse
  than the backtest means the backtest was optimistic, or the market changed.
- A negative paper result is the cheapest lesson this project can give you: it costs nothing.

## What it does not do

It does not place real orders. Real-money execution is not part of this project. Anything that
sends orders with your keys should be built, reviewed and tested separately, and only after paper
trading has shown a stable edge after costs.
