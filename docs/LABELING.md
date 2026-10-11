# Labeling

Implementation: `src/laya_crypto/labeling/triple_barrier.py`. Parameters: `configs/labeling.yaml`.
Tests: `tests/test_labels.py`.

## Primary decision: LONG / SHORT / WAIT

For a decision at the close of bar t:

1. **Entry** at the open of bar t+1, the first tradable price after the decision.
2. **Barriers** as fractions of the entry: `tp = tp_atr × ATR%_t`, `sl = sl_atr × ATR%_t` (ATR known at
   t; defaults 2 and 1). If `tp` would be below `min_barrier_pct` both are scaled up together, keeping
   their ratio. `method: triple_barrier_pct` uses fixed `tp_pct` / `sl_pct` instead.
3. **Path**: bars t+1 … t+H, `H = max_hold_bars` (default 24 × 5m = 2 h).
4. **Same-candle ambiguity**: when one candle reaches both barriers, the order is unknown.
   - `conservative_sl` (default): count it as the stop. TP-first is never assumed.
   - `resolve_1m`: look inside that candle with 1m klines (download `--timeframes 1m`); if the same
     minute still has both, count the stop.
   - `drop`: make the label missing.
5. **Vertical barrier**: no touch by t+H exits at the close of bar t+H.
6. **Costs**: round trip = entry fee + exit fee (taker or maker per `costs`) + 2 × (half spread +
   slippage), plus funding settled in (entry, exit] (longs pay positive rates).
7. **Missing data**: any missing bar in the path, a missing entry price or a missing ATR → no label.

Label = **LONG** if the long trade reaches its take-profit first and its net return after costs is
positive; **SHORT** likewise for the short; **WAIT** otherwise: the stop came first, the target came
too late, or the edge did not cover costs. WAIT is therefore a meaningful outcome ("no trade with
this barrier geometry pays"), not a leftover class. With `tp ≥ sl`, LONG and SHORT are mutually
exclusive (the config validator refuses `tp_atr < sl_atr`). `timeout_policy: sign_if_net_positive`
also labels a profitable timeout as a trade.

On the probe dataset (BTC/ETH/SOL, Mar–Jul 2025, defaults) the classes came out at about one third
each (WAIT 35%, LONG 33%, SHORT 32%), so no re-weighting is needed by default; `dataset.yaml:
wait_downsample` thins WAIT rows in TRAIN only if a configuration makes WAIT dominant.

## Experimental targets (stored in every dataset)

| column | meaning |
|---|---|
| `fwd_ret_15m`, `fwd_ret_30m`, `fwd_ret_60m` | log return from entry open to the close at that horizon |
| `long_tp_first`, `short_tp_first`, `long_sl_first`, `short_sl_first` | barrier outcome indicators |
| `long_net_ret`, `short_net_ret` | net trade return after costs and funding |
| `mfe`, `mae` | max favourable / adverse excursion over the horizon (upside / downside risk) |
| `long_ambiguous`, `short_ambiguous` | the exit candle touched both barriers |
| `label_end_time` | close of bar t+H, used for purging and embargo |

## Soft targets

`soft_targets: true` labels every row under several barrier scales (`soft_barrier_scales`, default
0.75×, 1×, 1.25×) and uses the share of scales giving each class as LAYA's `gold` distribution.
This expresses how sensitive a label is to the exact barrier choice; it is not a probability of
profit. Hard labels (default) are written as `expected`.

## What the labels do not do

They do not choose the best possible trade with hindsight, and they never use a future price as the
execution price: the entry is the next open, exits are the barrier prices (or the open, when the
backtest sees a gap through a barrier), and costs are always paid.
