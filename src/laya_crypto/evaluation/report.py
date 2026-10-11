"""Human-readable comparison report: baselines versus LAYA on identical rows and rules."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..config import project_path
from ..utils.io import atomic_write_text, read_json
from .scoring import comparison_table


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_comparison_report(dataset: str) -> Path:
    tab = comparison_table(dataset)
    lines = [f"# Model comparison - dataset `{dataset}`", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from the stored predictions.", "",
             "All models are scored on the same rows with the same labels, costs and execution model. "
             "`val` is used for model selection; `test` is the untouched final period and is reported, never tuned on. "
             "`expectancy` is the mean net return per LONG/SHORT signal after fees, spread, slippage and funding "
             "(labelled triple-barrier execution). The CI is a moving-block bootstrap of the per-decision mean.", ""]
    manifest = read_json(project_path("data", "datasets", dataset, "manifest.json")) or {}
    if manifest:
        lines += [f"Rows by split: {manifest.get('rows_by_split')}. Symbols: {len(manifest.get('symbols', []))}.", ""]
    if tab.empty:
        lines.append("No scored models yet.")
    for split in ("val", "test"):
        part = tab[tab["split"] == split].sort_values("log_loss")
        if part.empty:
            continue
        if part["n"].nunique() > 1:
            lines += [f"> **Warning:** models in this table were scored on different numbers of rows "
                      f"({', '.join(f'{m}: {n}' for m, n in zip(part['model'], part['n'], strict=True))}). "
                      "Only rows-equal comparisons are like for like; re-run "
                      "`evaluate --dataset <name> --checkpoint <run>/final --name laya-<run>` (no --max-rows) to score LAYA on every row.", ""]
        lines += [f"## {split}", "", "| model | n | acc | bal.acc | macro F1 | log loss | Brier | ECE | trades | "
                  "expectancy | win rate | PF | 95% CI per decision |", "|" + "---|" * 13]
        for r in part.itertuples(index=False):
            lines.append(f"| {r.model} | {r.n} | {_fmt(r.accuracy)} | {_fmt(r.balanced_accuracy)} | {_fmt(r.macro_f1)} | "
                         f"{_fmt(r.log_loss)} | {_fmt(r.brier)} | {_fmt(r.ece)} | {r.trades} | "
                         f"{_fmt(r.expectancy, 5)} | {_fmt(r.win_rate)} | {_fmt(r.profit_factor, 3)} | "
                         f"[{_fmt(r.per_decision_ci_low, 5)}, {_fmt(r.per_decision_ci_high, 5)}] |")
        lines.append("")
    bt_dir = project_path("reports", dataset, "backtests")
    bts = sorted(bt_dir.glob("*.json")) if bt_dir.exists() else []
    if bts:
        lines += ["## Backtests", "", "| model / split | trades | return % | max DD % | PF | win rate | Sharpe (daily, ann.) |"
                  " fees USDT |", "|" + "---|" * 8]
        for p in bts:
            b = read_json(p)
            lines.append(f"| {b['model']} / {b['split']} | {b['trades']} | {_fmt(b['return_pct'], 2)} | "
                         f"{_fmt(b['max_drawdown_pct'], 2)} | {_fmt(b.get('profit_factor'), 3)} | "
                         f"{_fmt(b.get('win_rate'), 3)} | {_fmt(b.get('sharpe_daily_annualised'), 2)} | "
                         f"{_fmt(b.get('fees_usdt'), 2)} |")
        lines.append("")
    laya = tab[tab["model"].str.startswith("laya") & (tab["split"] == "val")]
    base = tab[~tab["model"].str.startswith("laya") & (tab["split"] == "val") & (tab["model"] != "always_wait")]
    if not laya.empty and not base.empty:
        best_laya, best_base = laya.sort_values("log_loss").iloc[0], base.sort_values("log_loss").iloc[0]
        verdict = "better" if best_laya.log_loss < best_base.log_loss else "NOT better"
        lines += ["## Does fine-tuned LAYA beat the simpler models?", "",
                  f"On validation log loss, the best LAYA run ({best_laya.model}, {best_laya.log_loss:.4f}) is **{verdict}** "
                  f"than the best baseline ({best_base.model}, {best_base.log_loss:.4f}). "
                  "Check the test table and the seed spread before drawing conclusions; a simpler model winning is a "
                  "valid result.", ""]
    out = project_path("reports", dataset, "comparison.md")
    atomic_write_text(out, "\n".join(lines))
    return out
