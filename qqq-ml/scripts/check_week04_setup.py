"""Week-4 step 1-2: verify the HAR baseline and describe the evaluation slices.

* Recomputes RW / HAR scores from ``har_baselines.parquet`` with
  ``src.metrics`` (no retraining) and checks QLIKE 0.197 / RMSE 0.609.
* Describes the slices in ``config/evaluation.toml`` using realized
  variance only - no week-4 model output is read.

Writes ``reports/week04_definitions.md``.

Usage:
    python scripts/check_week04_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_week01_report import _md_table  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import metrics as M  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

EXPECTED = {"qlike": 0.197, "rmse_log": 0.609}


def main() -> None:
    pred = load_predictions("har_baselines")
    tbl = M.evaluate(pred).xs("all", level="fold").round(4)
    har = tbl.loc[("rv", "har")]
    ok = {k: abs(har[k] - v) < 0.0006 for k, v in EXPECTED.items()}
    print(tbl)
    print("baseline check:", "PASS" if all(ok.values()) else f"FAIL {ok}")

    y = pd.read_parquet(dl.PROCESSED_DIR / "targets_daily.parquet")
    rv = y["rv"].where(y["valid"].astype(bool))
    cfg = M.load_eval_config()
    oos = pd.DatetimeIndex(sorted(pred.loc[pred["target"] == "rv", "date"]
                                  .unique()))
    sl = M.eval_slices(oos, rv, cfg)
    vol = np.sqrt(252 * rv.reindex(oos)) * 100

    rows = []
    for key, p in cfg["periods"].items():
        m = sl[key]
        rows.append({"slice": key, "definition": f"{p['start']} → {p['end']}",
                     "OOS sessions": int(m.sum()),
                     "jump days inside": int((m & sl["jump_day"]).sum()),
                     "median ann. vol %": round(vol[m].median(), 1),
                     "max ann. vol %": round(vol[m].max(), 1)})
    j = cfg["jump_days"]
    rows.append({"slice": "jump_day",
                 "definition": f"RV(t) > {j['multiple']}× mean RV(t-{j['window']}..t-1)",
                 "OOS sessions": int(sl["jump_day"].sum()),
                 "jump days inside": int(sl["jump_day"].sum()),
                 "median ann. vol %": round(vol[sl["jump_day"]].median(), 1),
                 "max ann. vol %": round(vol[sl["jump_day"]].max(), 1)})
    rows.append({"slice": "all OOS", "definition": f"{oos[0]:%Y-%m-%d} → "
                 f"{oos[-1]:%Y-%m-%d}", "OOS sessions": len(oos),
                 "jump days inside": int(sl["jump_day"].sum()),
                 "median ann. vol %": round(vol.median(), 1),
                 "max ann. vol %": round(vol.max(), 1)})
    summary = pd.DataFrame(rows)

    by_year = sl["jump_day"].groupby(oos.year).agg(["sum", "size"])
    by_year.columns = ["jump days", "sessions"]
    by_year = by_year.reset_index(names="year")

    sens = pd.DataFrame([{
        "multiple": k,
        "jump days (OOS)": str(int(M.jump_days(rv, k, j["window"])
                                   .reindex(oos).fillna(False).astype(bool)
                                   .sum()))}
        for k in (1.5, 2.0, 2.5, 3.0)])

    lists = {}
    for key in cfg["periods"]:
        d = oos[sl[key] & sl["jump_day"]]
        lists[key] = ", ".join(f"{x:%m-%d}" for x in d) or "none"

    md = f"""# Week 4 評估切片定義（在看任何第 4 週模型結果之前）

設定檔：`config/evaluation.toml`。本檔只用到已實現變異數與第 3 週的基準預測。

## 基準核對

以 `src/metrics.py` 重算 `har_baselines.parquet`（不重新訓練）：

{_md_table(tbl.reset_index())}

日內 RV 的 HAR：QLIKE {har['qlike']:.4f}（預期 0.197）、RMSE {har['rmse_log']:.4f}（預期 0.609）→
**{'PASS' if all(ok.values()) else 'FAIL'}**

## 切片

{_md_table(summary)}

期間內的跳躍日：
""" + "\n".join(f"* `{k}`：{v}" for k, v in lists.items()) + f"""

跳躍日每年分布：

{_md_table(by_year)}

倍數敏感度（只供選門檻參考）：

{_md_table(sens)}
"""
    out = ROOT / "reports" / "week04_definitions.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
