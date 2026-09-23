"""Generate reports/week02_features.md (+ figures).

Needs ``python -m src.features`` to have been run. Also runs the week-2
tests and embeds their results.

Usage:
    python scripts/make_week02_report.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET_xml
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_week01_report import (BLUE, INK_2, ORANGE, _date_axis,  # noqa: E402
                                _md_table, _save)
from src import data_loader as dl  # noqa: E402
from src import features as F  # noqa: E402
from src.validation import WalkForwardSplit  # noqa: E402

REPORTS = ROOT / "reports"


def fig_rv_series(y: pd.DataFrame) -> str:
    vol = np.sqrt(252 * y["rv"]) * 100
    m = vol.rolling(21, min_periods=15).mean()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(vol.index, vol, color=BLUE, linewidth=0.6, alpha=0.45)
    ax.plot(m.index, m, color=BLUE, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 40, 80])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("QQQ daily realized volatility (thin: daily, thick: 21-day mean)")
    ax.set_ylabel("Annualized, % (log scale)")
    _date_axis(ax)
    return _save(fig, "w02_rv_series.png")


def fig_rv_hist(y: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bins = np.linspace(min(y["log_rv"].min(), y["log_rv_on"].min()),
                       max(y["log_rv"].max(), y["log_rv_on"].max()), 60)
    ax.hist(y["log_rv"].dropna(), bins=bins, histtype="step", linewidth=1.8,
            color=BLUE, label="Intraday RV")
    ax.hist(y["log_rv_on"].dropna(), bins=bins, histtype="step",
            linewidth=1.8, color=ORANGE, label="Intraday + overnight²")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Distribution of log RV")
    ax.set_xlabel("log RV (daily variance)")
    ax.set_ylabel("Sessions")
    return _save(fig, "w02_rv_hist.png")


def fig_overnight_share(y: pd.DataFrame) -> str:
    v = y.dropna(subset=["rv", "overnight"])
    share = (v["overnight"] ** 2).groupby(v.index.year).sum() / \
        v["rv_on"].groupby(v.index.year).sum() * 100
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(share.index.astype(str), share.to_numpy(), color=BLUE, width=0.8)
    for x, s in enumerate(share.to_numpy()):
        ax.text(x, s, f"{s:.0f}%", ha="center", va="bottom", color=INK_2,
                fontsize=8)
    ax.set_title("Overnight share of total daily variance, by year")
    ax.set_ylabel("Overnight² / (intraday RV + overnight²), %")
    ax.grid(axis="x", visible=False)
    return _save(fig, "w02_overnight_share.png")


def run_tests() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "r.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "tests/test_features.py", "tests/test_validation.py",
                        f"--junitxml={xml}"], cwd=ROOT, check=False,
                       capture_output=True)
        rows = []
        for tc in ET_xml.parse(xml).getroot().iter("testcase"):
            status = "passed"
            for tag in ("failure", "error", "skipped"):
                if tc.find(tag) is not None:
                    status = tag
            rows.append({"file": tc.get("classname").split(".")[-1],
                         "test": tc.get("name"), "result": status,
                         "seconds": round(float(tc.get("time", 0)), 1)})
    return pd.DataFrame(rows)


def main() -> None:
    P = dl.PROCESSED_DIR
    meta = json.loads((P / "features_meta.json").read_text(encoding="utf-8"))
    X = pd.read_parquet(P / "features_open.parquet")
    y = pd.read_parquet(P / "targets_daily.parquet")

    figs = {"series": fig_rv_series(y), "hist": fig_rv_hist(y),
            "share": fig_overnight_share(y)}

    ft = F.feature_table()
    ft = ft[ft["name"].isin(X.columns)]
    stats = X.describe().T
    ft = ft.assign(
        non_null=ft["name"].map(X.notna().sum()).astype(int),
        mean=ft["name"].map(stats["mean"]).map(lambda v: f"{v:.4g}"),
        std=ft["name"].map(stats["std"]).map(lambda v: f"{v:.4g}"),
    )[["name", "cutoff", "definition", "non_null", "mean", "std"]]

    ok = X.drop(columns=["overnight_gap"]).notna().all(axis=1) & y["valid"]
    usable = X.index[ok]
    wf = WalkForwardSplit()
    folds = wf.describe(usable).reset_index()

    vix_note = "未載入（以 --no-cboe 建置）"
    if meta["vol_index"]:
        md = F.MarketData.from_processed()
        stale = dl.align_prev_close(md.vol_index["vix"], X.index)["stale_days"]
        vix_note = (f"對齊到前一個 Cboe 交易日；落後天數分布："
                    f"{stale.value_counts().sort_index().to_dict()}；"
                    f"超過 {meta['config']['vix_max_stale_days']} 天設為 NaN")

    tests = run_tests()
    n_pass = int((tests["result"] == "passed").sum())

    rv_ann = np.sqrt(252 * y["rv"]) * 100
    md_txt = f"""# Week 2 特徵工程報告

產生時間：{pd.Timestamp.now(tz=dl.ET):%Y-%m-%d %H:%M} ET　·　特徵建置：{meta['built_at'][:16]}

程式：`src/features.py`（特徵、RV）、`src/validation.py`（walk-forward）。
矩陣：`data/processed/features_prev_close.parquet`、`features_open.parquet`；
目標（本週不建模，只用於洩漏檢查）：`targets_daily.parquet`。
檢視結論見 [`week02_review_notes.md`](week02_review_notes.md)。

## 1. 資訊截止點

| cutoff | 可用資訊 |
|---|---|
""" + "\n".join(f"| `{k}` | {v} |" for k, v in F.CUTOFF_DOC.items()) + f"""

預測日 t 的矩陣只包含 cutoff ≤ 決策時點的特徵；`features_prev_close` 不含 `overnight_gap`，
`features_open` 才有。

## 2. 特徵清單

設定：`{meta['config']}`

{_md_table(ft)}

VIX / VXN：{vix_note}。

## 3. 已實現波動（RV）

* `rv`：當日 5 分鐘 log 報酬平方和（開盤到收盤，只算日內）。
* `rv_on`：`rv` + 隔夜報酬²（前一交易日收盤 → 當日開盤，已加回股息；前一交易日缺資料時為 NaN）。
* HAR 特徵預設用 `rv`；`python -m src.features --rv-overnight` 切換成 `rv_on`。
* 年化 RV（√(252·rv)）：中位數 {rv_ann.median():.1f}%，P5 {rv_ann.quantile(.05):.1f}%，P95 {rv_ann.quantile(.95):.1f}%，最大 {rv_ann.max():.0f}%（{rv_ann.idxmax():%Y-%m-%d}）。
* 有效交易日（完整且補值比例 ≤ {meta['config']['max_fill_ratio']:.0%}）：{int(y['valid'].sum())} / {len(y)}；
  無效日：{', '.join(d.strftime('%Y-%m-%d') for d in y.index[~y['valid'].astype(bool)]) or '無'}。

![RV series]({figs['series']})

![RV distribution]({figs['hist']})

![overnight share]({figs['share']})

## 4. Walk-forward 切分

預設：訓練 3 年、測試 1 年、每次前移 1 年（滾動）。樣本 = 所有 prev_close 特徵齊全且目標有效的交易日
（{len(usable)} 天，{usable[0]:%Y-%m-%d} → {usable[-1]:%Y-%m-%d}）。
第一個測試年取「第一筆樣本 + 3 年」之後的第一個 1 月 1 日，確保每折訓練都滿 3 年。

{_md_table(folds)}

## 5. 測試結果

{n_pass} / {len(tests)} 通過。

{_md_table(tests)}

* **截斷測試**（`test_truncation_features_unchanged`）：對 4 種 cutoff，各在約 18 個日期
  （隨機 + 缺資料日隔天、無效日隔天、Cboe 獨有日期、VIX 缺口、除息日、半日交易日）刪除截止點之後的所有資料重算，
  當天每個特徵都必須不變。
* **反向對照**（`test_truncation_catches_a_leaky_feature`）：故意註冊一個用當日收盤的洩漏特徵，確認截斷測試抓得到。
* **標籤時間**：每個特徵的截止時間都早於標籤時間（當日收盤）。
* **切分器**：訓練與測試不重疊、訓練全部早於測試、每天最多被測一次、embargo、sklearn `cv=` 相容；
  補值與標準化的參數只由各折訓練段估計。
* 依要求**未**用「特徵與目標相關係數」判斷洩漏。
"""
    path = REPORTS / "week02_features.md"
    path.write_text(md_txt, encoding="utf-8")
    print(f"wrote {path}  ({n_pass}/{len(tests)} tests passed)")


if __name__ == "__main__":
    main()
