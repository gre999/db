"""Generate reports/week03_har.md (+ figures) from the saved predictions.

Run ``python -m src.models.har`` first.

Usage:
    python scripts/make_week03_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_week01_report import (BLUE, INK_2, MUTED, ORANGE,  # noqa: E402
                                _date_axis, _md_table, _save)
from src import data_loader as dl  # noqa: E402
from src import metrics as M  # noqa: E402
from src.models.base import PRED_DIR, load_predictions  # noqa: E402

REPORTS = ROOT / "reports"
AQUA = "#1baf7a"   # categorical slot 3 of the reference palette
MAIN = "rv"


def ann_vol(var: pd.Series) -> pd.Series:
    return np.sqrt(252 * var) * 100


def fig_series(pred: pd.DataFrame) -> str:
    p = pred[(pred["target"] == MAIN)].pivot(index="date", columns="model",
                                              values="y_pred_var")
    act = pred[(pred["target"] == MAIN) & (pred["model"] == "har")] \
        .set_index("date")["y_true_var"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4))
    for ax, (lo, hi), title in [
            (axes[0], (None, None), "Out-of-sample, all folds"),
            (axes[1], ("2020-02-01", "2020-06-30"), "Zoom: 2020 Feb-Jun")]:
        a = act.loc[lo:hi]
        h = p["har"].loc[lo:hi]
        ax.plot(a.index, ann_vol(a), color=MUTED, linewidth=0.8,
                label="Actual RV")
        ax.plot(h.index, ann_vol(h), color=BLUE, linewidth=1.5,
                label="HAR forecast")
        ax.set_yscale("log")
        ax.set_yticks([5, 10, 20, 40, 80])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylabel("Annualized vol, %")
        ax.set_title(title)
        _date_axis(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, "w03_forecast_series.png")


def fig_scatter(pred: pd.DataFrame) -> str:
    p = pred[pred["target"] == MAIN]
    lo = min(p["y_true_log"].min(), p["y_pred_log"].min())
    hi = max(p["y_true_log"].max(), p["y_pred_log"].max())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharex=True, sharey=True)
    for ax, m, title in [(axes[0], "rw", "Random walk"), (axes[1], "har", "HAR")]:
        q = p[p["model"] == m]
        ax.scatter(q["y_pred_log"], q["y_true_log"], s=8, color=BLUE,
                   alpha=0.35, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1)
        r = np.corrcoef(q["y_pred_log"], q["y_true_log"])[0, 1]
        ax.set_title(f"{title}  (corr {r:.2f})")
        ax.set_xlabel("Forecast log RV")
    axes[0].set_ylabel("Actual log RV")
    fig.tight_layout()
    return _save(fig, "w03_scatter.png")


def fig_coefs(coef: pd.DataFrame) -> str:
    c = coef[(coef["target"] == MAIN) & (coef["model"] == "har")]
    names = {"har_logrv_1d": ("β day", BLUE), "har_logrv_5d": ("β week", ORANGE),
             "har_logrv_22d": ("β month", AQUA)}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (param, (lab, col)) in enumerate(names.items()):
        d = c[c["param"] == param].sort_values("fold")
        x = d["fold"] + (i - 1) * 0.08
        ax.errorbar(x, d["value"], yerr=1.96 * d["se"], color=col, marker="o",
                    markersize=5, linewidth=1.5, capsize=3, label=lab)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    years = c.drop_duplicates("fold").set_index("fold")["train_end"] \
        .dt.year.add(1)
    ax.set_xticks(years.index, [f"{f}\n(test {y})" for f, y in years.items()])
    ax.set_title("HAR coefficients by fold (±1.96 Newey-West s.e.)")
    ax.set_ylabel("Coefficient")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right",
              bbox_to_anchor=(1, 1.0))
    return _save(fig, "w03_coefficients.png")


def fig_acf(pred: pd.DataFrame, nlags: int = 20) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, m, title in [(axes[0], "rw", "Random walk"), (axes[1], "har", "HAR")]:
        q = pred[(pred["target"] == MAIN) & (pred["model"] == m)] \
            .sort_values("date")
        e = q["y_true_log"] - q["y_pred_log"]
        r = M.acf(e, nlags)
        ax.bar(np.arange(1, nlags + 1), r, color=BLUE, width=0.7)
        band = 1.96 / np.sqrt(len(e))
        ax.axhspan(-band, band, color=MUTED, alpha=0.2, linewidth=0)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_title(f"{title} residual ACF")
        ax.set_xlabel("Lag (sessions)")
    axes[0].set_ylabel("Autocorrelation")
    fig.tight_layout()
    return _save(fig, "w03_residual_acf.png")


def coef_table(coef: pd.DataFrame, target: str) -> pd.DataFrame:
    c = coef[(coef["target"] == target) & (coef["model"] == "har")]
    val = c.pivot(index="fold", columns="param", values="value")
    se = c.pivot(index="fold", columns="param", values="se")
    info = c.drop_duplicates("fold").set_index("fold")
    out = pd.DataFrame(index=val.index)
    out["train"] = info["train_start"].dt.strftime("%Y-%m") + " → " + \
        info["train_end"].dt.strftime("%Y-%m")
    for p, lab in [("const", "b0"), ("har_logrv_1d", "β_d"),
                   ("har_logrv_5d", "β_w"), ("har_logrv_22d", "β_m")]:
        out[lab] = [f"{v:.3f} ({s:.3f})" for v, s in zip(val[p], se[p])]
    out["β_d+β_w+β_m"] = (val["har_logrv_1d"] + val["har_logrv_5d"]
                          + val["har_logrv_22d"]).round(3)
    out["R² train"] = val["r2_train"].round(3)
    out["σ² resid"] = val["resid_var"].round(3)
    return out.reset_index()


def main() -> None:
    pred = load_predictions("har_baselines")
    coef = pd.read_parquet(PRED_DIR / "har_baselines_coefficients.parquet")
    meta = json.loads((PRED_DIR / "har_baselines_meta.json").read_text(
        encoding="utf-8"))

    # Extra row: HAR without the bias correction (transparency for QLIKE).
    nc = pred[pred["model"] == "har"].copy()
    nc["model"] = "har_nocorr"
    nc["y_pred_var"] = np.exp(nc["y_pred_log"])
    allp = pd.concat([pred, nc], ignore_index=True)

    overall = M.evaluate(allp).xs("all", level="fold").round(4)
    rel = M.relative_to(M.evaluate(pred), "rw").xs("all", level="fold") \
        .xs("har", level="model").round(3)

    dm_rows = []
    for t in ("rv", "rv_on"):
        a = pred[(pred["target"] == t) & (pred["model"] == "har")] \
            .sort_values("date")
        b = pred[(pred["target"] == t) & (pred["model"] == "rw")] \
            .sort_values("date")
        for lname, la, lb in [
                ("squared log error", (a["y_true_log"] - a["y_pred_log"]) ** 2,
                 (b["y_true_log"] - b["y_pred_log"]) ** 2),
                ("QLIKE", M.qlike_loss(a["y_true_var"], a["y_pred_var"]),
                 M.qlike_loss(b["y_true_var"], b["y_pred_var"]))]:
            dm, pv = M.diebold_mariano(np.asarray(la), np.asarray(lb))
            dm_rows.append({"target": t, "loss": lname, "DM (HAR−RW)":
                            round(dm, 2), "p-value": f"{pv:.2g}"})
    dm_tbl = pd.DataFrame(dm_rows)

    wins = {m: bool(rel.loc[MAIN, m] < 1) for m in ("rmse_log", "mae_log",
                                                     "qlike")}
    wins_on = {m: bool(rel.loc["rv_on", m] < 1) for m in wins}

    yearly = M.evaluate(pred[pred["target"] == MAIN], by="year")
    yr = yearly.reset_index().pivot(index="year", columns="model",
                                    values=["rmse_log", "qlike", "n"])
    ytab = pd.DataFrame({
        "year": yr.index.astype(str),
        "n": yr[("n", "har")].astype(int).to_numpy(),
        "RMSE RW": yr[("rmse_log", "rw")].round(3).to_numpy(),
        "RMSE HAR": yr[("rmse_log", "har")].round(3).to_numpy(),
        "RMSE ratio": (yr[("rmse_log", "har")] / yr[("rmse_log", "rw")])
        .round(3).to_numpy(),
        "QLIKE RW": yr[("qlike", "rw")].round(3).to_numpy(),
        "QLIKE HAR": yr[("qlike", "har")].round(3).to_numpy(),
        "QLIKE ratio": (yr[("qlike", "har")] / yr[("qlike", "rw")])
        .round(3).to_numpy(),
    })

    lb = {}
    for m in ("rw", "har"):
        q = pred[(pred["target"] == MAIN) & (pred["model"] == m)] \
            .sort_values("date")
        lb[m] = M.ljung_box(q["y_true_log"] - q["y_pred_log"], 10)
    r1 = {m: M.acf((lambda q: q["y_true_log"] - q["y_pred_log"])(
        pred[(pred["target"] == MAIN) & (pred["model"] == m)]
        .sort_values("date")), 1)[0] for m in ("rw", "har")}

    ct = coef_table(coef, MAIN)
    ct_on = coef_table(coef, "rv_on")
    c = coef[(coef["target"] == MAIN) & (coef["model"] == "har")]
    betas = c[c["param"].str.startswith("har_")]
    n_pos = int((betas["value"] > 0).sum())
    n_sig = int((betas["value"] / betas["se"] > 1.96).sum())

    figs = {"series": fig_series(pred), "scatter": fig_scatter(pred),
            "coef": fig_coefs(coef), "acf": fig_acf(pred)}

    def ok(b):
        return "✅" if b else "❌"

    md = f"""# Week 3 HAR 基準模型

產生時間：{pd.Timestamp.now(tz=dl.ET):%Y-%m-%d %H:%M} ET　·　預測存檔：{meta['saved_at'][:16]}

程式：`src/models/har.py`、`src/models/base.py`、`src/metrics.py`。
預測：`data/processed/predictions/har_baselines.parquet`（{meta['rows']:,} 列）、
係數：`har_baselines_coefficients.parquet`。

## 完成標準

主結果（日內 RV）：HAR 勝過 random walk —
RMSE {ok(wins['rmse_log'])}、MAE {ok(wins['mae_log'])}、QLIKE {ok(wins['qlike'])}
（含隔夜版：RMSE {ok(wins_on['rmse_log'])}、MAE {ok(wins_on['mae_log'])}、QLIKE {ok(wins_on['qlike'])}）。

## 1. 設定

* **列 t = 預測日**。特徵來自 `features_prev_close`（t−1 收盤前可知），標籤為 t 日的 log RV（t 收盤後才知道）；
  等同「用 t 以前資訊預測 log RV(t+1)」。建立資料集時逐列檢查標籤時間 > 特徵截止時間。
* 目標：`rv`（日內，主結果）與 `rv_on`（日內 + 隔夜²）。含隔夜版的 HAR 自變數也改用 rv_on 計算。
* **Random walk**：log RV̂(t) = log RV(t−1)；變異數預測就是 RV(t−1)，不做偏誤修正。
* **HAR**：log RV(t) = b0 + β_d·log RV(t−1) + β_w·log(平均 RV t−5..t−1) + β_m·log(平均 RV t−22..t−1)，OLS；
  標準誤為 Newey-West（5 期）。變異數預測 = exp(預測 + σ²/2)，σ² = 該折訓練段殘差變異數。
* Walk-forward：{meta['splitter']}。兩個模型使用完全相同的樣本日期。
* 誤差：RMSE、MAE 在對數尺度；QLIKE 在變異數尺度。

## 2. 誤差比較（全部樣本外）

{_md_table(overall.reset_index())}

HAR / RW 比值（< 1 代表 HAR 較好）：

{_md_table(rel.reset_index())}

Diebold-Mariano 檢定（負值 = HAR 損失較小）：

{_md_table(dm_tbl)}

`har_nocorr` 是同一組 HAR 預測但不做 exp(σ²/2) 修正，用來看偏誤修正對 QLIKE 的影響。

## 3. 逐年誤差（日內 RV）

{_md_table(ytab)}

## 4. 各折係數

日內 RV（括號內為 Newey-West 標準誤）：

{_md_table(ct)}

三個 β 共 {len(betas)} 個估計值：{n_pos} 個為正，{n_sig} 個 t > 1.96。

![coefficients]({figs['coef']})

含隔夜 RV：

{_md_table(ct_on)}

## 5. 診斷圖

![forecast series]({figs['series']})

![scatter]({figs['scatter']})

![residual acf]({figs['acf']})

殘差自相關（日內 RV，對數尺度）：lag-1 ACF — RW {r1['rw']:.3f}、HAR {r1['har']:.3f}；
Ljung-Box Q(10) — RW {lb['rw'][0]:.1f}（p={lb['rw'][1]:.2g}）、HAR {lb['har'][0]:.1f}（p={lb['har'][1]:.2g}）。
"""
    path = REPORTS / "week03_har.md"
    path.write_text(md, encoding="utf-8")
    print(f"wrote {path}")
    print(overall)


if __name__ == "__main__":
    main()
