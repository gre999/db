"""Generate reports/week04_xgb.md (+ figures).

Reads ``har_baselines.parquet`` (RW, HAR - not retrained) and
``week4_models.parquet`` (HAR-X, XGBoost, RF). Run ``python -m
src.models.week4`` first.

Usage:
    python scripts/make_week04_report.py
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
AQUA, YELLOW = "#1baf7a", "#eda100"
MODELS = ["rw", "har", "harx", "xgb", "rf"]
LABEL = {"rw": "Random walk", "har": "HAR", "harx": "HAR-X", "xgb": "XGBoost",
         "rf": "Random forest"}
COLOR = {"har": BLUE, "xgb": ORANGE, "harx": AQUA, "rf": YELLOW}
SLICES = ["covid_2020", "tariff_2025", "jump_day"]


def load_all() -> tuple[pd.DataFrame, int]:
    base = load_predictions("har_baselines")
    base = base[base["target"] == "rv"]
    new = load_predictions("week4_models")
    pred = pd.concat([base, new], ignore_index=True)
    common = set.intersection(*[set(g) for _, g in pred.groupby("model")["date"]])
    n_drop = pred["date"].nunique() - len(common)
    pred = pred[pred["date"].isin(common)].sort_values(["model", "date"])
    return pred.reset_index(drop=True), n_drop


def losses(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred.assign(ql=M.qlike_loss(pred["y_true_var"], pred["y_pred_var"]),
                    se=(pred["y_true_log"] - pred["y_pred_log"]) ** 2)
    return p


def dm_table(p: pd.DataFrame, slices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    har = p[p["model"] == "har"].set_index("date")
    for m in ["rw", "harx", "xgb", "rf"]:
        q = p[p["model"] == m].set_index("date")
        for sname in ["all"] + SLICES:
            idx = har.index if sname == "all" else \
                har.index[slices.reindex(har.index)[sname].to_numpy()]
            if len(idx) < 10:
                continue
            for loss, col in [("QLIKE", "ql"), ("sq. log err", "se")]:
                dm, pv = M.diebold_mariano(q.loc[idx, col], har.loc[idx, col])
                rows.append({"model": LABEL[m], "slice": sname, "n": len(idx),
                             "loss": loss,
                             "mean diff (model−HAR)": f"{(q.loc[idx, col] - har.loc[idx, col]).mean():+.4f}",
                             "DM": round(dm, 2), "p-value": f"{pv:.2g}"})
    return pd.DataFrame(rows)


def fig_cum_qlike(p: pd.DataFrame, cfg: dict) -> str:
    har = p[p["model"] == "har"].set_index("date")["ql"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for per in cfg["periods"].values():
        ax.axvspan(pd.Timestamp(per["start"]), pd.Timestamp(per["end"]),
                   color=MUTED, alpha=0.18, linewidth=0)
    for m in ["harx", "xgb", "rf"]:
        d = (p[p["model"] == m].set_index("date")["ql"] - har).cumsum()
        ax.plot(d.index, d, color=COLOR[m], linewidth=1.6, label=LABEL[m])
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_title("Cumulative QLIKE difference vs HAR (below 0 = beating HAR)")
    ax.set_ylabel("Σ (QLIKE model − QLIKE HAR)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _date_axis(ax)
    return _save(fig, "w04_cum_qlike.png")


def fig_zoom(p: pd.DataFrame, cfg: dict) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.6))
    act = p[p["model"] == "har"].set_index("date")["y_true_var"]
    for ax, (key, per) in zip(axes, cfg["periods"].items()):
        lo = pd.Timestamp(per["start"]) - pd.Timedelta(days=14)
        hi = pd.Timestamp(per["end"]) + pd.Timedelta(days=7)
        a = act.loc[lo:hi]
        ax.plot(a.index, np.sqrt(252 * a) * 100, color=MUTED, linewidth=0.9,
                label="Actual RV")
        for m in ["har", "harx", "xgb"]:
            q = p[p["model"] == m].set_index("date")["y_pred_var"].loc[lo:hi]
            ax.plot(q.index, np.sqrt(252 * q) * 100, color=COLOR[m],
                    linewidth=1.5, label=LABEL[m])
        ax.axvspan(pd.Timestamp(per["start"]), pd.Timestamp(per["end"]),
                   color=MUTED, alpha=0.12, linewidth=0)
        ax.set_yscale("log")
        ax.set_yticks([5, 10, 20, 40, 80])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylabel("Annualized vol, %")
        ax.set_title(per["label"])
        _date_axis(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, lab, frameon=False, fontsize=8, ncol=4, loc="lower center")
    return _save(fig, "w04_jump_zoom.png")


def fig_importance(imp: pd.DataFrame, model: str) -> str:
    q = imp[imp["model"] == model]
    mean = q.groupby(["slice", "feature"])["importance"].mean().unstack(0)
    order = mean["all"].sort_values().index
    cols = [c for c in ["all", "jump_day"] if c in mean]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(order))
    h = 0.38
    for i, (c, col) in enumerate(zip(cols, [BLUE, ORANGE])):
        ax.barh(y + (i - 0.5) * h, mean.loc[order, c], height=h, color=col,
                label={"all": "All test days", "jump_day": "Jump days"}[c])
    ax.set_yticks(y, order)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Increase in RMSE (log) when shuffled, mean over folds")
    ax.set_title(f"{LABEL[model]}: permutation importance on test slices")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, f"w04_importance_{model}.png")


def rank_table(imp: pd.DataFrame, model: str) -> tuple[pd.DataFrame, float]:
    q = imp[(imp["model"] == model) & (imp["slice"] == "all")]
    r = q.pivot(index="feature", columns="fold", values="importance") \
        .rank(ascending=False).astype(int)
    r = r.loc[r.mean(axis=1).sort_values().index]
    corr = r.corr(method="spearman").to_numpy()
    mean_rho = corr[np.triu_indices_from(corr, 1)].mean()
    r.columns = [f"fold {c}" for c in r.columns]
    return r.reset_index(), float(mean_rho)


def main() -> None:
    pred, n_drop = load_all()
    p = losses(pred)
    cfg = M.load_eval_config()
    t = pd.read_parquet(dl.PROCESSED_DIR / "targets_daily.parquet")
    rv = t["rv"].where(t["valid"].astype(bool))
    dates = pd.DatetimeIndex(sorted(pred["date"].unique()))
    slices = M.eval_slices(dates, rv, cfg)

    overall = M.evaluate(pred).xs("all", level="fold").droplevel("target")
    overall = overall.loc[[m for m in MODELS if m in overall.index]]
    rel = (overall[["rmse_log", "mae_log", "qlike"]] /
           overall.loc["har", ["rmse_log", "mae_log", "qlike"]]).round(3)
    ov = overall.round(4).join(rel.add_suffix(" / HAR"))
    ov.index = [LABEL[m] for m in ov.index]

    yearly = M.evaluate(pred, by="year").reset_index()
    yq = yearly.pivot(index="year", columns="model", values="qlike")[MODELS]
    yq = yq.round(3)
    yq["XGB / HAR"] = (yq["xgb"] / yq["har"]).round(3)
    yq["HAR-X / HAR"] = (yq["harx"] / yq["har"]).round(3)
    yq = yq.rename(columns=LABEL).reset_index()

    ss = M.slice_scores(pred, slices).droplevel("target").reset_index()
    sq = ss.pivot(index="slice", columns="model", values="qlike")[MODELS] \
        .round(3)
    sr = ss.pivot(index="slice", columns="model", values="rmse_log")[MODELS] \
        .round(3)
    n_slice = ss.drop_duplicates("slice").set_index("slice")["n"]
    sq.insert(0, "n", n_slice)
    sr.insert(0, "n", n_slice)
    sq = sq.rename(columns=LABEL).reset_index()
    sr = sr.rename(columns=LABEL).reset_index()

    dm = dm_table(p, slices)
    dm_all = dm[(dm["slice"] == "all") & (dm["loss"] == "QLIKE")]

    imp = pd.read_parquet(PRED_DIR / "week4_models_importance.parquet")
    params = pd.read_parquet(PRED_DIR / "week4_models_params.parquet")
    coef = pd.read_parquet(PRED_DIR / "week4_models_coefficients.parquet")
    meta = json.loads((PRED_DIR / "week4_models_meta.json").read_text(
        encoding="utf-8"))

    rk_x, rho_x = rank_table(imp, "xgb")
    rk_r, rho_r = rank_table(imp, "rf")
    imp_tbl = imp.groupby(["model", "slice", "feature"])["importance"].mean() \
        .unstack("slice")
    imp_tbl = imp_tbl[[c for c in ["all"] + SLICES if c in imp_tbl]]
    top = {}
    for m in ["xgb", "rf", "harx"]:
        if m in imp_tbl.index.get_level_values(0):
            x = imp_tbl.loc[m].round(4).sort_values("all", ascending=False)
            top[m] = x.reset_index()

    def as_int(df, cols):
        df = df.copy()
        for c in cols:
            df[c] = df[c].map(lambda v: "None" if pd.isna(v) else str(int(v)))
        return df

    xp = as_int(params[params["model"] == "xgb"][
        ["fold", "test_year", "max_depth", "learning_rate", "n_estimators",
         "val_rmse"]].round(4), ["fold", "test_year", "max_depth",
                                 "n_estimators"])
    rp = as_int(params[params["model"] == "rf"][
        ["fold", "test_year", "max_depth", "min_samples_leaf", "max_features",
         "val_rmse"]].round(4), ["fold", "test_year", "max_depth",
                                 "min_samples_leaf"])
    hx_cols = ["har_logrv_1d", "har_logrv_5d", "har_logrv_22d", "vix_1d",
               "vxn_1d"]
    hx = coef[coef["param"].isin(hx_cols)]
    hxv = hx.pivot(index="fold", columns="param", values="value")[hx_cols]
    hxs = hx.pivot(index="fold", columns="param", values="se")[hx_cols]
    hxt = pd.DataFrame({c: [f"{v:.3f} ({s:.3f})" for v, s in
                            zip(hxv[c], hxs[c])] for c in hxv.columns},
                       index=hxv.index).reset_index()

    figs = {"cum": fig_cum_qlike(p, cfg), "zoom": fig_zoom(p, cfg),
            "imp_x": fig_importance(imp, "xgb"),
            "imp_r": fig_importance(imp, "rf")}

    def verdict(m):
        r = dm_all[dm_all["model"] == LABEL[m]].iloc[0]
        ratio = rel.loc[m, "qlike"]
        sig = float(r["p-value"]) < 0.05
        if ratio < 1 and sig:
            return f"**顯著較好**（QLIKE 比值 {ratio:.3f}，DM p = {r['p-value']}）"
        if ratio < 1:
            return f"較好但**不顯著**（QLIKE 比值 {ratio:.3f}，DM p = {r['p-value']}）"
        if sig:
            return f"**顯著較差**（QLIKE 比值 {ratio:.3f}，DM p = {r['p-value']}）"
        return f"較差但不顯著（QLIKE 比值 {ratio:.3f}，DM p = {r['p-value']}）"

    md = f"""# Week 4 XGBoost 波動模型

產生時間：{pd.Timestamp.now(tz=dl.ET):%Y-%m-%d %H:%M} ET　·　預測存檔：{meta['saved_at'][:16]}

程式：`src/models/trees.py`（XGBoost、隨機森林、調參、permutation importance）、
`src/models/har.py`（HAR-X）、`src/models/week4.py`（執行）。
預測：`data/processed/predictions/week4_models.parquet`；HAR、RW 讀自 `har_baselines.parquet`（未重新訓練）。
切片定義：`config/evaluation.toml`（在看模型結果前確認）；定義與基準核對見 `week04_definitions.md`。

## 結論：XGBoost 對 HAR（QLIKE，全部樣本外）

* XGBoost：{verdict('xgb')}
* 隨機森林：{verdict('rf')}
* HAR-X：{verdict('harx')}

分期間與切片的結果見第 3、4 節；解讀見 `week04_review_notes.md`。

## 1. 設定

* 目標：t 日的 log RV（日內，主結果），特徵來自 `features_prev_close`（t−1 收盤前可知）。
* **HAR-X**：HAR 三項 + log VIX + log VXN，OLS。
* **XGBoost / 隨機森林**：{len(meta['tree_features'])} 個特徵：`{', '.join(meta['tree_features'])}`。
* 調參只在每折訓練段內：訓練段最後一年當驗證段；XGBoost 用 early stopping 決定樹數，
  網格 max_depth × learning_rate；隨機森林網格 max_depth × min_samples_leaf × max_features。
  選定後以整個訓練段重新訓練。測試段不參與。
* 變異數預測 = exp(預測 + σ²/2)。HAR / HAR-X 的 σ² = 訓練段殘差變異數；樹模型的訓練段內殘差會嚴重低估，
  改用**驗證段**的殘差均方（仍在訓練段內）。
* 比較樣本：所有模型都有預測的日期（{len(dates)} 天；因特徵缺值少了 {n_drop} 天）。

## 2. 全部樣本外

{_md_table(ov.reset_index(names='model'))}

Diebold-Mariano（各模型對 HAR，QLIKE；負值 = 優於 HAR）：

{_md_table(dm_all.drop(columns=['slice', 'loss']))}

![cumulative qlike]({figs['cum']})

灰色區塊為兩段跳躍期間。曲線往下代表該段期間贏 HAR，往上代表輸。

## 3. 逐年 QLIKE

{_md_table(yq)}

## 4. 跳躍期間與跳躍日

QLIKE：

{_md_table(sq)}

RMSE（log）：

{_md_table(sr)}

Diebold-Mariano（對 HAR，各切片；樣本小，p 值僅供參考）：

{_md_table(dm[dm['slice'] != 'all'])}

![jump zoom]({figs['zoom']})

## 5. 特徵重要度（測試段 permutation importance）

以測試段打亂單一特徵後 log RMSE 的增加量，{imp['fold'].nunique()} 折平均。相關的特徵（HAR 三項、VIX/VXN）會互相分攤重要度。

XGBoost（各折排名平均 Spearman 相關 {rho_x:.2f}）：

![importance xgb]({figs['imp_x']})

{_md_table(top['xgb'])}

各折排名（1 = 最重要）：

{_md_table(rk_x)}

隨機森林（各折排名平均 Spearman 相關 {rho_r:.2f}）：

![importance rf]({figs['imp_r']})

{_md_table(top['rf'])}

HAR-X：

{_md_table(top['harx'])}

## 6. 每折選到的參數

XGBoost：

{_md_table(xp)}

隨機森林：

{_md_table(rp)}

HAR-X 係數（括號內為 Newey-West 標準誤；VIX、VXN 以 log 進入）：

{_md_table(hxt)}
"""
    out = REPORTS / "week04_xgb.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    print(ov)


if __name__ == "__main__":
    main()
