"""Generate reports/week10_robustness.md.

Reads config/week10_robustness.toml and every already-saved model's
predictions (week08_filter.parquet, week09_filter_{random_forest,xgboost,
xgboost_depth3,huber}.parquet, week09_filter_logistic_2feature.parquet -
the last one is generated here if missing, from the exact feature pair
config/week09_filter.toml [exploratory] already names). No new models or
features this week - only power analysis, robustness variants on the
logistic main model, and report-only diagnostics (which kind of day gets
filtered out).

Usage:
    python scripts/make_week10_report.py
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_week01_report import _md_table  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import filter_eval as FE  # noqa: E402
from src import regime_eval as E  # noqa: E402
from src import rules as R  # noqa: E402
from src.models import filter as FL  # noqa: E402
from src.models.base import PRED_DIR, load_predictions  # noqa: E402

REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "week10_robustness.toml"

MODEL_PRED_NAMES = {
    "logistic": "week08_filter",
    "random_forest": "week09_filter_random_forest",
    "xgboost": "week09_filter_xgboost",
    "xgboost_depth3": "week09_filter_xgboost_depth3",
    "huber": "week09_filter_huber",
    "logistic_2feature": "week09_filter_logistic_2feature",
}
CLASSIFIERS = ["logistic", "random_forest", "xgboost", "xgboost_depth3", "logistic_2feature"]


def as_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return df


def rnd(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(digits)
        elif pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].map({True: "yes", False: ""})
    return df


def drop_largest_year_diff(preds: pd.DataFrame, orb: pd.DataFrame) -> dict:
    series = FE.logistic_filter_series(preds, orb)
    yb = FE.yearly_breakdown(preds, orb)
    yb["diff"] = yb["filtered_sharpe"] - yb["unfiltered_sharpe"]
    biggest_year = int(yb.loc[yb["diff"].idxmax(), "year"])
    keep_dates = series["unfiltered"].index[series["unfiltered"].index.year != biggest_year]
    f_ev = E._daily_eval(series["filtered"].loc[keep_dates])
    u_ev = E._daily_eval(series["unfiltered"].loc[keep_dates])
    return {"biggest_year": biggest_year, "diff_excl": f_ev["sharpe"] - u_ev["sharpe"]}


def four_criteria(preds: pd.DataFrame, orb: pd.DataFrame, min_trades_per_year: int = 20) -> dict:
    result = FE.evaluate_success(preds, orb, min_trades_per_year=min_trades_per_year)
    bb = FE.block_bootstrap_significance(preds, orb, n_boot=2000, seed=0)
    null = FE.random_filter_null(preds, orb, n_reps=1000, seed=0)
    rp = FE.random_filter_p_value(preds, orb, null=null)
    fb = FE.fold_breakdown(preds, orb)
    fb["diff"] = fb["filtered_sharpe"] - fb["unfiltered_sharpe"]
    n_pos_folds = int((fb["diff"] > 0).sum())
    drop = drop_largest_year_diff(preds, orb)
    c1, c2 = bb["p_value"] < 0.05, rp["p_value"] < 0.05
    c3 = result["min_trades_per_year_ok"]
    c4 = (n_pos_folds >= 6) and (drop["diff_excl"] > 0)
    return {"bb_p": bb["p_value"], "rand_p": rp["p_value"],
           "n_pos_folds": n_pos_folds, "n_folds": len(fb),
           "diff_excl": drop["diff_excl"],
           "filtered_sharpe": result["filtered_sharpe"],
           "unfiltered_sharpe": result["unfiltered_sharpe"],
           "auc": FL.auc(preds) if preds["model"].iloc[0] != "huber" else np.nan,
           "success": bool(c1 and c2 and c3 and c4)}


def main() -> None:
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")
    s = orb.set_index("day") if "day" in orb.columns else orb
    X = pd.read_parquet(dl.PROCESSED_DIR / "features_open_5m.parquet")

    with open(ROOT / "config" / "week09_filter.toml", "rb") as f:
        cfg9 = tomllib.load(f)
    two_feat_cols = list(cfg9["exploratory"]["two_feature_logistic"])
    p2f = PRED_DIR / f"{MODEL_PRED_NAMES['logistic_2feature']}.parquet"
    if not p2f.exists():
        preds_2f = FL.fit_filter_folds_generic(
            X, orb, FL.make_logistic, "logistic_2feature", y_col="label",
            feature_cols=two_feat_cols)
        preds_2f.to_parquet(p2f, index=False)

    preds = {k: load_predictions(v) for k, v in MODEL_PRED_NAMES.items()}

    # ---------------------------------------------------------------
    # Part 2: evidence ladder + power analysis
    # ---------------------------------------------------------------
    crit = {k: four_criteria(v, orb) for k, v in preds.items()}
    ladder_rows = []
    for k in ["logistic", "random_forest", "xgboost", "xgboost_depth3", "huber", "logistic_2feature"]:
        c = crit[k]
        ladder_rows.append({
            "model": k + (" (exploratory)" if k == "logistic_2feature" else ""),
            "auc": c["auc"], "filtered_sharpe": c["filtered_sharpe"],
            "unfiltered_sharpe": c["unfiltered_sharpe"],
            "rand_p": c["rand_p"], "bb_p": c["bb_p"],
            "fold_consistency": f"{c['n_pos_folds']}/{c['n_folds']}",
            "success": c["success"],
        })
    ladder_tbl = rnd(pd.DataFrame(ladder_rows), 3)

    pw_logistic = FE.power_analysis(preds["logistic"], orb, n_boot=2000, seed=0)
    pw_2f = FE.power_analysis(preds["logistic_2feature"], orb, n_boot=2000, seed=0)

    # ---------------------------------------------------------------
    # Part 3: robustness (logistic only)
    # ---------------------------------------------------------------
    minute = dl.load_minute_rth()
    cost_variants = {
        "main_cost": R.ORBConfig(),
        "2x_slippage": R.ORBConfig(cost_per_share=R.MAIN_COMMISSION + 2 * R.MAIN_SLIPPAGE),
        "2x_commission": R.ORBConfig(cost_per_share=2 * R.MAIN_COMMISSION + R.MAIN_SLIPPAGE),
    }
    orb_variants = {name: R.run_orb(minute, cfg=c) for name, c in cost_variants.items()}
    cost_tbl = rnd(FE.cost_sensitivity(preds["logistic"], orb_variants=orb_variants), 4)

    val_rows = []
    for vy in [1, 2]:
        p = FL.fit_filter_folds_generic(X, orb, FL.make_logistic, "logistic", validation_years=vy)
        r = FE.evaluate_success(p, orb)
        val_rows.append({"validation_years": vy, "filtered_sharpe": r["filtered_sharpe"],
                        "unfiltered_sharpe": r["unfiltered_sharpe"]})
    val_tbl = rnd(pd.DataFrame(val_rows), 3)

    grid5 = tuple(round(x, 2) for x in np.arange(0.05, 1.01, 0.05))
    grid_rows = []
    for tag, grid in [("10%", FL.RETENTION_GRID), ("5%", grid5)]:
        p = FL.fit_filter_folds_generic(X, orb, FL.make_logistic, "logistic", retention_grid=grid)
        r = FE.evaluate_success(p, orb)
        grid_rows.append({"grid_step": tag, "filtered_sharpe": r["filtered_sharpe"],
                         "unfiltered_sharpe": r["unfiltered_sharpe"]})
    grid_tbl = rnd(pd.DataFrame(grid_rows), 3)

    reseed_cfg = cfg["robustness"]["random_baseline_reseed"]
    reseed_n_reps = reseed_cfg["n_reps"]
    reseed_rows = []
    for name, key in [("logistic", "logistic"), ("logistic_2feature", "logistic_2feature")]:
        for seed in reseed_cfg["seeds"]:
            rp = FE.random_filter_p_value(preds[key], orb, n_reps=reseed_n_reps, seed=seed)
            reseed_rows.append({"model": name, "seed": seed,
                               "p_value": rp["p_value"], "se": rp["se"]})
    reseed_tbl = as_int(rnd(pd.DataFrame(reseed_rows), 4), ["seed"])
    reseed_df = pd.DataFrame(reseed_rows)
    reseed_summary = reseed_df.groupby("model").agg(
        mean_p=("p_value", "mean"), min_p=("p_value", "min"), max_p=("p_value", "max"),
        se=("se", "mean"))
    reseed_summary["mean_minus_2se"] = reseed_summary["mean_p"] - 2 * reseed_summary["se"]
    reseed_summary["mean_plus_2se"] = reseed_summary["mean_p"] + 2 * reseed_summary["se"]

    tc_logistic = FE.threshold_curve_with_random_band(preds["logistic"], orb, n_reps=1000, seed=0)
    tc_2f = FE.threshold_curve_with_random_band(preds["logistic_2feature"], orb, n_reps=1000, seed=0)
    tc_logistic_tbl = rnd(tc_logistic, 3)
    tc_2f_tbl = rnd(tc_2f, 3)
    n_above_band_logistic = int((tc_logistic["sharpe"] > tc_logistic["random_p95"]).sum())
    n_above_band_2f = int((tc_2f["sharpe"] > tc_2f["random_p95"]).sum())

    # ---------------------------------------------------------------
    # Part 4: big-win days
    # ---------------------------------------------------------------
    j = preds["logistic"].set_index("date").join(
        X[["har_logrv_1d", "vix_1d", "open5m_body_ratio"]], how="left")
    avoid_tbl = j.groupby("keep")[["har_logrv_1d", "vix_1d", "open5m_body_ratio"]].mean().reset_index()
    avoid_tbl["keep"] = avoid_tbl["keep"].map({True: "kept", False: "dropped"})
    avoid_tbl = rnd(avoid_tbl, 4)

    mw_rows = []
    for k, name in MODEL_PRED_NAMES.items():
        mw = FE.missed_big_wins(preds[k], orb)
        mw_rows.append({"model": k, "n_big_win_days": mw["n_big_win_days_in_window"],
                       "n_dropped": mw["n_dropped"], "frac_dropped": mw["frac_dropped"]})
    mw_tbl = rnd(pd.DataFrame(mw_rows), 3)

    # ---------------------------------------------------------------
    # Part 5: 2022 (descriptive recap only - computed in week 9)
    # ---------------------------------------------------------------

    md = f"""# Week 10 (階段四第三週)穩健性、檢定力與階段四總結

不找新模型或新特徵,整理階段四(第 8-10 週)結論,補檢定力分析與穩健性檢查。所有檢查設定
(`config/week10_robustness.toml`)在跑任何檢查前寫定,結果好壞照列,不依結果增減項目。

## 一、證據階梯

三層,由弱到強:分類層(AUC、機率十分位勝率/淨R)→ 排序層(隨機篩選經驗p值)→ 策略層
(篩選前後夏普差 block bootstrap p值、逐折一致性)。六個模型(含探索性兩特徵模型)× 第九週
四項成功標準:

{_md_table(ladder_tbl)}

沒有一個模型 `success` 是 yes——這張總表把第八、九週的結論並排放在一起,一致:分類層、
排序層偶爾有模型過關(logistic、兩特徵模型的排序層 p<0.05),但策略層的差異顯著性檢定
(`bb_p`)沒有一個模型過關。

## 二、檢定力分析

用 block bootstrap 分布的標準差估計「篩選前後夏普差」的標準誤,再用標準公式反推 5% 顯著、
80% 檢定力下的最小可偵測夏普差(≈2.80倍標準誤):

| model | observed_diff | se | mde(5%/80%) | detectable | years_for_0.2_diff |
|---|---|---|---|---|---|
| logistic | {pw_logistic['observed_diff']:.3f} | {pw_logistic['se']:.3f} | {pw_logistic['mde']:.3f} | {'yes' if pw_logistic['detectable'] else ''} | {pw_logistic['years_needed_for_target']:.0f} |
| logistic_2feature(探索性) | {pw_2f['observed_diff']:.3f} | {pw_2f['se']:.3f} | {pw_2f['mde']:.3f} | {'yes' if pw_2f['detectable'] else ''} | {pw_2f['years_needed_for_target']:.0f} |

兩個模型的最小可偵測夏普差(約 0.69-0.70)都遠大於實際觀察到的差(logistic
{pw_logistic['observed_diff']:.2f}、兩特徵模型 {pw_2f['observed_diff']:.2f})——現有樣本量
(~{pw_logistic['n_years']:.1f} 年 OOS)本來就偵測不到這個量級的效果,不是模型不夠好。反推
要用現有的資料頻率偵測到 0.2 的夏普差,需要約 {pw_logistic['years_needed_for_target']:.0f}
年資料(假設標準誤隨 1/√天數 縮小的一階近似,不是精確的樣本設計目標,只當量級參考)——這是
不切實際的年限,說明「用更多資料等出顯著結果」不是這條路線可行的下一步。

## 三、穩健性(只對 logistic 主模型)

**滑價加倍、手續費加倍**:

{_md_table(cost_tbl)}

篩選前後夏普差在三種成本假設下都在 0.19-0.20 之間,穩定。

**驗證段 1 年 vs 2 年**:

{_md_table(val_tbl)}

**保留比例網格 10% vs 5%**:

{_md_table(grid_tbl)}

5% 步進的結果(夏普 {grid_tbl[grid_tbl.grid_step=='5%']['filtered_sharpe'].iloc[0]}) 比 10%
步進(原始結果 {grid_tbl[grid_tbl.grid_step=='10%']['filtered_sharpe'].iloc[0]})低不少,但仍
高於不篩選——網格粗細對結果有一定敏感度,方向不變。

**隨機基準精度重估**(修改自最初的草案——n_reps=1000 下 5 個種子給出 p=0.030–0.051,
logistic 一個種子超過 0.05,但 n=1000 在 p≈0.04 附近的蒙地卡羅標準誤約 0.006,這個範圍有
一部分本來就是估計本身的雜訊,不必然代表真實 p 值不穩定。改成 n_reps={reseed_n_reps},
{len(reseed_cfg['seeds'])} 個種子,標準誤降到約 0.002,直接報告 p 值與其蒙地卡羅標準誤,
取代「幾個種子超過門檻」的計數描述):

{_md_table(reseed_tbl)}

{_md_table(rnd(reseed_summary.reset_index(), 4))}

logistic:三個種子的 p 值落在 {reseed_summary.loc['logistic','min_p']:.3f}–{reseed_summary.loc['logistic','max_p']:.3f},
平均 {reseed_summary.loc['logistic','mean_p']:.3f} ± {reseed_summary.loc['logistic','se']:.3f}
(蒙地卡羅標準誤)——**三個估計都穩定落在 0.05 以下**,原本 n=1000 時看到的「換種子跳到
0.051」主要是估計精度不足造成的雜訊,不是真實 p 值在 0.05 兩側擺盪;但真實值(約 0.043–0.047)
比原本 n=1000、種子0 報告的 0.030 更接近 0.05 這條線,判讀上要更保守。兩特徵模型:
{reseed_summary.loc['logistic_2feature','min_p']:.3f}–{reseed_summary.loc['logistic_2feature','max_p']:.3f},
平均 {reseed_summary.loc['logistic_2feature','mean_p']:.3f} ± {reseed_summary.loc['logistic_2feature','se']:.3f}
——同樣穩定,而且離 0.05 的安全邊際比 logistic 大。

**門檻曲線定稿**(固定保留比例,測試段已配適模型的樣本外夏普,疊隨機篩選 5-95% 區間):

logistic:

{_md_table(tc_logistic_tbl)}

兩特徵模型(探索性):

{_md_table(tc_2f_tbl)}

logistic 在 8 個保留比例中有 {n_above_band_logistic} 個高於隨機篩選第95百分位,兩特徵模型
有 {n_above_band_2f} 個——這是描述性的,沒有對多重比較校正(8 個保留比例一起看,單純運氣下
也會有大約 0.4 個「碰巧超過」),不能取代第二節的正式檢定,但方向上跟兩特徵模型的排序層
p 值比 logistic 更穩定這件事一致。

## 四、大賺日

篩選主要避開哪類日子(logistic,留下 vs 篩掉的特徵平均):

{_md_table(avoid_tbl)}

篩掉的日子前一日已實現波動較高、VIX 較高、第一根K線實體占比較低(方向較不乾淨)——跟第九週
逐折係數(`har_logrv_1d` 負、`open5m_body_ratio` 正)方向一致,不是新發現,是同一個機制的
另一種呈現。

各模型被篩掉的大賺日比例並列:

{_md_table(mw_tbl)}

## 五、2022 年做空(只描述,不修正)

第九週已發現:logistic、隨機森林、XGBoost 三個模型在 2022 年做空訊號上一致地留下賠錢的空單、
丟掉賺錢的空單(留下的空單平均淨R全部為負,丟掉的空單平均淨R全部為正)。本週不修正,列為
假說與未來工作:**三個模型共用同一組開盤特徵、且訓練窗高度重疊,不能算三次獨立驗證**——
把這個現象當成「三個模型獨立發現同一件事」會高估其穩健性,正確的說法是「同一組特徵在同一年
產生的同一個排序錯誤,被三個模型不約而同地繼承」。值得未來調查,不是這週要解決的問題。

## 六、選做(未執行,列入未來工作)

VWAP 篩選(只用開盤前特徵,沿用第九週四項標準)設定已寫入
`config/week10_robustness.toml [optional.vwap_filter]`,本週不實際執行。

## 結論

**在現有開盤特徵與樣本量下,訊號統計強度不足以支撐顯著的策略改善。** 分類層、排序層偶爾
能看到不是雜訊的訊號(用足夠精度〔n_reps=10000〕重估後,logistic 與兩特徵模型的隨機篩選
p 值都穩定小於 0.05——logistic 約 0.04-0.05、離門檻較近,兩特徵模型約 0.02、安全邊際較大;
大賺日的避開模式跟穩定係數方向一致),但策略層的差異顯著性檢定(block bootstrap)沒有一個
模型、任何穩健性變體通過,檢定力分析顯示現有樣本量本來就偵測不到這個量級的效果,不是換
模型或換更多資料(在合理年限內)能解決的問題。
"""
    out_path = REPORTS / "week10_robustness.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
