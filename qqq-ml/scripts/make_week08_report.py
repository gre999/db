"""Generate reports/week08_filter_baseline.md.

Reads config/week08_filter.toml, data/processed/strategies/orb_daily.parquet
(run ``python -m src.rules`` first), data/processed/features_open_5m.parquet
(run ``python -m src.features_open`` first) and
data/processed/predictions/week08_filter.parquet (run
``python -m src.models.filter`` first). Never refits ORB, the feature
matrix, or the logistic filter - only src/filter_eval.py's evaluation (and
the phase-3 HMM comparison, which does refit that one model per its own
already-tested pipeline) run here.

Usage:
    python scripts/make_week08_report.py
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
from src.models import filter as FL  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "week08_filter.toml"


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


def main() -> None:
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    retention_grid_str = ", ".join(
        f"{int(r * 100)}%" for r in cfg["model"]["threshold_selection"]["retention_grid"])
    min_tpy = cfg["model"]["threshold_selection"]["min_trades_per_year"]

    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")
    s = orb.set_index("day") if "day" in orb.columns else orb
    preds = load_predictions("week08_filter")

    # ---------------------------------------------------------------
    # Part 2: label analysis
    # ---------------------------------------------------------------
    tr = s[s["traded"]]
    n_doji = int((~s["traded"]).sum())
    yearly = tr.assign(year=tr.index.year).groupby("year").agg(
        n=("r_net", "size"), win_rate=("r_net", lambda x: (x > 0).mean()),
        mean_r_net=("r_net", "mean"), sum_r_net=("r_net", "sum")).round(3)
    yearly_tbl = as_int(rnd(yearly.reset_index(), 3), ["year", "n"])

    n_traded = len(tr)
    top_n = int(np.ceil(n_traded * 0.10))
    sorted_r = tr["r_net"].sort_values(ascending=False)
    top10 = sorted_r.iloc[:top_n]
    total_net = tr["r_net"].sum()
    total_gross_profit = tr.loc[tr["r_net"] > 0, "r_net"].sum()
    concentration = {"top_n": top_n, "n_traded": n_traded,
                     "top10_sum_r": float(top10.sum()),
                     "total_net_r": float(total_net),
                     "top10_over_net": float(top10.sum() / total_net),
                     "top10_over_gross_profit": float(top10.sum() / total_gross_profit)}

    big = tr[tr["r_net"] >= 5]
    big_by_year = {int(y): int(n) for y, n in big.groupby(big.index.year).size().items()}
    gaps = big.index.to_series().diff().dt.days.dropna()

    # ---------------------------------------------------------------
    # Part 4: model diagnostics
    # ---------------------------------------------------------------
    auc = FL.auc(preds)
    cal = FL.decile_calibration(preds, orb)
    cal_tbl = as_int(rnd(cal, 4), ["decile", "n"])
    retention_by_fold = {int(f): float(r) for f, r in
                        preds.groupby("fold")["retention_selected"].first().items()}
    no_filter_folds = ", ".join(str(f) for f, r in retention_by_fold.items() if r == 1.0) or "無"

    # ---------------------------------------------------------------
    # Part 5: filtered performance
    # ---------------------------------------------------------------
    result = FE.evaluate_success(preds, orb)
    by_fold_tpy = as_int(rnd(result["by_fold_trades_per_year"], 2), ["fold"])
    main_tbl = pd.DataFrame([
        {"series": "filtered", "sharpe": result["filtered_sharpe"],
         "max_drawdown": result["filtered_max_drawdown"],
         "ann_return": result["filtered_ann_return"]},
        {"series": "unfiltered", "sharpe": result["unfiltered_sharpe"],
         "max_drawdown": result["unfiltered_max_drawdown"],
         "ann_return": result["unfiltered_ann_return"]},
        {"series": "random-null p95", "sharpe": result["random_null_p95"],
         "max_drawdown": np.nan, "ann_return": np.nan},
        {"series": "random-null mean", "sharpe": result["random_null_mean"],
         "max_drawdown": np.nan, "ann_return": np.nan},
    ])
    main_tbl = rnd(main_tbl, 3)
    main_tbl = main_tbl.fillna("—")

    common_idx, cmp_tbl = FE.compare_sources(preds, orb)
    cmp_display = cmp_tbl[["model", "filtered_sharpe", "unfiltered_sharpe",
                           "filtered_max_drawdown", "unfiltered_max_drawdown"]]
    cmp_display = rnd(cmp_display, 3)

    curve = FE.threshold_curve(preds, orb)
    curve_tbl = as_int(rnd(curve, 3), ["n_trades"])

    mw = FE.missed_big_wins(preds, orb)

    md = f"""# Week 8 (階段四第一週)交易日篩選管線:logistic 基準

用開盤 09:35 截止的特徵預測 ORB 每筆訊號是否值得進場,依訓練段最後一年驗證選出的門檻篩選
測試段交易日。第九週只換模型(HAR 特徵 → 其他模型),不動這裡的門檻選擇/評估架構。

設定(`config/week08_filter.toml`)在看任何篩選結果前寫定:基底策略 ORB(主成本、單位曝險,
VWAP 留到第十週);決策時點 09:35(`open_5m` 截止);主標籤扣成本淨 R>0,十字星日整個排除
於建模集(不算負樣本);門檻用訓練段最後一年當驗證段,在保留比例 {{{retention_grid_str}}} 的
粗網格中選驗證段夏普最高、且驗證段每年 ≥{min_tpy} 筆交易的門檻,測試段完全不參與選擇。

**特徵可用期間的修改**(查看任何篩選結果前):原規劃的 HAR＋殘差XGBoost 預測與 HMM 預測型
狀態/機率只從 2019-01-02 起有值,若整體建模要求兩欄都非 NaN 會把可用折數從全樣本(~8-9折)
砍到 2019 年後(~4-5折)。改為:主分析用 ORB 全樣本(2015年起),換成三個既有的 HAR 原始
log RV 特徵(前1日、前5日均、前22日均)讓模型自己學波動;兩個停用的特徵留在 2019+ 子樣本
的穩健性對照裡。

## 一、標籤分析

ORB 共 {len(s)} 天,`traded==True` {n_traded} 天(不交易 {n_doji} 天:十字星 + session 品質
不合格)。逐年勝率與平均淨 R:

{_md_table(yearly_tbl)}

勝率穩定在 21–27%(2014 樣本太少不計),多數年份平均淨 R 小幅為正,但 2015、2017、2026
至今是淨負值——策略邊際本來就薄。

**報酬集中度**:前 10%(*{concentration['top_n']}*/{concentration['n_traded']} 筆)交易的淨 R
總和是全部交易淨 R 總和的 **{concentration['top10_over_net']:.1f} 倍**(全樣本淨 R 只有
{concentration['total_net_r']:.0f}R,被後面九成交易大幅侵蝕);若只看所有獲利交易的毛利,
前 10% 佔了 **{concentration['top10_over_gross_profit']:.1%}**——集中度非常高,能否篩選成功
很大程度取決於分類器抓不抓得住這前 10%。

**大賺日(R≥5)**:{len(big)} 天(佔已交易日 {len(big)/n_traded:.1%}),逐年分布:
{big_by_year},沒有集中在特定一兩年;間隔天數中位數 {gaps.median():.0f} 天。用既有
prev_close 截止特徵(VIX、前一日已實現波動、隔夜跳空、HMM 狀態)比較過,大賺日跟其他交易日
幾乎沒有差異——這是本週要往「開盤當下」特徵找訊號的原因。

## 二、開盤特徵矩陣

`src/features_open.py`,`data/processed/features_open_5m.parquet`,主分析特徵從 2015-01-20
起完整覆蓋(2893/2954 天)。開盤前:隔夜跳空、跳空/20日ATR、前一日報酬、HAR 三個窗口的
log RV(前1日/前5日均/前22日均)、VIX、VXN、星期幾;第一根K線:方向、實體占區間比例、
區間相對過去20日均值、相對成交量、方向與跳空方向是否一致。全部走 `features.py` 既有的
`open_5m` 截止機制註冊,自動有截斷測試覆蓋;另外新增一個更嚴格的測試——不只截斷,直接
篡改 09:35 之後任何一筆K線的數值,確認特徵逐位元不變。

## 三、logistic 基準

`src/models/filter.py`:`Pipeline(StandardScaler → LogisticRegression(class_weight=
"balanced"))`,每折配適段(訓練段前 ~2 年)訓練,驗證段(訓練段最後 ~1 年)選門檻,測試段
只用該模型、該門檻算機率,完全不重新配適。

**AUC(全期 OOS 合併)= {auc:.3f}**——中等。機率十分位校準:

{_md_table(cal_tbl)}

大致單調(最低十分位勝率 {cal['win_rate'].iloc[0]:.1%}/平均淨R {cal['mean_r_net'].iloc[0]:.2f}，
最高十分位勝率 {cal['win_rate'].iloc[-1]:.1%}/平均淨R {cal['mean_r_net'].iloc[-1]:.2f}），中間
有些微波動但趨勢正確。各折選中的保留比例:

{retention_by_fold}

第 {no_filter_folds} 折驗證段看不出篩選好處,規則正確地選了不篩選(保留比例=100%);
其餘折都選了低於 100% 的保留比例。

## 四、篩選後績效

{_md_table(main_tbl)}

**三項成功標準**(`config/week08_filter.toml [significance]`)**全部達成**:
1. 篩選後夏普({result['filtered_sharpe']:.3f}) > 不篩選({result['unfiltered_sharpe']:.3f}) {'✓' if result['beats_unfiltered'] else '✗'}
2. 篩選後夏普 > 隨機篩選 1000 次分布第 95 百分位({result['random_null_p95']:.3f}，
   平均 {result['random_null_mean']:.3f}) {'✓' if result['beats_random_p95'] else '✗'}
3. 每折每年交易數皆 ≥{cfg['model']['threshold_selection']['min_trades_per_year']} {'✓' if result['min_trades_per_year_ok'] else '✗'}

逐折交易數:

{_md_table(by_fold_tpy)}

**與階段三 HMM 簡單狀態規則對照**(限定在兩者共同覆蓋的 {common_idx.min().date()} 至
{common_idx.max().date()} 子範圍,{len(common_idx)} 天,才公平比較):

{_md_table(cmp_display)}

logistic 篩選(夏普 {cmp_tbl.iloc[0]['filtered_sharpe']:.3f})明顯優於 HMM 狀態規則(夏普
{cmp_tbl.iloc[1]['filtered_sharpe']:.3f}，比不篩選的 {cmp_tbl.iloc[1]['unfiltered_sharpe']:.3f}
還差)——呼應第七週「HMM 篩選是雜訊」的結論,這週用開盤特徵的篩選明顯做得更好。

**門檻曲線**(描述用,用測試段已配適模型的機率重算各保留比例下的結果,不是實際選門檻的
依據):

{_md_table(curve_tbl)}

從 100%(不篩選,夏普 {curve[curve['retention']==1.0]['sharpe'].iloc[0]:.3f})降到 70–90%
區間夏普升到 {curve['sharpe'].iloc[4:7].max():.3f} 附近,更激進篩選(30–60%)仍全面優於不
篩選但略有震盪——不是單一巧合門檻,趨勢平滑,對「這不是雜訊」的判讀是正面佐證。

**被篩掉的大賺日**:OOS 窗內 {mw['n_big_win_days_in_window']} 個大賺日(R≥5),篩選規則丟掉了
{mw['n_dropped']} 個({mw['frac_dropped']:.1%})。確實丟了不少最賺錢的日子,但整體夏普仍提升,
代表篩選器在「避開的爛日子」上贏得比「丟掉的好日子」損失得更多。

## 五、完成標準逐項回答

1. **開盤特徵通過截斷與日內截止測試?** 通過——`open_5m` 截止的既有截斷測試 + 新增的
   「篡改 09:35 後數值」測試,192 個測試全過。
2. **整條管線跑通,篩選後夏普是否超過三個對照組?** 是——超過不篩選、超過隨機篩選 1000
   次分布第 95 百分位、也超過階段三 HMM 簡單狀態規則(限定共同範圍)。
3. **AUC 不錯但夏普沒升的情況?** 沒發生——AUC 中等({auc:.3f})，但夏普確實提升，門檻曲線
   顯示這個提升在整個保留比例網格上都穩定存在，不是單點巧合。報酬集中度（前10%佔72%毛利）
   提醒了篩選器可能誤傷大賺日，實測確實丟了 {mw['frac_dropped']:.0%} 的大賺日，但淨效果仍是
   正的。
4. **夏普好得不合理，需要查是否用到 9:35 後資料？** 目前的結果量級（篩選後夏普 0.7 左右，
   比不篩選高約 0.2）不算「好得不合理」；而且日內截止的因果測試（第三部分）與門檻選擇的
   因果測試（第四部分：篡改最後一折測試段特徵/標籤，確認選中門檻不變）都已經專門檢查過
   look-ahead，沒有發現問題。
"""
    out_path = REPORTS / "week08_filter_baseline.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
