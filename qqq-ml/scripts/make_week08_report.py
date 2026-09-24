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

    X = pd.read_parquet(dl.PROCESSED_DIR / "features_open_5m.parquet")
    coefs = FL.coefficient_table(X, orb)
    coef_cols = [c for c in coefs.columns if c not in
                ("fold", "intercept", "retention_selected")]
    coef_summary = []
    for c in coef_cols:
        v = coefs[c]
        coef_summary.append({"feature": c, "mean_abs_coef": v.abs().mean(),
                            "mean_coef": v.mean(),
                            "sign_consistency": max((v > 0).mean(), (v < 0).mean())})
    coef_summary = pd.DataFrame(coef_summary).sort_values(
        "mean_abs_coef", ascending=False).reset_index(drop=True)
    coef_summary_tbl = rnd(coef_summary, 3)
    body_ratio_row = coef_summary[coef_summary["feature"] == "open5m_body_ratio"].iloc[0]
    logrv1d_row = coef_summary[coef_summary["feature"] == "har_logrv_1d"].iloc[0]
    range_rel_row = coef_summary[coef_summary["feature"] == "open5m_range_rel"].iloc[0]

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

    # ---------------------------------------------------------------
    # Pre-conclusion checks (requested before finalizing the report)
    # ---------------------------------------------------------------
    null = FE.random_filter_null(preds, orb, n_reps=1000, seed=0)
    rp = FE.random_filter_p_value(preds, orb, null=null)
    bb = FE.block_bootstrap_significance(preds, orb, n_boot=2000, seed=0)

    yb = FE.yearly_breakdown(preds, orb)
    yb["diff"] = yb["filtered_sharpe"] - yb["unfiltered_sharpe"]
    yb_tbl = as_int(rnd(yb, 3), ["year", "n_days"])
    n_positive_years = int((yb["diff"] > 0).sum())
    worst_year = yb.loc[yb["diff"].idxmin()]
    best_year = yb.loc[yb["diff"].idxmax()]

    fb = FE.fold_breakdown(preds, orb)
    fb["diff"] = fb["filtered_sharpe"] - fb["unfiltered_sharpe"]
    fb_tbl = as_int(rnd(fb, 3), ["fold", "n_days"])

    cs = FE.cost_sensitivity(preds)
    cs_tbl = rnd(cs, 4)
    gap_main = cs[cs["cost_config"] == "main_cost"]["sharpe_gap"].iloc[0]
    gap_gross = cs[cs["cost_config"] == "gross_no_cost"]["sharpe_gap"].iloc[0]
    gap_2x = cs[cs["cost_config"] == "2x_slippage"]["sharpe_gap"].iloc[0]

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

**逐折標準化 logistic 係數**(哪些特徵最重要、方向在各折是否一致;`sign_consistency` 是
9 折中同號的比例):

{_md_table(coef_summary_tbl)}

穩定驅動的兩個特徵是 `open5m_body_ratio`(9 折全部同號為正,平均係數量級最大,
{body_ratio_row['mean_abs_coef']:.3f})和 `har_logrv_1d`(9 折全部同號為負,
{logrv1d_row['mean_abs_coef']:.3f})。`open5m_range_rel` 反而不穩定(只有
{range_rel_row['sign_consistency']:.0%} 折同號,接近隨機),不是主要驅動者。

**機制**:`open5m_body_ratio` 為正——第一根 5 分鐘K線實體占區間比例越高(開盤方向越乾淨、
少影線),ORB 進場當天訊號淨賺的機率越高,這跟 ORB 自己的動能假設(方向乾淨的開盤代表動能
續航機率高)方向一致,不是意外的相關性。`har_logrv_1d` 為負——前一日已實現波動越高,當天
訊號反而越可能虧,較合理的解釋是波動群聚(昨天波動高、今天通常也高)讓當天走勢更容易來回
震盪、提早碰到停損,而不是乾淨地跑滿 10R,不是「停損距離太大」的機制(ORB 的停損/目標都是
用當天自己第一根K線的區間定義,不是用前一日波動定義)。

## 四、篩選後績效(點估計)

{_md_table(main_tbl)}

**三項預先設定的成功標準**(`config/week08_filter.toml [significance]`)**字面上全部達成**:
1. 篩選後夏普({result['filtered_sharpe']:.3f}) > 不篩選({result['unfiltered_sharpe']:.3f}) {'✓' if result['beats_unfiltered'] else '✗'}
2. 篩選後夏普 > 隨機篩選 1000 次分布第 95 百分位({result['random_null_p95']:.3f}，
   平均 {result['random_null_mean']:.3f}) {'✓' if result['beats_random_p95'] else '✗'}
3. 每折每年交易數皆 ≥{cfg['model']['threshold_selection']['min_trades_per_year']} {'✓' if result['min_trades_per_year_ok'] else '✗'}

**但這三項標準本身沒有一項是「篩選前後夏普差是否顯著不為零」的檢定**——第五節的追加檢定
會補上這個檢定,結論會修正這裡的「達成」。

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

## 五、定稿前追加檢定

第四節的三項標準都只比較點估計,沒有一項檢定「篩選後夏普 - 不篩選夏普」這個差本身是不是
在抽樣變異下也可能出現。定稿前補四項檢查:

**1. Block bootstrap 顯著性**:篩選後與不篩選夏普差(觀察值 {bb['observed_sharpe_diff']:.3f})
的移動區塊拔靴(區塊長度 {bb['block_size']} 天,{bb['n_boot']} 次)**p 值 = {bb['p_value']:.3f},
不顯著**。對照隨機篩選基準的經確 p 值:實際篩選夏普({rp['observed_sharpe']:.3f})在
{rp['n_reps']} 次隨機篩選中排第 {rp['rank_from_top']} 高,p = {rp['p_value']:.3f}。這兩個檢定
問不同問題——隨機篩選基準問「篩同樣天數,選這些特定的天跟隨機選有沒有差」(p={rp['p_value']:.3f},
較小);block bootstrap 問「篩選後與不篩選的夏普差,考慮這段日報酬本身的序列相關抽樣變異,是
不是明顯不是零」(p={bb['p_value']:.3f},不顯著)。**篩選能排出比隨機更好的天,不代表夏普提升
這個點估計本身站得住腳**。

**2. 逐年拆解**({n_positive_years}/{len(yb)} 年篩選後夏普 > 不篩選,方向大致一致,但匯總效果
被兩個極端年份主導):

{_md_table(yb_tbl)}

{int(worst_year['year'])} 年篩選後大幅更差(差 {worst_year['diff']:.2f},filtered
{worst_year['filtered_sharpe']:.2f} vs unfiltered {worst_year['unfiltered_sharpe']:.2f}——
那年篩選器選錯了方向);{int(best_year['year'])} 年篩選後大幅更好(差 {best_year['diff']:.2f},
且是只到 9 月的部分年度)。這兩個極端值互相部分抵銷但沒有完全抵銷,淨效果偏正,這也解釋了
第 1 點 block bootstrap 為何不顯著——整體提升不是穩健地分散在各年,是被少數幾折的極端結果
撐著。逐折版本:

{_md_table(fb_tbl)}

**3. 逐折係數同號一致性**:已經在第三節報告——`open5m_body_ratio`、`har_logrv_1d` 兩者都是
9 折全部同號,且與成本無關(見下一項),是排序能力有訊號的正面證據;但這兩個特徵是看完全部
9 折結果後才挑出來的,只能算探索性發現,不是預先設定的模型。

**4. 成本敏感度**:篩選前後的夏普差在毛報酬(不扣成本)、主成本、2倍滑價下幾乎不變:

{_md_table(cs_tbl)}

差值分別是毛報酬 {gap_gross:.3f}、主成本 {gap_main:.3f}、2倍滑價 {gap_2x:.3f}——提升確實不是
「交易變少、省成本」的假象,這項是四項追加檢查中唯一完全正面的結果。

## 六、結論

**排序能力:初步證據。** 隨機篩選基準 p={rp['p_value']:.3f}、兩個特徵(`open5m_body_ratio`、
`har_logrv_1d`)9 折同號、成本敏感度顯示提升與成本無關——三項都指向分類器對「哪些訊號日
比較值得進場」有一些真實的排序能力,`open5m_body_ratio` 的機制(開盤方向越乾淨,ORB 的動能
假設越成立)也跟策略邏輯吻合,不是巧合相關。

**策略層面夏普改善:證據不足。** Block bootstrap p={bb['p_value']:.3f},不顯著;匯總提升被
{int(worst_year['year'])}、{int(best_year['year'])} 兩個極端年份主導,不是穩健分散在各年的
效果。

**三項預先設定的成功標準字面上全部達成,但不足以稱為「成功」**——原標準沒有一項是「篩選前後
夏普差是否顯著不為零」的檢定,這是第五節追加檢定才補上的,結果推翻了字面上的「達成」結論。

## 七、給第九週的建議

1. **成功標準加入 block bootstrap p<0.05**(篩選前後夏普差),在跑模型前寫進設定檔,跟本週
   其餘標準一樣不能看過結果再決定。
2. **{int(worst_year['year'])} 年拆多空**,看篩選器是在多頭訊號、空頭訊號,還是兩者上都選
   錯了方向,才能判斷是特徵在那年失效還是機制本身有方向性的弱點。
3. **`open5m_body_ratio` + `har_logrv_1d` 兩特徵模型是看完本週全部結果後才挑出來的,下週
   只能列為探索性對照,不能當作預先設定的主模型**——主模型還是本週全部特徵的 logistic
   基準,換模型時維持同一組特徵,只換分類器本身。

## 八、完成標準逐項回答

1. **開盤特徵通過截斷與日內截止測試?** 通過——`open_5m` 截止的既有截斷測試 + 新增的
   「篡改 09:35 後數值」測試,198 個測試全過。
2. **整條管線跑通,篩選後夏普是否超過三個對照組?** 點估計上是——超過不篩選、超過隨機篩選
   1000 次分布第 95 百分位、也超過階段三 HMM 簡單狀態規則(限定共同範圍);但 block
   bootstrap 顯示篩選後與不篩選的夏普差本身不顯著(見第五、六節),不能只看點估計就宣告
   通過。
3. **AUC 不錯但夏普沒升的情況?** 沒發生——AUC 中等({auc:.3f})，門檻曲線顯示點估計上的提升
   在整個保留比例網格上都存在。報酬集中度（前10%佔72%毛利）提醒了篩選器可能誤傷大賺日，
   實測確實丟了 {mw['frac_dropped']:.0%} 的大賺日，但這不是本週真正的問題所在——真正的問題
   是點估計的統計顯著性不足（第五、六節）。
4. **夏普好得不合理，需要查是否用到 9:35 後資料？** 目前的結果量級不算「好得不合理」；日內
   截止的因果測試（第三部分）與門檻選擇的因果測試（第四部分：篡改最後一折測試段特徵/標籤，
   確認選中門檻不變）都已經專門檢查過 look-ahead，沒有發現問題——這方面的因果性是乾淨的，
   問題完全在統計顯著性，不在資料洩漏。
"""
    out_path = REPORTS / "week08_filter_baseline.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
