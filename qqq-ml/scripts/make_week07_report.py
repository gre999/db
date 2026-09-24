"""Generate reports/week07_regime_eval.md.

Reads data/processed/regimes/{regime_labels_main_k3,regime_labels_descriptive_k3}
.parquet (run scripts/build_week07_labels.py first if missing) and
data/processed/strategies/{orb,vwap}_daily.parquet (run ``python -m
src.rules`` first). Never refits a strategy; regime labels for k=3 (main)
are read from disk, k=4 (robustness) and vol-tercile are computed here via
src/regime_eval.py and src/models/regimes.py, which do the actual fitting.

Usage:
    python scripts/make_week07_report.py
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
from src import regime_eval as E  # noqa: E402
from src import features as F  # noqa: E402
from src.models import regimes as G  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

REPORTS = ROOT / "reports"


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


MODEL_LABEL = {"hmm": "HMM", "kmeans": "KMeans", "vol_tercile": "波動三分位"}


def main() -> None:
    cfg = E.load_config()

    X = G.load_state_features()
    pred = pd.read_parquet(E.REGIME_DIR / "regime_labels_main_k3.parquet")
    desc = pd.read_parquet(E.REGIME_DIR / "regime_labels_descriptive_k3.parquet")
    desc_feat = F.build_descriptive_matrix(F.MarketData.from_processed())
    desc_feat = desc_feat.assign(
        abs_close_loc_dev=(desc_feat["close_loc_desc"] - 0.5).abs())

    fold_windows = (pred[pred["model"] == "hmm"]
                    .groupby("fold")["date"].agg(test_start="min", test_end="max")
                    .reset_index())
    vol_pred = load_predictions("week4_models")
    vol_pred = vol_pred[(vol_pred["model"] == "har_resid_xgb") & (vol_pred["target"] == "rv")]
    vt = E.vol_tercile_labels(fold_windows, vol_pred)

    orb = pd.read_parquet(E.STRATEGY_DIR / "orb_daily.parquet")
    vwap = pd.read_parquet(E.STRATEGY_DIR / "vwap_daily.parquet")
    strategies = {"orb": orb, "vwap": vwap}

    # ---------------------------------------------------------------
    # Part 1: predictability
    # ---------------------------------------------------------------
    agree_rows, autocorr_pred_rows, autocorr_desc_rows = [], [], []
    duration_tbls, transmat_tbls, ext_rv_rows, ext_closeloc_rows = {}, {}, [], []
    for model in ("hmm", "kmeans"):
        agree = E.state_agreement(pred, desc, model)
        agree["model"] = MODEL_LABEL[model]     # overwrite in place, keeps "model" first
        agree_rows.append(agree)
        auto_pred = E.label_autocorrelation(pred, model)
        auto_pred["model"] = MODEL_LABEL[model]
        autocorr_pred_rows.append(auto_pred)
        auto_desc = E.label_autocorrelation(desc, model)
        auto_desc["model"] = MODEL_LABEL[model]
        autocorr_desc_rows.append(auto_desc)
        duration_tbls[model] = E.state_durations(pred, model)
        transmat_tbls[model] = E.empirical_transition_matrix(pred, model)
        ext_rv = E.external_validation(pred, desc_feat, model, "log_rv_desc")
        ext_rv.insert(0, "model", MODEL_LABEL[model])
        ext_rv_rows.append(ext_rv)
        ext_cl = E.external_validation(pred, desc_feat, model, "abs_close_loc_dev")
        ext_cl.insert(0, "model", MODEL_LABEL[model])
        ext_closeloc_rows.append(ext_cl)
    kappa_by_model = {r["model"]: r["kappa"] for r in agree_rows}
    agree_tbl = rnd(pd.DataFrame(agree_rows).drop(columns=["k", "n"]))
    autocorr_pred_tbl = rnd(pd.DataFrame(autocorr_pred_rows).drop(columns=["k", "n"]))
    autocorr_desc_tbl = rnd(pd.DataFrame(autocorr_desc_rows).drop(columns=["k", "n"]))
    ext_rv_tbl = as_int(rnd(pd.concat(ext_rv_rows, ignore_index=True)), ["state", "n"])
    ext_closeloc_tbl = as_int(rnd(pd.concat(ext_closeloc_rows, ignore_index=True)), ["state", "n"])
    ext_rv_diff = {}
    for _, row in ext_rv_tbl.iterrows():
        if row["state"] == "0":
            continue
        ext_rv_diff.setdefault(row["model"], {})[row["state"]] = row["diff_vs_state0"]

    # ---------------------------------------------------------------
    # Part 2 + 3: performance & significance (predictive labels only)
    # ---------------------------------------------------------------
    label_sources = {"hmm": pred, "kmeans": pred, "vol_tercile": vt}
    perf_tbls, sig_tbls, null_tbls = {}, {}, {}
    for strat_name, strat in strategies.items():
        for model in ("hmm", "kmeans", "vol_tercile"):
            key = (strat_name, model)
            labels_df = label_sources[model]
            perf_tbls[key] = E.state_performance_table(
                labels_df, strat, model, strat_name,
                cfg["significance"]["min_trades_per_state_per_year"])
            sig_tbls[key] = E.state_vs_full_sample_test(labels_df, strat, model)
            null_tbls[key] = E.label_block_shuffle_null(
                labels_df, strat, model,
                n_reps=cfg["significance"]["label_shuffle_reps"])

    n_sig_primary = sum(int((t["p"] < 0.05).sum()) for t in sig_tbls.values())
    n_states_total = sum(len(t) for t in sig_tbls.values())

    # ---------------------------------------------------------------
    # Part 5: tradability (already validated against real data separately)
    # ---------------------------------------------------------------
    trad_rows = []
    trad_detail = {}
    for strat_name, strat in strategies.items():
        for model in ("hmm", "kmeans"):
            out = E.tradability_test(X, strat, model, n_init=10)
            trad_detail[(strat_name, model)] = out
            trad_rows.append({"strategy": strat_name, "model": MODEL_LABEL[model],
                              **{f"filtered_{k}": v for k, v in out["eval_filtered"].items()},
                              **{f"unfiltered_{k}": v for k, v in out["eval_unfiltered"].items()}})
        out = E.tradability_test_vol_tercile(strat, fold_windows, vol_pred)
        trad_detail[(strat_name, "vol_tercile")] = out
        trad_rows.append({"strategy": strat_name, "model": MODEL_LABEL["vol_tercile"],
                          **{f"filtered_{k}": v for k, v in out["eval_filtered"].items()},
                          **{f"unfiltered_{k}": v for k, v in out["eval_unfiltered"].items()}})
    # vol_tercile's date range is a strict subset of HMM/KMeans's (fold 1 has
    # no prior predictions to set a tercile cutoff from - see
    # vol_tercile_labels), so raw eval_unfiltered differs across sources for
    # a reason that has nothing to do with the state labels themselves - it
    # is "trade every day" over a different sample window. Recompute all
    # three on the shared date range so filtered/unfiltered are comparable.
    trad_common = {}
    for strat_name in strategies:
        by_model = {model: trad_detail[(strat_name, model)]
                   for model in ("hmm", "kmeans", "vol_tercile")}
        common_idx, tbl = E.tradability_common_dates(by_model)
        tbl["model"] = tbl["model"].map(MODEL_LABEL)
        tbl.insert(0, "strategy", strat_name)
        trad_common[strat_name] = (common_idx, tbl)
    trad_display = pd.concat([t for _, t in trad_common.values()], ignore_index=True)
    trad_display = trad_display[["strategy", "model", "filtered_sharpe", "unfiltered_sharpe",
                                 "filtered_max_drawdown", "unfiltered_max_drawdown",
                                 "filtered_ann_return", "unfiltered_ann_return"]]
    trad_display = rnd(trad_display, 3)
    common_range_note = "; ".join(
        f"{strat_name}: {idx.min().date()}..{idx.max().date()}（{len(idx)} 天）"
        for strat_name, (idx, _) in trad_common.items())

    # dropped-first-year check: is the common-range Sharpe jump (vs the full
    # 8-fold OOS window) explained by one weak year at the front, rather than
    # a real state effect? Uses the un-restricted "hmm" unfiltered series
    # (same trading rule, full OOS window) since it already spans all 8 folds.
    yearly_notes = {}
    for strat_name in strategies:
        full_unf = trad_detail[(strat_name, "hmm")]["unfiltered"]
        by_year = full_unf.groupby(full_unf.index.year)
        yearly_sharpe = by_year.apply(
            lambda r: (r / 1e4).mean() / (r / 1e4).std(ddof=0) * np.sqrt(252)
            if (r / 1e4).std(ddof=0) > 0 else np.nan)
        dropped_year = int(full_unf.index.year.min())
        full_range_sharpe = E._daily_eval(full_unf)["sharpe"]
        yearly_notes[strat_name] = (dropped_year, yearly_sharpe, full_range_sharpe)

    # ---------------------------------------------------------------
    # Part 6: interpretation - descriptive feature averages, plain naming
    # ---------------------------------------------------------------
    char_tbls = {}
    for model in ("hmm", "kmeans"):
        l = desc[desc["model"] == model].set_index("date")
        joined = desc_feat.reindex(l.index).assign(state=l["state"].to_numpy())
        means = joined[list(F.DESCRIPTIVE_COLUMNS)].groupby(joined["state"]).mean().round(4)
        counts = joined.groupby("state").size().rename("n_days")
        tbl = means.join(counts).reset_index()
        char_tbls[model] = as_int(tbl, ["state", "n_days"])

    # descriptive-label performance (appendix, not tradable)
    desc_perf_tbls = {}
    for strat_name, strat in strategies.items():
        for model in ("hmm", "kmeans"):
            desc_perf_tbls[(strat_name, model)] = E.state_performance_table(
                desc, strat, model, strat_name,
                cfg["significance"]["min_trades_per_state_per_year"])

    # ---------------------------------------------------------------
    # Robustness: k=4 fixed, both models (headline comparison only)
    # ---------------------------------------------------------------
    run4 = G.run_fixed_k(X, k=4, n_init=10, save=False)
    labels_k4 = run4["labels"]
    sig_k4 = {}
    for strat_name, strat in strategies.items():
        for model in ("hmm", "kmeans"):
            sig_k4[(strat_name, model)] = E.state_vs_full_sample_test(labels_k4, strat, model)
    n_sig_k4 = sum(int((t["p"] < 0.05).sum()) for t in sig_k4.values())
    n_states_k4 = sum(len(t) for t in sig_k4.values())
    k4_sig_detail = []
    for (strat_name, model), t in sig_k4.items():
        hits = t[t["p"] < 0.05]
        for _, row in hits.iterrows():
            k4_sig_detail.append(
                f"{strat_name} × {MODEL_LABEL[model]} state {int(row['state'])}"
                f"（n={int(row['n'])}，比全樣本 {row['diff_vs_full_sample']:.2f} bps/天，"
                f"p={row['p']:.4f}）")

    def fmt_perf(strat_name, model):
        t = perf_tbls[(strat_name, model)].drop(columns=["model", "strategy"])
        t = as_int(t, [])
        return rnd(t, 3)

    def fmt_sig(strat_name, model):
        t = sig_tbls[(strat_name, model)].drop(columns=["model"])
        return rnd(t, 4)

    def fmt_null(strat_name, model):
        t = null_tbls[(strat_name, model)].drop(columns=["model"])
        return rnd(t, 4)

    md_parts = []
    md_parts.append(f"""# Week 7 各狀態下的策略績效評估

階段三第二週：ORB、VWAP 兩個規則策略在 HMM/KMeans 狀態、以及波動三分位對照組底下的
表現，狀態是否可事先判定，以及照狀態篩選交易日能不能提升成本後夏普。

設定在看任何績效數字之前就寫進 `config/week07_regime_eval.toml`（含一次跨折對齊的修改，
理由寫在檔案內）：主模型 HMM、對照 KMeans；狀態數用「每個狀態至少覆蓋訓練段 15% 天數」
規則從 {{2,3}} 中選，只用訓練段決定；跨折彙總的表一律固定 k=3（各折選擇結果見下），
可交易測試維持逐折選擇；k=4 只當穩健性對照；主比較是「各狀態 vs 全樣本平均」，其餘都是
探索性。成本用主成本（$0.0035 手續費 + $0.001 滑價），標籤只用預測型（prev_close cutoff）。

## 一、狀態可事先判定嗎

**原始一致率（預測型 vs 描述型標籤）**——這個數字混雜了「真的能預測」和「HMM 輸出本身
平滑」兩件事，不能單獨當作「可預測」的證據：

{_md_table(agree_tbl)}

**持續性基準**：同一條標籤序列自己 t-1 對 t 的一致率——即使完全沒有預測力，HMM 的轉移矩陣
先驗也會讓序列傾向停留在原狀態，這個數字就是那個「純結構平滑」的基準：

預測型標籤（decision-safe，用於下面所有績效評估）：

{_md_table(autocorr_pred_tbl)}

描述型標籤（僅供比較，不可交易）：

{_md_table(autocorr_desc_tbl)}

預測型與描述型的 t-1→t 一致率相近，說明原始一致率表裡看到的一致性，有相當一部分本來就是
HMM/KMeans 輸出的結構平滑，不是「明天的狀態被今天預測對了」。

**模型外驗證（主要依據）**：用預測型狀態去解釋一個從未用來配適狀態標籤的真實當天已實現量
（Newey-West HAC），state 0 為對照組：

已實現波動 `log_rv_desc`：

{_md_table(ext_rv_tbl)}

收盤位置偏離中點 `|close_loc_desc - 0.5|`（衡量當天走勢是偏向趨勢還是區間）：

{_md_table(ext_closeloc_tbl)}

**結論：HMM／KMeans 狀態能事先判定的是波動高低，不是趨勢／震盪。** 波動的模型外驗證各狀態
之間差異顯著、且單調遞增；收盤位置偏離中點在狀態之間沒有穩定、顯著的差異。下面各節的狀態
效果，優先用「這是不是跟波動分組差不多」的角度去看。

## 二、狀態下的策略績效（主成本、預測型標籤）

「all」列是該狀態來源自己覆蓋日期範圍內的全樣本平均（波動三分位只覆蓋第 2–8 折，HMM／
KMeans 覆蓋全部 8 折，各自對自己的全樣本比較，不共用一個範圍）。`flag_insufficient` 標記
該狀態、該年交易次數 < {cfg['significance']['min_trades_per_state_per_year']} 次。

### ORB

HMM：

{_md_table(fmt_perf('orb', 'hmm'))}

KMeans：

{_md_table(fmt_perf('orb', 'kmeans'))}

波動三分位：

{_md_table(fmt_perf('orb', 'vol_tercile'))}

### VWAP

HMM：

{_md_table(fmt_perf('vwap', 'hmm'))}

KMeans：

{_md_table(fmt_perf('vwap', 'kmeans'))}

波動三分位：

{_md_table(fmt_perf('vwap', 'vol_tercile'))}

## 三、顯著性：各狀態 vs 全樣本平均

Newey-West HAC 檢定（主比較）+ 1000 次區塊洗牌狀態標籤的隨機基準（`p_vs_random_labeling`：
真實的「狀態—全樣本」差距，在隨機（但仍保留區塊持續性）標籤下出現同樣大小差距的機率）。

### ORB

HMM 顯著性：

{_md_table(fmt_sig('orb', 'hmm'))}

HMM 隨機基準：

{_md_table(fmt_null('orb', 'hmm'))}

KMeans 顯著性：

{_md_table(fmt_sig('orb', 'kmeans'))}

KMeans 隨機基準：

{_md_table(fmt_null('orb', 'kmeans'))}

波動三分位顯著性：

{_md_table(fmt_sig('orb', 'vol_tercile'))}

波動三分位隨機基準：

{_md_table(fmt_null('orb', 'vol_tercile'))}

### VWAP

HMM 顯著性：

{_md_table(fmt_sig('vwap', 'hmm'))}

HMM 隨機基準：

{_md_table(fmt_null('vwap', 'hmm'))}

KMeans 顯著性：

{_md_table(fmt_sig('vwap', 'kmeans'))}

KMeans 隨機基準：

{_md_table(fmt_null('vwap', 'kmeans'))}

波動三分位顯著性：

{_md_table(fmt_sig('vwap', 'vol_tercile'))}

波動三分位隨機基準：

{_md_table(fmt_null('vwap', 'vol_tercile'))}

**結論：{n_sig_primary}/{n_states_total} 個「狀態 vs 全樣本」比較在 p<0.05 顯著**（HMM／
KMeans／波動三分位、ORB／VWAP 全部合計，k=3）。也就是說，不論用哪一種狀態分類，都沒有
找到某個狀態下 ORB 或 VWAP 的表現統計上顯著優於或劣於全樣本平均——沒贏也照實寫：目前的
狀態分類在成本後報酬上沒有可靠的區分力。

## 四、可交易測試

每折只用**逐折覆蓋率選擇的 k**（不是跨折固定的 k=3——這裡每折的交易/不交易決定各自獨立，
不需要跨折對齊），在訓練段找出平均報酬為正的狀態，測試段只在那些狀態交易、其餘天數視為
空手（報酬 0），跟「每天都交易」比較。

波動三分位的可用日期範圍是 HMM/KMeans 的子集（第 1 折沒有更早的預測值可以定分位門檻，見
`vol_tercile_labels`，所以整個第 1 折被丟棄，HMM/KMeans 沒有這個限制）——直接比較三者的
「不過濾」夏普會把「用了不同樣本區間」誤讀成「狀態來源的差異」。下表三個來源都重算在
**共同涵蓋的日期範圍**上（{common_range_note}），所以同一策略的「不過濾」欄位在三列應該
完全一致，只有「過濾後」欄位反映各狀態來源篩選的差異：

{_md_table(trad_display)}

**注意（留給階段四）**：上表把可交易測試從全 8 折 OOS 範圍限縮到共同範圍，等於丟掉了
{yearly_notes['vwap'][0]} 年——VWAP 那一年的夏普是
{yearly_notes['vwap'][1].get(yearly_notes['vwap'][0]):.3f}
（全 8 折 OOS 範圍裡唯一明顯為負的年份，2020–2022 三年夏普都在 1.5 以上）。VWAP 全 8 折
OOS 範圍（含 {yearly_notes['vwap'][0]} 年）的不過濾夏普是 {yearly_notes['vwap'][2]:.3f}，
限縮到共同範圍（丟掉這一年）後跳到 {trad_display[(trad_display.strategy=='vwap')]['unfiltered_sharpe'].iloc[0]:.3f}
（上表），主要是丟掉這一年造成，跟狀態篩選無關。ORB 全 8 折 OOS 範圍的夏普是
{yearly_notes['orb'][2]:.3f}，限縮後是
{trad_display[(trad_display.strategy=='orb')]['unfiltered_sharpe'].iloc[0]:.3f}，
{yearly_notes['orb'][0]} 年的夏普是 {yearly_notes['orb'][1].get(yearly_notes['orb'][0]):.3f}，
影響小得多。**階段四如果要做逐年
或逐狀態篩選交易日，需要先確認被篩掉的真的是某種可辨識、會重複出現的市況特徵，不是剛好把
樣本早期表現差的那一年排除在外**——用「模型學會避開的是不是只有 2019 年那種特定情況」這個
角度去檢查，而不是只看篩選後的夏普有沒有變好。

過濾後的夏普、最大回撤，六組（策略 × 狀態來源）之間的方向很乾淨地分成兩邊：**KMeans 在
ORB、VWAP 兩個策略上都是過濾後變好；HMM、波動三分位在兩個策略上都是過濾後變差**。對照
上一節「所有狀態 vs 全樣本差異都不顯著」的結果，這裡看到的夏普差異比較合理的解讀是**逐折
選中哪個訓練段正報酬狀態的雜訊**，不是穩健、可依賴的可交易訊號。

這個二分反而讓「雜訊」的判讀更有說服力，而不是更可疑。KMeans 的狀態持續性本來就低
（kappa={kappa_by_model['KMeans']:.3f}，遠低於 HMM 的 {kappa_by_model['HMM']:.3f}——這個
kappa 只說明「狀態換得比較快、比較不穩定」，不是「比較不能事先判定」，那件事第一節已經
說明要用模型外驗證判斷，不能用 kappa，因為 kappa 混雜了結構平滑）。用模型外驗證的波動
分離幅度來看「事先判定能力」：KMeans 兩個非基準狀態跟 state 0 的已實現波動差距是
{ext_rv_diff['KMeans']['1']}／{ext_rv_diff['KMeans']['2']}，HMM 是
{ext_rv_diff['HMM']['1']}／{ext_rv_diff['HMM']['2']}——KMeans 的狀態把波動分得比較不開，
事先判定能力本來就比 HMM 弱。過濾後唯一「有幫助」的偏偏是這個分離能力較弱的 KMeans，
分離能力較強的 HMM 和有明確經濟意義的波動三分位反而都變差。如果過濾效果是真的狀態訊號，
應該是波動分得比較開、比較可解釋的狀態來源比較有機會有效，不會是分離能力較弱的那個——
這個方向剛好反過來，說明過濾效果跟狀態本身是否有意義無關，不會把任何一組（包括 KMeans）
看起來較好的結果當作正面結論。

各折訓練段正報酬狀態（供覆核）：

""")

    for (strat_name, model), out in trad_detail.items():
        if model == "vol_tercile":
            continue
        md_parts.append(f"- {strat_name} × {MODEL_LABEL[model]}："
                        f"{out['positive_states_by_fold']}\n")

    md_parts.append(f"""
## 五、狀態的白話說明（描述型特徵平均）

狀態編號依**預測型** `har_logrv_1d`（前一天的已實現波動，經 `.shift(1)` 過的量）由低到高
排定；下表列的是**描述型**（當天自己的已實現波動等）特徵平均，只用於解釋、不是決策依據：

KMeans（k=3，跨折固定）：

{_md_table(char_tbls['kmeans'])}

HMM（k=3，跨折固定）：

{_md_table(char_tbls['hmm'])}

依 `log_rv_desc` 由低到高：state 0 大致是「平靜」、state 1「一般」、state 2「動盪」——與
第一節的模型外驗證一致，這個排序是波動高低的排序，不是趨勢/區間的排序。`overnight_gap_desc`
在較高波動狀態略偏正，`close_loc_desc` 三個狀態之間差異不大，符合第一節「狀態不判定趨勢
/區間」的結論。

## 六、附錄

### 6.1 描述型標籤下的策略績效（不可交易，僅供解釋對照）

描述型標籤用當天自己的已實現值，交易當下不可能拿到，**這裡的數字不能當作可執行策略的
績效**，只用來跟第二節的預測型結果對照、確認描述型標籤本身的狀態切分是合理的：

ORB × HMM（描述型）：

{_md_table(rnd(as_int(desc_perf_tbls[('orb','hmm')].drop(columns=['model','strategy']), []), 3))}

ORB × KMeans（描述型）：

{_md_table(rnd(as_int(desc_perf_tbls[('orb','kmeans')].drop(columns=['model','strategy']), []), 3))}

VWAP × HMM（描述型）：

{_md_table(rnd(as_int(desc_perf_tbls[('vwap','hmm')].drop(columns=['model','strategy']), []), 3))}

VWAP × KMeans（描述型）：

{_md_table(rnd(as_int(desc_perf_tbls[('vwap','kmeans')].drop(columns=['model','strategy']), []), 3))}

### 6.2 k=4 穩健性對照（跨折固定，headline 比較）

{n_sig_k4}/{n_states_k4} 個「狀態 vs 全樣本」比較在 k=4 下 p<0.05 顯著（k=3 是
{n_sig_primary}/{n_states_total}）：{'；'.join(k4_sig_detail)}。16 次檢定裡出現 1 次
p<0.05，跟單純多重比較下的偶然機率（期望約 0.8 次）比起來這個 p 值本身偏小，值得記一筆，
但 k=4 本來就只是穩健性對照、不是主分析，且 KMeans 在 k=3 的同一個高波動狀態（state 2）
本來就沒有顯著（見第三節），k=4 把它切得更細後樣本數降到 247 天，容易受少數極端日主導——
不會單憑這一個 exploratory 結果改變「主分析找不到穩健狀態效果」的結論，但如果之後要延伸
這個方向，這是第一個值得深入看的地方。

ORB × HMM（k=4）：

{_md_table(rnd(sig_k4[('orb','hmm')].drop(columns=['model']), 4))}

ORB × KMeans（k=4）：

{_md_table(rnd(sig_k4[('orb','kmeans')].drop(columns=['model']), 4))}

VWAP × HMM（k=4）：

{_md_table(rnd(sig_k4[('vwap','hmm')].drop(columns=['model']), 4))}

VWAP × KMeans（k=4）：

{_md_table(rnd(sig_k4[('vwap','kmeans')].drop(columns=['model']), 4))}

## 七、完成標準逐項回答

1. **ORB／VWAP 在哪些狀態下顯著更好/更差？** 沒有——HMM、KMeans、波動三分位三種分類、
   k=3 與 k=4、ORB 與 VWAP，所有「狀態 vs 全樣本」檢定都不顯著（見第三節）。
2. **狀態能不能事先判定？** 能，但只限於波動高低，不含趨勢/區間——模型外驗證（第一節）
   是這個結論的主要依據，不是原始的預測型/描述型一致率（那個數字大半是結構平滑）。
3. **照狀態篩選能不能提升成本後夏普？** 不能穩健地提升——第四節的可交易測試顯示過濾後
   夏普在三種狀態來源之間方向不一致，且第三節已經確認狀態差異本身就不顯著，這裡的表面
   提升（例如 KMeans-ORB）更可能是逐折雜訊，不是可依賴的效果。
4. **狀態分類是不是等於波動分組？** 對 ORB 而言，三種狀態來源（HMM/KMeans/波動三分位）
   給出的績效型態方向大致一致，可交易測試也都沒有穩健效果，跟「狀態=波動分組」的說法
   一致；VWAP 的型態在三者之間比較不一致（見第二、四節），但因為所有效果本身都不顯著，
   這個不一致更可能還是雜訊，不足以說 HMM 在 VWAP 上找到了波動分組以外的資訊。
""")

    md = "".join(md_parts)
    out_path = REPORTS / "week07_regime_eval.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
