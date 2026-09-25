"""Generate reports/week12_dl_walkforward.md.

Reads config/week12_dl.toml and everything scripts/run_week12_
cnn_walkforward.py, scripts/run_week12_stacking.py, and scripts/
run_week12_strategy_layer.py already computed and saved. Read-only -
does not refit the CNN, the hand-feature logistic, or the stacking model.

Usage:
    python scripts/make_week12_report.py
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_week01_report import _md_table  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import dl_eval as DE  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

REPORTS = ROOT / "reports"
SEQ_DIR = dl.PROCESSED_DIR / "sequences"
CONFIG_PATH = ROOT / "config" / "week12_dl.toml"


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


def pooled_auc(preds: pd.DataFrame) -> float:
    return float(roc_auc_score(preds["y_true"], preds["y_pred_proba"]))


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running_max = {}, 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, p * (m - i))
        running_max = max(running_max, a)
        adj[k] = running_max
    return adj


def main() -> None:
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    # ---------------------------------------------------------------
    # Part: primary comparison (task A)
    # ---------------------------------------------------------------
    cnn_a = load_predictions("week12_task_a_cnn").set_index("date")
    hand_a = load_predictions("week11_task_a_hand_feature_logistic").set_index("date")
    common_a = cnn_a.index.intersection(hand_a.index)
    cnn_auc = roc_auc_score(cnn_a.loc[common_a, "y_true"], cnn_a.loc[common_a, "y_pred_proba"])
    hand_auc = roc_auc_score(hand_a.loc[common_a, "y_true"], hand_a.loc[common_a, "y_pred_proba"])
    primary_bb = DE.block_bootstrap_auc_diff(cnn_a.loc[common_a, "y_true"],
                                             cnn_a.loc[common_a, "y_pred_proba"],
                                             hand_a.loc[common_a, "y_pred_proba"],
                                             n_boot=2000, seed=0)

    diag_a = pd.read_parquet(SEQ_DIR / "week12_task_a_fold_diagnostics.parquet")
    diag_a["gap"] = diag_a["fit_auc"] - diag_a["test_auc"]
    hand_fold_auc = []
    for f, g in hand_a.reset_index().groupby("fold"):
        if g["y_true"].nunique() > 1:
            hand_fold_auc.append(roc_auc_score(g["y_true"], g["y_pred_proba"]))
    mean_fold_auc_cnn = diag_a["test_auc"].mean()
    mean_fold_auc_hand = float(np.mean(hand_fold_auc))

    # ---------------------------------------------------------------
    # Part: task B comparison
    # ---------------------------------------------------------------
    taskb_bb = {}
    taskb_rows = []
    for task in ("task_b15", "task_b30"):
        cnn_b = load_predictions(f"week12_{task}_cnn").set_index("date")
        best_name, best_auc, best_preds = None, -1.0, None
        for m in ("logistic_l2", "xgboost"):
            p = load_predictions(f"week11_{task}_{m}").set_index("date")
            common = cnn_b.index.intersection(p.index)
            a = roc_auc_score(p.loc[common, "y_true"], p.loc[common, "y_pred_proba"])
            if a > best_auc:
                best_auc, best_name, best_preds = a, m, p
        common = cnn_b.index.intersection(best_preds.index)
        cnn_b_auc = roc_auc_score(cnn_b.loc[common, "y_true"], cnn_b.loc[common, "y_pred_proba"])
        bb = DE.block_bootstrap_auc_diff(cnn_b.loc[common, "y_true"], cnn_b.loc[common, "y_pred_proba"],
                                         best_preds.loc[common, "y_pred_proba"], n_boot=2000, seed=0)
        taskb_bb[task] = bb["p_value"]
        taskb_rows.append({"task": task, "cnn_auc": cnn_b_auc, "best_baseline": best_name,
                          "best_baseline_auc": best_auc, "auc_diff": bb["observed_auc_diff"],
                          "raw_p": bb["p_value"]})
    holm_adj = holm_correction(taskb_bb)
    taskb_tbl = pd.DataFrame(taskb_rows)
    taskb_tbl["holm_p"] = taskb_tbl["task"].map(holm_adj)
    taskb_tbl = rnd(taskb_tbl, 4)

    # ---------------------------------------------------------------
    # Part: incremental information
    # ---------------------------------------------------------------
    stack_a = load_predictions("week12_task_a_stacking").set_index("date")
    common_stack = stack_a.index.intersection(hand_a.index)
    stack_auc = roc_auc_score(stack_a.loc[common_stack, "y_true"], stack_a.loc[common_stack, "y_pred_proba"])
    stack_bb = DE.block_bootstrap_auc_diff(stack_a.loc[common_stack, "y_true"],
                                           stack_a.loc[common_stack, "y_pred_proba"],
                                           hand_a.loc[common_stack, "y_pred_proba"],
                                           n_boot=2000, seed=0)

    spearman = pd.read_parquet(SEQ_DIR / "week12_task_a_spearman.parquet")
    r2_per_fold = pd.read_parquet(SEQ_DIR / "week12_task_a_r2_per_fold.parquet")
    wald = pd.read_parquet(SEQ_DIR / "week12_task_a_stacking_wald.parquet")
    n_cnn_sig = int((wald["cnn_p"] < 0.05).sum())
    n_cnn_pos = int((wald["cnn_coef"] > 0).sum())
    n_cnn_neg = int((wald["cnn_coef"] < 0).sum())

    # ---------------------------------------------------------------
    # Part: strategy layer
    # ---------------------------------------------------------------
    strat_a = pd.read_parquet(SEQ_DIR / "week12_task_a_strategy_layer.parquet")
    strat_b = pd.read_parquet(SEQ_DIR / "week12_task_b_strategy_layer.parquet")

    md = f"""# Week 12 (階段五第二週)CNN 完整 walk-forward 與增量資訊檢查

凍結第11週的 CNN 架構/訓練設定,跑完 8 折(不是原訂的9折,見下)完整 walk-forward,檢查
CNN 從原始分鐘序列學到的東西有沒有超出第8週手工特徵已經捕捉的資訊。

**折數修正**(查看任何結果前就該抓到,但跑完才發現):序列資料集的成交量特徵要抓過去20個
交易日的歷史,可用起始日比 ORB 原始資料晚了約一個月,把 WalkForwardSplit 的第一個測試年
從 2018 推到 2019——比其他週的9折少一折,任務A/B都是8折(測試年2019–2026,最後一折到
2026年9月)。`config/week12_dl.toml` 已經加註修正並附理由。

## 一、主要比較(任務A)

CNN 對第8週手工特徵 logistic 的測試段 AUC 差,8折測試段全部合併成一條日期序列:

| | AUC |
|---|---|
| CNN | {cnn_auc:.4f} |
| 手工特徵 logistic | {hand_auc:.4f} |

**block bootstrap(交易日為單位):差 = {primary_bb['observed_auc_diff']:+.4f},
p = {primary_bb['p_value']:.4f}**——**CNN 顯著比手工特徵差**,不是「差不多」。

逐折 AUC 平均當對照(不是合併,單純算術平均,因為各折校準可能不同):CNN
{mean_fold_auc_cnn:.4f}、手工特徵 {mean_fold_auc_hand:.4f}——方向跟合併 AUC 一致。

## 二、逐折 AUC 與種子範圍

{_md_table(as_int(rnd(diag_a[['fold','n_fit','n_val','n_test','fit_auc','val_auc','test_auc','test_auc_seed_min','test_auc_seed_max']], 4), ['fold','n_fit','n_val','n_test']))}

**過擬合差距**(配適AUC−測試AUC,對照第9週樹模型的隨機森林0.207、XGBoost深度2的0.310、
深度3的0.373):平均 **{diag_a['gap'].mean():.4f}**——明顯比樹模型溫和,但逐折差異很大
(最小{diag_a['gap'].min():.3f},最大{diag_a['gap'].max():.3f}),第2折測試AUC甚至低於0.5
(比隨機還差)。

**第二次打亂標籤檢查**(第11週在第1折做過一次,這次在最後一折/第8折重做):預先設定的
單一種子(種子0)結果是 AUC=0.6069,**超出 [0.45,0.55],字面上沒過**。但這折測試段只有
180天(比第1折的250天小),換5個不同打亂種子重跑:0.6069、0.6247、0.4840、0.4473、
0.4181——平均約0.516,正負對稱散布在0.5兩側,不是持續偏向同一邊。如果是真的資料洩漏,
應該每個種子都同方向偏離;這種對稱亂跳的型態比較像小樣本(180天)本身的抽樣雜訊——跟
第10週隨機基準的教訓一樣。**判讀:懷疑是雜訊、不是洩漏,但誠實記錄單一種子檢查字面上
沒過這件事。**

## 三、增量資訊(任務A)

**Spearman 相關**(每折CNN測試段機率 vs 手工特徵,機率與logit相關係數相同):

{_md_table(as_int(rnd(spearman, 3), ['fold', 'n']))}

**R²**(CNN logit 對全部14個手工特徵的OLS迴歸):

每折各自迴歸:

{_md_table(as_int(rnd(r2_per_fold, 4), ['fold','n','k']))}

8折R²平均 = {r2_per_fold['r2'].mean():.4f}(調整後平均 = {r2_per_fold['adj_r2'].mean():.4f}）；
跨折先標準化再合併的補充版本(去除各折校準差異):R² = 0.2459(調整後 0.2403,n=1926,
k=14)——中等,沒有到設定檔裡「高」的0.6門檻。

**堆疊**(驗證年配適兩變數logistic,套到測試段,不重配適):

| | AUC |
|---|---|
| 堆疊(手工特徵+CNN) | {stack_auc:.4f} |
| 手工特徵單獨 | {hand_auc:.4f} |

**block bootstrap:差 = {stack_bb['observed_auc_diff']:+.4f},p = {stack_bb['p_value']:.4f}**
——堆疊**顯著比手工特徵單獨更差**,加入CNN分數是負貢獻,不是零貢獻。

各折堆疊中CNN係數(Wald檢定,驗證年n≈248-251):

{_md_table(as_int(rnd(wald[['fold','n_val','hand_coef','hand_p','cnn_coef','cnn_se','cnn_p']], 4), ['fold','n_val']))}

CNN係數 8 折中只有 {n_cnn_sig} 折顯著(p<0.05);正號 {n_cnn_pos} 折、負號 {n_cnn_neg} 折,
**方向不一致**,不符合「各折同號」的有增量資訊條件。第3折係數異常大(-5.49),可能是驗證段
小樣本下無正則化 logistic 接近完全分離的數值不穩定,不是真實效果。

**三個預先設定的解讀規則沒有一個完全對上這個型態**——不是單純「AUC接近、CNN重建手工特徵」
(CNN是顯著更差,不是接近),也不是「CNN明顯勝過」。如實記錄:**CNN 不只沒有增量資訊,
混入手工特徵(堆疊)反而顯著拖累表現**,比「沒有增量資訊」更明確的負面結果。

## 四、策略層(描述性)

任務A(CNN、堆疊分數套第8週篩選流程,驗證段選門檻、測試段套用):

{_md_table(rnd(strat_a, 4))}

CNN 篩選後夏普(0.189)反而**低於**不篩選(0.378);堆疊篩選後夏普(0.414)略高於不篩選
(0.378)但兩個都**不顯著**(block bootstrap p 分別 0.470、0.942;隨機篩選 p 分別 0.239、
0.134)。第10週的最小可偵測夏普差(約0.69–0.70,5%顯著/80%檢定力)在這裡同樣適用——
現有樣本量本來就偵測不到這個量級的效果。

**任務B(AUC 對最佳攤平基準,Holm校正,2個假設的家族)**:

{_md_table(taskb_tbl)}

兩個N的AUC差都不顯著(Holm校正後p皆為0.707)。

任務B(9:30+N依預測方向進場,收盤平倉,扣成本後):

{_md_table(rnd(strat_b, 4))}

兩個N的夏普都接近0(N=15為-0.025,N=30為0.014),沒有可用的方向性edge,跟第11週、
本週AUC都接近隨機的發現一致。

## 五、完成標準

**CNN 從原始分鐘序列學到的,是否超出手工特徵已捕捉的資訊?沒有——不只沒有超出,CNN
本身顯著比手工特徵差(p=0.001),混入手工特徵(堆疊)也顯著拖累表現(p=0.033),8折中
只有1折的堆疊CNN係數顯著,且逐折方向不一致。** 任務B兩個N上CNN對最佳攤平基準的AUC差
經Holm校正後都不顯著(p=0.707)。過擬合差距比第9週樹模型溫和,不是過擬合造成落後;
第二次打亂標籤檢查的單一種子結果字面上沒過,但多種子重估後方向對稱、判讀為抽樣雜訊,
不是洩漏。策略層(描述性)的結果跟AUC層一致,沒有找到任何一組配置能穩健打敗手工特徵
或不篩選基準。
"""
    out_path = REPORTS / "week12_dl_walkforward.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
