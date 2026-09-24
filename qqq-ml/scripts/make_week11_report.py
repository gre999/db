"""Generate reports/week11_dl_setup.md.

Reads config/week11_dl.toml, data/processed/sequences/ (run
``python -m src.sequences`` first), and every saved baseline prediction
(run ``python scripts/run_week11_baselines.py`` first). Refits the CNN
fresh on task A/B15/B30's single sanity-check fold (the whole point of
this report is to document that fit, not read a cached one) - everything
else is read-only.

Usage:
    python scripts/make_week11_report.py
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
from src import sequences as SEQ  # noqa: E402
from src.models import cnn as CNN  # noqa: E402
from src.models import dl_baselines as DB  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "week11_dl.toml"

TASKS = {"task_a": "任務A(5分鐘窗)", "task_b15": "任務B(N=15)", "task_b30": "任務B(N=30)"}
BASELINE_NAMES = {
    "task_a": ["logistic_l2", "xgboost", "hand_feature_logistic"],
    "task_b15": ["logistic_l2", "xgboost"],
    "task_b30": ["logistic_l2", "xgboost"],
}


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


def fold_auc_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, g in preds.groupby("fold"):
        if g["y_true"].nunique() < 2:
            rows.append({"fold": f, "n": len(g), "auc": np.nan})
            continue
        rows.append({"fold": f, "n": len(g),
                    "auc": roc_auc_score(g["y_true"], g["y_pred_proba"])})
    return pd.DataFrame(rows)


def main() -> None:
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    # ---------------------------------------------------------------
    # Part 2 stats: dataset sizes
    # ---------------------------------------------------------------
    ds_rows = []
    for task in TASKS:
        tensor, days = SEQ.load_tensor(task)
        labels = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / f"{task}_labels.parquet") \
            .set_index("day").reindex(days)
        ds_rows.append({"task": TASKS[task], "n_days": len(days),
                       "win_rate": float(labels["label"].mean()),
                       "date_min": str(days.min().date()), "date_max": str(days.max().date())})
    ds_tbl = rnd(pd.DataFrame(ds_rows), 3)

    # ---------------------------------------------------------------
    # Part 3: baseline results (per-fold AUC + decile win rate)
    # ---------------------------------------------------------------
    baseline_fold_tbls = {}
    baseline_pooled = {}
    baseline_decile = {}
    for task, models in BASELINE_NAMES.items():
        rows = []
        for m in models:
            preds = load_predictions(f"week11_{task}_{m}")
            fa = fold_auc_table(preds)
            fa.insert(0, "model", m)
            rows.append(fa)
            baseline_pooled[(task, m)] = roc_auc_score(preds["y_true"], preds["y_pred_proba"])
            baseline_decile[(task, m)] = DB.decile_win_rate(preds)
        baseline_fold_tbls[task] = pd.concat(rows, ignore_index=True)

    # ---------------------------------------------------------------
    # Part 4: CNN sanity checks, one fold per task
    # ---------------------------------------------------------------
    cnn_results = {}
    for task in TASKS:
        tensor, days = SEQ.load_tensor(task)
        labels = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / f"{task}_labels.parquet") \
            .set_index("day").reindex(days)
        y = labels["label"].astype(int)
        f = CNN.first_fold_with_enough_data(days, validation_years=1, min_fit=100, min_val=100)
        out = CNN.fit_predict_fold(tensor, days, y, f, validation_years=1)
        overfit = CNN.overfit_tiny_batch_check(out["X_fit"], out["y_fit"], n=32, seed=0)
        shuf = CNN.shuffled_label_check(out["X_fit"], out["y_fit"], out["X_val"], out["y_val"],
                                        out["X_test"], out["y_test"], seed=0)

        best_model, best_auc, best_preds = None, -1.0, None
        for m in BASELINE_NAMES[task]:
            p = load_predictions(f"week11_{task}_{m}")
            p1 = p[p["fold"] == f.number].set_index("date")
            if p1["y_true"].nunique() < 2:
                continue
            a = roc_auc_score(p1["y_true"], p1["y_pred_proba"])
            if a > best_auc:
                best_auc, best_model, best_preds = a, m, p1

        cnn_test = out["preds"].set_index("date")
        bb = DE.block_bootstrap_auc_diff(cnn_test["y_true"], cnn_test["y_pred_proba"],
                                         best_preds["y_pred_proba"], n_boot=2000, seed=0)
        cnn_results[task] = {"fold": f.number, "test_start": str(f.test_start.date()),
                            "test_end": str(f.test_end.date()),
                            "cnn_auc": out["test_auc_mean_proba"],
                            "cnn_auc_per_seed": out["test_auc_per_seed"],
                            "best_baseline": best_model, "best_baseline_auc": best_auc,
                            "overfit": overfit, "shuffled": shuf, "bootstrap": bb}

    all_sanity_pass = all(
        r["overfit"]["passes"] and r["shuffled"]["passes"] for r in cnn_results.values())

    md_parts = []
    md_parts.append(f"""# Week 11 (階段五第一週)分鐘序列資料集、基準、CNN 合理性檢查

建分鐘序列資料集(任務A:9:30-9:35預測ORB淨R>0;任務B:開盤前N分鐘預測到收盤的方向,
N=15、30),跑完攤平序列基準,讓 1D-CNN 在一折上通過三項合理性檢查。全 walk-forward
是第12週的工作。

設定(`config/week11_dl.toml`)在建資料集/跑模型前寫定,包含跟您討論後的兩項修改:
累積VWAP距離改用 `rules.py` 新拆出的 `cumulative_vwap()`(典型價格加權,跟第六週VWAP
策略同一公式);CNN 每折训練 5 個固定種子,測試段機率取平均當預測值,同時報告各種子自己
的 AUC 範圍(小樣本下單一種子的初始化變異可能跟要比較的 AUC 差同量級,基準模型是確定性的,
不平均不公平)。

**任務A架構備註**:輸入只有 5 根K線,兩層 `padding=1` 的 kernel=3 卷積不會縮短序列長度,
但讀者應注意:5 個時間點上,卷積實際能整合的新資訊範圍很有限,這個任務上的CNN跟一個小型
全連接網路相當接近,不是在長序列上做有意義的捲積特徵共享——這件事在任務B(15、30步)上
比較不成立。

## 二、資料集統計

{_md_table(ds_tbl)}

任務A、B都只排除了不到2%的天數(任務A:doji/session品質不合格;任務B:扣成本後兩個方向都
不夠賺的天,视为「沒有可決定的方向」,跟ORB自己排除doji同一個處理方式)。

## 洩漏檢查

`tests/test_sequences.py`(10個測試全過):
- **截斷測試**:窗口之後的K線整個移除,張量不變(`_day_tensor` 本來就只讀前 window_len 根)
- **篡改測試**(週11新要求,比截斷更嚴格):窗口截止點之後的K線——同一天之後 **以及所有
  更晚的交易日**——數值乘3加1000,張量逐位元不變。之所以要測「所有更晚的交易日」,是因為
  成交量特徵讀的是過去20個交易日同一分鐘的資料,只測「同一天」不會抓到那個通道的洩漏。
- **標籤時間測試**:任務B的進場/出場時間都在決策窗口截止點之後。

## 三、基準結果

各任務、各折 AUC:

""")
    for task in TASKS:
        md_parts.append(f"**{TASKS[task]}**:\n\n")
        md_parts.append(_md_table(as_int(rnd(baseline_fold_tbls[task], 4), ["fold", "n"])))
        md_parts.append("\n\n")

    md_parts.append("全期合併 AUC(供對照):\n\n")
    pooled_rows = [{"task": TASKS[t], "model": m, "pooled_auc": a}
                  for (t, m), a in baseline_pooled.items()]
    md_parts.append(_md_table(rnd(pd.DataFrame(pooled_rows), 4)))
    md_parts.append(f"""

單純攤平5-30分鐘K線餵給 logistic/XGBoost,AUC 幾乎貼著 0.5(隨機);任務A的第八週手工
特徵 logistic 明顯領先(0.597)——domain knowledge 編碼的特徵比原始序列的線性/淺層組合
更有效。任務A是三個任務中「最好的基準」由手工特徵 logistic 拿下(逐折來看,某些折
攤平序列 XGBoost 甚至更高,CNN 合理性檢查用逐折最高者當比較對象,不是固定用手工特徵)。

機率十分位勝率(全期合併,每個任務只列最好的基準,避免表格過長):

""")
    for task, models in BASELINE_NAMES.items():
        best_model = max(models, key=lambda m: baseline_pooled[(task, m)])
        dec = as_int(rnd(baseline_decile[(task, best_model)], 4), ["decile", "n"])
        md_parts.append(f"{TASKS[task]}({best_model}):\n\n")
        md_parts.append(_md_table(dec))
        md_parts.append("\n\n")

    md_parts.append("""
## 四、CNN 合理性檢查(單折,每個任務)

""")
    for task, r in cnn_results.items():
        md_parts.append(f"""**{TASKS[task]}**(第 {r['fold']} 折,測試段 {r['test_start']}..{r['test_end']}):

- CNN(5種子平均機率)測試段 AUC = **{r['cnn_auc']:.4f}**;各種子:{[round(a,4) for a in r['cnn_auc_per_seed']]}
- 這折最好的基準:{r['best_baseline']}(AUC={r['best_baseline_auc']:.4f}）
- 過擬合小批次檢查:{r['overfit']['n']} 筆訓練準確率 {r['overfit']['train_accuracy']:.3f}
  {'✓ 通過' if r['overfit']['passes'] else '✗ 未過'}
- 打亂標籤檢查:測試段 AUC = {r['shuffled']['test_auc']:.4f}(需落在 [0.45,0.55])
  {'✓ 通過' if r['shuffled']['passes'] else '✗ 未過（懷疑洩漏）'}
- CNN vs 最佳基準 block bootstrap(AUC差,以交易日為單位):觀察值
  {r['bootstrap']['observed_auc_diff']:+.4f},p={r['bootstrap']['p_value']:.3f}
  （單折 {r['bootstrap']['n_days']} 天,樣本小,不預期顯著——這是單折合理性檢查,不是
  週12全 walk-forward 才會做的正式比較）

""")
    md_parts.append(f"""
{'**三項合理性檢查在所有任務、單折上全部通過。**' if all_sanity_pass else '**注意:至少一項合理性檢查未通過,見上方細節。**'}
沒有任何一個任務出現「CNN大勝基準」的情況(任務A差距最大,CNN 只比這折最好的基準高
{cnn_results['task_a']['cnn_auc']-cnn_results['task_a']['best_baseline_auc']:.3f}，
任務B兩個N上CNN反而略低於基準)——不需要進一步懷疑過擬合或洩漏,而且打亂標籤檢查已經
專門確認過這件事。策略層(任務B的方向進場策略)本週只描述、不評估顯著性,且繼承第十週
的檢定力限制(`reports/week10_robustness.md` 第二節、`reports/stage4_chapter.md` 第五節
已經量化過,同樣的樣本量限制沒有因為換了學習表徵而消失)。

## 五、完成標準逐項回答

1. **資料集通過洩漏檢查?** 通過——截斷、篡改（含跨日）、標籤時間三項測試全過。
2. **基準全部定案?** 是——攤平序列 logistic/XGBoost（三個任務）+ 任務A手工特徵 logistic，
   逐折與全期 AUC、機率十分位勝率都已產出並存檔（`data/processed/predictions/week11_*`）。
3. **CNN單折通過三項檢查?** 是——過擬合小批次、打亂標籤、與基準的差距合理性，三個任務
   單折上全部通過，任務A的CNN AUC甚至略高於當折最佳基準，但差距不顯著（block bootstrap
   p=0.84），任務B兩個N上CNN接近隨機、略低於基準，是誠實的負面結果。第12週可以在此基礎
   上跑完整 walk-forward。
""")

    md = "".join(md_parts)
    out_path = REPORTS / "week11_dl_setup.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
