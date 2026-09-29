"""Week 14 Part 3: fill reports/final_report_template.md's {{PLACEHOLDER}}
tokens from reports/results_summary.json (+ a handful of already-verified
historical figures that predate results_summary.json's existence, cited
inline below) and write reports/final_report.md. No new computation here -
every number is read from an already-generated file.

Sharpe figures used throughout are the ARITHMETIC-MEAN definition
(results_summary.json's main "sharpe"/"diff" fields), matching the
template's own "統計慣例說明" section - never the "_compounding" fields.

Usage:
    python scripts/make_final_report.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"


def pct(x: float) -> str:
    return f"{x:+.4f}"


def main() -> None:
    d = json.loads((REPORTS / "results_summary.json").read_text(encoding="utf-8"))
    rg = d["robustness_gaps"]

    p2 = d["phase2"]["rows"][0]
    p2_sec = d["phase2"]["rows"][1]
    p3_orb = d["phase3"]["rows"][0]
    p3_vwap = d["phase3"]["rows"][1]
    p4 = d["phase4"]["rows"][0]
    p5 = d["phase5"]["rows"][0]
    p2_rand = rg["phase2"]["random_baseline"]
    p3_2x = {r["strategy"]: r for r in rg["phase3"]["cost_doubling"]["rows"]}
    p4_r = {r["target_r"]: r for r in rg["phase4"]["target_r_sensitivity"]["rows"]}
    p5_gaps = rg["phase5"]

    # Historical figures verified against source reports/commits during
    # week 6/10/12/13 (not stored in results_summary.json, which only
    # started in week 12) - citations noted where each was checked.
    HIST = {
        "P3_ORB_REPL_SHARPE": "1.012", "P3_ORB_PAPER_SHARPE": "1.13",     # week06_regimes.md 第二節
        "P3_VWAP_REPL_SHARPE": "2.339", "P3_VWAP_PAPER_SHARPE": "2.1",    # week06_regimes.md 第二節
        "P10_YEARS_NEEDED": "104",                                        # week10_robustness.md 第五節
        "P5_FOLD2_AUC": "0.4483",                                         # week12_task_a_fold_diagnostics.parquet fold 2
    }

    null_pctile = (p2_rand["p_value"])
    null_pctile_str = f"{(1 - null_pctile) * 100:.1f}"

    values = {
        "REPORT_DATE": dt.date.today().isoformat(),

        "P2_QLIKE_DIFF": pct(p2["prediction_layer"]["diff"]),
        "P2_DM_P": f"{p2['prediction_layer']['p_value']:.5f}",
        "P2_STRAT_DIFF": pct(p2["strategy_layer"]["diff"]),
        "P2_STRAT_P": f"{p2['strategy_layer']['p_value']:.4f}",
        "P2_STRAT_DIFF_SEC": pct(p2_sec["strategy_layer"]["diff"]),
        "P2_STRAT_P_SEC": f"{p2_sec['strategy_layer']['p_value']:.4f}",
        "P2_MDE": f"{p2['strategy_layer']['mde']:.4f}",
        "P2_RANDOM_NREPS": str(p2_rand["n_reps"]),
        "P2_RANDOM_OBSERVED": f"{p2_rand['observed_sharpe']:.4f}",
        "P2_RANDOM_PERCENTILE": null_pctile_str,
        "P2_RANDOM_P": f"{p2_rand['p_value']:.4f}",

        "P3_HMM_T1": f"{p3_orb['prediction_layer']['states'][0]['t']:.2f}",
        "P3_HMM_T2": f"{p3_orb['prediction_layer']['states'][1]['t']:.2f}",
        "P3_ORB_FILTERED": f"{p3_orb['strategy_layer']['sharpe']:.4f}",
        "P3_ORB_UNFILTERED": f"{p3_orb['strategy_layer']['unfiltered_sharpe']:.4f}",
        "P3_ORB_DIFF": pct(p3_orb["strategy_layer"]["diff"]),
        "P3_ORB_P": f"{p3_orb['strategy_layer']['p_value']:.4f}",
        "P3_VWAP_FILTERED": f"{p3_vwap['strategy_layer']['sharpe']:.4f}",
        "P3_VWAP_UNFILTERED": f"{p3_vwap['strategy_layer']['unfiltered_sharpe']:.4f}",
        "P3_VWAP_DIFF": pct(p3_vwap["strategy_layer"]["diff"]),
        "P3_VWAP_P": f"{p3_vwap['strategy_layer']['p_value']:.4f}",
        "P3_ORB_2X_DIFF": pct(p3_2x["orb"]["diff"]),
        "P3_VWAP_2X_DIFF": pct(p3_2x["vwap"]["diff"]),

        "P4_AUC": f"{p4['prediction_layer']['value']:.4f}",
        "P4_STRAT_A": f"{p4['strategy_layer']['sharpe']:.4f}",
        "P4_STRAT_B": f"{p4['strategy_layer']['unfiltered_sharpe']:.4f}",
        "P4_STRAT_DIFF": pct(p4["strategy_layer"]["diff"]),
        "P4_STRAT_P": f"{p4['strategy_layer']['p_value']:.4f}",
        "P4_MDE": f"{p4['strategy_layer']['mde']:.4f}",
        "P4_5R_DIFF": pct(p4_r[5.0]["diff"]),
        "P4_5R_P": f"{p4_r[5.0]['p_value']:.4f}",
        "P4_15R_DIFF": pct(p4_r[15.0]["diff"]),
        "P4_15R_P": f"{p4_r[15.0]['p_value']:.4f}",

        "P5_AUC_DIFF": pct(p5["prediction_layer"]["diff"]),
        "P5_AUC_P": f"{p5['prediction_layer']['p_value']:.4f}",
        "P5_STRAT_A": f"{p5['strategy_layer']['sharpe']:.4f}",
        "P5_STRAT_B": f"{p5['strategy_layer']['unfiltered_sharpe']:.4f}",
        "P5_STRAT_DIFF": pct(p5["strategy_layer"]["diff"]),
        "P5_STRAT_P": f"{p5['strategy_layer']['p_value']:.4f}",
        "P5_YEARLY_BETTER": str(p5_gaps["n_years_filtered_better"]),
        "P5_YEARLY_TOTAL": str(p5_gaps["n_years_total"]),
        "P5_2X_DIFF": pct(p5_gaps["cost_doubling"]["diff"]),

        "P345_MDE_RANGE": "0.69–0.79",
    }
    values.update(HIST)

    # ---------------------------------------------------------------
    # Embedded tables/figure (markdown blocks, not simple scalars)
    # ---------------------------------------------------------------
    direction_rows = d["_direction_check"]
    tbl_direction = ["| 階段/列 | 算術版差 | 複利版差 | 方向一致? |", "|---|---|---|---|"]
    for r in direction_rows:
        tbl_direction.append(f"| {r['phase']}/{r['row']} | {r['diff_arithmetic']:+.4f} | "
                             f"{r['diff_compounding']:+.4f} | {'是' if r['direction_match'] else '否'} |")
    tbl_direction.append("\nTable: 算術版與複利版年化Sharpe差的方向核對(第13週)")
    values["TBL_DIRECTION_CHECK"] = "\n".join(tbl_direction)

    values["TBL_P4_OVERFIT"] = "\n".join([
        "| 模型 | 配適段AUC | 測試段AUC | 差距 |", "|---|---|---|---|",
        "| logistic | 0.667 | 0.612 | 0.056 |",
        "| 隨機森林 | 0.828 | 0.620 | 0.207 |",
        "| XGBoost(深度2) | 0.901 | 0.591 | 0.310 |",
        "| XGBoost(深度3,附錄) | 0.965 | 0.591 | 0.373 |",
        "\nTable: 第9週樹模型過擬合差距(配適段AUC−測試段AUC)",
    ])

    why_hard_rows = [
        ("phase2", p2["strategy_layer"]["diff"], p2["strategy_layer"]["p_value"],
         p2["strategy_layer"]["mde"], "有檢定力支持的沒有實質差異"),
        ("phase3", p3_orb["strategy_layer"]["diff"], p3_orb["strategy_layer"]["p_value"],
         p3_orb["strategy_layer"]["mde"], "偵測不到"),
        ("phase4", p4["strategy_layer"]["diff"], p4["strategy_layer"]["p_value"],
         p4["strategy_layer"]["mde"], "偵測不到"),
        ("phase5", p5["strategy_layer"]["diff"], p5["strategy_layer"]["p_value"],
         p5["strategy_layer"]["mde"], "偵測不到"),
    ]
    tbl_why = ["| 階段 | 策略層觀察差 | p值 | MDE | 達MDE? | 判讀 |", "|---|---|---|---|---|---|"]
    for name, diff, p, mde, interp in why_hard_rows:
        tbl_why.append(f"| {name} | {diff:+.4f} | {p:.4f} | {mde:.4f} | "
                       f"{'是' if abs(diff) >= mde else '否'} | {interp} |")
    tbl_why.append("\nTable: 策略層觀察差 vs 最小可偵測差(算術平均Sharpe定義),全部讀自 results_summary.json")
    values["TBL_WHY_HARD"] = "\n".join(tbl_why)

    values["FIG_WHY_HARD"] = "\n".join([
        "![各階段策略層觀察差(算術平均Sharpe,保留正負號)與±最小可偵測差區間;資料範圍"
        "見各階段自己的OOS walk-forward測試段(階段二2015-2026、階段三2020-2026共同範圍、"
        "階段四2018-2026、階段五2019-2026)](figures/w14_diff_vs_mde.png)",
    ])

    corrections = [
        (5, "20日歷史波動部位權重的前視偏誤",
         "進入階段三之前主動做的全面前視偏誤複查(不是被結果異常倒逼發現)",
         "hist20變體的滾動窗納入當天自己的收盤到收盤報酬;修正前Sharpe 1.4715、修正後1.1568",
         "修正後HAR、HAR+殘差XGB、20日歷史波動三者Sharpe統計上無顯著差異"),
        (6, "HMM樣本外狀態解碼漏了測試段第一天的轉移步驟",
         "寫test_forward_filter_init_dist_seeds_without_lookahead這個測試時發現(commit c3e15b1),"
         "在看任何第6週結果之前就抓到並修正",
         "forward_filter手寫前向演算法,測試段第一天原本被當成全新序列起點,沒有從訓練段"
         "最後一天的信念狀態往前轉移",
         "修正後才產生week06_regimes.md報告的全部HMM樣本外狀態數字"),
        (6, "VWAP論文複現誤用ORB的部位規則",
         "看到異常結果後查出:第一次複現Sharpe 3.26、最大回撤-35.9%,遠超論文的2.1/-0.094",
         "VWAP複現誤套用ORB的「每筆風險1%權益、4倍槓桿上限」規則,而不是VWAP自己論文的"
         "「100%權益、不加槓桿、股數當日開盤算死」規則",
         "改為1倍槓桿、每天開盤算一次股數後,複現Sharpe 2.339、最大回撤-0.0957,同論文"
         "2.1/-0.094同一量級"),
        (7, "跨狀態來源比較的日期範圍不一致",
         "config/week07_regime_eval.toml內建的跨折對齊修正,以及報告內波動三分位只覆蓋"
         "第2-8折的說明",
         "三個狀態來源(HMM、KMeans、波動三分位)的可用日期範圍不同,直接比較「不篩選」"
         "欄位不是同一組交易日",
         "改成在共同涵蓋日期範圍(2020-01-02..2026-09-22,1689天)上比較"),
        (8, "原始三項成功標準沒有一項是差異顯著性檢定",
         "使用者在定稿前要求補四項檢查(block bootstrap、逐年拆解、成本敏感度),推翻了"
         "字面「三項全過」的結論",
         "篩選後Sharpe 0.707對不篩選0.517字面達成,但block bootstrap p=0.485不顯著,"
         "效果集中在兩個極端年份部分抵銷",
         "第9週起把block bootstrap p<0.05明訂為四項成功標準之一"),
        (10, "隨機篩選基準重估種子的蒙地卡羅精度不足",
         "n_reps=1000時5個種子的p值在0.030-0.051間跳動,當時解讀成「排序層不穩健」",
         "n=1000在p≈0.04附近標準誤約0.006,5個種子0.021的全距大半是估計精度不足的雜訊",
         "改用n_reps=10000重估,3個種子穩定落在0.042-0.047,排序層結果其實穩定"),
    ]
    tbl_corr = ["| 週 | 標題 | 發現方式 | 影響範圍與修正後結論 |", "|---|---|---|---|"]
    for wk, title, found, scope, concl in corrections:
        tbl_corr.append(f"| {wk} | {title} | {found} | {scope}。**修正後**:{concl} |")
    tbl_corr.append("\nTable: 修正紀錄(逐條查證過原始commit/報告文字)")
    values["TBL_CORRECTIONS"] = "\n".join(tbl_corr)

    # ---------------------------------------------------------------
    # Fill
    # ---------------------------------------------------------------
    template = (REPORTS / "final_report_template.md").read_text(encoding="utf-8")
    out = template
    for key, val in values.items():
        out = out.replace("{{" + key + "}}", str(val))

    remaining = [tok for tok in out.split("{{")[1:]]
    remaining_keys = [tok.split("}}")[0] for tok in remaining if "}}" in tok]
    if remaining_keys:
        raise ValueError(f"unfilled placeholders remain: {remaining_keys}")

    out_path = REPORTS / "final_report.md"
    out_path.write_text(out, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
