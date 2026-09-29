"""Week 13 (config/week13_wrapup.toml [results_summary]): recompute the
headline prediction-layer and strategy-layer numbers for phases 2-5 from
each week's own saved prediction/backtest files, and write reports/
results_summary.json. Never refits any model - every number here is
either read directly from an already-saved file or recomputed by calling
the exact evaluation function each week's own report used (Diebold-Mariano
for QLIKE, block bootstrap for Sharpe/AUC diffs), so it can be cross-
checked against that week's report text.

Phase 1 (weeks 1-2, data/features) has no predictive or strategy claim and
is excluded, matching final_outline.md's phase-support-paragraph scope
(phase 2-5 only).

Usage:
    python scripts/collect_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import backtest as BT  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import dl_eval as DE  # noqa: E402
from src import filter_eval as FE  # noqa: E402
from src import metrics as M  # noqa: E402
from src import regime_eval as E  # noqa: E402
from src.models import regimes as G  # noqa: E402
from src.models.base import load_predictions  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPORTS = ROOT / "reports"
SEQ_DIR = dl.PROCESSED_DIR / "sequences"


def jsonify(obj):
    """Recursively convert to JSON-safe types. NaN -> None (json.dumps'
    default allow_nan=True would otherwise emit a bare `NaN` token, which
    is valid for Python's own parser but not standard JSON - e.g.
    src.regime_eval._strategy_score leaves mean_r_net/mean_n_segments NaN
    for whichever strategy they don't apply to)."""
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return None if isinstance(v, float) and np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


# =========================================================================
# Phase 2 (weeks 3-5): HAR/XGBoost volatility forecasting -> position sizing
# =========================================================================
def _source_qlike(name: str, inputs: dict, cfg) -> float:
    """Same QLIKE definition as scripts/make_week05_report.py's
    source_qlike (variance forecast vs realized close-to-close variance) -
    reused here rather than duplicated logic, since hist20 has no walk-
    forward predictions file and isn't part of the DM-test pool."""
    idx = inputs["common_index"]
    rv_on = inputs["rv_on"].reindex(idx)
    if name == "hist20":
        var_cc = inputs["daily"]["ret_cc"].rolling(
            cfg.hist_window, min_periods=cfg.hist_window).var(ddof=0).reindex(idx)
    elif name in BT.MODEL_SOURCES:
        from src import strategies as S
        p = inputs["preds"][name].reindex(idx)
        var_cc = S.cc_variance_forecast(p, inputs["ratios"])
    else:
        raise ValueError(name)
    ok = var_cc.notna() & rv_on.notna() & (var_cc > 0) & (rv_on > 0)
    return float(M.qlike_loss(rv_on[ok], var_cc[ok]).mean())


def phase2() -> dict:
    inputs = BT.load_inputs()
    cfg = BT.BacktestConfig()
    out = BT.run(cfg, inputs=inputs, save=False)
    bts = out["backtests"]

    pred = pd.concat([load_predictions("har_baselines"), load_predictions("week4_models")],
                     ignore_index=True)
    pred = pred[pred["target"] == "rv"]
    common = set.intersection(*[set(g) for _, g in pred.groupby("model")["date"]])
    pred = pred[pred["date"].isin(common)]
    ql = pred.assign(ql=M.qlike_loss(pred["y_true_var"], pred["y_pred_var"]))
    har = ql[ql["model"] == "har"].set_index("date")["ql"]
    xgb = ql[ql["model"] == "har_resid_xgb"].set_index("date")["ql"]
    dm, dm_p = M.diebold_mariano(xgb.loc[har.index], har)

    def strat_row(a: str, b: str) -> dict:
        ret_a, ret_b = bts[a]["net_return"], bts[b]["net_return"]
        obs, p = BT.block_bootstrap_sharpe_diff(ret_a, ret_b, block_size=20, n_boot=3000, seed=0)
        mde = BT.mde_from_returns(ret_a, ret_b, block_size=20, n_boot=3000, seed=0)
        eval_a, eval_b = BT.evaluate(bts[a], cfg.vol_target), BT.evaluate(bts[b], cfg.vol_target)
        diff_comp = float(eval_a["sharpe"] - eval_b["sharpe"])
        return {"sharpe": BT.arithmetic_sharpe(ret_a), "sharpe_compounding": float(eval_a["sharpe"]),
               "unfiltered_sharpe": BT.arithmetic_sharpe(ret_b),
               "unfiltered_sharpe_compounding": float(eval_b["sharpe"]),
               "max_drawdown": float(eval_a["max_drawdown"]),
               "diff": obs, "diff_compounding": diff_comp,
               "direction_match": bool(np.sign(obs) == np.sign(diff_comp)),
               "p_value": p, "mde": mde["mde"],
               "mde_note": "事後補算,僅供判讀,不改變原結論(week5未算MDE)"}

    qlike_hist20 = _source_qlike("hist20", inputs, cfg)
    qlike_xgb_point = _source_qlike("har_resid_xgb", inputs, cfg)

    return {
        "weeks": [3, 4, 5],
        "rows": [
            {
                "row": "primary", "label": "HAR+殘差XGB vs HAR",
                "prediction_layer": {"metric": "QLIKE (target=y_true_var, week3/4預測檔自己的RTH-only已實現變異數)",
                                     "value_a": float(xgb.mean()),
                                     "value_b": float(har.mean()),
                                     "diff": float((xgb.loc[har.index] - har).mean()),
                                     "test": "Diebold-Mariano", "dm_stat": float(dm),
                                     "p_value": float(dm_p),
                                     "cross_check_note": "此列QLIKE跟下面secondary列的QLIKE數字不能直接比較:這列的目標變數是y_true_var(RTH-only已實現變異數,week3/4預測檔本身的定義),secondary列的目標變數是week5 backtest層的rv_on(=rv+隔夜報酬平方,收盤到收盤變異數)。查過src/strategies.py的scale_ratios/cc_variance_forecast docstring,確認是目標變數定義不同,不是計算錯誤。"},
                "strategy_layer": strat_row("har_resid_xgb", "har"),
            },
            {
                "row": "secondary", "label": "HAR+殘差XGB vs 20日歷史波動",
                "prediction_layer": {"metric": "QLIKE (target=rv_on, week5 backtest層的收盤到收盤變異數,含隔夜跳空)",
                                     "value_a": qlike_xgb_point,
                                     "value_b": qlike_hist20,
                                     "diff": qlike_xgb_point - qlike_hist20,
                                     "note": "hist20不是walk-forward預測模型(用當天自己的滾動已實現變異數當預測),不適用DM檢定架構,只報點估計,無p值"},
                "strategy_layer": strat_row("har_resid_xgb", "hist20"),
            },
        ],
    }


# =========================================================================
# Phase 3 (weeks 6-7): HMM/KMeans market-state labels
# =========================================================================
def phase3() -> dict:
    X = G.load_state_features()
    pred = pd.read_parquet(E.REGIME_DIR / "regime_labels_main_k3.parquet")
    orb = pd.read_parquet(E.STRATEGY_DIR / "orb_daily.parquet")
    vwap = pd.read_parquet(E.STRATEGY_DIR / "vwap_daily.parquet")

    # prediction layer: model-external validation (predictive HMM state vs
    # actual realized volatility, Newey-West HAC), reused directly from
    # week07_regime_eval.md's own table - recomputed here via the same
    # function, not refit.
    from src import features as F
    desc_feat = F.build_descriptive_matrix(F.MarketData.from_processed())
    ext_rv = E.external_validation(pred, desc_feat, "hmm", "log_rv_desc")
    pred_layer = {"metric": "predictive-HMM-state vs realized log_rv_desc (Newey-West HAC)",
                 "reference_state": 0,
                 "states": ext_rv[ext_rv["state"] != 0][["state", "diff_vs_state0", "se", "t", "p", "n"]]
                          .to_dict(orient="records")}

    rows = []
    for strat_name, strat in (("orb", orb), ("vwap", vwap)):
        hmm_out = E.tradability_test(X, strat, "hmm", n_init=10)
        kmeans_out = E.tradability_test(X, strat, "kmeans", n_init=10)
        fold_windows = (pred[pred["model"] == "hmm"]
                        .groupby("fold")["date"].agg(test_start="min", test_end="max")
                        .reset_index())
        vol_pred = load_predictions("week4_models")
        vol_pred = vol_pred[(vol_pred["model"] == "har_resid_xgb") & (vol_pred["target"] == "rv")]
        vt = E.vol_tercile_labels(fold_windows, vol_pred)
        vt_out = E.tradability_test_vol_tercile(strat, fold_windows, vol_pred)
        common_idx, common_tbl = E.tradability_common_dates(
            {"hmm": hmm_out, "kmeans": kmeans_out, "vol_tercile": vt_out})

        f_common = hmm_out["filtered"].loc[common_idx] / 1e4
        u_common = hmm_out["unfiltered"].loc[common_idx] / 1e4
        obs, p = BT.block_bootstrap_sharpe_diff(f_common, u_common, block_size=20, n_boot=3000, seed=0)
        mde = BT.mde_from_returns(f_common, u_common, block_size=20, n_boot=3000, seed=0)
        hmm_row = common_tbl[common_tbl["model"] == "hmm"].iloc[0]
        diff_comp = float(hmm_row["filtered_sharpe"] - hmm_row["unfiltered_sharpe"])

        rows.append({
            "row": strat_name, "label": f"HMM 狀態篩選 {strat_name.upper()} vs 不篩選",
            "date_range": f"{common_idx.min().date()}..{common_idx.max().date()} ({len(common_idx)} 天)",
            "prediction_layer": pred_layer,
            "strategy_layer": {"sharpe": BT.arithmetic_sharpe(f_common),
                              "sharpe_compounding": float(hmm_row["filtered_sharpe"]),
                              "unfiltered_sharpe": BT.arithmetic_sharpe(u_common),
                              "unfiltered_sharpe_compounding": float(hmm_row["unfiltered_sharpe"]),
                              "max_drawdown": float(hmm_row["filtered_max_drawdown"]),
                              "diff": obs, "diff_compounding": diff_comp,
                              "direction_match": bool(np.sign(obs) == np.sign(diff_comp)),
                              "p_value": p, "mde": mde["mde"],
                              "mde_note": "事後補算,僅供判讀,不改變原結論(week6-7未算MDE)"},
        })

    return {"weeks": [6, 7], "rows": rows}


# =========================================================================
# Phase 4 (weeks 8-10): trading-day filter (logistic vs unfiltered ORB)
# =========================================================================
def phase4() -> dict:
    preds = load_predictions("week08_filter")
    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")

    result = FE.evaluate_success(preds, orb, min_trades_per_year=20)
    # n_boot=2000, seed=0 matches week08/10's own calls exactly (see
    # scripts/make_week08_report.py:154, make_week10_report.py:79,135) -
    # kept identical here (not week13's own n_boot=3000/5000 choices) so
    # this row reproduces, not just approximates, the published numbers.
    bb = FE.block_bootstrap_significance(preds, orb, n_boot=2000, seed=0)
    pw = FE.power_analysis(preds, orb, n_boot=2000, seed=0)
    series = FE.logistic_filter_series(preds, orb)
    filt, unfilt = series["filtered"] / 1e4, series["unfiltered"] / 1e4
    diff_comp = float(result["filtered_sharpe"] - result["unfiltered_sharpe"])

    return {
        "weeks": [8, 9, 10],
        "rows": [{
            "row": "primary", "label": "logistic 篩選 ORB vs 不篩選",
            "prediction_layer": {"metric": "AUC (pooled OOS)",
                                 "value": float(roc_auc_score(preds["y_true"], preds["y_pred_proba"]))},
            "strategy_layer": {"sharpe": BT.arithmetic_sharpe(filt),
                              "sharpe_compounding": float(result["filtered_sharpe"]),
                              "unfiltered_sharpe": BT.arithmetic_sharpe(unfilt),
                              "unfiltered_sharpe_compounding": float(result["unfiltered_sharpe"]),
                              "max_drawdown": float(result["filtered_max_drawdown"]),
                              "diff": float(bb["observed_sharpe_diff"]), "diff_compounding": diff_comp,
                              "direction_match": bool(np.sign(bb["observed_sharpe_diff"]) == np.sign(diff_comp)),
                              "p_value": float(bb["p_value"]), "mde": float(pw["mde"])},
        }],
    }


# =========================================================================
# Phase 5 (weeks 11-12): CNN on minute sequences
# =========================================================================
def phase5() -> dict:
    cnn_a = load_predictions("week12_task_a_cnn").set_index("date")
    hand_a = load_predictions("week11_task_a_hand_feature_logistic").set_index("date")
    common = cnn_a.index.intersection(hand_a.index)

    cnn_auc = roc_auc_score(cnn_a.loc[common, "y_true"], cnn_a.loc[common, "y_pred_proba"])
    hand_auc = roc_auc_score(hand_a.loc[common, "y_true"], hand_a.loc[common, "y_pred_proba"])
    # n_boot=5000 here (not week12's original 2000) per config/week13_wrapup.toml
    # [stage5.auc_power_analysis] - deliberately bumped for SE precision, not a
    # cross-check mismatch. p_value drifts from week12's reported 0.0010 to
    # ~0.0008 here; both are far below 0.05 and the conclusion is unchanged.
    bb = DE.block_bootstrap_auc_diff(cnn_a.loc[common, "y_true"], cnn_a.loc[common, "y_pred_proba"],
                                     hand_a.loc[common, "y_pred_proba"], block_size=20, n_boot=5000, seed=0)
    pw = DE.auc_power_analysis(cnn_a.loc[common, "y_true"], cnn_a.loc[common, "y_pred_proba"],
                               hand_a.loc[common, "y_pred_proba"], block_size=20, n_boot=5000, seed=0)

    strat = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / "week12_task_a_strategy_layer.parquet")
    cnn_strat = strat[strat["model"] == "cnn"].iloc[0]
    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")
    cnn_filter_preds = load_predictions("week12_task_a_cnn_filter")
    cnn_series = FE.logistic_filter_series(cnn_filter_preds, orb)
    cnn_filt, cnn_unfilt = cnn_series["filtered"] / 1e4, cnn_series["unfiltered"] / 1e4
    diff_comp = float(cnn_strat["filtered_sharpe"] - cnn_strat["unfiltered_sharpe"])
    diff_arith = BT.arithmetic_sharpe(cnn_filt) - BT.arithmetic_sharpe(cnn_unfilt)

    return {
        "weeks": [11, 12],
        "rows": [{
            "row": "primary", "label": "CNN vs 手工特徵 logistic (task A)",
            "prediction_layer": {"metric": "AUC (pooled, 2019-2026共同日期)",
                                 "value_a": float(cnn_auc), "value_b": float(hand_auc),
                                 "diff": float(bb["observed_auc_diff"]), "p_value": float(bb["p_value"]),
                                 "mde": float(pw["mde"])},
            "strategy_layer": {"sharpe": BT.arithmetic_sharpe(cnn_filt),
                              "sharpe_compounding": float(cnn_strat["filtered_sharpe"]),
                              "unfiltered_sharpe": BT.arithmetic_sharpe(cnn_unfilt),
                              "unfiltered_sharpe_compounding": float(cnn_strat["unfiltered_sharpe"]),
                              "max_drawdown": float(cnn_strat["filtered_max_drawdown"]),
                              "diff": diff_arith, "diff_compounding": diff_comp,
                              "direction_match": bool(np.sign(diff_arith) == np.sign(diff_comp)),
                              "p_value": float(cnn_strat["bb_p"]),
                              "mde": "見week10 MDE~0.7(策略層樣本量與階段四相同量級,未在week12重算)"},
        }],
    }


def robustness_gaps() -> dict:
    """Week 14 (config/week14_final.toml [gapfill.*]): fold the five
    robustness-matrix gap-fill results (scripts/run_week14_robustness.py)
    into results_summary.json. Read-only here - the actual computation
    already ran and wrote data/processed/sequences/week14_*; this just
    re-shapes it under a stable key. None of these results contradict any
    earlier week's conclusion (see reports/final_report.md's discussion),
    so no correction-log entry was needed for them."""
    gaps = json.loads((SEQ_DIR / "week14_robustness_gaps.json").read_text(encoding="utf-8"))
    phase3_yearly = pd.read_parquet(SEQ_DIR / "week14_phase3_yearly.parquet")

    return {
        "phase2": {
            "random_baseline": {
                **gaps["phase2_random_baseline"],
                "design_note": "打亂har_resid_xgb的y_pred_var在日期間的對應(fold/校準比例"
                              "不動),接回同一套vol_target部位規則,p=null>=observed的比例"
                              "(單尾,跟week8 random_filter_p_value同一套邏輯)。這個檢定回答"
                              "的問題跟階段二主比較(HAR+殘差XGB vs HAR)不同——這裡問的是"
                              "『用任何一組跟真實預測同尺度但日期打亂的變異數估計去做波動"
                              "目標部位,能不能跟這個模型的實際表現一樣好』,不是『這個模型"
                              "比另一個模型準不準』。p=0.031單一檢定,不是本專案原本四項"
                              "標準的一部分,解讀要保守。",
            },
        },
        "phase3": {
            "cost_doubling": {
                **gaps["phase3_cost_doubling"],
                "design_note": "ORBConfig/VWAPConfig的cost_per_share加倍(0.0045→0.009/股),"
                              "run_orb/run_vwap只在記憶體內重算,不覆蓋orb_daily.parquet/"
                              "vwap_daily.parquet。tradability_test內部會重新呼叫"
                              "fit_hmm_fold_main,但該函式random_state=0寫死、確定性配適,"
                              "重跑會重現跟regime_labels_main_k3.parquet完全相同的狀態指派"
                              "(不是產生新的模型決定),只有策略自己的bps_return因成本改變"
                              "而不同。ORB/VWAP兩者在雙倍成本下,篩選vs不篩選的方向"
                              "(filtered更差)跟主成本下完全一致,結論不變。",
            },
            "yearly": phase3_yearly.to_dict(orient="records"),
        },
        "phase4": {
            "target_r_sensitivity": gaps["phase4_target_r_sensitivity"],
        },
        "phase5": {
            "yearly": gaps["phase5_yearly_and_cost"]["yearly"],
            "n_years_filtered_better": gaps["phase5_yearly_and_cost"]["n_years_filtered_better"],
            "n_years_total": gaps["phase5_yearly_and_cost"]["n_years_total"],
            "cost_doubling": {
                **gaps["phase5_yearly_and_cost"]["cost_2x"],
                "design_note": "CNN(task A)篩選keep決定不變(不重新訓練CNN),只把ORBConfig"
                              "的cost_per_share加倍重跑run_orb,套用同一組CNN keep決定——"
                              "雙倍成本下篩選仍比不篩選差(diff同號),跟主成本下的結論一致。",
            },
        },
    }


def _direction_check(summary: dict) -> list[dict]:
    """Week 13 (per-user follow-up): after switching results_summary.json's
    main sharpe/diff fields from compounding to arithmetic, confirm every
    row's DIRECTION (which side is higher) is unchanged - list each row's
    arithmetic vs compounding diff sign side by side rather than asserting
    it silently."""
    rows = []
    for phase_name in ("phase2", "phase3", "phase4", "phase5"):
        for row in summary[phase_name]["rows"]:
            sl = row["strategy_layer"]
            if "diff_compounding" not in sl:
                continue
            rows.append({"phase": phase_name, "row": row["row"], "label": row["label"],
                        "diff_arithmetic": sl["diff"], "diff_compounding": sl["diff_compounding"],
                        "direction_match": sl["direction_match"]})
    return rows


def main() -> None:
    phases = {"phase2": phase2(), "phase3": phase3(), "phase4": phase4(), "phase5": phase5()}
    direction_check = _direction_check(phases)
    n_flipped = sum(1 for r in direction_check if not r["direction_match"])

    summary = {
        "_cross_check_notes": [
            "改用算術平均年化Sharpe(mean/std*sqrt(252),BT.arithmetic_sharpe)當「sharpe」/"
            "「unfiltered_sharpe」/「diff」欄位的主要定義,跟block bootstrap顯著性檢定、"
            "最小可偵測差(mde)用的是同一個定義,所以表中兩個sharpe相減就等於diff,不需要"
            "另外換算。「sharpe_compounding」/「unfiltered_sharpe_compounding」/"
            "「diff_compounding」是複利/幾何年化報酬版本(BT.evaluate/E._daily_eval,"
            "equity[-1]**(252/n)-1),對照用——**各週報告本身顯示的都是複利版本,這裡沒有"
            "改各週報告,只有這份程式產生的總表換了主要口徑**,方法論說明見"
            "final_outline.md。",
            f"方向檢查(誰高誰低,算術版 vs 複利版):{len(direction_check)} 個策略層比較"
            f"全部檢查過,{n_flipped} 個方向不同({'無' if n_flipped == 0 else '見下'})——"
            "見 _direction_check 欄位逐列比較。",
            "階段二 primary/secondary 兩列的 QLIKE 用不同目標變數(y_true_var vs rv_on)"
            "計算,見各列自己的 cross_check_note,不能跨列比較。",
        ],
        "_direction_check": direction_check,
        **phases,
        "robustness_gaps": robustness_gaps(),
    }
    summary = jsonify(summary)
    p = REPORTS / "results_summary.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p} - direction check: {n_flipped}/{len(direction_check)} flipped")


if __name__ == "__main__":
    main()
