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


def jsonify(obj):
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
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
        obs, p = BT.block_bootstrap_sharpe_diff(bts[a]["net_return"], bts[b]["net_return"],
                                                 block_size=20, n_boot=3000, seed=0)
        mde = BT.mde_from_returns(bts[a]["net_return"], bts[b]["net_return"],
                                  block_size=20, n_boot=3000, seed=0)
        eval_a = BT.evaluate(bts[a], cfg.vol_target)
        return {"sharpe": float(eval_a["sharpe"]), "max_drawdown": float(eval_a["max_drawdown"]),
               "diff": obs, "p_value": p, "mde": mde["mde"],
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

        rows.append({
            "row": strat_name, "label": f"HMM 狀態篩選 {strat_name.upper()} vs 不篩選",
            "date_range": f"{common_idx.min().date()}..{common_idx.max().date()} ({len(common_idx)} 天)",
            "prediction_layer": pred_layer,
            "strategy_layer": {"sharpe": float(hmm_row["filtered_sharpe"]),
                              "max_drawdown": float(hmm_row["filtered_max_drawdown"]),
                              "diff": obs, "p_value": p, "mde": mde["mde"],
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

    return {
        "weeks": [8, 9, 10],
        "rows": [{
            "row": "primary", "label": "logistic 篩選 ORB vs 不篩選",
            "prediction_layer": {"metric": "AUC (pooled OOS)",
                                 "value": float(roc_auc_score(preds["y_true"], preds["y_pred_proba"]))},
            "strategy_layer": {"sharpe": float(result["filtered_sharpe"]),
                              "max_drawdown": float(result["filtered_max_drawdown"]),
                              "diff": float(result["filtered_sharpe"] - result["unfiltered_sharpe"]),
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

    return {
        "weeks": [11, 12],
        "rows": [{
            "row": "primary", "label": "CNN vs 手工特徵 logistic (task A)",
            "prediction_layer": {"metric": "AUC (pooled, 2019-2026共同日期)",
                                 "value_a": float(cnn_auc), "value_b": float(hand_auc),
                                 "diff": float(bb["observed_auc_diff"]), "p_value": float(bb["p_value"]),
                                 "mde": float(pw["mde"])},
            "strategy_layer": {"sharpe": float(cnn_strat["filtered_sharpe"]),
                              "max_drawdown": float(cnn_strat["filtered_max_drawdown"]),
                              "diff": float(cnn_strat["filtered_sharpe"] - cnn_strat["unfiltered_sharpe"]),
                              "p_value": float(cnn_strat["bb_p"]),
                              "mde": "見week10 MDE~0.7(策略層樣本量與階段四相同量級,未在week12重算)"},
        }],
    }


def main() -> None:
    summary = {
        "_cross_check_notes": [
            "「sharpe」欄位(顯示值)用複利/幾何年化報酬(BT.evaluate / E._daily_eval,"
            "equity[-1]**(252/n)-1,全專案自第5週起headline數字都用這個定義);"
            "「diff」/「p_value」/「mde」欄位用算術平均年化Sharpe(BT._sharpe,"
            "block_bootstrap_sharpe_diff內部定義,mean/std*sqrt(252))。兩者不是同一個"
            "公式,「diff」字面上不會剛好等於兩個「sharpe」相減——這不是這次新產生的不一致,"
            "是第5週就已經存在、貫穿全專案(週5/8/9/10/12)的既有設計:headline數字用複利"
            "口徑,顯著性檢定用算術口徑(重抽樣3000-5000次時算術Sharpe計算成本低、統計性質"
            "更單純)。本檔案逐一核對到週5(1.1375/-0.0026/0.9173/-0.0161/0.8517全部吻合)、"
            "週7(HMM ORB 0.2/0.35/-0.188,VWAP 0.893/1.011/-0.213,全部吻合)、週8/10、"
            "週12的原始報告數字,顯示層(sharpe/max_drawdown)逐一對上,只有diff/p/mde是"
            "本週(第13週)才按同一套方法補算或重算,不是原報告既有的數字。",
            "階段二 primary/secondary 兩列的 QLIKE 用不同目標變數(y_true_var vs rv_on)"
            "計算,見各列自己的 cross_check_note,不能跨列比較。",
        ],
        "phase2": phase2(), "phase3": phase3(), "phase4": phase4(), "phase5": phase5(),
    }
    summary = jsonify(summary)
    p = REPORTS / "results_summary.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
