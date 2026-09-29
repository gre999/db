"""Week 14 (config/week14_final.toml [gapfill.*]): fill the five robustness-
matrix gaps identified in Part 1. Every check here reuses an already-
frozen model/rule (week 5's position-sizing weight function, week 7's
HMM state labels, week 8's logistic filter's keep decisions, week 12's
CNN filter's keep decisions) - nothing is refit, no new research question,
no changed conclusion. Results are printed and written to data/processed/
sequences/week14_*.parquet|json for scripts/collect_results.py to fold
into reports/results_summary.json.

Usage:
    python scripts/run_week14_robustness.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import backtest as BT  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import filter_eval as FE  # noqa: E402
from src import regime_eval as E  # noqa: E402
from src import rules as R  # noqa: E402
from src import strategies as S  # noqa: E402
from src.models import regimes as G  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

log = logging.getLogger(__name__)
SEQ_DIR = dl.PROCESSED_DIR / "sequences"


# =========================================================================
# Gap 1: phase 2 random baseline - shuffle the model's predicted variance
# across dates (keeping fold/ratio calibration attached to each date as-is),
# run the SAME vol-target position rule, repeat n_reps times.
# =========================================================================
def phase2_random_baseline(n_reps: int = 1000, seed: int = 0) -> dict:
    inputs = BT.load_inputs()
    cfg = BT.BacktestConfig()
    idx = inputs["common_index"]
    p = inputs["preds"]["har_resid_xgb"].reindex(idx)

    w_real = BT.build_weight("har_resid_xgb", inputs, cfg)
    bt_real = BT.run_backtest(w_real, inputs["daily"], timing=cfg.timing,
                              cost_bps=cfg.cost_bps, band=cfg.band)
    observed = BT.arithmetic_sharpe(bt_real["net_return"])

    rng = np.random.default_rng(seed)
    y_pred_var = p["y_pred_var"].to_numpy()
    null = np.empty(n_reps)
    for i in range(n_reps):
        shuffled = p.copy()
        shuffled["y_pred_var"] = rng.permutation(y_pred_var)
        var_cc = S.cc_variance_forecast(shuffled, inputs["ratios"])
        w = S.vol_target_weight(var_cc, cfg.vol_target, cfg.leverage_cap)
        bt = BT.run_backtest(w, inputs["daily"], timing=cfg.timing,
                             cost_bps=cfg.cost_bps, band=cfg.band)
        null[i] = BT.arithmetic_sharpe(bt["net_return"])

    n_at_or_above = int(np.sum(null >= observed))
    p_value = n_at_or_above / n_reps
    se = float(np.sqrt(p_value * (1 - p_value) / n_reps))
    return {"observed_sharpe": observed, "null_mean": float(null.mean()),
           "null_p95": float(np.percentile(null, 95)), "p_value": p_value, "se": se,
           "n_reps": n_reps,
           "design_note": "打亂har_resid_xgb的y_pred_var在日期間的對應(fold/校準比例不動),"
                          "接回同一套vol_target部位規則,p=null>=observed的比例(單尾,跟week8"
                          "random_filter_p_value同一套邏輯)"}


# =========================================================================
# Gap 2: phase 3 cost doubling - double ORB/VWAP's cost_per_share, rerun
# run_orb/run_vwap in memory (never overwrites the saved *_daily.parquet),
# re-evaluate tradability_test with the ALREADY-FROZEN HMM state labels.
# =========================================================================
def phase3_cost_doubling() -> dict:
    X = G.load_state_features()
    orb_2x = R.run_orb(cfg=replace(R.ORBConfig(), cost_per_share=2 * R.ORBConfig().cost_per_share))
    vwap_2x = R.run_vwap(cfg=replace(R.VWAPConfig(), cost_per_share=2 * R.VWAPConfig().cost_per_share))

    rows = []
    for strat_name, strat_2x in (("orb", orb_2x), ("vwap", vwap_2x)):
        out = E.tradability_test(X, strat_2x, "hmm", n_init=10)
        f, u = out["filtered"] / 1e4, out["unfiltered"] / 1e4
        obs, p = BT.block_bootstrap_sharpe_diff(f, u, block_size=20, n_boot=3000, seed=0)
        rows.append({"strategy": strat_name, "filtered_sharpe": BT.arithmetic_sharpe(f),
                    "unfiltered_sharpe": BT.arithmetic_sharpe(u), "diff": obs, "p_value": p,
                    "cost_per_share": (2 * R.ORBConfig().cost_per_share if strat_name == "orb"
                                     else 2 * R.VWAPConfig().cost_per_share)})
    return {"rows": rows,
           "design_note": "ORBConfig/VWAPConfig的cost_per_share加倍,run_orb/run_vwap"
                          "只在記憶體內重算(不覆蓋orb_daily.parquet/vwap_daily.parquet)。"
                          "tradability_test內部會重新呼叫fit_hmm_fold_main,但該函式"
                          "random_state=0寫死、確定性配適,重跑會重現跟regime_labels_main_k3"
                          ".parquet完全相同的狀態指派(不是產生新的模型決定),只有策略"
                          "自己的bps_return因成本改變而不同——跟主成本下的tradability_test"
                          "結果比對方向"}


# =========================================================================
# Gap 3: phase 3 yearly activation - state_performance_by_year exists and
# is tested but was never wired into any report. No new computation design,
# just calling it and saving the output.
# =========================================================================
def phase3_yearly_activation() -> dict:
    pred = pd.read_parquet(E.REGIME_DIR / "regime_labels_main_k3.parquet")
    orb = pd.read_parquet(E.STRATEGY_DIR / "orb_daily.parquet")
    vwap = pd.read_parquet(E.STRATEGY_DIR / "vwap_daily.parquet")
    rows = []
    for strat_name, strat in (("orb", orb), ("vwap", vwap)):
        for model in ("hmm", "kmeans"):
            rows.append(E.state_performance_by_year(pred, strat, model, strat_name))
    tbl = pd.concat(rows, ignore_index=True)
    return tbl


# =========================================================================
# Gap 4: phase 4 ORB target_r sensitivity - rerun run_orb at target_r=5/15
# (10 is the existing default, not rerun), apply week 8's ALREADY-FROZEN
# keep decisions (fit on target_r=10 labels) unchanged.
# =========================================================================
def phase4_target_r_sensitivity() -> dict:
    preds = load_predictions("week08_filter")  # frozen keep decisions, fit under target_r=10
    rows = []
    for target_r in (5.0, 15.0):
        orb_r = R.run_orb(cfg=replace(R.ORBConfig(), target_r=target_r))
        result = FE.evaluate_success(preds, orb_r, min_trades_per_year=20)
        bb = FE.block_bootstrap_significance(preds, orb_r, n_boot=2000, seed=0)
        series = FE.logistic_filter_series(preds, orb_r)
        filt, unfilt = series["filtered"] / 1e4, series["unfiltered"] / 1e4
        rows.append({"target_r": target_r,
                    "filtered_sharpe": BT.arithmetic_sharpe(filt),
                    "unfiltered_sharpe": BT.arithmetic_sharpe(unfilt),
                    "filtered_sharpe_compounding": result["filtered_sharpe"],
                    "unfiltered_sharpe_compounding": result["unfiltered_sharpe"],
                    "diff": BT.arithmetic_sharpe(filt) - BT.arithmetic_sharpe(unfilt),
                    "p_value": bb["p_value"]})
    return {"rows": rows,
           "design_note": "target_r=5/15在記憶體內重算run_orb(10是既有預設,不重算),"
                          "套用week08_filter.parquet裡以target_r=10標籤訓練出的keep決定"
                          "(不重新訓練篩選模型、不重新選門檻),檢驗同一批被選中的日子換算"
                          "到不同停利定義下,篩選vs不篩選的方向是否一致"}


# =========================================================================
# Gap 5: phase 5 yearly + cost doubling - CNN(task A) filter's yearly
# Sharpe breakdown, and cost-doubled ORB applied to the SAME frozen CNN
# keep decisions (CNN's score doesn't depend on cost; only ORB's own
# bps_return does).
# =========================================================================
def phase5_yearly_and_cost() -> dict:
    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")
    cnn_preds = load_predictions("week12_task_a_cnn_filter")

    series = FE.logistic_filter_series(cnn_preds, orb)
    filt = series["filtered"]
    yearly_rows = []
    for yr, g in filt.groupby(filt.index.year):
        u = series["unfiltered"].reindex(g.index)
        yearly_rows.append({"year": int(yr), "n_days": len(g),
                           "filtered_sharpe": BT.arithmetic_sharpe(g / 1e4),
                           "unfiltered_sharpe": BT.arithmetic_sharpe(u / 1e4)})
    yearly = pd.DataFrame(yearly_rows)
    n_better = int((yearly["filtered_sharpe"] > yearly["unfiltered_sharpe"]).sum())

    orb_2x = R.run_orb(cfg=replace(R.ORBConfig(), cost_per_share=2 * R.ORBConfig().cost_per_share))
    series_2x = FE.logistic_filter_series(cnn_preds, orb_2x)
    f2, u2 = series_2x["filtered"] / 1e4, series_2x["unfiltered"] / 1e4
    obs2x, p2x = BT.block_bootstrap_sharpe_diff(f2, u2, block_size=20, n_boot=2000, seed=0)

    return {"yearly": yearly.to_dict(orient="records"),
           "n_years_filtered_better": n_better, "n_years_total": len(yearly),
           "cost_2x": {"filtered_sharpe": BT.arithmetic_sharpe(f2),
                      "unfiltered_sharpe": BT.arithmetic_sharpe(u2),
                      "diff": obs2x, "p_value": p2x},
           "design_note": "CNN(task A)篩選keep決定不變(不重新訓練CNN),逐年用同一組keep"
                          "天數拆解Sharpe;成本加倍版本只把ORBConfig的cost_per_share加倍"
                          "重跑run_orb,套用同一組CNN keep決定,因為CNN分數本身不受交易"
                          "成本影響"}


def jsonify(obj):
    """NaN -> None so the output is standard-compliant JSON, not just
    Python-json-module-readable (see scripts/collect_results.py's copy of
    this same fix for the reason)."""
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
    return obj


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Gap 1: phase2 random baseline (n_reps=1000)...")
    g1 = phase2_random_baseline(n_reps=1000, seed=0)
    log.info("  observed=%.4f null_mean=%.4f null_p95=%.4f p=%.4f (se=%.4f)",
             g1["observed_sharpe"], g1["null_mean"], g1["null_p95"], g1["p_value"], g1["se"])

    log.info("Gap 2: phase3 cost doubling...")
    g2 = phase3_cost_doubling()
    for r in g2["rows"]:
        log.info("  %s: filtered=%.4f unfiltered=%.4f diff=%.4f p=%.4f",
                 r["strategy"], r["filtered_sharpe"], r["unfiltered_sharpe"], r["diff"], r["p_value"])

    log.info("Gap 3: phase3 yearly activation...")
    g3 = phase3_yearly_activation()
    log.info("  %d rows", len(g3))

    log.info("Gap 4: phase4 target_r sensitivity...")
    g4 = phase4_target_r_sensitivity()
    for r in g4["rows"]:
        log.info("  target_r=%.0f: filtered=%.4f unfiltered=%.4f diff=%.4f p=%.4f",
                 r["target_r"], r["filtered_sharpe"], r["unfiltered_sharpe"], r["diff"], r["p_value"])

    log.info("Gap 5: phase5 yearly + cost doubling...")
    g5 = phase5_yearly_and_cost()
    log.info("  %d/%d years filtered>unfiltered; cost_2x diff=%.4f p=%.4f",
             g5["n_years_filtered_better"], g5["n_years_total"],
             g5["cost_2x"]["diff"], g5["cost_2x"]["p_value"])

    out = {"phase2_random_baseline": g1, "phase3_cost_doubling": g2,
          "phase4_target_r_sensitivity": g4, "phase5_yearly_and_cost": g5}
    p_json = SEQ_DIR / "week14_robustness_gaps.json"
    p_json.write_text(json.dumps(jsonify(out), indent=2, ensure_ascii=False), encoding="utf-8")
    p_parquet = SEQ_DIR / "week14_phase3_yearly.parquet"
    g3.to_parquet(p_parquet, index=False)
    log.info("wrote %s, %s", p_json, p_parquet)


if __name__ == "__main__":
    main()
