"""Week 12 (descriptive only): task A's CNN/stacking scores routed
through week 8's threshold-selection machinery (src.models.filter.
select_threshold_posthoc - never refits anything, just picks a keep/drop
cutoff from each fold's validation-window scores), then evaluated with
the exact same src.filter_eval functions every phase-4 week used. Task
B: enter at 09:30+N in the predicted direction, flatten at the close.

Usage:
    python scripts/run_week12_strategy_layer.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402
from src import filter_eval as FE  # noqa: E402
from src import regime_eval as E  # noqa: E402
from src import sequences as SEQ  # noqa: E402
from src.models import filter as FL  # noqa: E402
from src.models.base import PRED_DIR, load_predictions  # noqa: E402
from src.validation import WalkForwardSplit  # noqa: E402

log = logging.getLogger(__name__)


def build_task_a_filter_preds(model_name: str, test_preds: pd.DataFrame,
                              val_preds: pd.DataFrame, orb: pd.DataFrame) -> pd.DataFrame:
    s = orb.set_index("day") if "day" in orb.columns else orb
    rows = []
    for f_num, g_test in test_preds.groupby("fold"):
        g_val = val_preds[val_preds["fold"] == f_num]
        if g_val.empty:
            continue
        val_score = g_val.set_index("date")["y_pred_proba"]
        bps_val_full = s["bps_return"].reindex(val_score.index).fillna(0.0)
        best = FL.select_threshold_posthoc(val_score, val_score.index, bps_val_full)
        for _, row in g_test.iterrows():
            p = row["y_pred_proba"]
            rows.append({"date": row["date"], "fold": f_num, "model": model_name,
                        "target": "task_a", "y_true": row["y_true"],
                        "y_pred_proba": p, "threshold": best["threshold"],
                        "retention_selected": best["retention"],
                        "keep": p >= best["threshold"]})
    return pd.DataFrame(rows)


def task_b_directional_strategy(preds: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Enter at 09:30+N in the predicted direction (prob>=0.5 -> long,
    else short), flatten at the close. Net-of-cost Sharpe/max drawdown,
    full-calendar convention (every non-covered day contributes 0)."""
    j = preds.set_index("date").join(labels[["raw_bps", "cost_bps"]], how="inner")
    pred_dir = np.where(j["y_pred_proba"] >= 0.5, 1, -1)
    strategy_bps = pred_dir * j["raw_bps"] - j["cost_bps"]
    series = pd.Series(strategy_bps.to_numpy(), index=j.index).sort_index()
    ev = E._daily_eval(series)
    win_rate = float((strategy_bps > 0).mean())
    return {"n_days": len(series), "sharpe": ev["sharpe"], "max_drawdown": ev["max_drawdown"],
           "ann_return": ev["ann_return"], "win_rate": win_rate}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")

    # ---------------------------------------------------------------
    # Task A: CNN, stacking
    # ---------------------------------------------------------------
    cnn_test = load_predictions("week12_task_a_cnn")
    cnn_val = load_predictions("week12_task_a_cnn_val")
    stack_test = load_predictions("week12_task_a_stacking")
    # stacking has no saved val_preds - rebuild a val-scored version isn't
    # needed for filtering (threshold search uses the SAME validation
    # window the stacking model was ITSELF fit on; approximate by scoring
    # the stacking model's own fitted probability on its own validation
    # rows - already implicit in ST.fit_stacking_fold, so reuse cnn_val's
    # dates/fold structure with the hand+cnn val scores is out of scope
    # here; instead this descriptive layer for stacking selects its
    # threshold from the STACKED VALIDATION SCORE, i.e. the same value
    # the Wald test was fit on - already available via a quick recompute.
    from src.models import stacking as ST
    from src.models.filter import OPEN_FEATURE_COLUMNS, make_logistic

    tensor, days = SEQ.load_tensor("task_a")
    Xopen = pd.read_parquet(dl.PROCESSED_DIR / "features_open_5m.parquet")[list(OPEN_FEATURE_COLUMNS)]
    Xopen = Xopen.reindex(days).dropna()
    labels_a = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / "task_a_labels.parquet") \
        .set_index("day").reindex(Xopen.index)
    y = labels_a["label"].astype(int)
    cnn_val_idx = cnn_val.set_index("date")

    stack_val_rows = []
    for f in WalkForwardSplit().folds(Xopen.index):
        train_dates = Xopen.index[f.train_idx]
        validation_start = f.test_start - pd.DateOffset(years=1)
        fit_dates = train_dates[train_dates < validation_start]
        val_dates = train_dates[train_dates >= validation_start]
        if len(fit_dates) < 30 or len(val_dates) < 30:
            continue
        hand_model = make_logistic(y.loc[fit_dates])
        hand_model.fit(Xopen.loc[fit_dates], y.loc[fit_dates])
        hand_val_proba = pd.Series(hand_model.predict_proba(Xopen.loc[val_dates])[:, 1], index=val_dates)
        cnn_val_f = cnn_val_idx[cnn_val_idx["fold"] == f.number]
        common = hand_val_proba.index.intersection(cnn_val_f.index)
        if len(common) < 30:
            continue
        X_val = np.column_stack([ST.logit(hand_val_proba.loc[common].to_numpy()),
                                 ST.logit(cnn_val_f.loc[common, "y_pred_proba"].to_numpy())])
        wald = ST.wald_test(X_val, y.loc[common].to_numpy())
        stack_val_proba = wald["model"].predict_proba(X_val)[:, 1]
        for d, p in zip(common, stack_val_proba):
            stack_val_rows.append({"date": d, "fold": f.number, "y_pred_proba": float(p)})
    stack_val = pd.DataFrame(stack_val_rows)

    filter_preds = {
        "cnn": build_task_a_filter_preds("cnn", cnn_test, cnn_val, orb),
        "stacking": build_task_a_filter_preds("stacking", stack_test, stack_val, orb),
    }

    strategy_rows = []
    for name, preds in filter_preds.items():
        if preds.empty:
            continue
        p = PRED_DIR / f"week12_task_a_{name}_filter.parquet"
        preds.to_parquet(p, index=False)
        result = FE.evaluate_success(preds, orb, min_trades_per_year=20)
        bb = FE.block_bootstrap_significance(preds, orb, n_boot=2000, seed=0)
        rp = FE.random_filter_p_value(preds, orb,
                                      null=FE.random_filter_null(preds, orb, n_reps=10000, seed=0))
        strategy_rows.append({"model": name, "filtered_sharpe": result["filtered_sharpe"],
                             "unfiltered_sharpe": result["unfiltered_sharpe"],
                             "filtered_max_drawdown": result["filtered_max_drawdown"],
                             "bb_p": bb["p_value"], "rand_p": rp["p_value"],
                             "rand_se": rp["se"]})
        log.info("task A %s: filtered_sharpe=%.3f unfiltered=%.3f bb_p=%.3f rand_p=%.4f",
                 name, result["filtered_sharpe"], result["unfiltered_sharpe"],
                 bb["p_value"], rp["p_value"])

    strat_tbl = pd.DataFrame(strategy_rows)
    p_strat = dl.PROCESSED_DIR / "sequences" / "week12_task_a_strategy_layer.parquet"
    strat_tbl.to_parquet(p_strat, index=False)
    log.info("wrote %s", p_strat)

    # ---------------------------------------------------------------
    # Task B: directional strategy
    # ---------------------------------------------------------------
    taskb_rows = []
    for task, n in [("task_b15", 15), ("task_b30", 30)]:
        cnn_preds = load_predictions(f"week12_{task}_cnn")
        labels_b = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / f"{task}_labels.parquet") \
            .set_index("day")
        out = task_b_directional_strategy(cnn_preds, labels_b)
        out["task"] = task
        taskb_rows.append(out)
        log.info("%s directional strategy: sharpe=%.3f max_dd=%.3f win_rate=%.3f",
                 task, out["sharpe"], out["max_drawdown"], out["win_rate"])
    taskb_tbl = pd.DataFrame(taskb_rows)
    p_b = dl.PROCESSED_DIR / "sequences" / "week12_task_b_strategy_layer.parquet"
    taskb_tbl.to_parquet(p_b, index=False)
    log.info("wrote %s", p_b)


if __name__ == "__main__":
    main()
