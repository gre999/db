"""Week 12: task A incremental-information check (config/week12_dl.toml
[incremental_info]) - Spearman correlations, R² regressions, and the
validation-window-only stacking test.

Reads data/processed/predictions/week12_task_a_cnn{,_val}.parquet (run
scripts/run_week12_cnn_walkforward.py first), data/processed/
features_open_5m.parquet, and data/processed/sequences/task_a_labels
.parquet. Refits week 8's hand-feature logistic per fold (to get its
validation-window scores, which week 11's baseline run never saved -
only test-fold predictions) - never refits or touches the CNN itself.

Writes:
- data/processed/predictions/week12_task_a_stacking.parquet (test-fold)
- data/processed/sequences/week12_task_a_stacking_wald.parquet (per-fold
  Wald test on the stacking model's coefficients)
- data/processed/sequences/week12_task_a_spearman.parquet
- data/processed/sequences/week12_task_a_r2_per_fold.parquet
  (+ prints the pooled-standardized R²)

Usage:
    python scripts/run_week12_stacking.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402
from src import sequences as SEQ  # noqa: E402
from src.models import stacking as ST  # noqa: E402
from src.models.base import PRED_DIR, load_predictions  # noqa: E402
from src.models.filter import OPEN_FEATURE_COLUMNS, make_logistic  # noqa: E402
from src.validation import WalkForwardSplit  # noqa: E402

log = logging.getLogger(__name__)
SPEARMAN_FEATURES = ["open5m_body_ratio", "open5m_direction", "open5m_range_rel",
                     "har_logrv_1d"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tensor, days = SEQ.load_tensor("task_a")
    Xopen = pd.read_parquet(dl.PROCESSED_DIR / "features_open_5m.parquet")[list(OPEN_FEATURE_COLUMNS)]
    Xopen = Xopen.reindex(days).dropna()
    labels = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / "task_a_labels.parquet") \
        .set_index("day").reindex(Xopen.index)
    y = labels["label"].astype(int)

    cnn_test = load_predictions("week12_task_a_cnn").set_index("date")
    cnn_val = load_predictions("week12_task_a_cnn_val").set_index("date")

    stack_rows, wald_rows = [], []
    for f in WalkForwardSplit().folds(Xopen.index):
        train_dates = Xopen.index[f.train_idx]
        test_dates = Xopen.index[f.test_idx]
        validation_start = f.test_start - pd.DateOffset(years=1)
        fit_dates = train_dates[train_dates < validation_start]
        val_dates = train_dates[train_dates >= validation_start]
        if len(fit_dates) < 30 or len(val_dates) < 30:
            continue

        hand_model = make_logistic(y.loc[fit_dates])
        hand_model.fit(Xopen.loc[fit_dates], y.loc[fit_dates])
        hand_val_proba = pd.Series(hand_model.predict_proba(Xopen.loc[val_dates])[:, 1],
                                   index=val_dates)
        hand_test_proba = pd.Series(hand_model.predict_proba(Xopen.loc[test_dates])[:, 1],
                                    index=test_dates)

        cnn_val_f = cnn_val[cnn_val["fold"] == f.number]
        cnn_test_f = cnn_test[cnn_test["fold"] == f.number]

        val_common = hand_val_proba.index.intersection(cnn_val_f.index)
        test_common = hand_test_proba.index.intersection(cnn_test_f.index)
        if len(val_common) < 30 or len(test_common) == 0:
            continue

        out = ST.fit_stacking_fold(
            hand_val_proba.loc[val_common].to_numpy(),
            cnn_val_f.loc[val_common, "y_pred_proba"].to_numpy(),
            y.loc[val_common].to_numpy(),
            hand_test_proba.loc[test_common].to_numpy(),
            cnn_test_f.loc[test_common, "y_pred_proba"].to_numpy(),
        )
        for d, p in zip(test_common, out["test_proba"]):
            stack_rows.append({"date": d, "fold": f.number, "model": "stacking",
                              "target": "task_a", "y_true": int(y.loc[d]),
                              "y_pred_proba": float(p)})
        wald_rows.append({"fold": f.number, "n_val": len(val_common),
                         "intercept": out["intercept"],
                         "hand_coef": out["hand_coef"], "hand_se": out["hand_se"],
                         "hand_p": out["hand_p"],
                         "cnn_coef": out["cnn_coef"], "cnn_se": out["cnn_se"],
                         "cnn_p": out["cnn_p"]})
        log.info("fold %d: n_val=%d hand_coef=%.3f(p=%.3f) cnn_coef=%.3f(p=%.3f)",
                 f.number, len(val_common), out["hand_coef"], out["hand_p"],
                 out["cnn_coef"], out["cnn_p"])

    stack_preds = pd.DataFrame(stack_rows)
    wald_tbl = pd.DataFrame(wald_rows)
    p1 = PRED_DIR / "week12_task_a_stacking.parquet"
    p2 = dl.PROCESSED_DIR / "sequences" / "week12_task_a_stacking_wald.parquet"
    stack_preds.to_parquet(p1, index=False)
    wald_tbl.to_parquet(p2, index=False)
    log.info("wrote %s, %s", p1, p2)

    # ---------------------------------------------------------------
    # Spearman + R^2 (both use the CNN's TEST-fold predictions)
    # ---------------------------------------------------------------
    cnn_test_df = cnn_test.reset_index()
    spearman = ST.spearman_by_fold(cnn_test_df, Xopen, SPEARMAN_FEATURES)
    p3 = dl.PROCESSED_DIR / "sequences" / "week12_task_a_spearman.parquet"
    spearman.to_parquet(p3, index=False)
    log.info("wrote %s", p3)

    r2_per_fold = ST.r_squared_per_fold(cnn_test_df, Xopen, list(OPEN_FEATURE_COLUMNS))
    p4 = dl.PROCESSED_DIR / "sequences" / "week12_task_a_r2_per_fold.parquet"
    r2_per_fold.to_parquet(p4, index=False)
    log.info("wrote %s", p4)

    r2_pooled = ST.r_squared_pooled_standardized(cnn_test_df, Xopen, list(OPEN_FEATURE_COLUMNS))
    log.info("pooled-standardized R^2: %.4f (adj %.4f, n=%d, k=%d)",
             r2_pooled["r2"], r2_pooled["adj_r2"], r2_pooled["n"], r2_pooled["k"])


if __name__ == "__main__":
    main()
