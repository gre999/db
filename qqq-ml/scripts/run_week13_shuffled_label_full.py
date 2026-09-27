"""Week 13 (config/week13_wrapup.toml [stage5.shuffled_label_full]):
expand week 11/12's single-fold shuffled-label sanity check to all 8 task-A
folds x 5 shuffle seeds (40 runs total). Re-derives each fold's fit/val/
test split exactly as src.models.cnn.fit_predict_fold does (same
SEQ.standardize + days.isin masks) WITHOUT retraining the real 5-seed
ensemble - only the label-shuffled single-seed diagnostic model
(src.models.cnn.shuffled_label_check) is fit, 40 times.

Writes data/processed/sequences/week13_shuffled_label_full.parquet
(columns: fold, shuffle_seed, test_auc, passes).

Usage:
    python scripts/run_week13_shuffled_label_full.py
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
from src.models import cnn as CNN  # noqa: E402
from src.validation import WalkForwardSplit  # noqa: E402

log = logging.getLogger(__name__)
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tensor, days = SEQ.load_tensor("task_a")
    labels = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / "task_a_labels.parquet") \
        .set_index("day").reindex(days)
    y = labels["label"].astype(int)

    rows = []
    for f in WalkForwardSplit().folds(days):
        train_dates = days[f.train_idx]
        test_dates = days[f.test_idx]
        validation_start = f.test_start - pd.DateOffset(years=1)
        fit_dates = train_dates[train_dates < validation_start]
        val_dates = train_dates[train_dates >= validation_start]

        std_tensor, _, _ = SEQ.standardize(tensor, days, fit_dates)
        fit_mask, val_mask, test_mask = (days.isin(fit_dates), days.isin(val_dates),
                                         days.isin(test_dates))
        X_fit, y_fit = std_tensor[fit_mask], y.to_numpy()[fit_mask]
        X_val, y_val = std_tensor[val_mask], y.to_numpy()[val_mask]
        X_test, y_test = std_tensor[test_mask], y.to_numpy()[test_mask]

        for s in SHUFFLE_SEEDS:
            out = CNN.shuffled_label_check(X_fit, y_fit, X_val, y_val, X_test, y_test, seed=s)
            rows.append({"fold": f.number, "shuffle_seed": s,
                        "test_auc": out["test_auc"], "passes": out["passes"]})
            log.info("fold %d seed %d: test_auc=%.4f passes=%s",
                     f.number, s, out["test_auc"], out["passes"])

    df = pd.DataFrame(rows)
    p = dl.PROCESSED_DIR / "sequences" / "week13_shuffled_label_full.parquet"
    df.to_parquet(p, index=False)
    log.info("wrote %s: mean=%.4f sd=%.4f min=%.4f max=%.4f n_outside_band=%d/%d",
             p, df["test_auc"].mean(), df["test_auc"].std(ddof=1),
             df["test_auc"].min(), df["test_auc"].max(),
             (~df["passes"]).sum(), len(df))


if __name__ == "__main__":
    main()
