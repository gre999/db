"""Week 12: full 9-fold CNN walk-forward for tasks A/B15/B30 (frozen
week-11 architecture/training - config/week12_dl.toml [inherited_from_
week11]).

Writes, per task:
- data/processed/predictions/week12_{task}_cnn.parquet (test-fold preds)
- data/processed/predictions/week12_{task}_cnn_val.parquet (validation-
  window preds - needed for week 12's stacking step, never used to pick a
  threshold or refit anything)
- data/processed/sequences/week12_{task}_fold_diagnostics.parquet (per-
  fold fit/val/test AUC, per-seed test AUC range, per-seed early-stopping
  epoch)

Usage:
    python scripts/run_week12_cnn_walkforward.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402
from src import sequences as SEQ  # noqa: E402
from src.models import cnn as CNN  # noqa: E402
from src.models.base import PRED_DIR  # noqa: E402
from src.validation import WalkForwardSplit  # noqa: E402

log = logging.getLogger(__name__)
TASKS = ("task_a", "task_b15", "task_b30")


def run_task(task: str) -> None:
    tensor, days = SEQ.load_tensor(task)
    labels = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / f"{task}_labels.parquet") \
        .set_index("day").reindex(days)
    y = labels["label"].astype(int)

    test_rows, val_rows, diag_rows = [], [], []
    folds = WalkForwardSplit().folds(days)
    for f in folds:
        t0 = time.time()
        out = CNN.fit_predict_fold(tensor, days, y, f, validation_years=1)
        test_rows.append(out["preds"])
        val_rows.append(out["val_preds"])
        diag_rows.append({
            "fold": f.number, "n_fit": len(out["X_fit"]), "n_val": len(out["X_val"]),
            "n_test": len(out["X_test"]),
            "fit_auc": out["fit_auc_mean_proba"], "val_auc": out["val_auc_mean_proba"],
            "test_auc": out["test_auc_mean_proba"],
            "test_auc_seed_min": float(np.min(out["test_auc_per_seed"])),
            "test_auc_seed_max": float(np.max(out["test_auc_per_seed"])),
            "epochs_per_seed": list(out["epochs_per_seed"]),
        })
        log.info("%s fold %d done in %.1fs: fit_auc=%.4f val_auc=%.4f test_auc=%.4f",
                 task, f.number, time.time() - t0, out["fit_auc_mean_proba"],
                 out["val_auc_mean_proba"], out["test_auc_mean_proba"])

    test_preds = pd.concat(test_rows, ignore_index=True)
    val_preds = pd.concat(val_rows, ignore_index=True)
    diag = pd.DataFrame(diag_rows)

    p1 = PRED_DIR / f"week12_{task}_cnn.parquet"
    p2 = PRED_DIR / f"week12_{task}_cnn_val.parquet"
    p3 = dl.PROCESSED_DIR / "sequences" / f"week12_{task}_fold_diagnostics.parquet"
    test_preds.to_parquet(p1, index=False)
    val_preds.to_parquet(p2, index=False)
    diag.to_parquet(p3, index=False)
    log.info("wrote %s, %s, %s", p1, p2, p3)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()
