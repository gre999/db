"""Week 14 (reproducibility spot-check): retrain ONE fold of task A's CNN
(fixed seeds 0-4, deterministic mode - the same code path as scripts/
run_week12_cnn_walkforward.py) and confirm the resulting test-fold
predictions match the already-saved data/processed/predictions/
week12_task_a_cnn.parquet exactly for that fold. This is a targeted
check of "does training reproduce byte-for-byte given the same seeds and
inputs," not a full CNN retrain - run in a clean venv (pinned
requirements, same torch build) per README.md's "可重現性" section.

Usage:
    python scripts/verify_cnn_determinism.py [--fold N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402
from src import sequences as SEQ  # noqa: E402
from src.models import cnn as CNN  # noqa: E402
from src.models.base import load_predictions  # noqa: E402
from src.validation import WalkForwardSplit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    args = ap.parse_args()

    tensor, days = SEQ.load_tensor("task_a")
    labels = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / "task_a_labels.parquet") \
        .set_index("day").reindex(days)
    y = labels["label"].astype(int)

    folds = list(WalkForwardSplit().folds(days))
    f = next(fold for fold in folds if fold.number == args.fold)

    print(f"Retraining fold {args.fold} (5 seeds, deterministic mode)...")
    out = CNN.fit_predict_fold(tensor, days, y, f, validation_years=1)
    fresh = out["preds"].set_index("date")["y_pred_proba"]

    saved = load_predictions("week12_task_a_cnn").set_index("date")
    saved_fold = saved[saved["fold"] == args.fold]["y_pred_proba"]

    common = fresh.index.intersection(saved_fold.index)
    if len(common) != len(fresh) or len(common) != len(saved_fold):
        raise SystemExit(f"FAILED: date mismatch - fresh={len(fresh)} saved={len(saved_fold)} "
                         f"common={len(common)}")

    diff = (fresh.loc[common] - saved_fold.loc[common]).abs()
    max_diff = float(diff.max())
    print(f"n_days={len(common)} max_abs_diff={max_diff:.2e} "
         f"(0.0 = byte-for-byte identical)")

    if not np.allclose(fresh.loc[common].to_numpy(), saved_fold.loc[common].to_numpy(),
                       rtol=0, atol=1e-12):
        raise SystemExit(f"FAILED: fold {args.fold} predictions do not match the saved file "
                         f"(max diff {max_diff:.2e})")
    print(f"PASSED: fold {args.fold}'s freshly-retrained predictions exactly match "
         f"data/processed/predictions/week12_task_a_cnn.parquet")


if __name__ == "__main__":
    main()
