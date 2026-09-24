"""Week 11: fit and save every baseline for tasks A/B15/B30.

Reads data/processed/sequences/ (run ``python -m src.sequences`` first)
and data/processed/features_open_5m.parquet (task A's hand-feature
baseline). Writes data/processed/predictions/week11_{task}_{model}
.parquet.

Usage:
    python scripts/run_week11_baselines.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402
from src import sequences as SEQ  # noqa: E402
from src.models import dl_baselines as DB  # noqa: E402
from src.models.base import PRED_DIR  # noqa: E402
from src.models.filter import OPEN_FEATURE_COLUMNS, make_logistic, make_xgboost  # noqa: E402

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

TASKS = ("task_a", "task_b15", "task_b30")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        tensor, days = SEQ.load_tensor(task)
        labels = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / f"{task}_labels.parquet") \
            .set_index("day").reindex(days)
        y = labels["label"].astype(int)
        X = pd.DataFrame(DB.flatten(tensor), index=days)

        for model_name, factory in [("logistic_l2", DB.make_logistic_l2),
                                    ("xgboost", make_xgboost)]:
            preds = DB.fit_predict_folds(X, y, factory, model_name, task)
            p = PRED_DIR / f"week11_{task}_{model_name}.parquet"
            preds.to_parquet(p, index=False)
            log.info("wrote %s (%d rows)", p, len(preds))

    # task A's extra hand-feature baseline (week 8's OPEN_FEATURE_COLUMNS)
    tensor, days = SEQ.load_tensor("task_a")
    Xopen = pd.read_parquet(dl.PROCESSED_DIR / "features_open_5m.parquet")[list(OPEN_FEATURE_COLUMNS)]
    Xopen = Xopen.reindex(days).dropna()
    labels_a = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / "task_a_labels.parquet") \
        .set_index("day").reindex(Xopen.index)
    y = labels_a["label"].astype(int)
    preds = DB.fit_predict_folds(Xopen, y, make_logistic, "hand_feature_logistic", "task_a")
    p = PRED_DIR / "week11_task_a_hand_feature_logistic.parquet"
    preds.to_parquet(p, index=False)
    log.info("wrote %s (%d rows)", p, len(preds))


if __name__ == "__main__":
    main()
