"""Week 8 (phase 4, week 1): the trade-filter's opening (09:35-cutoff)
feature matrix.

Every column is either a registered ``src/features.py`` feature (cutoff
<= "open_5m", automatic truncation-test coverage - see
``tests/test_features.py::test_truncation_features_unchanged`` and
``test_open_5m_features_unchanged_by_later_minute_data``) or an "extra"
whose own producer is responsible for its information timing.

MAIN analysis features (``OPEN_FEATURE_COLUMNS``) cover ORB's full sample
from ~2015 (20/22-day rolling warm-up) - includes the three raw HAR log-RV
windows (t-1, t-5-avg, t-22-avg) so the model can learn volatility
information itself from data that's available for the whole sample.

ROBUSTNESS-ONLY extras (``EXTRA_COLUMNS``): HAR+residual-XGB's t
volatility forecast (``src/models/har.py`` via the saved ``week4_models``
predictions) and the HMM's predictive state/probability
(``src/models/regimes.py`` via ``regime_labels_main_k3``) are both already
point-in-time by construction of their own walk-forward pipelines, known
as of session t-1's close (cutoff "prev_close" here) - but only exist from
2019-01-02 (their own walk-forward warm-up), years after ORB's 2014-12-22
start. Amended before looking at any filtering result (reason: this
cross-feature date-coverage gap, found while building this matrix - see
config/week08_filter.toml): the main analysis uses the full 2015+ sample
without these two; a 2019+-subsample robustness comparison (with vs
without them, both runs on the same subsample) uses them. Both are saved
in the same file so downstream code chooses per analysis rather than
rebuilding anything.

Stored separately from the daily (prev_close/open) matrices that
``src/features.py::build_and_save`` writes - a different, later cutoff, so
mixing the two by accident is structurally impossible either way, but a
distinct file makes it obvious:
``data/processed/features_open_5m.parquet``.

Usage::

    python -m src.features_open
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import data_loader as dl
from src import features as F
from src.models.base import load_predictions

log = logging.getLogger(__name__)

# Main analysis: pre-open (gap, prior-day return/vol at 3 HAR windows,
# VIX/VXN, dow) + first-5-min-bar (direction, body/range, relative range,
# relative volume, gap agreement) - full ORB sample from ~2015. See
# config/week08_filter.toml part 3 for the spec these implement.
OPEN_FEATURE_COLUMNS = (
    "overnight_gap", "gap_atr_ratio", "ret_cc_1d",
    "har_logrv_1d", "har_logrv_5d", "har_logrv_22d",
    "vix_1d", "vxn_1d", "day_of_week",
    "open5m_direction", "open5m_body_ratio", "open5m_range_rel",
    "open5m_rel_volume", "open5m_gap_agree",
)
# Robustness-only (2019+ subsample, with-vs-without comparison - not part
# of the main analysis's feature set; see module docstring).
EXTRA_COLUMNS = ("har_resid_xgb_pred_var", "hmm_state", "hmm_state_prob")


def load_extras(processed_dir: Path = dl.PROCESSED_DIR,
                pred: pd.DataFrame | None = None,
                hmm_labels: pd.DataFrame | None = None
                ) -> dict[str, tuple[pd.Series, str]]:
    """Stage-2 model outputs as ``build_feature_matrix`` extras: (series
    indexed by target day, cutoff). ``hmm_state`` is included alongside
    ``hmm_state_prob`` because the probability alone is ambiguous (0.99 of
    being in the calm state means something different from 0.99 of being
    in the turbulent one) - a small, explicit scope addition beyond the
    literal "HMM predictive state probability" spec.

    ``pred``/``hmm_labels`` let callers (tests) inject data instead of
    reading from disk - same pattern as
    ``src.regime_eval.vol_tercile_labels``'s ``pred`` parameter."""
    if pred is None:
        pred = load_predictions("week4_models")
    pred = pred[(pred["model"] == "har_resid_xgb") & (pred["target"] == "rv")]
    y_pred_var = pred.set_index("date")["y_pred_var"]

    if hmm_labels is None:
        hmm_labels = pd.read_parquet(Path(processed_dir) / "regimes" /
                                     "regime_labels_main_k3.parquet")
    hmm = hmm_labels[hmm_labels["model"] == "hmm"].set_index("date")
    return {
        "har_resid_xgb_pred_var": (y_pred_var, "prev_close"),
        "hmm_state": (hmm["state"].astype(float), "prev_close"),
        "hmm_state_prob": (hmm["prob"], "prev_close"),
    }


def build(data: F.MarketData | None = None, processed_dir: Path = dl.PROCESSED_DIR,
         cboe_dir: Path | None = dl.CBOE_DIR,
         pred: pd.DataFrame | None = None,
         hmm_labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """The open_5m feature matrix, rows restricted to sessions that have
    minute data (same convention as ``src/features.py::build_and_save``)."""
    if data is None:
        data = F.MarketData.from_processed(processed_dir, cboe_dir)
    extras = load_extras(processed_dir, pred, hmm_labels)
    X = F.build_feature_matrix(data, "open_5m", names=list(OPEN_FEATURE_COLUMNS),
                               extras=extras)
    days = pd.DatetimeIndex(data.bars5["day"].unique())
    return X.loc[X.index.isin(days)]


def build_and_save(processed_dir: Path = dl.PROCESSED_DIR,
                   cboe_dir: Path | None = dl.CBOE_DIR) -> Path:
    X = build(processed_dir=processed_dir, cboe_dir=cboe_dir)
    p = Path(processed_dir) / "features_open_5m.parquet"
    X.to_parquet(p)
    return p


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = build_and_save()
    log.info("wrote %s", p)


if __name__ == "__main__":
    main()
