"""Tests for src/features_open.py: the open_5m (09:35-cutoff) trade-filter
feature matrix - main-analysis columns vs the 2019+-only robustness extras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src import features_open as FO
from tests import synth


@pytest.fixture(scope="module")
def data() -> F.MarketData:
    return synth.make_market_data()


def _fake_pred(data: F.MarketData, start: str) -> pd.DataFrame:
    days = data.calendar[data.calendar >= pd.Timestamp(start)]
    rng = np.random.default_rng(3)
    return pd.DataFrame({"date": days, "model": "har_resid_xgb", "target": "rv",
                        "y_pred_var": rng.uniform(1e-5, 2e-4, len(days))})


def _fake_hmm_labels(data: F.MarketData, start: str) -> pd.DataFrame:
    days = data.calendar[data.calendar >= pd.Timestamp(start)]
    rng = np.random.default_rng(4)
    return pd.DataFrame({"date": days, "model": "hmm", "k": 3,
                        "state": rng.integers(0, 3, len(days)),
                        "prob": rng.uniform(0.4, 1.0, len(days))})


def test_build_output_has_every_declared_column(data):
    pred = _fake_pred(data, "2018-01-01")
    hmm = _fake_hmm_labels(data, "2018-01-01")
    X = FO.build(data, pred=pred, hmm_labels=hmm)
    assert set(FO.OPEN_FEATURE_COLUMNS) <= set(X.columns)
    assert set(FO.EXTRA_COLUMNS) <= set(X.columns)
    days = pd.DatetimeIndex(data.bars5["day"].unique())
    assert X.index.isin(days).all()


def test_main_columns_available_well_before_extras_start(data):
    """The whole point of the week 8 amendment: main-analysis columns must
    not depend on the 2019+-only stage-2-model extras' start date."""
    extras_start = "2019-06-01"        # later than the synth data's early years
    pred = _fake_pred(data, extras_start)
    hmm = _fake_hmm_labels(data, extras_start)
    X = FO.build(data, pred=pred, hmm_labels=hmm)

    early_day = pd.Timestamp("2016-06-01")
    early_day = X.index[X.index.get_indexer([early_day], method="nearest")[0]]
    main_row = X.loc[early_day, list(FO.OPEN_FEATURE_COLUMNS)]
    assert main_row.notna().all(), main_row[main_row.isna()]
    extra_row = X.loc[early_day, list(FO.EXTRA_COLUMNS)]
    assert extra_row.isna().all()

    late_day = X.index[X.index >= pd.Timestamp(extras_start)][5]
    assert X.loc[late_day, list(FO.EXTRA_COLUMNS)].notna().all()


def test_extras_are_prev_close_cutoff_not_leaking_same_day(data):
    """Regression guard: the extras dict must declare cutoff="prev_close",
    not something later, or build_feature_matrix's own cutoff check
    (test_extras_respect_cutoff in test_features.py) would never catch a
    same-day-leaking extra passed at "open_5m"."""
    pred = _fake_pred(data, "2018-01-01")
    hmm = _fake_hmm_labels(data, "2018-01-01")
    extras = FO.load_extras(pred=pred, hmm_labels=hmm)
    for name in FO.EXTRA_COLUMNS:
        assert extras[name][1] == "prev_close", name


def test_hmm_state_and_prob_both_present_and_distinct(data):
    pred = _fake_pred(data, "2018-01-01")
    hmm = _fake_hmm_labels(data, "2018-01-01")
    X = FO.build(data, pred=pred, hmm_labels=hmm)
    rows = X.dropna(subset=["hmm_state", "hmm_state_prob"])
    assert rows["hmm_state"].isin([0.0, 1.0, 2.0]).all()
    assert rows["hmm_state_prob"].between(0, 1).all()
