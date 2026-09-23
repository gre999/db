"""Walk-forward splitter and fold-local pipeline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from src.validation import WalkForwardSplit, make_pipeline, walk_forward_predict
from tests import synth


@pytest.fixture(scope="module")
def dates() -> pd.DatetimeIndex:
    return synth.calendar()          # 2014-12-22 .. 2020-12-31 sessions


def test_default_folds(dates):
    wf = WalkForwardSplit()
    tbl = wf.describe(dates)
    assert list(tbl["test_start"].astype(str)) == [
        "2018-01-02", "2019-01-02", "2020-01-02"]
    assert str(tbl.loc[1, "train_start"]) == "2015-01-02"
    assert str(tbl.loc[1, "train_end"]) == "2017-12-29"
    assert (tbl["n_test"] > 240).all() and (tbl["n_train"] > 740).all()


@pytest.mark.parametrize("kw", [
    {}, {"train_years": 2, "step_years": 1}, {"expanding": True},
    {"embargo_days": 5}, {"test_years": 2, "step_years": 2},
])
def test_no_overlap_and_time_order(dates, kw):
    folds = WalkForwardSplit(**kw).folds(dates)
    assert folds
    prev_test_start = None
    for f in folds:
        assert not set(f.train_idx) & set(f.test_idx)
        assert dates[f.train_idx].max() < dates[f.test_idx].min()
        assert np.all(np.diff(f.train_idx) > 0) and np.all(np.diff(f.test_idx) > 0)
        if prev_test_start is not None:
            assert f.test_start > prev_test_start
        prev_test_start = f.test_start
    tests = np.concatenate([f.test_idx for f in folds])
    assert len(tests) == len(set(tests))      # each day tested at most once


def test_rolling_window_length(dates):
    for f in WalkForwardSplit(train_years=3).folds(dates):
        span = f.train_end - f.train_start
        assert pd.Timedelta(days=3 * 365 - 10) < span <= pd.Timedelta(days=3 * 366)


def test_expanding_anchored(dates):
    folds = WalkForwardSplit(expanding=True).folds(dates)
    assert all(f.train_start == dates[0] for f in folds)


def test_embargo_gap(dates):
    base = WalkForwardSplit().folds(dates)
    emb = WalkForwardSplit(embargo_days=5).folds(dates)
    for a, b in zip(base, emb):
        assert len(a.train_idx) - len(b.train_idx) == 5
        assert b.test_idx[0] - b.train_idx[-1] == 6


def test_partial_last_fold_and_min_test(dates):
    short = dates[dates <= "2020-01-20"]            # 12 test days in 2020
    assert len(WalkForwardSplit().folds(short)) == 2
    assert len(WalkForwardSplit(min_test_days=5).folds(short)) == 3


def test_unsorted_dates_rejected(dates):
    with pytest.raises(ValueError):
        WalkForwardSplit().folds(dates[::-1])


def _xy(dates, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(len(dates), 3)), index=dates,
                     columns=["a", "b", "c"])
    X.loc[X.sample(frac=0.05, random_state=1).index, "a"] = np.nan
    # Drift in feature scale over time: fold-local scaling must differ.
    X["b"] = X["b"] * (1 + np.arange(len(dates)) / 200)
    y = pd.Series(X["b"].to_numpy() * 0.5 + rng.normal(size=len(dates)),
                  index=dates)
    return X, y


def test_preprocessing_fitted_on_train_only(dates):
    X, y = _xy(dates)
    wf = WalkForwardSplit()
    preds, fitted = walk_forward_predict(make_pipeline(LinearRegression()),
                                         X, y, wf)
    for f, pipe in zip(wf.folds(dates), fitted):
        Xtr = X.iloc[f.train_idx]
        med = Xtr.median()
        np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, med)
        imputed = Xtr.fillna(med)
        np.testing.assert_allclose(pipe.named_steps["scale"].mean_,
                                   imputed.mean())
    assert preds.index.is_monotonic_increasing
    assert set(preds["fold"]) == {1, 2, 3}
    assert preds.index.min() >= pd.Timestamp("2018-01-01")


def test_sklearn_cv_compatible(dates):
    X, y = _xy(dates)
    scores = cross_val_score(make_pipeline(DummyRegressor()), X, y,
                             cv=WalkForwardSplit(), scoring="neg_mean_squared_error")
    assert len(scores) == 3
