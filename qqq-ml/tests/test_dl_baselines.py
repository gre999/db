"""Tests for src/models/dl_baselines.py: flattened-sequence baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import dl_baselines as DB
from src.models.filter import make_xgboost
from src.validation import WalkForwardSplit


def _make_xy(seed: int = 0, start="2015-01-01", end="2022-12-31", n_features: int = 20):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    signal = rng.normal(0, 1, n)
    X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(n_features)}, index=days)
    X["f0"] = signal
    p_win = 1 / (1 + np.exp(-(0.8 * signal)))
    y = pd.Series((rng.random(n) < p_win).astype(int), index=days)
    return X, y


@pytest.fixture(scope="module")
def xy():
    return _make_xy()


def test_flatten_shape():
    tensor = np.zeros((10, 5, 4))
    flat = DB.flatten(tensor)
    assert flat.shape == (10, 20)


def test_fit_predict_folds_output_schema_and_detects_signal(xy):
    X, y = xy
    preds = DB.fit_predict_folds(X, y, DB.make_logistic_l2, "logistic_l2", "task_test")
    assert not preds.empty
    assert set(preds.columns) == {"date", "fold", "model", "target", "y_true", "y_pred_proba"}
    assert (preds["model"] == "logistic_l2").all()
    assert preds["y_pred_proba"].between(0, 1).all()

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(preds["y_true"], preds["y_pred_proba"])
    assert auc > 0.55, auc


def test_fit_predict_folds_never_uses_test_window_for_fitting(xy):
    """Perturb the last fold's test-window X/y and confirm every OTHER
    fold's predictions are completely unchanged (the model-fit window
    never overlaps the test window by construction, but this verifies it
    end to end rather than assuming it from the code alone)."""
    X, y = xy
    folds = WalkForwardSplit().folds(X.index)
    last = folds[-1]
    test_dates = X.index[last.test_idx]

    before = DB.fit_predict_folds(X, y, DB.make_logistic_l2, "logistic_l2", "task_test")

    X2 = X.copy()
    X2.loc[test_dates] += 1000.0
    y2 = y.copy()
    y2.loc[test_dates] = 1 - y2.loc[test_dates]

    after = DB.fit_predict_folds(X2, y2, DB.make_logistic_l2, "logistic_l2", "task_test")

    before_other = before[before["fold"] != last.number].reset_index(drop=True)
    after_other = after[after["fold"] != last.number].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_other, after_other)


def test_fit_predict_folds_works_with_xgboost_factory(xy):
    X, y = xy
    preds = DB.fit_predict_folds(X, y, make_xgboost, "xgboost", "task_test")
    assert not preds.empty
    assert (preds["model"] == "xgboost").all()


def test_decile_win_rate_monotonic_on_planted_signal(xy):
    X, y = xy
    preds = DB.fit_predict_folds(X, y, DB.make_logistic_l2, "logistic_l2", "task_test")
    dec = DB.decile_win_rate(preds)
    assert dec["win_rate"].iloc[-1] > dec["win_rate"].iloc[0]
    assert dec["n"].sum() == len(preds)


def test_make_logistic_l2_uses_l2_penalty():
    model = DB.make_logistic_l2()
    assert model.named_steps["clf"].penalty == "l2"
    assert model.named_steps["clf"].class_weight == "balanced"
