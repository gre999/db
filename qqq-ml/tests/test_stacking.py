"""Tests for src/models/stacking.py: the incremental-information check.

test_wald_test_matches_statsmodels_logit uses statsmodels as a
cross-check only (requirements-dev.txt, dev/test-only - src/models/
stacking.py itself has no statsmodels dependency).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import stacking as ST


def _make_binary_data(seed=0, n=600, k=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, k))
    true_beta = np.array([0.8, -0.5])[:k]
    logits = 0.2 + X @ true_beta
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return X, y


# --------------------------------------------------------------- logit
def test_logit_is_inverse_of_sigmoid():
    p = np.array([0.1, 0.5, 0.9])
    l = ST.logit(p)
    back = 1 / (1 + np.exp(-l))
    assert np.allclose(back, p, atol=1e-5)


# ----------------------------------------------------------- Wald test
def test_wald_test_matches_statsmodels_logit():
    statsmodels = pytest.importorskip("statsmodels.api")
    X, y = _make_binary_data(seed=1)
    out = ST.wald_test(X, y)

    Xd = np.column_stack([np.ones(len(X)), X])
    sm_model = statsmodels.Logit(y, Xd).fit(disp=0)

    assert np.allclose(out["coef"], np.asarray(sm_model.params), atol=1e-4)
    assert np.allclose(out["se"], np.asarray(sm_model.bse), atol=1e-4)
    assert np.allclose(out["p"], np.asarray(sm_model.pvalues), atol=1e-3)


def test_wald_test_recovers_known_effect_and_null():
    rng = np.random.default_rng(2)
    n = 2000
    x_real = rng.normal(0, 1, n)
    x_noise = rng.normal(0, 1, n)      # unrelated to y
    logits = 1.5 * x_real
    p = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    X = np.column_stack([x_real, x_noise])
    out = ST.wald_test(X, y)
    assert out["p"][1] < 0.001          # x_real: strongly significant
    assert out["p"][2] > 0.05           # x_noise: not significant


# ------------------------------------------------------------- stacking
def test_fit_stacking_fold_applies_validation_fit_to_test_unchanged():
    rng = np.random.default_rng(3)
    n_val, n_test = 300, 200
    hand_val = rng.uniform(0.05, 0.95, n_val)
    cnn_val = rng.uniform(0.05, 0.95, n_val)
    logits = 0.5 * ST.logit(hand_val) + 1.2 * ST.logit(cnn_val)
    p_val = 1 / (1 + np.exp(-logits))
    y_val = (rng.random(n_val) < p_val).astype(int)

    hand_test = rng.uniform(0.05, 0.95, n_test)
    cnn_test = rng.uniform(0.05, 0.95, n_test)

    out = ST.fit_stacking_fold(hand_val, cnn_val, y_val, hand_test, cnn_test)
    assert len(out["test_proba"]) == n_test
    assert out["cnn_coef"] > 0            # planted a strong positive CNN effect
    assert out["cnn_p"] < 0.05

    # applying the SAME fitted model manually must reproduce test_proba exactly
    X_test = np.column_stack([ST.logit(hand_test), ST.logit(cnn_test)])
    manual = out["wald"]["model"].predict_proba(X_test)[:, 1]
    assert np.allclose(out["test_proba"], manual)


def test_fit_stacking_fold_null_case_cnn_not_significant():
    rng = np.random.default_rng(4)
    n_val, n_test = 300, 200
    hand_val = rng.uniform(0.05, 0.95, n_val)
    cnn_val = rng.uniform(0.05, 0.95, n_val)   # pure noise, unrelated to y
    logits = 1.0 * ST.logit(hand_val)
    p_val = 1 / (1 + np.exp(-logits))
    y_val = (rng.random(n_val) < p_val).astype(int)
    hand_test = rng.uniform(0.05, 0.95, n_test)
    cnn_test = rng.uniform(0.05, 0.95, n_test)

    out = ST.fit_stacking_fold(hand_val, cnn_val, y_val, hand_test, cnn_test)
    assert out["cnn_p"] > 0.05


# --------------------------------------------------------- Spearman / R2
def _fake_preds_and_features(seed=5, n_days=500, n_folds=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-01", periods=n_days)
    fold = pd.Series(np.array_split(np.arange(n_days), n_folds))
    fold_of = np.concatenate([[i + 1] * len(idx) for i, idx in enumerate(
        np.array_split(np.arange(n_days), n_folds))])
    feat_a = rng.normal(0, 1, n_days)
    feat_b = rng.normal(0, 1, n_days)
    cnn_logit = 0.9 * feat_a + rng.normal(0, 0.3, n_days)   # strongly tied to feat_a
    cnn_proba = 1 / (1 + np.exp(-cnn_logit))
    preds = pd.DataFrame({"date": days, "fold": fold_of, "model": "cnn",
                         "y_true": rng.integers(0, 2, n_days),
                         "y_pred_proba": cnn_proba})
    features = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b}, index=days)
    return preds, features


def test_spearman_by_fold_detects_the_planted_relationship():
    preds, features = _fake_preds_and_features()
    out = ST.spearman_by_fold(preds, features, ["feat_a", "feat_b"])
    assert (out["feat_a"].abs() > 0.5).all()
    assert (out["feat_a"].abs() > out["feat_b"].abs()).all()


def test_r_squared_per_fold_and_pooled_standardized_are_high_for_planted_signal():
    preds, features = _fake_preds_and_features()
    per_fold = ST.r_squared_per_fold(preds, features, ["feat_a", "feat_b"])
    assert (per_fold["r2"] > 0.5).all()
    assert (per_fold["adj_r2"] <= per_fold["r2"]).all()

    pooled = ST.r_squared_pooled_standardized(preds, features, ["feat_a", "feat_b"])
    assert pooled["r2"] > 0.5
    assert pooled["adj_r2"] <= pooled["r2"]
