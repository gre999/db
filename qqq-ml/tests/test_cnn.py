"""Tests for src/models/cnn.py: the 1D-CNN and its sanity checks.

Uses small synthetic sequence data with reduced epochs/patience (passed
via train_kwargs) so tests stay fast - the module's own default
hyperparameters (config/week11_dl.toml) are exercised for real on actual
data separately, not re-run at full scale here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import cnn as CNN
from src.validation import WalkForwardSplit

FAST = {"max_epochs": 15, "patience": 3, "batch_size": 16}


def _make_sequence_data(seed=0, n_days=900, window_len=5):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-01", periods=n_days)
    tensor = rng.normal(0, 1, size=(n_days, window_len, CNN.SEQ.N_CHANNELS))
    signal = tensor[:, 0, 0]        # first timestep, first channel drives the label
    p_win = 1 / (1 + np.exp(-1.2 * signal))
    y = pd.Series((rng.random(n_days) < p_win).astype(int), index=days)
    return tensor, pd.DatetimeIndex(days), y


def test_small_cnn_param_count():
    model = CNN.SmallCNN()
    assert CNN.n_params(model) == 1809


def test_train_one_seed_is_deterministic():
    tensor, days, y = _make_sequence_data()
    X, yv = tensor[:200], y.to_numpy()[:200]
    Xv, yvv = tensor[200:260], y.to_numpy()[200:260]
    m1 = CNN.train_one_seed(X, yv, Xv, yvv, seed=0, **FAST)
    m2 = CNN.train_one_seed(X, yv, Xv, yvv, seed=0, **FAST)
    p1 = CNN.predict_proba(m1, Xv)
    p2 = CNN.predict_proba(m2, Xv)
    assert np.array_equal(p1, p2)


def test_train_one_seed_different_seeds_usually_differ():
    tensor, days, y = _make_sequence_data()
    X, yv = tensor[:200], y.to_numpy()[:200]
    Xv, yvv = tensor[200:260], y.to_numpy()[200:260]
    m0 = CNN.train_one_seed(X, yv, Xv, yvv, seed=0, **FAST)
    m1 = CNN.train_one_seed(X, yv, Xv, yvv, seed=1, **FAST)
    p0 = CNN.predict_proba(m0, Xv)
    p1 = CNN.predict_proba(m1, Xv)
    assert not np.array_equal(p0, p1)


def test_fit_predict_fold_output_schema_and_averaging():
    tensor, days, y = _make_sequence_data(n_days=900)
    splitter = WalkForwardSplit(train_years=2, test_years=1, step_years=1,
                                first_test_start="2017-06-01")
    f = next(iter(splitter.folds(days)))
    out = CNN.fit_predict_fold(tensor, days, y, f, validation_years=1,
                               seeds=(0, 1, 2), **FAST)
    preds = out["preds"]
    assert set(preds.columns) >= {"date", "fold", "model", "y_true", "y_pred_proba"}
    assert (preds["model"] == "cnn").all()
    assert len(out["test_auc_per_seed"]) == 3
    # the reported preds probability must be the mean across the 3 seeds
    manual_mean = np.mean([CNN.predict_proba(m, out["X_test"]) for m in out["models"]], axis=0)
    assert np.allclose(preds["y_pred_proba"].to_numpy(), manual_mean)


def test_overfit_tiny_batch_check_passes_on_separable_data():
    tensor, days, y = _make_sequence_data(seed=3, n_days=300)
    X, yv = tensor, y.to_numpy()
    out = CNN.overfit_tiny_batch_check(X, yv, n=32, seed=0, max_epochs=300)
    assert out["passes"], out
    assert out["train_accuracy"] >= 0.95


def test_shuffled_label_check_gives_near_chance_auc():
    tensor, days, y = _make_sequence_data(seed=4, n_days=900)
    X, yv = tensor[:400], y.to_numpy()[:400]
    Xv, yvv = tensor[400:460], y.to_numpy()[400:460]
    Xt, yt = tensor[460:560], y.to_numpy()[460:560]
    out = CNN.shuffled_label_check(X, yv, Xv, yvv, Xt, yt, seed=0, **FAST)
    assert 0.35 <= out["test_auc"] <= 0.65, out   # loose bound - small sample, still informative


def test_first_fold_with_enough_data_finds_a_qualifying_fold():
    tensor, days, y = _make_sequence_data(n_days=1500)
    f = CNN.first_fold_with_enough_data(days, validation_years=1, min_fit=100, min_val=50)
    assert f is not None
    assert len(f.test_idx) > 0
