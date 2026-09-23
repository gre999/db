"""Week-4 model tests: in-fold tuning, fold isolation, HAR-X, importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import features as F
from src.models import week4
from src.models.base import make_dataset, validate_predictions
from src.models.har import VOL_INDEX, harx_model
from src.models.trees import (TREE_FEATURES, RFModel, XGBModel, inner_split,
                              permutation_importance)
from src.validation import WalkForwardSplit
from tests import synth

SMALL = {"xgb": XGBModel(grid={"max_depth": [2, 3], "learning_rate": [0.1]}),
         "rf": RFModel(grid={"max_depth": [4], "min_samples_leaf": [20],
                             "max_features": [0.33, 0.66]})}


@pytest.fixture(scope="module")
def data():
    md = synth.make_market_data()
    X = F.build_feature_matrix(md, "prev_close")
    tg = F.build_targets(md)
    X, y = make_dataset(X, tg, "rv")
    return X, y, tg["rv"].where(tg["valid"].astype(bool))


@pytest.fixture(scope="module")
def out(data):
    X, y, rv = data
    models = {"harx": harx_model(), **{k: clone(v) for k, v in SMALL.items()}}
    return week4.run(X=X, y=y, rv_calendar=rv, models=models, n_repeats=2,
                     save=False)


def _train_slice(data):
    X, y, _ = data
    ok = X[TREE_FEATURES].notna().all(axis=1) & y.notna()
    f = WalkForwardSplit().folds(X.index[ok])[0]
    Xs, ys = X[ok], y[ok]
    return Xs.iloc[f.train_idx], ys.iloc[f.train_idx], Xs.iloc[f.test_idx], \
        ys.iloc[f.test_idx]


def test_inner_split_is_last_year_of_training(data):
    Xtr, *_ = _train_slice(data)
    tr, va = inner_split(Xtr.index)
    assert tr.sum() + va.sum() == len(Xtr) and not (tr & va).any()
    assert Xtr.index[tr].max() < Xtr.index[va].min()
    span = Xtr.index[va].max() - Xtr.index[va].min()
    assert pd.Timedelta(days=350) < span <= pd.Timedelta(days=366)
    assert Xtr.index[va].max() == Xtr.index.max()


@pytest.mark.parametrize("name", ["xgb", "rf"])
def test_tuning_records_and_resid_var(data, name):
    Xtr, ytr, *_ = _train_slice(data)
    m = clone(SMALL[name]).fit(Xtr, ytr)
    g = m.grid_results_
    assert len(g) == 2
    assert m.best_params_["val_rmse"] == pytest.approx(g["val_rmse"].min())
    assert m.resid_var_ == pytest.approx(m.best_params_["val_mse"])
    if name == "xgb":
        assert 1 <= m.best_params_["n_estimators"] <= XGBModel.max_estimators


@pytest.mark.parametrize("name", ["xgb", "rf"])
def test_test_segment_never_used(data, name):
    """Changing the test slice must not change tuning or the fitted model."""
    X, y, rv = data
    Xtr, ytr, Xte, yte = _train_slice(data)
    a = clone(SMALL[name]).fit(Xtr, ytr)
    X2, y2 = X.copy(), y.copy()
    rng = np.random.default_rng(3)
    X2.loc[Xte.index, TREE_FEATURES] = rng.normal(
        size=(len(Xte), len(TREE_FEATURES)))
    y2.loc[Xte.index] = rng.normal(-9, 3, len(Xte))
    b = clone(SMALL[name]).fit(X2.loc[Xtr.index], y2.loc[Xtr.index])
    assert a.best_params_.keys() == b.best_params_.keys()
    for k, v in a.best_params_.items():      # multi-threaded RF: last-digit noise
        assert b.best_params_[k] == pytest.approx(v, rel=1e-9, nan_ok=True), k
    np.testing.assert_allclose(a.predict(Xte), b.predict(Xte), rtol=1e-9)
    # and through the runner: fold-1 params identical
    o1 = week4.run(X=X, y=y, rv_calendar=rv, models={name: clone(SMALL[name])},
                   n_repeats=1, save=False)
    o2 = week4.run(X=X2, y=y2, rv_calendar=rv, models={name: clone(SMALL[name])},
                   n_repeats=1, save=False)
    p1 = o1["params"].query("fold == 1").drop(columns=["fold"])
    p2 = o2["params"].query("fold == 1").drop(columns=["fold"])
    pd.testing.assert_frame_equal(p1.reset_index(drop=True),
                                  p2.reset_index(drop=True), rtol=1e-9)


def test_harx_is_ols_on_log_vix(data):
    Xtr, ytr, *_ = _train_slice(data)
    m = harx_model().fit(Xtr, ytr)
    Z = Xtr[m.features].copy()
    for c in VOL_INDEX:
        Z[c] = np.log(Z[c])
    Z.insert(0, "const", 1.0)
    beta, *_ = np.linalg.lstsq(Z.to_numpy(), ytr.to_numpy(), rcond=None)
    np.testing.assert_allclose(m.coef_.to_numpy(), beta)
    assert list(m.coef_.index)[-2:] == VOL_INDEX
    np.testing.assert_allclose(Xtr["vix_1d"], data[0].loc[Xtr.index, "vix_1d"])


def test_permutation_importance(data):
    Xtr, ytr, Xte, yte = _train_slice(data)
    m = clone(SMALL["xgb"]).fit(Xtr, ytr)
    imp = permutation_importance(m, Xte, yte, n_repeats=3).set_index("feature")
    assert set(imp.index) == set(TREE_FEATURES)
    assert imp.loc["har_logrv_1d", "importance"] > 0.05
    assert imp["importance"].idxmax() in ("har_logrv_1d", "har_logrv_5d")


def test_outputs(out):
    pred = out["pred"]
    validate_predictions(pred)
    d = pred.groupby("model")["date"].apply(frozenset)
    assert d["harx"] == d["xgb"] == d["rf"]
    assert set(out["params"]["model"]) == {"xgb", "rf"}
    assert out["params"].groupby("model")["fold"].nunique().eq(2).all()
    assert set(out["coefficients"]["model"]) == {"harx"}
    imp = out["importance"]
    assert {"all", "covid_2020", "jump_day"} <= set(imp["slice"])
    assert (imp.groupby(["model", "fold", "slice"]).size() > 0).all()
