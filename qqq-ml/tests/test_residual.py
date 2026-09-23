"""HAR-plus-residual model: fold isolation, and the extrapolation claim."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import features as F
from src.models import week4
from src.models.base import make_dataset, validate_predictions
from src.models.har import HAR_LOG, HARModel
from src.models.residual import HARResidual
from src.models.trees import XGBModel
from tests import synth

SMALL_TREE = XGBModel(feature_list=HAR_LOG,
                      grid={"max_depth": [3], "learning_rate": [0.1]})


def _har_features(idx, x, rng) -> pd.DataFrame:
    """The 3 HAR columns as noisy scalings of ``x`` (not perfectly collinear)."""
    return pd.DataFrame({c: x * s + rng.normal(0, 0.02, len(x)) for c, s in
                         zip(HAR_LOG, [1.0, 0.8, 0.5])}, index=idx)


def test_predict_is_har_plus_tree_residual():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2016-01-01", periods=400)
    x = rng.uniform(-2, 0, len(idx))
    X = _har_features(idx, x, rng)
    y = pd.Series(x + rng.normal(0, 0.05, len(idx)), index=idx)

    m = HARResidual(har=HARModel(), tree=clone(SMALL_TREE)).fit(X, y)
    np.testing.assert_allclose(m.predict(X),
                               m.har_.predict(X) + m.tree_.predict(X))
    # tree target was the HAR residual, not y itself
    np.testing.assert_allclose(m.tree_.n_train_, len(X))
    assert m.resid_var_ == pytest.approx(m.tree_.resid_var_)
    assert set(m.features) == set(HAR_LOG)


def test_rescues_tree_extrapolation_beyond_training_range():
    """The motivating claim: HAR's linear extrapolation saves an XGBoost
    residual correction from the flat prediction a lone tree gives once the
    input leaves the range it was trained on (e.g. the 2020 RV spike)."""
    rng = np.random.default_rng(1)
    idx_tr = pd.bdate_range("2016-01-01", periods=500)
    idx_te = pd.bdate_range(idx_tr[-1] + pd.Timedelta(days=1), periods=20)

    x_tr = rng.uniform(-2, 0, len(idx_tr))
    y_tr = pd.Series(x_tr + rng.normal(0, 0.05, len(idx_tr)), index=idx_tr)
    Xtr = _har_features(idx_tr, x_tr, rng)

    x_te = np.full(len(idx_te), 3.0)          # far beyond the training range
    y_te = pd.Series(x_te, index=idx_te)      # noiseless continuation of the trend
    Xte = _har_features(idx_te, x_te, rng)

    plain = clone(SMALL_TREE).fit(Xtr, y_tr)
    combo = HARResidual(har=HARModel(), tree=clone(SMALL_TREE)).fit(Xtr, y_tr)

    err_plain = np.abs(plain.predict(Xte) - y_te.to_numpy()).mean()
    err_combo = np.abs(combo.predict(Xte) - y_te.to_numpy()).mean()
    assert err_plain > 1.0            # tree alone collapses near the top of its range
    assert err_combo < 0.5            # HAR extrapolates the trend, tree just corrects
    assert err_combo < err_plain / 2


def test_test_segment_never_used():
    """Changing the test slice must not change the fitted HAR or tree stage."""
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2016-01-01", periods=500)
    x = rng.uniform(-2, 0, len(idx))
    X = _har_features(idx, x, rng)
    y = pd.Series(x + rng.normal(0, 0.05, len(idx)), index=idx)
    tr, te = idx[:450], idx[450:]

    a = HARResidual(har=HARModel(), tree=clone(SMALL_TREE)).fit(X.loc[tr], y.loc[tr])
    X2, y2 = X.copy(), y.copy()
    X2.loc[te, HAR_LOG] = rng.normal(size=(len(te), len(HAR_LOG)))
    y2.loc[te] = rng.normal(-9, 3, len(te))
    b = HARResidual(har=HARModel(), tree=clone(SMALL_TREE)).fit(X2.loc[tr], y2.loc[tr])

    np.testing.assert_allclose(a.har_.coef_.to_numpy(), b.har_.coef_.to_numpy())
    np.testing.assert_allclose(a.predict(X.loc[te]), b.predict(X.loc[te]), rtol=1e-9)


def test_runs_through_week4():
    md = synth.make_market_data()
    X = F.build_feature_matrix(md, "prev_close")
    tg = F.build_targets(md)
    X, y = make_dataset(X, tg, "rv")
    rv = tg["rv"].where(tg["valid"].astype(bool))
    models = {"har_resid_xgb": HARResidual(har=HARModel(), tree=clone(SMALL_TREE))}
    out = week4.run(X=X, y=y, rv_calendar=rv, models=models, n_repeats=2,
                    save=False)
    validate_predictions(out["pred"])
    assert set(out["params"]["model"]) == {"har_resid_xgb"}
    assert out["pred"]["y_pred_log"].notna().all()
