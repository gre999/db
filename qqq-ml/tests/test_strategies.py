"""Position sizing: fold-only calibration, vol targeting, the VRP overlay."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import strategies as S

FOLDS = pd.DataFrame({"fold": [1, 2], "train_start": pd.to_datetime(
    ["2016-01-01", "2017-01-01"]), "train_end": pd.to_datetime(
    ["2016-12-31", "2017-12-31"])})


def test_scale_ratios_uses_training_window_only():
    idx = pd.bdate_range("2016-01-01", "2018-12-31")
    rng = np.random.default_rng(0)
    rv = pd.Series(rng.uniform(1e-5, 5e-5, len(idx)), index=idx)
    rv_on = rv * 1.3     # true ratio 1.3 everywhere
    ratios = S.scale_ratios(rv, rv_on, FOLDS)
    np.testing.assert_allclose(ratios.to_numpy(), [1.3, 1.3], rtol=1e-9)

    # perturbing rv_on OUTSIDE every fold's training window must not move it
    rv_on2 = rv_on.copy()
    outside = idx[(idx < "2016-01-01") | (idx > "2017-12-31")]
    rv_on2.loc[outside] *= 100
    ratios2 = S.scale_ratios(rv, rv_on2, FOLDS)
    pd.testing.assert_series_equal(ratios, ratios2, check_names=False)


def test_cc_variance_forecast_applies_matching_fold_ratio():
    pred = pd.DataFrame({"fold": [1, 1, 2], "y_pred_var": [1e-4, 2e-4, 3e-4]},
                        index=pd.bdate_range("2019-01-01", periods=3))
    ratios = pd.Series({1: 1.5, 2: 2.0})
    out = S.cc_variance_forecast(pred, ratios)
    np.testing.assert_allclose(out.to_numpy(), [1.5e-4, 3e-4, 6e-4])


def test_cc_variance_forecast_missing_fold_raises():
    pred = pd.DataFrame({"fold": [3], "y_pred_var": [1e-4]},
                        index=[pd.Timestamp("2019-01-01")])
    with pytest.raises(KeyError):
        S.cc_variance_forecast(pred, pd.Series({1: 1.0}))


def test_vol_target_weight_formula_and_cap():
    # daily var such that annualized vol = 10%, 20%, 40%
    var = pd.Series([(0.10 ** 2) / 252, (0.20 ** 2) / 252, (0.40 ** 2) / 252])
    w = S.vol_target_weight(var, vol_target=0.20, leverage_cap=1.5)
    np.testing.assert_allclose(w.to_numpy(), [1.5, 1.0, 0.5], atol=1e-6)


def test_historical_vol_weight_is_causal():
    idx = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(1)
    ret = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    w = S.historical_vol_weight(ret, window=20, vol_target=0.15, leverage_cap=3.0)
    assert w.iloc[:19].isna().all()          # not enough history yet
    assert w.iloc[19:].notna().all()

    ret2 = ret.copy()
    ret2.iloc[40:] = ret2.iloc[40:] * 50      # blow up only the "future"
    w2 = S.historical_vol_weight(ret2, window=20, vol_target=0.15, leverage_cap=3.0)
    pd.testing.assert_series_equal(w.iloc[:40], w2.iloc[:40])


def test_buy_and_hold_weight_is_one():
    idx = pd.bdate_range("2020-01-01", periods=5)
    w = S.buy_and_hold_weight(idx)
    assert (w == 1.0).all()


def test_variance_risk_premium_scale():
    # VXN = 20 -> implied annualized var = 0.04; forecast daily var such
    # that annualized = 0.03 -> premium = 0.01
    vxn = pd.Series([20.0])
    var_cc = pd.Series([0.03 / 252])
    prem = S.variance_risk_premium(vxn, var_cc)
    np.testing.assert_allclose(prem.to_numpy(), [0.01], atol=1e-9)


def test_premium_thresholds_and_strategy_n_weight():
    idx = pd.bdate_range("2016-01-01", "2017-12-31")
    premium = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)
    thr = S.premium_thresholds(premium, FOLDS, q=0.5)
    # median of a linear ramp within each fold's own window
    for f in FOLDS.itertuples():
        m = (premium.index >= f.train_start) & (premium.index <= f.train_end)
        assert thr[f.fold] == pytest.approx(premium[m].median())

    base_w = pd.Series(1.0, index=idx)
    fold = pd.Series(np.where(idx.year == 2016, 1, 2), index=idx)
    out = S.strategy_n_weight(base_w, premium, fold, thr, derate=0.6)
    below = premium < fold.map(thr)
    assert (out[below] == 0.6).all()
    assert (out[~below] == 1.0).all()
