"""Backtest execution: timing, no-trade band, cost, evaluation, bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest as B
from src import strategies as S


def _daily(idx, **cols):
    return pd.DataFrame(cols, index=idx)


def test_close_timing_basic_no_cost_no_band():
    idx = pd.bdate_range("2020-01-01", periods=2)
    w = pd.Series([1.0, 0.5], index=idx)
    daily = _daily(idx, ret_cc=[0.01, -0.02])
    bt = B.run_backtest(w, daily, timing="close", cost_bps=0, band=0)
    np.testing.assert_allclose(bt["weight_held"], [1.0, 0.5])
    np.testing.assert_allclose(bt["net_return"], [0.01, -0.01])
    np.testing.assert_allclose(bt["gross_return"], bt["net_return"])


def test_cost_calculation_correct():
    idx = pd.bdate_range("2020-01-01", periods=3)
    w = pd.Series([1.0, 1.0, 0.0], index=idx)
    daily = _daily(idx, ret_cc=[0.0, 0.0, 0.0])
    bt = B.run_backtest(w, daily, timing="close", cost_bps=100, band=0, w0=0.0)
    np.testing.assert_allclose(bt["turnover"], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(bt["cost"], [0.01, 0.0, 0.01])
    np.testing.assert_allclose(bt["net_return"], [-0.01, 0.0, -0.01])


def test_no_trade_band_skips_small_changes():
    idx = pd.bdate_range("2020-01-01", periods=4)
    w = pd.Series([1.0, 1.03, 1.10, 1.20], index=idx)
    daily = _daily(idx, ret_cc=[0.0] * 4)
    bt = B.run_backtest(w, daily, timing="close", cost_bps=100, band=0.05, w0=1.0)
    np.testing.assert_allclose(bt["weight_held"], [1.0, 1.0, 1.10, 1.20])
    np.testing.assert_allclose(bt["turnover"], [0.0, 0.0, 0.10, 0.10], atol=1e-9)


def test_next_open_timing_splits_overnight_and_intraday_legs():
    idx = pd.bdate_range("2020-01-01", periods=2)
    w = pd.Series([1.0, 2.0], index=idx)
    daily = _daily(idx, ret_overnight=[0.01, 0.02], ret_intraday=[0.03, 0.04],
                  ret_cc=[0.0, 0.0])
    bt = B.run_backtest(w, daily, timing="next_open", cost_bps=0, band=0, w0=0.0)
    # day 1: overnight leg earned by w0=0, intraday leg by the new weight 1.0
    assert bt["gross_return"].iloc[0] == pytest.approx(0.0 * 0.01 + 1.0 * 0.03)
    # day 2: overnight leg earned by yesterday's held weight (1.0), intraday by 2.0
    assert bt["gross_return"].iloc[1] == pytest.approx(1.0 * 0.02 + 2.0 * 0.04)


def test_missing_return_columns_raise():
    idx = pd.bdate_range("2020-01-01", periods=2)
    w = pd.Series([1.0, 1.0], index=idx)
    daily = _daily(idx[:1], ret_cc=[0.01])   # second date missing entirely
    with pytest.raises(ValueError):
        B.run_backtest(w, daily, timing="close")


def test_evaluate_matches_manual_sharpe_and_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=4)
    bt = pd.DataFrame({"weight_held": [1.0] * 4, "turnover": [0.0] * 4,
                       "net_return": [0.01, -0.02, 0.01, 0.01]}, index=idx)
    bt["equity"] = (1 + bt["net_return"]).cumprod()
    ev = B.evaluate(bt, vol_target=0.15)
    r = bt["net_return"]
    ann_vol = r.std(ddof=0) * np.sqrt(252)
    ann_ret = (1 + r).prod() ** (252 / 4) - 1
    assert ev["ann_vol"] == pytest.approx(ann_vol)
    assert ev["sharpe"] == pytest.approx(ann_ret / ann_vol)
    dd = (bt["equity"] / bt["equity"].cummax() - 1).min()
    assert ev["max_drawdown"] == pytest.approx(dd)
    assert ev["tracking_error"] == pytest.approx(ann_vol - 0.15)


def test_perfect_foresight_has_lowest_tracking_error():
    """The motivating claim for including a perfect-foresight upper bound:
    with the true variance in hand, vol targeting should land closest to
    the target - closer than a materially noisy forecast of the same
    variance."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2016-01-01", periods=2000)
    lv = np.zeros(len(idx))
    for i in range(1, len(lv)):
        lv[i] = 0.97 * lv[i - 1] + 0.2 * rng.normal()
    true_var = (0.0009 * np.exp(lv)) ** 1        # daily variance, log-AR(1) vol
    ret_cc = rng.normal(0, np.sqrt(true_var))
    daily = _daily(idx, ret_cc=ret_cc)

    true_var_s = pd.Series(true_var, index=idx)
    noisy_var_s = true_var_s * np.exp(rng.normal(0, 1.0, len(idx)))

    w_perfect = S.vol_target_weight(true_var_s, vol_target=0.15, leverage_cap=10.0)
    w_noisy = S.vol_target_weight(noisy_var_s, vol_target=0.15, leverage_cap=10.0)

    bt_perfect = B.run_backtest(w_perfect, daily, timing="close", cost_bps=0)
    bt_noisy = B.run_backtest(w_noisy, daily, timing="close", cost_bps=0)
    ev_perfect = B.evaluate(bt_perfect, vol_target=0.15)
    ev_noisy = B.evaluate(bt_noisy, vol_target=0.15)

    assert abs(ev_perfect["tracking_error"]) < abs(ev_noisy["tracking_error"])


def test_block_bootstrap_sharpe_diff():
    rng = np.random.default_rng(3)
    n = 1000
    common = rng.normal(0, 0.01, n)
    a = 0.0015 + common + rng.normal(0, 0.001, n)   # clearly higher Sharpe
    b = 0.0000 + common + rng.normal(0, 0.001, n)
    ra, rb = pd.Series(a), pd.Series(b)
    obs, p = B.block_bootstrap_sharpe_diff(ra, rb, block_size=20, n_boot=500, seed=1)
    assert obs > 0
    assert p < 0.05

    obs0, p0 = B.block_bootstrap_sharpe_diff(ra, ra, block_size=20, n_boot=200, seed=1)
    assert obs0 == pytest.approx(0.0)
    assert p0 == pytest.approx(1.0)


def test_block_bootstrap_sharpe_diff_dist_matches_the_p_value_function():
    """block_bootstrap_sharpe_diff_dist is the shared implementation behind
    block_bootstrap_sharpe_diff - same seed must give the same observed
    diff and a p-value recomputable from its own distribution."""
    rng = np.random.default_rng(4)
    n = 800
    common = rng.normal(0, 0.01, n)
    a = pd.Series(0.001 + common + rng.normal(0, 0.001, n))
    b = pd.Series(common + rng.normal(0, 0.001, n))

    obs, p = B.block_bootstrap_sharpe_diff(a, b, block_size=20, n_boot=300, seed=2)
    obs2, diffs = B.block_bootstrap_sharpe_diff_dist(a, b, block_size=20, n_boot=300, seed=2)
    assert obs2 == pytest.approx(obs)
    assert diffs.shape == (300,)
    centered = diffs - diffs.mean()
    manual_p = float(np.mean(np.abs(centered) >= abs(obs)))
    assert manual_p == pytest.approx(p)


def test_regress_weight_diff_on_return_detects_return_timing():
    """A weight difference that's genuinely aligned with the return it earns
    (return timing) must come back with a significant positive slope close
    to the true one; a weight difference independent of the return must not."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    w_b = pd.Series(1.0, index=idx)

    true_slope = 0.05
    diff = pd.Series(rng.normal(0, 0.3, len(idx)), index=idx)
    ret = pd.Series(0.0003 + true_slope * diff.to_numpy()
                    + rng.normal(0, 0.01, len(idx)), index=idx)
    res = B.regress_weight_diff_on_return(w_b + diff, w_b, ret)
    assert res["slope"] == pytest.approx(true_slope, rel=0.3)
    assert res["slope_p"] < 0.01

    ret_indep = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    res_noise = B.regress_weight_diff_on_return(w_b + diff, w_b, ret_indep)
    assert res_noise["slope_p"] > 0.10
