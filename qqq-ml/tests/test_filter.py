"""Tests for src/models/filter.py: the logistic trade-day filter baseline.

Uses synthetic X/orb data (not real features/strategy output) - filter.py
operates purely on already-built DataFrames, decoupled from MarketData, so
a compact synthetic generator is enough and keeps these tests fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import regime_eval as E
from src.features_open import OPEN_FEATURE_COLUMNS
from src.models import filter as FL
from src.validation import WalkForwardSplit


def _make_data(seed: int = 0, start="2015-01-01", end="2022-12-31",
              doji_frac: float = 0.15):
    """Synthetic X (open_5m-style columns) + orb (day/traded/r_net/
    bps_return), with one real column genuinely predictive of the label so
    AUC/calibration tests have something to detect."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    signal = rng.normal(0, 1, n)

    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in OPEN_FEATURE_COLUMNS},
                     index=days)
    X["open5m_body_ratio"] = signal      # the one genuinely predictive column

    doji = rng.random(n) < doji_frac
    p_win = 1 / (1 + np.exp(-(0.8 * signal)))
    win = rng.random(n) < p_win
    r_net = np.where(win, rng.uniform(0.5, 3.0, n), rng.uniform(-2.0, -0.1, n))
    bps_return = np.where(doji, 0.0, r_net * 20.0)   # arbitrary R->bps scale

    orb = pd.DataFrame({"day": days, "traded": ~doji, "r_net": np.where(doji, np.nan, r_net),
                        "bps_return": bps_return}).set_index("day")
    return X, orb


@pytest.fixture(scope="module")
def data():
    return _make_data()


# --------------------------------------------------------------- labels
def test_build_labels_drops_doji_and_matches_r_net_sign(data):
    _, orb = data
    labels = FL.build_labels(orb)
    assert labels.index.isin(orb.index[orb["traded"]]).all()
    assert not labels.index.isin(orb.index[~orb["traded"]]).any()
    assert (labels["label"] == (labels["r_net"] > 0)).all()


# --------------------------------------------------------- validation score
def test_validation_score_zeros_dropped_signal_days():
    idx = pd.bdate_range("2020-01-01", periods=10)
    bps = pd.Series([10, -10, 5, 5, -5, 0, 20, -20, 15, -15], index=idx, dtype=float)
    signal_dates = idx[:8]          # last 2 days are "doji", never tradeable
    kept = idx[[0, 2, 3, 6, 8]]     # subset of signal_dates only
    sharpe, tpy = FL._validation_score(bps, kept, signal_dates, min_trades_per_year=1)
    manual = bps.copy()
    manual.loc[signal_dates.difference(kept)] = 0.0
    assert sharpe == pytest.approx(E._daily_eval(manual)["sharpe"])
    assert tpy == pytest.approx(len(kept) / bps.index.year.nunique())


# ----------------------------------------------------------- end to end
def test_fit_filter_folds_output_shape_and_keep_matches_threshold(data):
    X, orb = data
    preds = FL.fit_filter_folds(X, orb)
    assert not preds.empty
    assert set(preds["fold"]) == {f.number for f in WalkForwardSplit().folds(X.index)
                                  if len(X.index[f.train_idx]) >= 30}
    for _, row in preds.iterrows():
        assert row["keep"] == (row["y_pred_proba"] >= row["threshold"])
    # every fold's chosen retention is one of the declared candidates
    assert preds["retention_selected"].isin(FL.RETENTION_GRID).all()


def test_validation_years_changes_the_split_without_breaking_output(data):
    """Week 10 robustness knob: a 2-year validation window should still
    produce a well-formed predictions frame, with a smaller model-fit
    window per fold (so somewhat different, but still valid, choices)."""
    X, orb = data
    preds_1y = FL.fit_filter_folds(X, orb)
    preds_2y = FL.fit_filter_folds_generic(X, orb, FL.make_logistic, "logistic",
                                           validation_years=2)
    assert not preds_2y.empty
    assert preds_2y["retention_selected"].isin(FL.RETENTION_GRID).all()
    for _, row in preds_2y.iterrows():
        assert row["keep"] == (row["y_pred_proba"] >= row["threshold"])
    # 2-year validation leaves less fit data, so at least as many folds get
    # skipped (never more test-fold coverage than the 1-year version)
    assert set(preds_2y["fold"]) <= set(preds_1y["fold"])


def test_fit_filter_folds_never_uses_test_fold_for_threshold_or_fit(data):
    """Perturb the LAST fold's own test-window features/labels and confirm
    that fold's chosen threshold/retention is unchanged - the model fit and
    threshold selection only ever see the model-fit and validation windows,
    both strictly inside the training period (same causality-check pattern
    as test_run_fixed_k_uses_same_k_every_fold / test_vol_tercile_labels_
    uses_training_window_cutoffs_only elsewhere in this project)."""
    X, orb = data
    folds = WalkForwardSplit().folds(X.index)
    last = folds[-1]
    test_dates = X.index[last.test_idx]

    before = FL.fit_filter_folds(X, orb)
    before_choice = before[before["fold"] == last.number][
        ["threshold", "retention_selected"]].iloc[0]

    X2 = X.copy()
    X2.loc[test_dates, list(OPEN_FEATURE_COLUMNS)] += 1000.0
    orb2 = orb.copy()
    orb2.loc[test_dates, "bps_return"] = orb2.loc[test_dates, "bps_return"] * 0 + 999.0
    orb2.loc[test_dates, "r_net"] = 999.0

    after = FL.fit_filter_folds(X2, orb2)
    after_choice = after[after["fold"] == last.number][
        ["threshold", "retention_selected"]].iloc[0]

    assert before_choice["threshold"] == pytest.approx(after_choice["threshold"])
    assert before_choice["retention_selected"] == after_choice["retention_selected"]
    # earlier folds' output must be completely untouched
    pd.testing.assert_frame_equal(
        before[before["fold"] != last.number].reset_index(drop=True),
        after[after["fold"] != last.number].reset_index(drop=True))


def test_retention_grid_always_offers_no_filter_option():
    assert FL.RETENTION_GRID[-1] == 1.00


# --------------------------------------------------------------- AUC / calibration
def test_coefficient_table_matches_fit_filter_folds_and_finds_planted_signal(data):
    X, orb = data
    preds = FL.fit_filter_folds(X, orb)
    coefs = FL.coefficient_table(X, orb)

    assert set(coefs["fold"]) == set(preds["fold"])
    assert set(OPEN_FEATURE_COLUMNS) <= set(coefs.columns)
    assert (coefs["retention_selected"].to_numpy()
           == preds.groupby("fold")["retention_selected"].first().to_numpy()).all()

    # the one genuinely predictive column should stand out and have a
    # consistent sign across folds (it's the only column driving the label
    # in the synthetic fixture)
    signal_col = coefs["open5m_body_ratio"]
    other_cols = [c for c in OPEN_FEATURE_COLUMNS if c != "open5m_body_ratio"]
    assert (signal_col.abs().median() >
           coefs[other_cols].abs().to_numpy().mean() * 2)
    assert (signal_col > 0).all() or (signal_col < 0).all()


def test_auc_and_calibration_detect_the_planted_signal(data):
    X, orb = data
    preds = FL.fit_filter_folds(X, orb)
    a = FL.auc(preds)
    assert a > 0.55, a       # planted signal should be clearly detectable

    cal = FL.decile_calibration(preds, orb)
    assert cal["win_rate"].iloc[-1] > cal["win_rate"].iloc[0]
    assert cal["mean_r_net"].iloc[-1] > cal["mean_r_net"].iloc[0]
    assert cal["n"].sum() == len(preds)


# ------------------------------------------------------- week 9: tree models
@pytest.mark.parametrize("model_key", ["random_forest", "xgboost", "xgboost_depth3"])
def test_tree_models_detect_the_planted_signal(data, model_key):
    X, orb = data
    factory, y_col = FL.MODEL_FACTORIES[model_key]
    preds = FL.fit_filter_folds_generic(X, orb, factory, model_key, y_col=y_col)
    assert not preds.empty
    assert (preds["model"] == model_key).all()
    a = FL.auc(preds)
    assert a > 0.55, a


def test_huber_regression_baseline_ranks_by_predicted_r_net(data):
    X, orb = data
    factory, y_col = FL.MODEL_FACTORIES["huber"]
    assert y_col == "r_net"
    preds = FL.fit_filter_folds_generic(X, orb, factory, "huber", y_col=y_col)
    assert not preds.empty
    # kept days should have a higher win rate than dropped days if the
    # regression's ranking carries the planted signal at all
    kept_win_rate = preds.loc[preds["keep"], "y_true"].mean()
    dropped = preds.loc[~preds["keep"], "y_true"]
    if len(dropped):
        assert kept_win_rate >= dropped.mean()


def test_random_forest_scale_pos_weight_not_applicable_but_class_weight_set():
    model = FL.make_random_forest()
    assert model.class_weight == "balanced"
    assert model.max_depth == 4
    assert model.min_samples_leaf == 20


def test_xgboost_scale_pos_weight_computed_from_y():
    y_imbalanced = pd.Series([1] * 20 + [0] * 80)   # 20 pos, 80 neg
    model = FL.make_xgboost(y_imbalanced)
    assert model.scale_pos_weight == pytest.approx(80 / 20)
    assert model.max_depth == 2       # main-analysis depth, not the appendix one


def test_xgboost_depth3_factory_overrides_depth_only():
    model = FL.MODEL_FACTORIES["xgboost_depth3"][0](None)
    assert model.max_depth == 3
    assert model.learning_rate == 0.05


def test_tree_model_never_uses_test_fold_for_threshold_or_fit(data):
    """Same causality check as test_fit_filter_folds_never_uses_test_fold_
    for_threshold_or_fit, generalized: _fit_fold is shared code, but this
    confirms it actually generalizes rather than assuming it - perturb the
    last fold's test data and require XGBoost's chosen threshold/retention
    for that fold is unchanged."""
    X, orb = data
    factory, y_col = FL.MODEL_FACTORIES["xgboost"]
    folds = WalkForwardSplit().folds(X.index)
    last = folds[-1]
    test_dates = X.index[last.test_idx]

    before = FL.fit_filter_folds_generic(X, orb, factory, "xgboost", y_col=y_col)
    before_choice = before[before["fold"] == last.number][
        ["threshold", "retention_selected"]].iloc[0]

    X2 = X.copy()
    X2.loc[test_dates, list(OPEN_FEATURE_COLUMNS)] += 1000.0
    orb2 = orb.copy()
    orb2.loc[test_dates, "bps_return"] = 999.0
    orb2.loc[test_dates, "r_net"] = 999.0

    after = FL.fit_filter_folds_generic(X2, orb2, factory, "xgboost", y_col=y_col)
    after_choice = after[after["fold"] == last.number][
        ["threshold", "retention_selected"]].iloc[0]

    assert before_choice["threshold"] == pytest.approx(after_choice["threshold"])
    assert before_choice["retention_selected"] == after_choice["retention_selected"]


def test_fold_overfitting_table_reports_fit_and_test_auc(data):
    X, orb = data
    out = FL.fold_overfitting_table(X, orb, FL.make_logistic, "logistic")
    assert not out.empty
    assert {"fold", "model", "fit_auc", "test_auc"} <= set(out.columns)
    assert out["fit_auc"].between(0, 1).all()
    assert out["test_auc"].between(0, 1).all()
    # the planted signal is genuinely linear/simple, so fit and test AUC
    # shouldn't be wildly different for logistic on this synthetic fixture
    assert (out["fit_auc"] - out["test_auc"]).abs().median() < 0.25


def test_permutation_importance_table_ranks_the_planted_signal_highest(data):
    X, orb = data
    out = FL.permutation_importance_table(X, orb, FL.make_logistic, "logistic",
                                          n_repeats=5)
    assert not out.empty
    assert set(OPEN_FEATURE_COLUMNS) <= set(out.columns)
    mean_importance = out[list(OPEN_FEATURE_COLUMNS)].mean()
    assert mean_importance.idxmax() == "open5m_body_ratio"
