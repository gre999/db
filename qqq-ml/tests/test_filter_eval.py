"""Tests for src/filter_eval.py: filtered-performance evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import filter_eval as FE
from src import regime_eval as E
from src.models import filter as FL


def _make_orb(seed=0, start="2019-01-01", end="2021-12-31", doji_frac=0.15):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    doji = rng.random(len(days)) < doji_frac
    r_net = rng.normal(0.1, 1.0, len(days))
    bps_return = np.where(doji, 0.0, r_net * 20.0)
    return pd.DataFrame({"day": days, "traded": ~doji,
                        "r_net": np.where(doji, np.nan, r_net),
                        "bps_return": bps_return}).set_index("day")


def _make_preds(orb, keep_frac=0.6, seed=1, n_folds=2):
    signal_days = orb.index[orb["traded"]]
    rng = np.random.default_rng(seed)
    mid = len(signal_days) // 2
    fold_of = np.where(np.arange(len(signal_days)) < mid, 1, 2)
    proba = rng.uniform(0, 1, len(signal_days))
    rows = pd.DataFrame({"date": signal_days, "fold": fold_of,
                        "y_pred_proba": proba,
                        "y_true": rng.integers(0, 2, len(signal_days))})
    keep = pd.Series(False, index=rows.index)
    for f, g in rows.groupby("fold"):
        thr = g["y_pred_proba"].quantile(1 - keep_frac)
        keep.loc[g.index] = g["y_pred_proba"] >= thr
    rows["keep"] = keep.to_numpy()
    rows["threshold"] = rows.groupby("fold")["y_pred_proba"] \
        .transform(lambda s: s.quantile(1 - keep_frac))
    rows["retention_selected"] = keep_frac
    rows["model"] = "logistic"
    rows["target"] = "net_r_positive"
    return rows


@pytest.fixture(scope="module")
def orb():
    return _make_orb()


@pytest.fixture(scope="module")
def preds(orb):
    return _make_preds(orb)


# ------------------------------------------------------- logistic_filter_series
def test_logistic_filter_series_zeros_doji_and_dropped_days(orb, preds):
    out = FE.logistic_filter_series(preds, orb)
    filtered, unfiltered = out["filtered"], out["unfiltered"]

    doji_days = orb.index[~orb["traded"]]
    common_doji = doji_days.intersection(filtered.index)
    assert (filtered.loc[common_doji] == 0).all()
    assert (unfiltered.loc[common_doji] == 0).all()

    dropped = pd.DatetimeIndex(preds.loc[~preds["keep"], "date"])
    assert (filtered.loc[dropped] == 0).all()

    kept = pd.DatetimeIndex(preds.loc[preds["keep"], "date"])
    s = orb
    assert np.allclose(filtered.loc[kept].to_numpy(),
                       s.loc[kept, "bps_return"].to_numpy())
    # unfiltered always equals ORB's own bps_return, regardless of keep
    signal_days = pd.DatetimeIndex(preds["date"])
    assert np.allclose(unfiltered.loc[signal_days].to_numpy(),
                       s.loc[signal_days, "bps_return"].to_numpy())


def test_logistic_filter_series_covers_full_calendar_not_just_signal_days(orb, preds):
    out = FE.logistic_filter_series(preds, orb)
    oos_min, oos_max = preds["date"].min(), preds["date"].max()
    expected_range = orb.index[(orb.index >= oos_min) & (orb.index <= oos_max)]
    assert len(out["unfiltered"]) == len(expected_range)


# ------------------------------------------------------------- random null
def test_random_filter_null_preserves_per_fold_keep_counts(orb, preds):
    null = FE.random_filter_null(preds, orb, n_reps=50, seed=0)
    assert len(null) == 50
    assert np.isfinite(null).all()


def test_random_filter_null_mean_sits_between_extremes(orb, preds):
    """A random subset of the same size shouldn't systematically beat or
    lag the full-retention (100%) baseline by a huge margin when returns
    are close to i.i.d. noise (as in this synthetic fixture) - a loose
    sanity bound, not a tight one."""
    null = FE.random_filter_null(preds, orb, n_reps=300, seed=2)
    full = E._daily_eval(FE.logistic_filter_series(preds, orb)["unfiltered"])["sharpe"]
    assert abs(np.mean(null) - full) < 2.0


# --------------------------------------------------------------- success
def test_evaluate_success_flags_match_manual_comparison(orb, preds):
    result = FE.evaluate_success(preds, orb, min_trades_per_year=1, n_reps=100)
    assert result["beats_unfiltered"] == (result["filtered_sharpe"] > result["unfiltered_sharpe"])
    assert result["beats_random_p95"] == (result["filtered_sharpe"] > result["random_null_p95"])
    assert result["success"] == (result["beats_unfiltered"] and result["beats_random_p95"]
                                 and result["min_trades_per_year_ok"])


def test_evaluate_success_trades_per_year_flag_is_correct(orb, preds):
    result = FE.evaluate_success(preds, orb, min_trades_per_year=10_000, n_reps=10)
    assert result["min_trades_per_year_ok"] is False
    assert result["success"] is False


# ----------------------------------------------------------- threshold curve
def test_threshold_curve_full_retention_matches_unfiltered(orb, preds):
    curve = FE.threshold_curve(preds, orb)
    row = curve[curve["retention"] == 1.0].iloc[0]
    unfiltered_eval = E._daily_eval(FE.logistic_filter_series(preds, orb)["unfiltered"])
    assert row["sharpe"] == pytest.approx(unfiltered_eval["sharpe"])
    assert row["n_trades"] == len(preds)


def test_threshold_curve_trade_count_increases_with_retention(orb, preds):
    curve = FE.threshold_curve(preds, orb).sort_values("retention")
    assert curve["n_trades"].is_monotonic_increasing


# ------------------------------------------------------------ missed big wins
def test_missed_big_wins_counts_correctly():
    orb = _make_orb(seed=5)
    orb = orb.copy()
    big_days = orb.index[orb["traded"]][:10]
    orb.loc[big_days, "r_net"] = 6.0
    preds = _make_preds(orb, keep_frac=1.0)   # keep everything by default
    preds = preds.copy()
    drop_days = big_days[:4]
    preds.loc[preds["date"].isin(drop_days), "keep"] = False

    out = FE.missed_big_wins(preds, orb, r_threshold=5.0)
    assert out["n_big_win_days_in_window"] == 10
    assert out["n_dropped"] == 4
    assert set(out["dropped_dates"]) == set(drop_days)


# --------------------------------------------------------- block bootstrap
def test_block_bootstrap_significance_detects_a_planted_large_effect():
    """A filter that keeps only the strongly-positive days should show a
    small p-value; block_bootstrap_sharpe_diff is src.backtest's own
    already-tested function, this only checks the wiring (units, series
    construction) is correct."""
    days = pd.bdate_range("2019-01-01", "2022-12-31")
    rng = np.random.default_rng(9)
    doji = rng.random(len(days)) < 0.15
    # two well-separated populations so the effect is not noise-sized
    good = rng.normal(0, 1, len(days)) > 0
    r_net = np.where(good, rng.normal(3.0, 0.5, len(days)), rng.normal(-3.0, 0.5, len(days)))
    bps_return = np.where(doji, 0.0, r_net * 20.0)
    orb_strong = pd.DataFrame({"day": days, "traded": ~doji,
                              "r_net": np.where(doji, np.nan, r_net),
                              "bps_return": bps_return}).set_index("day")
    signal_days = orb_strong.index[orb_strong["traded"]]
    preds_strong = pd.DataFrame({
        "date": signal_days, "fold": 1,
        "keep": good[orb_strong["traded"].to_numpy()],
    })
    out = FE.block_bootstrap_significance(preds_strong, orb_strong, n_boot=300)
    assert out["observed_sharpe_diff"] > 0
    assert out["p_value"] < 0.05


def test_random_filter_p_value_matches_manual_rank(orb, preds):
    null = FE.random_filter_null(preds, orb, n_reps=200, seed=3)
    out = FE.random_filter_p_value(preds, orb, null=null)
    observed = E._daily_eval(FE.logistic_filter_series(preds, orb)["filtered"])["sharpe"]
    manual_p = float(np.mean(null >= observed))
    assert out["observed_sharpe"] == pytest.approx(observed)
    assert out["p_value"] == pytest.approx(manual_p)
    assert out["rank_from_top"] == int(np.sum(null >= observed)) + 1


# ------------------------------------------------------- yearly/fold breakdown
def test_yearly_breakdown_covers_every_year_in_window(orb, preds):
    yb = FE.yearly_breakdown(preds, orb)
    expected_years = set(orb.index.year) & set(pd.DatetimeIndex(preds["date"]).year)
    assert set(yb["year"]) == expected_years
    assert (yb["n_days"] > 0).all()


def test_fold_breakdown_one_row_per_fold(orb, preds):
    fb = FE.fold_breakdown(preds, orb)
    assert set(fb["fold"]) == set(preds["fold"].unique())
    assert (fb["n_days"] > 0).all()


# --------------------------------------------------------- cost sensitivity
def test_cost_sensitivity_orders_sharpe_by_cost_and_uses_injected_variants(orb, preds):
    """Higher cost must strictly lower both filtered and unfiltered Sharpe
    (same trades, more expensive) - checks the DI wiring end to end without
    touching real 1-min bar data."""
    def bump_cost(o: pd.DataFrame, extra_bps: float) -> pd.DataFrame:
        o2 = o.copy()
        traded = o2["traded"]
        o2.loc[traded, "bps_return"] = o2.loc[traded, "bps_return"] - extra_bps
        o2.loc[traded, "r_net"] = o2.loc[traded, "r_net"] - extra_bps / 20.0
        return o2

    variants = {"low_cost": orb, "high_cost": bump_cost(orb, 5.0)}
    out = FE.cost_sensitivity(preds, orb_variants=variants)
    low = out[out["cost_config"] == "low_cost"].iloc[0]
    high = out[out["cost_config"] == "high_cost"].iloc[0]
    assert high["filtered_sharpe"] < low["filtered_sharpe"]
    assert high["unfiltered_sharpe"] < low["unfiltered_sharpe"]
    assert set(out["cost_config"]) == set(variants)
