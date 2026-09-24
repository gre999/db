"""Week 7 state predictability: agreement/kappa, duration, transition matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import regime_eval as E


def _labels(dates, states, fold=1, model="hmm", k=3):
    return pd.DataFrame({"date": dates, "fold": fold, "model": model, "k": k,
                        "state": states, "prob": 1.0})


def test_state_agreement_perfect_match():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(0)
    states = rng.integers(0, 3, 100)
    pred = _labels(idx, states)
    desc = _labels(idx, states)
    res = E.state_agreement(pred, desc, "hmm")
    assert res["observed_agreement"] == pytest.approx(1.0)
    assert res["kappa"] == pytest.approx(1.0)


def test_state_agreement_independent_labels_near_chance():
    idx = pd.bdate_range("2020-01-01", periods=20000)
    rng = np.random.default_rng(1)
    pred = _labels(idx, rng.integers(0, 3, len(idx)))
    desc = _labels(idx, rng.integers(0, 3, len(idx)))
    res = E.state_agreement(pred, desc, "hmm")
    assert res["kappa"] == pytest.approx(0.0, abs=0.03)
    assert res["observed_agreement"] == pytest.approx(res["expected_marginal"], abs=0.02)


def test_state_agreement_uses_inner_join_on_date():
    idx1 = pd.bdate_range("2020-01-01", periods=10)
    idx2 = pd.bdate_range("2020-01-06", periods=10)   # partial overlap
    pred = _labels(idx1, [0] * 10)
    desc = _labels(idx2, [0] * 10)
    res = E.state_agreement(pred, desc, "hmm")
    assert res["n"] == len(idx1.intersection(idx2))


def test_state_durations_counts_consecutive_runs():
    idx = pd.bdate_range("2020-01-01", periods=7)
    states = [0, 0, 0, 1, 1, 0, 0]
    labels = _labels(idx, states)
    out = E.state_durations(labels, "hmm").set_index("state")
    assert out.loc[0, "n_spells"] == 2
    assert out.loc[0, "mean_days"] == pytest.approx((3 + 2) / 2)
    assert out.loc[1, "n_spells"] == 1
    assert out.loc[1, "mean_days"] == pytest.approx(2)


def test_empirical_transition_matrix_correctness():
    idx = pd.bdate_range("2020-01-01", periods=6)
    states = [0, 0, 1, 1, 1, 0]
    labels = _labels(idx, states)
    tm = E.empirical_transition_matrix(labels, "hmm")
    # transitions observed: 0->0, 0->1, 1->1, 1->1, 1->0 (5 total)
    assert tm.loc[0, 0] == pytest.approx(1 / 2)   # from state0 (2 obs): 1 stay, 1 leave
    assert tm.loc[0, 1] == pytest.approx(1 / 2)
    assert tm.loc[1, 1] == pytest.approx(2 / 3)   # from state1 (3 obs): 2 stay, 1 leave
    assert tm.loc[1, 0] == pytest.approx(1 / 3)


def _fake_predictions(idx):
    rng = np.random.default_rng(7)
    return pd.DataFrame({"date": idx, "y_pred_var": rng.uniform(1e-5, 5e-5, len(idx))})


def _fold_windows(test_starts_ends):
    return pd.DataFrame([{"fold": i + 1, "test_start": s, "test_end": e}
                        for i, (s, e) in enumerate(test_starts_ends)])


def test_vol_tercile_labels_uses_expanding_window_before_test_start():
    idx = pd.bdate_range("2015-01-01", "2019-12-31")
    pred = _fake_predictions(idx)
    fw = _fold_windows([("2018-01-01", "2018-12-31"), ("2019-01-01", "2019-12-31")])
    labels1 = E.vol_tercile_labels(fw, pred)

    # perturbing fold 2's test window must not move fold 1's labels/cutoffs
    pred2 = pred.set_index("date")
    fold2_dates = pred2.loc["2019-01-01":"2019-12-31"].index
    rng = np.random.default_rng(8)
    pred2.loc[fold2_dates, "y_pred_var"] = rng.uniform(100, 200, len(fold2_dates))
    labels2 = E.vol_tercile_labels(fw, pred2.reset_index())

    l1 = labels1[labels1["fold"] == 1].sort_values("date")
    l2 = labels2[labels2["fold"] == 1].sort_values("date")
    np.testing.assert_array_equal(l1["state"].to_numpy(), l2["state"].to_numpy())


def test_vol_tercile_labels_drops_fold_with_no_prior_data():
    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    pred = _fake_predictions(idx)
    # fold 1's test_start is the very first date in `pred` - no prior data
    fw = _fold_windows([("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31")])
    labels = E.vol_tercile_labels(fw, pred, min_train_days=50)
    assert 1 not in set(labels["fold"])
    assert 2 in set(labels["fold"])


def test_vol_tercile_labels_roughly_balanced():
    idx = pd.bdate_range("2015-01-01", "2022-12-31")
    pred = _fake_predictions(idx)
    fw = _fold_windows([("2018-01-01", "2018-12-31"), ("2019-01-01", "2019-12-31"),
                       ("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31"),
                       ("2022-01-01", "2022-12-31")])
    labels = E.vol_tercile_labels(fw, pred)
    counts = labels["state"].value_counts(normalize=True)
    for s in (0, 1, 2):
        assert counts.get(s, 0) == pytest.approx(1 / 3, abs=0.1)


def test_label_autocorrelation_matches_manual_calc():
    idx = pd.bdate_range("2020-01-01", periods=6)
    states = [0, 0, 1, 1, 1, 0]
    labels = _labels(idx, states)
    res = E.label_autocorrelation(labels, "hmm")
    # pairs (t-1,t): (0,0),(0,1),(1,1),(1,1),(1,0) -> 3/5 agree
    assert res["n"] == 5
    assert res["observed_agreement"] == pytest.approx(3 / 5)


def test_label_autocorrelation_high_for_persistent_sequence():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(3)
    states, cur = [], 0
    for _ in range(200):
        if rng.random() < 0.05:            # rare switches -> persistent
            cur = rng.integers(0, 3)
        states.append(cur)
    labels = _labels(idx, states)
    res = E.label_autocorrelation(labels, "hmm")
    assert res["kappa"] > 0.7


def test_external_validation_recovers_known_state_means():
    idx = pd.bdate_range("2020-01-01", periods=3000)
    rng = np.random.default_rng(4)
    states = rng.integers(0, 3, len(idx))
    true_means = {0: 0.0, 1: 1.0, 2: 2.0}
    y = np.array([true_means[s] for s in states]) + rng.normal(0, 0.5, len(idx))
    labels = _labels(idx, states)
    desc = pd.DataFrame({"y": y}, index=idx)

    out = E.external_validation(labels, desc, "hmm", "y").set_index("state")
    assert out.loc[0, "mean"] == pytest.approx(0.0, abs=0.1)
    assert out.loc[1, "diff_vs_state0"] == pytest.approx(1.0, abs=0.15)
    assert out.loc[2, "diff_vs_state0"] == pytest.approx(2.0, abs=0.15)
    assert out.loc[1, "p"] < 0.001
    assert out.loc[2, "p"] < 0.001


def test_external_validation_null_case_not_significant():
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(5)
    states = rng.integers(0, 3, len(idx))
    y = rng.normal(0, 1, len(idx))            # independent of state
    labels = _labels(idx, states)
    desc = pd.DataFrame({"y": y}, index=idx)

    out = E.external_validation(labels, desc, "hmm", "y").set_index("state")
    assert out.loc[1, "p"] > 0.05
    assert out.loc[2, "p"] > 0.05


def test_external_validation_only_uses_overlapping_dates():
    idx1 = pd.bdate_range("2020-01-01", periods=10)
    idx2 = pd.bdate_range("2020-01-06", periods=10)
    labels = _labels(idx1, [0, 1] * 5)
    desc = pd.DataFrame({"y": range(10)}, index=idx2)
    out = E.external_validation(labels, desc, "hmm", "y")
    assert out.loc[0, "n"] + out.loc[1, "n"] == len(idx1.intersection(idx2))


def _orb_strategy(dates, traded, r_net, bps):
    return pd.DataFrame({"day": dates, "traded": traded,
                        "r_gross": r_net, "r_net": r_net, "bps_return": bps})


def test_state_performance_table_basic_counts_and_all_row():
    idx = pd.bdate_range("2020-01-01", periods=8)
    states = [0, 0, 0, 0, 1, 1, 1, 1]
    labels = _labels(idx, states)
    traded = [True, True, False, True, True, True, True, False]
    r_net = [1.0, -1.0, np.nan, 2.0, 0.5, -0.5, 3.0, np.nan]
    bps = [10, -10, 0, 20, 5, -5, 30, 0]
    strat = _orb_strategy(idx, traded, r_net, bps)

    out = E.state_performance_table(labels, strat, "hmm", "orb",
                                    min_trades_per_year=0).set_index("state")
    assert out.loc[0, "n_days"] == 4
    assert out.loc[0, "n_traded"] == 3
    assert out.loc[0, "mean_bps"] == pytest.approx(np.mean([10, -10, 0, 20]))
    assert out.loc[0, "win_rate"] == pytest.approx(2 / 3)   # days 0,3 positive of 3 traded
    assert out.loc["all", "n_days"] == 8
    assert out.loc["all", "n_traded"] == 6


def test_state_performance_table_orb_and_vwap_extra_columns():
    idx = pd.bdate_range("2020-01-01", periods=4)
    labels = _labels(idx, [0, 0, 1, 1])

    orb = _orb_strategy(idx, [True] * 4, [1.0, 2.0, -1.0, 0.5], [10, 20, -10, 5])
    out_orb = E.state_performance_table(labels, orb, "hmm", "orb",
                                        min_trades_per_year=0)
    assert {"mean_r_gross", "mean_r_net"} <= set(out_orb.columns)
    assert "mean_n_segments" not in out_orb.columns

    vwap = pd.DataFrame({"day": idx, "traded": [True] * 4,
                        "n_segments": [10, 20, 5, 8],
                        "cost": [0.1, 0.2, 0.05, 0.08],
                        "first_entry_price": [100.0] * 4,
                        "bps_return": [10, 20, -10, 5]})
    out_vwap = E.state_performance_table(labels, vwap, "hmm", "vwap",
                                         min_trades_per_year=0)
    assert {"mean_n_segments", "mean_cost_bps"} <= set(out_vwap.columns)
    row0 = out_vwap.set_index("state").loc[0]
    assert row0["mean_n_segments"] == pytest.approx(15)
    assert row0["mean_cost_bps"] == pytest.approx(
        np.mean([0.1 / 100 * 1e4, 0.2 / 100 * 1e4]))


def test_flag_insufficient_trades_per_year():
    idx = pd.bdate_range("2020-01-01", periods=10)
    labels = _labels(idx, [0] * 10)
    strat = _orb_strategy(idx, [True] * 3 + [False] * 7,
                          [1.0, 1.0, 1.0] + [np.nan] * 7,
                          [1] * 3 + [0] * 7)
    out = E.state_performance_table(labels, strat, "hmm", "orb",
                                    min_trades_per_year=20).set_index("state")
    assert out.loc[0, "flag_insufficient"]      # only 3 trades in 1 year, < 20
    out2 = E.state_performance_table(labels, strat, "hmm", "orb",
                                     min_trades_per_year=2)
    assert not out2.set_index("state").loc[0, "flag_insufficient"]


def test_state_performance_by_year_and_fold_have_expected_shape():
    idx = pd.bdate_range("2020-06-01", "2021-06-30", freq="B")
    n = len(idx)
    states = [0 if i % 2 == 0 else 1 for i in range(n)]
    fold = [1 if d.year == 2020 else 2 for d in idx]
    labels = pd.DataFrame({"date": idx, "fold": fold, "model": "hmm", "k": 2,
                          "state": states, "prob": 1.0})
    strat = _orb_strategy(idx, [True] * n, [0.5] * n, [5] * n)

    by_year = E.state_performance_by_year(labels, strat, "hmm", "orb",
                                          min_trades_per_year=0)
    assert set(by_year["year"]) == {2020, 2021}
    assert set(by_year["state"]) == {0, 1}

    by_fold = E.state_performance_by_fold(labels, strat, "hmm", "orb",
                                          min_trades_per_year=0)
    assert set(by_fold["fold"]) == {1, 2}


def test_state_vs_full_sample_test_recovers_known_effect_and_matches_manual_calc():
    idx = pd.bdate_range("2020-01-01", periods=3000)
    rng = np.random.default_rng(6)
    states = rng.integers(0, 3, len(idx))
    true_bps = {0: -5.0, 1: 0.0, 2: 8.0}
    bps = np.array([true_bps[s] for s in states]) + rng.normal(0, 10, len(idx))
    labels = _labels(idx, states)
    strat = pd.DataFrame({"day": idx, "traded": True, "bps_return": bps})

    out = E.state_vs_full_sample_test(labels, strat, "hmm").set_index("state")
    full_mean = bps.mean()
    for s in (0, 1, 2):
        manual = bps[states == s].mean() - full_mean
        assert out.loc[s, "diff_vs_full_sample"] == pytest.approx(manual, abs=1e-8)
    assert out.loc[0, "p"] < 0.01
    assert out.loc[2, "p"] < 0.01


def test_state_vs_full_sample_test_null_case_not_significant():
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(7)
    states = rng.integers(0, 3, len(idx))
    bps = rng.normal(0, 10, len(idx))       # independent of state
    labels = _labels(idx, states)
    strat = pd.DataFrame({"day": idx, "traded": True, "bps_return": bps})
    out = E.state_vs_full_sample_test(labels, strat, "hmm").set_index("state")
    assert (out["p"] > 0.05).all()


def test_label_block_shuffle_null_detects_real_effect():
    idx = pd.bdate_range("2020-01-01", periods=1500)
    rng = np.random.default_rng(8)
    states = rng.integers(0, 2, len(idx))
    bps = np.where(states == 1, 15.0, -15.0) + rng.normal(0, 5, len(idx))
    labels = _labels(idx, states, k=2)
    strat = pd.DataFrame({"day": idx, "traded": True, "bps_return": bps})
    out = E.label_block_shuffle_null(labels, strat, "hmm", block_size=10,
                                     n_reps=300, seed=1).set_index("state")
    assert out.loc[1, "p_vs_random_labeling"] < 0.01
    assert out.loc[0, "p_vs_random_labeling"] < 0.01


def test_label_block_shuffle_null_no_effect_is_not_tiny_p():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    rng = np.random.default_rng(9)
    states = rng.integers(0, 2, len(idx))
    bps = rng.normal(0, 10, len(idx))
    labels = _labels(idx, states, k=2)
    strat = pd.DataFrame({"day": idx, "traded": True, "bps_return": bps})
    out = E.label_block_shuffle_null(labels, strat, "hmm", block_size=10,
                                     n_reps=300, seed=2).set_index("state")
    assert (out["p_vs_random_labeling"] > 0.1).all()


def test_daily_eval_matches_manual_calc():
    ret_bps = pd.Series([100.0, -50.0, 100.0, -50.0])   # 1%, -0.5%, ...
    out = E._daily_eval(ret_bps)
    r = ret_bps.to_numpy() / 1e4
    equity = np.cumprod(1 + r)
    assert out["ann_return"] == pytest.approx(equity[-1] ** (252 / 4) - 1)
    assert out["max_drawdown"] == pytest.approx(
        (equity / np.maximum.accumulate(equity) - 1).min())


def test_positive_states_from_training():
    states = np.array([0, 0, 0, 1, 1, 1])
    bps = np.array([-1.0, -2.0, 1.0, 1.0, 2.0, -0.5])   # state0 mean<0, state1 mean>0
    pos = E._positive_states_from_training(states, bps)
    assert pos == {1}


def test_tradability_test_only_trades_training_positive_states():
    idx = pd.bdate_range("2015-01-01", "2023-12-31")
    rng = np.random.default_rng(11)
    n = len(idx)
    regime = rng.integers(0, 2, n)
    log_rv = np.where(regime == 0, rng.normal(-10.0, 0.3, n),
                      rng.normal(-8.0, 0.3, n))
    X = pd.DataFrame({
        "har_logrv_1d": log_rv, "close_loc_1d": rng.uniform(0, 1, n),
        "volume_rel_20d": rng.normal(1.0, 0.2, n),
        "overnight_gap_1d": rng.normal(0.0, 0.005, n),
    }, index=idx)
    # bps_return is strongly tied to the SAME regime driving log_rv, so the
    # fitted HMM's low-vol state should come out training-positive and the
    # high-vol state training-negative (or vice versa) - deterministically
    # one side, not a coin flip
    bps = np.where(regime == 0, 20.0, -20.0) + rng.normal(0, 5, n)
    strat = pd.DataFrame({"day": idx, "traded": True, "bps_return": bps})

    out = E.tradability_test(X, strat, "hmm", k_choices=(2, 3), n_init=3)
    assert out["eval_filtered"]["n"] <= out["eval_unfiltered"]["n"]
    assert set(out["filtered"].index) == set(out["unfiltered"].index)
    # every fold with a decision made a nonempty choice one way or another
    assert len(out["positive_states_by_fold"]) > 0


def test_tradability_test_vol_tercile_matches_vol_tercile_labels_folds():
    idx = pd.bdate_range("2015-01-01", "2022-12-31")
    pred = _fake_predictions(idx)
    fw = _fold_windows([("2018-01-01", "2018-12-31"), ("2019-01-01", "2019-12-31"),
                       ("2020-01-01", "2020-12-31")])
    rng = np.random.default_rng(12)
    strat = pd.DataFrame({"day": idx, "traded": True,
                         "bps_return": rng.normal(0, 10, len(idx))})
    out = E.tradability_test_vol_tercile(strat, fw, pred)
    assert set(out["positive_states_by_fold"]).issubset({1, 2, 3})
    assert out["eval_filtered"]["n"] <= out["eval_unfiltered"]["n"]


def test_tradability_common_dates_restricts_to_shared_range_and_matches_manual_eval():
    idx_wide = pd.bdate_range("2019-01-01", "2019-01-31")   # 23 sessions
    idx_narrow = idx_wide[5:]                                # missing first 5
    ret = pd.Series(np.arange(len(idx_wide), dtype=float), index=idx_wide)

    results = {
        "wide": {"filtered": ret, "unfiltered": ret},
        "narrow": {"filtered": ret.loc[idx_narrow], "unfiltered": ret.loc[idx_narrow]},
    }
    common, tbl = E.tradability_common_dates(results)
    assert list(common) == list(idx_narrow)
    assert len(common) == len(idx_wide) - 5

    manual = E._daily_eval(ret.loc[idx_narrow])
    for _, row in tbl.iterrows():
        assert row["unfiltered_n"] == manual["n"]
        assert row["unfiltered_sharpe"] == pytest.approx(manual["sharpe"])
        assert row["filtered_sharpe"] == pytest.approx(manual["sharpe"])
    # the "wide" source's own un-restricted eval used a longer series, so
    # restricting it must actually change its number, not just relabel it
    wide_full_eval = E._daily_eval(ret)
    wide_row = tbl[tbl["model"] == "wide"].iloc[0]
    assert wide_row["unfiltered_n"] != wide_full_eval["n"]


def test_empirical_transition_matrix_ignores_fold_boundary():
    """states = [0, 1, 1, 0], folds = [1, 1, 2, 2]: the boundary step
    (row1->row2) would be a spurious (1->1) "stay" if counted - it must be
    dropped, leaving state 1 with a single counted transition, (1->0)."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    labels = _labels(idx, [0, 1, 1, 0])
    labels.loc[2:, "fold"] = 2
    tm = E.empirical_transition_matrix(labels, "hmm")
    assert tm.loc[0, 1] == pytest.approx(1.0)      # row0(f1) -> row1(f1)
    assert tm.loc[1, 0] == pytest.approx(1.0)      # row2(f2) -> row3(f2)
    assert tm.loc[1, 1] == pytest.approx(0.0)      # the boundary step, excluded
