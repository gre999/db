"""Regime clustering: fold-only fitting, causal HMM filtering, relabeling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from src.models import regimes as G
from src.validation import WalkForwardSplit


@pytest.fixture(scope="module")
def synth_X() -> pd.DataFrame:
    idx = pd.bdate_range("2015-01-01", "2023-12-31")
    rng = np.random.default_rng(0)
    n = len(idx)
    regime = rng.integers(0, 2, n)
    log_rv = np.where(regime == 0, rng.normal(-10.0, 0.3, n),
                      rng.normal(-8.0, 0.3, n))
    X = pd.DataFrame({
        "har_logrv_1d": log_rv,
        "close_loc_1d": rng.uniform(0, 1, n),
        "volume_rel_20d": rng.normal(1.0, 0.2, n),
        "overnight_gap_1d": rng.normal(0.0, 0.005, n),
    }, index=idx)
    return X


def _toy_hmm() -> GaussianHMM:
    m = GaussianHMM(n_components=2, covariance_type="diag")
    m.startprob_ = np.array([0.5, 0.5])
    m.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    m.means_ = np.array([[0.0], [5.0]])
    m.covars_ = np.array([[1.0], [1.0]])
    m.n_features = 1
    return m


# --------------------------------------------------------------------------
# Relabeling
# --------------------------------------------------------------------------
def test_relabel_by_vol_orders_states_ascending():
    raw = np.array([0, 0, 1, 1, 2, 2])
    vol = np.array([5.0, 5.0, 1.0, 1.0, 3.0, 3.0])
    mapping = G.relabel_by_vol(raw, vol, 3)
    assert mapping[1] == 0     # lowest vol -> rank 0
    assert mapping[2] == 1
    assert mapping[0] == 2     # highest vol -> rank k-1


def test_reorder_prob_columns_matches_label_mapping():
    mapping = np.array([2, 0, 1])          # old->new
    probs = np.array([[0.7, 0.2, 0.1]])    # columns = old states 0,1,2
    out = G.reorder_prob_columns(probs, mapping)
    # new state 0 came from old state 1 (mapping[1]==0) -> prob 0.2
    # new state 1 came from old state 2 (mapping[2]==1) -> prob 0.1
    # new state 2 came from old state 0 (mapping[0]==2) -> prob 0.7
    np.testing.assert_allclose(out, [[0.2, 0.1, 0.7]])


# --------------------------------------------------------------------------
# Forward filter: correctness and causality
# --------------------------------------------------------------------------
def test_forward_filter_matches_independent_reference_implementation():
    model = _toy_hmm()
    X = np.array([[0.1], [0.2], [4.8], [4.5], [0.3]])

    B = np.array([[norm.pdf(x[0], model.means_[s, 0], 1.0) for s in range(2)]
                 for x in X])
    ref = np.zeros((5, 2))
    a = model.startprob_ * B[0]
    ref[0] = a / a.sum()
    for t in range(1, 5):
        a = (ref[t - 1] @ model.transmat_) * B[t]
        ref[t] = a / a.sum()

    alpha, final = G.forward_filter(model, X)
    np.testing.assert_allclose(alpha, ref, atol=1e-10)
    np.testing.assert_allclose(final, ref[-1], atol=1e-10)


def test_forward_filter_is_causal_not_smoothing():
    """The defining property this module needs from the HMM stage: filtered
    probability at t must not depend on observations after t (unlike
    hmmlearn's own .predict()/.predict_proba(), which are Viterbi /
    forward-backward and do look ahead)."""
    model = _toy_hmm()
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (20, 1))
    alpha1, _ = G.forward_filter(model, X)

    X2 = X.copy()
    X2[15:] = 999.0
    alpha2, _ = G.forward_filter(model, X2)
    np.testing.assert_allclose(alpha1[:15], alpha2[:15])
    assert not np.allclose(alpha1[15:], alpha2[15:])   # later days DO see it


def test_forward_filter_init_dist_seeds_without_lookahead():
    """Continuing a sequence with init_dist=<prior period's final belief>
    must reproduce exactly what a single unbroken forward pass would give -
    i.e. splitting train/test at the fold boundary costs nothing."""
    model = _toy_hmm()
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (30, 1))
    whole, _ = G.forward_filter(model, X)

    part1, final1 = G.forward_filter(model, X[:12])
    part2, _ = G.forward_filter(model, X[12:], init_dist=final1)
    np.testing.assert_allclose(whole[:12], part1, atol=1e-10)
    np.testing.assert_allclose(whole[12:], part2, atol=1e-10)


# --------------------------------------------------------------------------
# Fold isolation (fit uses the training window only)
# --------------------------------------------------------------------------
def test_run_test_slice_never_affects_fold_fit_or_labels(synth_X):
    """Perturbing the LAST fold's test slice must not move any EARLIER
    fold's fit or labels. (Perturbing an early fold's test slice is not a
    valid probe here: in a rolling walk-forward split, an early fold's test
    window legitimately becomes part of a LATER fold's training window, so
    it's supposed to move later folds - that's not leakage.)"""
    out1 = G.run(X=synth_X, save=False)
    folds = WalkForwardSplit().folds(synth_X.index)
    last = folds[-1]
    te_last = synth_X.index[last.test_idx]

    X2 = synth_X.copy()
    rng = np.random.default_rng(3)
    X2.loc[te_last, :] = rng.normal(size=(len(te_last), X2.shape[1]))
    out2 = G.run(X=X2, save=False)

    earlier_folds = [f.number for f in folds[:-1]]
    sel1 = out1["selection"][out1["selection"]["fold"].isin(earlier_folds)] \
        .reset_index(drop=True)
    sel2 = out2["selection"][out2["selection"]["fold"].isin(earlier_folds)] \
        .reset_index(drop=True)
    pd.testing.assert_frame_equal(sel1, sel2)

    l1 = out1["labels"][out1["labels"]["fold"].isin(earlier_folds)] \
        .sort_values(["model", "date"])
    l2 = out2["labels"][out2["labels"]["fold"].isin(earlier_folds)] \
        .sort_values(["model", "date"])
    np.testing.assert_array_equal(l1["state"].to_numpy(), l2["state"].to_numpy())


def test_truncation_labels_and_probs_unchanged_before_cutoff(synth_X):
    """Delete everything after a cutoff inside fold 1's test window; every
    date at or before the cutoff must keep the same state and probability -
    the day-level analogue of test_features.py's truncation check."""
    out_full = G.run(X=synth_X, save=False)
    f0 = WalkForwardSplit().folds(synth_X.index)[0]
    te0 = synth_X.index[f0.test_idx]
    cutoff = te0[100]                      # well inside the test window

    out_trunc = G.run(X=synth_X.loc[:cutoff], save=False)

    l_full = out_full["labels"]
    l_trunc = out_trunc["labels"]
    m = l_full.merge(l_trunc, on=["date", "model"], suffixes=("_full", "_trunc"))
    assert len(m) == len(l_trunc)          # every truncated-run row matched
    assert (m["state_full"] == m["state_trunc"]).all()
    hmm = m[m["model"] == "hmm"]
    np.testing.assert_allclose(hmm["prob_full"], hmm["prob_trunc"], atol=1e-8)


def test_standardization_fit_on_training_window_only(synth_X):
    f0 = WalkForwardSplit().folds(synth_X.index)[0]
    Xtr = synth_X.iloc[f0.train_idx]
    km = G.fit_kmeans_fold(Xtr)
    hmm = G.fit_hmm_fold(Xtr)
    np.testing.assert_allclose(km["scaler"].mean_, Xtr.mean().to_numpy())
    np.testing.assert_allclose(hmm["scaler"].mean_, Xtr.mean().to_numpy())


# --------------------------------------------------------------------------
# Week 7: coverage-based k selection (main analysis config)
# --------------------------------------------------------------------------
def test_state_coverage_sums_to_one():
    states = np.array([0, 0, 0, 1, 1, 2])
    cov = G.state_coverage(states, 3)
    np.testing.assert_allclose(cov, [0.5, 1 / 3, 1 / 6])
    assert cov.sum() == pytest.approx(1.0)


def test_select_k_by_coverage_picks_largest_qualifying_k():
    # k=2: balanced (qualifies); k=3: one tiny state (fails at 15%)
    def fit_fn(Xtr, k):
        n = len(Xtr)
        if k == 2:
            states = np.array([0] * (n // 2) + [1] * (n - n // 2))
        else:
            states = np.array([0] * (n - 5) + [1] * 3 + [2] * 2)
        return {"k": k}, states

    Xtr = pd.DataFrame({"x": range(100)})
    sel = G.select_k_by_coverage(fit_fn, Xtr, k_choices=(2, 3), min_state_frac=0.15)
    assert sel["k"] == 2
    assert sel["fits"][3]["min_coverage"] < 0.15
    assert sel["fits"][2]["min_coverage"] >= 0.15


def test_select_k_by_coverage_falls_back_to_smallest_when_none_qualify():
    def fit_fn(Xtr, k):
        n = len(Xtr)
        states = np.array([0] * (n - 1) + [1] * 0 + [k - 1] * 1)  # tiny minority always
        return {"k": k}, states

    Xtr = pd.DataFrame({"x": range(50)})
    sel = G.select_k_by_coverage(fit_fn, Xtr, k_choices=(2, 3), min_state_frac=0.15)
    assert sel["k"] == 2                    # falls back to min(k_choices)


def test_fit_hmm_fold_main_uses_training_window_only(synth_X):
    f0 = WalkForwardSplit().folds(synth_X.index)[0]
    Xtr = synth_X.iloc[f0.train_idx]
    Xte = synth_X.iloc[f0.test_idx]

    fit1 = G.fit_hmm_fold_main(Xtr, n_init=2)
    X2 = synth_X.copy()
    rng = np.random.default_rng(9)
    X2.loc[Xte.index, :] = rng.normal(size=(len(Xte), X2.shape[1]))
    fit2 = G.fit_hmm_fold_main(X2.iloc[f0.train_idx], n_init=2)

    assert fit1["k"] == fit2["k"]
    np.testing.assert_allclose(fit1["model"].means_, fit2["model"].means_)
    np.testing.assert_allclose(fit1["final_dist"], fit2["final_dist"])


def test_fit_hmm_fold_fixed_k_ignores_coverage(synth_X):
    """Main cross-fold tables force k=3 in every fold regardless of that
    fold's own coverage-selection outcome - fixed_k must not apply the
    15%-coverage fallback at all."""
    f0 = WalkForwardSplit().folds(synth_X.index)[0]
    Xtr = synth_X.iloc[f0.train_idx]
    fit = G.fit_hmm_fold_fixed_k(Xtr, k=3, n_init=2)
    assert fit["k"] == 3
    assert fit["model"].n_components == 3


def test_fit_kmeans_fold_fixed_k_ignores_coverage(synth_X):
    f0 = WalkForwardSplit().folds(synth_X.index)[0]
    Xtr = synth_X.iloc[f0.train_idx]
    fit = G.fit_kmeans_fold_fixed_k(Xtr, k=3, n_init=2)
    assert fit["k"] == 3
    assert fit["model"].n_clusters == 3


def test_fit_kmeans_fold_main_respects_min_state_frac(synth_X):
    f0 = WalkForwardSplit().folds(synth_X.index)[0]
    Xtr = synth_X.iloc[f0.train_idx]
    fit = G.fit_kmeans_fold_main(Xtr, k_choices=(2, 3), min_state_frac=0.15)
    assert fit["k"] in (2, 3)
    cov = np.array(fit["coverage_by_k"][fit["k"]])
    assert cov.min() >= 0.15 - 1e-9 or fit["k"] == 2


def test_run_fixed_k_uses_same_k_every_fold(synth_X):
    out = G.run_fixed_k(synth_X, k=3, n_init=2, save=False)
    labels = out["labels"]
    assert (labels["k"] == 3).all()
    for model in ("hmm", "kmeans"):
        s = labels[labels["model"] == model]["state"]
        assert s.min() >= 0 and s.max() <= 2


def test_run_fixed_k_works_on_a_differently_named_vol_column(synth_X):
    """Stand-in for the descriptive matrix, which ranks by log_rv_desc
    instead of har_logrv_1d."""
    X = synth_X.rename(columns={"har_logrv_1d": "log_rv_desc"})
    out = G.run_fixed_k(X, k=2, n_init=2, vol_feature="log_rv_desc", save=False)
    assert (out["labels"]["k"] == 2).all()


# --------------------------------------------------------------------------
# End to end sanity
# --------------------------------------------------------------------------
def test_run_produces_ranked_states_and_full_oos_coverage(synth_X):
    out = G.run(X=synth_X, k_choices=(2, 3), save=False)
    labels = out["labels"]
    assert set(labels["model"]) == {"kmeans", "hmm"}
    for (model, k), g in labels.groupby(["model", "k"]):
        assert g["state"].min() >= 0 and g["state"].max() < k
    dates_per_model = labels.groupby("model")["date"].apply(frozenset)
    assert dates_per_model["kmeans"] == dates_per_model["hmm"]
    hmm_rows = labels[labels["model"] == "hmm"]
    assert hmm_rows["prob"].between(0, 1).all()
