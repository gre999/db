"""Week 6: market-regime clustering on the predictive state features.

Two families, both fit **only on each fold's training window** (never the
test window, never another fold's data), on
``src.features.PREDICTIVE_STATE_COLUMNS`` (log RV, close location, relative
volume, overnight gap - all lagged, decision-safe):

* KMeans, k in {2, 3, 4}, picked by silhouette score on the training data.
* Gaussian HMM, n_states in {2, 3, 4}, picked by BIC on the training data.

Folds come from :class:`~src.validation.WalkForwardSplit`, the same
splitter every other model in this project uses.

Out-of-sample state assignment:

* KMeans: ``.predict()`` on standardized test rows is safe - each row's
  label depends only on that row's own features and the already-fitted,
  fixed cluster centers, never on other test rows.
* HMM: **never** call ``.predict()`` or ``.predict_proba()`` on a test
  slice - both run Viterbi / forward-backward over the *whole* sequence
  passed in, so an early test day's state would be informed by later test
  days that haven't "happened" yet. Test-slice states instead come from
  :func:`forward_filter` (the forward algorithm only), run one day at a
  time. The belief state from the end of the training window seeds the
  first test day's filter, so the test period doesn't artificially reset
  to the model's unconditional prior at the fold boundary - that seed is
  itself computed entirely from training data, so this carries no leakage.

States are relabeled 0..k-1 by ascending mean log-RV **computed on the
training window only**, so state numbering means the same thing ("0 = calm
... k-1 = turbulent") in every fold instead of KMeans/HMM's arbitrary
per-fit indexing.

Usage::

    python -m src.models.regimes
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import data_loader as dl
from src import features as F
from src.validation import WalkForwardSplit

log = logging.getLogger(__name__)
STATE_FEATURES = list(F.PREDICTIVE_STATE_COLUMNS)
VOL_FEATURE = "har_logrv_1d"       # ranks states low -> high volatility
K_CHOICES = (2, 3, 4)
OUT_DIR = dl.PROCESSED_DIR / "regimes"


# --------------------------------------------------------------------------
# State relabeling (fixes KMeans'/HMM's arbitrary per-fit state indexing)
# --------------------------------------------------------------------------
def relabel_by_vol(raw_states: np.ndarray, vol: np.ndarray, k: int) -> np.ndarray:
    """``old -> new`` map: state indices ranked by ascending mean ``vol``."""
    means = [vol[raw_states == i].mean() if (raw_states == i).any() else np.inf
            for i in range(k)]
    order = np.argsort(means)                 # order[rank] = old_state
    mapping = np.empty(k, dtype=int)
    mapping[order] = np.arange(k)              # mapping[old_state] = rank
    return mapping


def _invert(mapping: np.ndarray) -> np.ndarray:
    inv = np.empty_like(mapping)
    inv[mapping] = np.arange(len(mapping))
    return inv


def reorder_prob_columns(probs: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Reorder a (T, k) prob matrix's columns from raw state to ranked state."""
    return probs[:, _invert(mapping)]


# --------------------------------------------------------------------------
# HMM forward filtering (causal - no Viterbi, no forward-backward)
# --------------------------------------------------------------------------
def _emission_logprob(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """(n_samples, n_states) Gaussian log-likelihoods under each state."""
    k = model.n_components
    out = np.empty((len(X), k))
    for i in range(k):
        mean = model.means_[i]
        ct = model.covariance_type
        if ct == "diag":
            cov = np.diag(model.covars_[i])
        elif ct == "full":
            cov = model.covars_[i]
        elif ct == "spherical":
            cov = np.eye(len(mean)) * model.covars_[i]
        elif ct == "tied":
            cov = model.covars_
        else:
            raise ValueError(f"unsupported covariance_type {ct!r}")
        out[:, i] = multivariate_normal.logpdf(X, mean=mean, cov=cov,
                                               allow_singular=True)
    return out


def forward_filter(model: GaussianHMM, X: np.ndarray,
                   init_dist: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Causal HMM filtering: P(state_t | obs_1..t), never obs_{>t}.

    ``init_dist`` overrides ``model.startprob_`` as the belief state before
    the first observation of ``X`` - pass a previous period's final
    filtered distribution to continue one sequence across a boundary
    (e.g. train -> test) without resetting to the unconditional prior.
    Returns (filtered_probs (len(X), k), final_dist (k,)).
    """
    log_b = _emission_logprob(model, X)
    A = model.transmat_
    T, k = log_b.shape
    alpha = np.empty((T, k))
    b0 = np.exp(log_b[0] - log_b[0].max())
    if init_dist is None:
        # true sequence start: startprob_ IS the t=0 prior, no transition
        a0 = model.startprob_ * b0
    else:
        # continuing a sequence: init_dist is the belief *after* the last
        # observation already folded in, so the new first observation still
        # needs its own transition step, exactly like every t >= 1 below
        a0 = (init_dist @ A) * b0
    alpha[0] = a0 / a0.sum()
    for t in range(1, T):
        pred = alpha[t - 1] @ A
        bt = np.exp(log_b[t] - log_b[t].max())
        w = pred * bt
        alpha[t] = w / w.sum()
    return alpha, alpha[-1]


# --------------------------------------------------------------------------
# Per-fold fit / predict
# --------------------------------------------------------------------------
def fit_kmeans_fold(Xtr: pd.DataFrame, k_choices=K_CHOICES,
                    random_state: int = 0) -> dict:
    scaler = StandardScaler().fit(Xtr.to_numpy())
    Ztr = scaler.transform(Xtr.to_numpy())
    best = None
    for k in k_choices:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Ztr)
        sil = silhouette_score(Ztr, km.labels_)
        if best is None or sil > best["silhouette"]:
            best = {"k": k, "model": km, "silhouette": float(sil)}
    mapping = relabel_by_vol(best["model"].labels_, Xtr[VOL_FEATURE].to_numpy(),
                             best["k"])
    return {"scaler": scaler, "model": best["model"], "k": best["k"],
           "silhouette": best["silhouette"], "mapping": mapping}


def predict_kmeans(fit: dict, Xte: pd.DataFrame) -> np.ndarray:
    Z = fit["scaler"].transform(Xte.to_numpy())
    raw = fit["model"].predict(Z)
    return fit["mapping"][raw]


def fit_hmm_fold(Xtr: pd.DataFrame, k_choices=K_CHOICES,
                 covariance_type: str = "diag", n_iter: int = 200,
                 random_state: int = 0) -> dict:
    scaler = StandardScaler().fit(Xtr.to_numpy())
    Ztr = scaler.transform(Xtr.to_numpy())
    best = None
    for k in k_choices:
        model = GaussianHMM(n_components=k, covariance_type=covariance_type,
                            n_iter=n_iter, tol=1e-3, random_state=random_state)
        model.fit(Ztr)
        bic = float(model.bic(Ztr))
        if best is None or bic < best["bic"]:
            best = {"k": k, "model": model, "bic": bic}
    train_probs, final_dist = forward_filter(best["model"], Ztr)
    train_states = train_probs.argmax(axis=1)
    mapping = relabel_by_vol(train_states, Xtr[VOL_FEATURE].to_numpy(), best["k"])
    return {"scaler": scaler, "model": best["model"], "k": best["k"],
           "bic": best["bic"], "mapping": mapping, "final_dist": final_dist}


def state_coverage(states: np.ndarray, k: int) -> np.ndarray:
    """Fraction of rows in each of the k states, ranked-state order."""
    n = len(states)
    return np.array([(states == i).sum() / n for i in range(k)])


def select_k_by_coverage(fit_fn, Xtr: pd.DataFrame, k_choices=(2, 3),
                         min_state_frac: float = 0.15) -> dict:
    """Fit at every k in ``k_choices`` and pick the **largest** one whose
    smallest training-window state still covers >= ``min_state_frac`` of
    the training days; falls back to the smallest ``k_choices`` if none
    qualify. Uses the training window only - never the test slice.

    ``fit_fn(Xtr, k) -> (fit, train_states)``; ``fit`` is whatever that
    model family needs to remember, ``train_states`` is the raw (not yet
    vol-ranked) hard state assignment on ``Xtr``.
    """
    fits = {}
    for k in sorted(k_choices):
        fit, train_states = fit_fn(Xtr, k)
        cov = state_coverage(train_states, k)
        fits[k] = {"fit": fit, "train_states": train_states, "coverage": cov,
                  "min_coverage": float(cov.min())}
    qualifying = [k for k in sorted(k_choices, reverse=True)
                 if fits[k]["min_coverage"] >= min_state_frac]
    chosen_k = qualifying[0] if qualifying else min(k_choices)
    return {"k": chosen_k, "fits": fits}


def _hmm_best_of_n_init(Ztr: np.ndarray, k: int, covariance_type: str = "diag",
                        n_iter: int = 200, n_init: int = 10, seed0: int = 0
                        ) -> tuple[GaussianHMM, float]:
    """Fit a k-state Gaussian HMM ``n_init`` times (different random
    restarts), keep the one with the highest training log-likelihood."""
    best = None
    for i in range(n_init):
        m = GaussianHMM(n_components=k, covariance_type=covariance_type,
                        n_iter=n_iter, tol=1e-3, random_state=seed0 + i)
        m.fit(Ztr)
        ll = float(m.score(Ztr))
        if best is None or ll > best[1]:
            best = (m, ll)
    return best


def fit_hmm_fold_main(Xtr: pd.DataFrame, k_choices=(2, 3),
                      min_state_frac: float = 0.15, n_init: int = 10,
                      covariance_type: str = "diag", n_iter: int = 200) -> dict:
    """Week-7 main HMM selection: the largest k in ``k_choices`` whose
    smallest training-window state covers >= ``min_state_frac`` of days
    (else falls back to ``min(k_choices)``); each k fit with ``n_init``
    random restarts, keeping the highest training log-likelihood. Decided
    entirely from the training window.
    """
    scaler = StandardScaler().fit(Xtr.to_numpy())
    Ztr = scaler.transform(Xtr.to_numpy())
    vol = Xtr[VOL_FEATURE].to_numpy()

    def fit_fn(_, k):
        model, ll = _hmm_best_of_n_init(Ztr, k, covariance_type, n_iter, n_init)
        train_states = forward_filter(model, Ztr)[0].argmax(axis=1)
        return {"model": model, "loglik": ll}, train_states

    sel = select_k_by_coverage(fit_fn, Xtr, k_choices, min_state_frac)
    k = sel["k"]
    chosen = sel["fits"][k]["fit"]
    train_probs, final_dist = forward_filter(chosen["model"], Ztr)
    mapping = relabel_by_vol(train_probs.argmax(axis=1), vol, k)
    return {"scaler": scaler, "model": chosen["model"], "k": k,
           "loglik": chosen["loglik"], "mapping": mapping,
           "final_dist": final_dist,
           "coverage_by_k": {kk: v["coverage"].tolist()
                            for kk, v in sel["fits"].items()}}


def fit_kmeans_fold_main(Xtr: pd.DataFrame, k_choices=(2, 3),
                         min_state_frac: float = 0.15, n_init: int = 10,
                         random_state: int = 0) -> dict:
    """Week-7 main KMeans selection: same coverage rule as
    :func:`fit_hmm_fold_main`, for a like-for-like comparison."""
    scaler = StandardScaler().fit(Xtr.to_numpy())
    Ztr = scaler.transform(Xtr.to_numpy())
    vol = Xtr[VOL_FEATURE].to_numpy()

    def fit_fn(_, k):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Ztr)
        return {"model": km}, km.labels_

    sel = select_k_by_coverage(fit_fn, Xtr, k_choices, min_state_frac)
    k = sel["k"]
    chosen = sel["fits"][k]["fit"]
    mapping = relabel_by_vol(chosen["model"].labels_, vol, k)
    return {"scaler": scaler, "model": chosen["model"], "k": k, "mapping": mapping,
           "coverage_by_k": {kk: v["coverage"].tolist()
                            for kk, v in sel["fits"].items()}}


def fit_hmm_fold_fixed_k(Xtr: pd.DataFrame, k: int, n_init: int = 10,
                         covariance_type: str = "diag", n_iter: int = 200,
                         vol_feature: str = VOL_FEATURE) -> dict:
    """Fit at a k fixed across every fold - for the main cross-fold-aggregated
    analysis, where per-fold coverage-selected k (2 or 3) would leave state
    numbering incomparable across folds ("k=2's state 1" != "k=3's state 1").
    Same best-of-``n_init`` fitting as :func:`fit_hmm_fold_main`, no
    coverage check. ``vol_feature`` picks the ranking column - pass
    ``"log_rv_desc"`` when fitting on the descriptive feature matrix."""
    scaler = StandardScaler().fit(Xtr.to_numpy())
    Ztr = scaler.transform(Xtr.to_numpy())
    model, ll = _hmm_best_of_n_init(Ztr, k, covariance_type, n_iter, n_init)
    train_probs, final_dist = forward_filter(model, Ztr)
    mapping = relabel_by_vol(train_probs.argmax(axis=1), Xtr[vol_feature].to_numpy(), k)
    return {"scaler": scaler, "model": model, "k": k, "loglik": ll,
           "mapping": mapping, "final_dist": final_dist}


def fit_kmeans_fold_fixed_k(Xtr: pd.DataFrame, k: int, n_init: int = 10,
                            random_state: int = 0,
                            vol_feature: str = VOL_FEATURE) -> dict:
    """KMeans counterpart of :func:`fit_hmm_fold_fixed_k`."""
    scaler = StandardScaler().fit(Xtr.to_numpy())
    Ztr = scaler.transform(Xtr.to_numpy())
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Ztr)
    mapping = relabel_by_vol(km.labels_, Xtr[vol_feature].to_numpy(), k)
    return {"scaler": scaler, "model": km, "k": k, "mapping": mapping}


def filter_hmm_oos(fit: dict, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Day-by-day filtered probabilities for a fold's test slice (ranked states)."""
    Zte = fit["scaler"].transform(Xte.to_numpy())
    probs, _ = forward_filter(fit["model"], Zte, init_dist=fit["final_dist"])
    probs = reorder_prob_columns(probs, fit["mapping"])
    states = probs.argmax(axis=1)
    return states, probs


def transmat_ranked(fit: dict) -> np.ndarray:
    """The fitted HMM's transition matrix, rows/cols reordered to ranked states."""
    inv = _invert(fit["mapping"])
    A = fit["model"].transmat_
    return A[inv][:, inv]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def load_state_features(processed_dir: Path = dl.PROCESSED_DIR) -> pd.DataFrame:
    X = F.build_feature_matrix(
        F.MarketData.from_processed(processed_dir), "prev_close",
        names=STATE_FEATURES)
    return X.dropna()


def run(X: pd.DataFrame | None = None, splitter: WalkForwardSplit = WalkForwardSplit(),
       k_choices=K_CHOICES, save: bool = True,
       processed_dir: Path = dl.PROCESSED_DIR) -> dict:
    X = load_state_features(processed_dir) if X is None else X
    label_rows, transmat_rows, sel_rows = [], [], []

    for f in splitter.folds(X.index):
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]

        km = fit_kmeans_fold(Xtr, k_choices)
        km_states = predict_kmeans(km, Xte)
        for d, s in zip(Xte.index, km_states):
            label_rows.append({"date": d, "fold": f.number, "model": "kmeans",
                              "k": km["k"], "state": int(s), "prob": np.nan})
        sel_rows.append({"fold": f.number, "model": "kmeans", "k": km["k"],
                        "criterion": "silhouette", "value": km["silhouette"]})

        hmm = fit_hmm_fold(Xtr, k_choices)
        hmm_states, hmm_probs = filter_hmm_oos(hmm, Xte)
        top_prob = hmm_probs[np.arange(len(hmm_probs)), hmm_states]
        for d, s, p in zip(Xte.index, hmm_states, top_prob):
            label_rows.append({"date": d, "fold": f.number, "model": "hmm",
                              "k": hmm["k"], "state": int(s), "prob": float(p)})
        sel_rows.append({"fold": f.number, "model": "hmm", "k": hmm["k"],
                        "criterion": "bic", "value": hmm["bic"]})

        A = transmat_ranked(hmm)
        for i in range(hmm["k"]):
            for j in range(hmm["k"]):
                transmat_rows.append({"fold": f.number, "k": hmm["k"],
                                     "from_state": i, "to_state": j,
                                     "prob": float(A[i, j])})
        log.info("fold %s: kmeans k=%d (sil=%.3f), hmm k=%d (bic=%.1f)",
                 f.number, km["k"], km["silhouette"], hmm["k"], hmm["bic"])

    labels = pd.DataFrame(label_rows)
    transmat = pd.DataFrame(transmat_rows)
    selection = pd.DataFrame(sel_rows)

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(OUT_DIR / "regime_labels.parquet", index=False)
        transmat.to_parquet(OUT_DIR / "hmm_transmat.parquet", index=False)
        selection.to_parquet(OUT_DIR / "model_selection.parquet", index=False)
    return {"labels": labels, "transmat": transmat, "selection": selection}


def run_fixed_k(X: pd.DataFrame, k: int = 3,
                splitter: WalkForwardSplit = WalkForwardSplit(),
                n_init: int = 10, vol_feature: str = VOL_FEATURE,
                save: bool = False, out_name: str | None = None,
                processed_dir: Path = dl.PROCESSED_DIR) -> dict:
    """Same k in every fold (:func:`fit_hmm_fold_fixed_k` /
    :func:`fit_kmeans_fold_fixed_k`) - for cross-fold-aggregated analysis
    (week 7's main performance tables), where per-fold coverage-selected k
    would leave state numbering incomparable across folds. Works on any
    feature matrix sharing ``vol_feature`` and the other state-feature
    columns - pass the descriptive matrix (``vol_feature="log_rv_desc"``)
    to get descriptive-state labels with the same fold structure.
    """
    label_rows, transmat_rows = [], []
    for f in splitter.folds(X.index):
        Xtr, Xte = X.iloc[f.train_idx], X.iloc[f.test_idx]

        km = fit_kmeans_fold_fixed_k(Xtr, k, n_init, vol_feature=vol_feature)
        km_states = predict_kmeans(km, Xte)
        for d, s in zip(Xte.index, km_states):
            label_rows.append({"date": d, "fold": f.number, "model": "kmeans",
                              "k": k, "state": int(s), "prob": np.nan})

        hmm = fit_hmm_fold_fixed_k(Xtr, k, n_init, vol_feature=vol_feature)
        hmm_states, hmm_probs = filter_hmm_oos(hmm, Xte)
        top_prob = hmm_probs[np.arange(len(hmm_probs)), hmm_states]
        for d, s, p in zip(Xte.index, hmm_states, top_prob):
            label_rows.append({"date": d, "fold": f.number, "model": "hmm",
                              "k": k, "state": int(s), "prob": float(p)})

        A = transmat_ranked(hmm)
        for i in range(k):
            for j in range(k):
                transmat_rows.append({"fold": f.number, "k": k,
                                     "from_state": i, "to_state": j,
                                     "prob": float(A[i, j])})
        log.info("fixed-k=%d fold %s done", k, f.number)

    labels = pd.DataFrame(label_rows)
    transmat = pd.DataFrame(transmat_rows)
    if save:
        name = out_name or f"fixed_k{k}"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(OUT_DIR / f"regime_labels_{name}.parquet", index=False)
        transmat.to_parquet(OUT_DIR / f"hmm_transmat_{name}.parquet", index=False)
    return {"labels": labels, "transmat": transmat}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = run()
    print(out["selection"])
    print(out["labels"].groupby(["model", "k"]).size())


if __name__ == "__main__":
    main()
