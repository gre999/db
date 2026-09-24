"""Week 7 (phase 3, week 2): does the regime label actually help?

Reads ``config/week07_regime_eval.toml`` for every setting used here (main
model, state-count rule, cost, which comparison is "primary" vs
exploratory) - fixed before any performance number in this module was
looked at; any change is written into that file with a reason.

Sections, matching the plan in ``reports/week07_regime_eval.md``:

1. State predictability - predictive vs descriptive label agreement,
   state duration, empirical (OOS, day-to-day) transition matrix.
2. Per-state strategy performance (:func:`state_performance_table`).
3. Significance - state vs full-sample-mean, block bootstrap and a
   label-shuffle random baseline.
4. Tradability - trade only in training-positive states, per-fold
   coverage-selected k (no cross-fold alignment needed there).

This module reads already-computed daily strategy results
(``data/processed/strategies/``) and regime labels
(``data/processed/regimes/``) - it never refits a strategy or a
clustering model itself.
"""
from __future__ import annotations

import argparse
import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import data_loader as dl
from src.models.base import load_predictions
from src.models.har import newey_west_cov
from src.validation import WalkForwardSplit

log = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "week07_regime_eval.toml"
REGIME_DIR = dl.PROCESSED_DIR / "regimes"
STRATEGY_DIR = dl.PROCESSED_DIR / "strategies"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# --------------------------------------------------------------------------
# 0. Volatility-tercile baseline ("simple state")
# --------------------------------------------------------------------------
def vol_tercile_labels(fold_windows: pd.DataFrame, pred: pd.DataFrame | None = None,
                       pred_name: str = "week4_models",
                       model: str = "har_resid_xgb", target: str = "rv",
                       min_train_days: int = 50) -> pd.DataFrame:
    """Naive 3-state baseline: HAR+residual-XGB's own t-1 variance forecast
    (already computed, ``data/processed/predictions/{pred_name}.parquet``),
    split into low/mid/high terciles using cutoffs from an **expanding
    window of everything strictly before that fold's test start** (33rd/
    67th percentile), applied to that fold's test window. Same long-format
    schema as :func:`~src.models.regimes.run_fixed_k`'s output (``date,
    fold, model, k, state, prob``; ``model="vol_tercile"``, ``prob``
    unused) so it lines up with HMM/KMeans everywhere else in this module.

    ``fold_windows`` must have the exact same fold numbers/test windows as
    the HMM/KMeans labels being compared against - pass e.g.
    ``pred_labels.groupby("fold")["date"].agg(test_start="min",
    test_end="max").reset_index()`` from the main-k3 label table, so all
    three arms are compared fold-for-fold. ``week4_models.parquet`` only
    holds *out-of-sample* predictions starting 2019 (it is itself a
    walk-forward run's test-period output), so there is no valid prior
    data to set a threshold from for whichever fold's test window starts
    earliest (typically 2019) - that fold is silently dropped here; the
    report states this explicitly rather than filling it with an
    in-sample threshold.

    This exists to answer: does HMM's state carry information beyond a
    volatility forecast the project already has? If HMM's per-state
    performance pattern looks like this baseline's, state classification
    is just volatility grouping under another name.
    """
    if pred is None:
        pred = load_predictions(pred_name)
        pred = pred[(pred["model"] == model) & (pred["target"] == target)]
    p = pred.set_index("date").sort_index() if "date" in pred.columns else pred.sort_index()

    rows = []
    for _, fb in fold_windows.iterrows():
        test_start, test_end = pd.Timestamp(fb["test_start"]), pd.Timestamp(fb["test_end"])
        tr = p.loc[p.index < test_start]
        te = p.loc[(p.index >= test_start) & (p.index <= test_end)]
        if len(tr) < min_train_days or len(te) == 0:
            continue
        q1, q2 = tr["y_pred_var"].quantile([1 / 3, 2 / 3])
        state = pd.cut(te["y_pred_var"], bins=[-np.inf, q1, q2, np.inf],
                       labels=[0, 1, 2]).astype(int)
        for d, s in zip(te.index, state):
            rows.append({"date": d, "fold": int(fb["fold"]), "model": "vol_tercile",
                        "k": 3, "state": int(s), "prob": np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. State predictability
# --------------------------------------------------------------------------
def _agreement_and_kappa(a: pd.Series, b: pd.Series, k: int) -> dict:
    """Shared agreement/kappa computation for two aligned state series."""
    n = len(a)
    observed = float((a.to_numpy() == b.to_numpy()).mean())
    a_freq = a.value_counts(normalize=True).reindex(range(k), fill_value=0.0)
    b_freq = b.value_counts(normalize=True).reindex(range(k), fill_value=0.0)
    expected_marginal = float((a_freq * b_freq).sum())
    expected_uniform = 1.0 / k
    kappa = ((observed - expected_marginal) / (1 - expected_marginal)
            if expected_marginal < 1 else np.nan)
    return {"k": k, "n": n, "observed_agreement": observed,
           "expected_uniform": expected_uniform,
           "expected_marginal": expected_marginal, "kappa": kappa}


def state_agreement(pred_labels: pd.DataFrame, desc_labels: pd.DataFrame,
                    model: str = "hmm") -> dict:
    """Predictive-vs-descriptive state agreement for one model, vs chance.

    Both inputs are the long-format label frames :func:`~src.models.
    regimes.run_fixed_k` produces (``date, fold, model, k, state, prob``);
    only rows for ``model`` are used, joined on ``date`` (inner - the two
    feature matrices can have slightly different warm-up lengths).

    Returns observed agreement, two chance baselines (uniform 1/k, and the
    marginal-frequency-weighted expectation), and Cohen's kappa (agreement
    beyond the marginal-frequency baseline; 0 = chance, 1 = perfect). **Read
    this together with** :func:`label_autocorrelation` **on the descriptive
    labels**: an HMM's descriptive states are themselves a filtered output
    with the same transition-matrix prior as the predictive ones, so part
    of this agreement is structural smoothness, not forecast skill -
    :func:`label_autocorrelation` gives the "smoothness alone" baseline to
    net out. The report's "predictable in advance" conclusion should rest
    on :func:`external_validation`, not on this number alone.
    """
    p = pred_labels[pred_labels["model"] == model].set_index("date")["state"]
    d = desc_labels[desc_labels["model"] == model].set_index("date")["state"]
    joined = pd.DataFrame({"pred": p, "desc": d}).dropna()
    k = int(max(joined["pred"].max(), joined["desc"].max()) + 1)
    return {"model": model, **_agreement_and_kappa(joined["pred"], joined["desc"], k)}


def label_autocorrelation(labels: pd.DataFrame, model: str = "hmm") -> dict:
    """Day-to-day self-consistency of ONE label sequence: agreement/kappa
    between state(t) and state(t-1), within the same fold only (same
    boundary rule as :func:`empirical_transition_matrix`).

    This is the calibration baseline for :func:`state_agreement`: even a
    sequence with no real forecasting skill will show nonzero t-vs-(t-1)
    self-consistency purely from being a filtered/smoothed output (the
    HMM's transition-matrix prior favors staying in the same state).
    Running this on the *descriptive* labels and comparing to the
    predictive-vs-descriptive number from :func:`state_agreement` shows
    how much of that agreement is shared structural smoothness rather than
    genuine next-day predictability.
    """
    l = labels[labels["model"] == model].sort_values("date")
    k = int(l["state"].max() + 1)
    prev_vals, cur_vals = [], []
    prev_state, prev_fold = None, None
    for _, row in l.iterrows():
        if prev_state is not None and row["fold"] == prev_fold:
            prev_vals.append(prev_state)
            cur_vals.append(row["state"])
        prev_state, prev_fold = row["state"], row["fold"]
    prev_s = pd.Series(prev_vals)
    cur_s = pd.Series(cur_vals)
    return {"model": model, **_agreement_and_kappa(prev_s, cur_s, k)}


def state_durations(labels: pd.DataFrame, model: str = "hmm") -> pd.DataFrame:
    """Mean/median consecutive-day run length in each state, OOS label
    sequence in date order (folds concatenated - see the week-6 state time
    series, same convention)."""
    l = labels[labels["model"] == model].sort_values("date")
    states = l["state"].to_numpy()
    runs = []
    cur, length = states[0], 1
    for s in states[1:]:
        if s == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = s, 1
    runs.append((cur, length))
    df = pd.DataFrame(runs, columns=["state", "run_length"])
    out = df.groupby("state")["run_length"].agg(["mean", "median", "count"])
    out.columns = ["mean_days", "median_days", "n_spells"]
    return out.reset_index()


def empirical_transition_matrix(labels: pd.DataFrame, model: str = "hmm"
                                ) -> pd.DataFrame:
    """Day-to-day transition frequencies actually observed OOS (not the
    fitted HMM's own transmat_ - this is realized, model-free counting).
    Only consecutive calendar rows within the same fold are counted (a gap
    at a fold boundary is not a same-day transition)."""
    l = labels[labels["model"] == model].sort_values("date")
    k = int(l["state"].max() + 1)
    counts = np.zeros((k, k))
    prev_state, prev_fold = None, None
    for _, row in l.iterrows():
        if prev_state is not None and row["fold"] == prev_fold:
            counts[int(prev_state), int(row["state"])] += 1
        prev_state, prev_fold = row["state"], row["fold"]
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts),
                      where=row_sums > 0)
    out = pd.DataFrame(probs, index=range(k), columns=range(k))
    out.index.name = "from_state"
    out.columns.name = "to_state"
    return out


def external_validation(labels: pd.DataFrame, desc: pd.DataFrame, model: str,
                        outcome: str, lags: int | None = None) -> pd.DataFrame:
    """The report's "states are predictable in advance" claim rests here,
    not on :func:`state_agreement`.

    OLS of ``outcome`` (a column of the *descriptive* feature matrix - an
    actual, same-day realized value such as ``log_rv_desc`` or day t's own
    ``|close_loc_desc - 0.5|`` - never used to fit the predictive state
    labels, which only ever see information from before t's close) on
    predictive-state dummies, Newey-West HAC standard errors (daily
    observations are serially correlated, especially given state
    persistence). State 0 is the reference level; each other state's row
    is its mean difference from state 0, with a t-stat and two-sided
    p-value. A model whose states correspond to a real, externally
    verifiable difference in what actually happens that day should show
    significant, monotonically-ordered differences here - independent of
    how the states were derived.
    """
    l = labels[labels["model"] == model].set_index("date")["state"]
    y = desc[outcome].dropna()
    idx = l.index.intersection(y.index)
    l, y = l.loc[idx], y.loc[idx]
    k = int(l.max() + 1)
    n = len(y)
    Z = np.column_stack([np.ones(n)] +
                        [(l == s).astype(float).to_numpy() for s in range(1, k)])
    yv = y.to_numpy()
    beta, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    resid = yv - Z @ beta
    lag = lags if lags is not None else int(np.floor(n ** (1 / 3)))
    se = np.sqrt(np.diag(newey_west_cov(Z, resid, lag)))
    tstat = beta / se
    pval = 2 * stats.norm.sf(np.abs(tstat))
    rows = [{"state": 0, "mean": float(beta[0]), "diff_vs_state0": 0.0,
            "se": np.nan, "t": np.nan, "p": np.nan,
            "n": int((l == 0).sum())}]
    for i, s in enumerate(range(1, k), start=1):
        rows.append({"state": s, "mean": float(beta[0] + beta[i]),
                    "diff_vs_state0": float(beta[i]), "se": float(se[i]),
                    "t": float(tstat[i]), "p": float(pval[i]),
                    "n": int((l == s).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Per-state strategy performance
# --------------------------------------------------------------------------
def _strategy_score(g: pd.DataFrame, strategy_name: str,
                    min_trades_per_year: int = 20) -> dict:
    """One row's worth of scores for a group of days (a state, or "all")."""
    traded = g[g["traded"]]
    ret = g["bps_return"] / 1e4
    sd = ret.std(ddof=0)
    sharpe = float(ret.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan
    out = {"n_days": len(g), "n_traded": len(traded),
          "mean_bps": float(g["bps_return"].mean()), "sharpe": sharpe,
          "win_rate": (float((traded["bps_return"] > 0).mean())
                      if len(traded) else np.nan)}
    if strategy_name == "orb":
        out["mean_r_gross"] = float(traded["r_gross"].mean()) if len(traded) else np.nan
        out["mean_r_net"] = float(traded["r_net"].mean()) if len(traded) else np.nan
    elif strategy_name == "vwap":
        out["mean_n_segments"] = (float(traded["n_segments"].mean())
                                  if len(traded) else np.nan)
        cost_bps = traded["cost"] / traded["first_entry_price"] * 1e4
        out["mean_cost_bps"] = float(cost_bps.mean()) if len(traded) else np.nan
    n_years = max(g.index.year.nunique(), 1)
    out["trades_per_year"] = len(traded) / n_years
    out["flag_insufficient"] = out["trades_per_year"] < min_trades_per_year
    return out


def _join_labels_and_strategy(labels: pd.DataFrame, strategy: pd.DataFrame,
                              model: str) -> pd.DataFrame:
    l = labels[labels["model"] == model].set_index("date")["state"]
    s = strategy.set_index("day") if "day" in strategy.columns else strategy
    return s.join(l, how="inner")


def state_performance_table(labels: pd.DataFrame, strategy: pd.DataFrame,
                            model: str, strategy_name: str,
                            min_trades_per_year: int = 20) -> pd.DataFrame:
    """Per-state performance for one (label source, strategy) pair, plus an
    ``"all"`` row - that source's own full-sample average, **restricted to
    the same dates that source's states cover** (vol_tercile only covers
    folds 2-8; HMM/KMeans cover all 8 - each is compared against its own
    date range, not a shared one). This ``"all"`` row is the config's
    primary comparison target: does each state differ from it?
    """
    j = _join_labels_and_strategy(labels, strategy, model)
    rows = [{"state": int(s), **_strategy_score(g, strategy_name, min_trades_per_year)}
           for s, g in j.groupby("state")]
    rows.append({"state": "all",
                **_strategy_score(j, strategy_name, min_trades_per_year)})
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    df.insert(1, "strategy", strategy_name)
    return df


def state_performance_by_year(labels: pd.DataFrame, strategy: pd.DataFrame,
                              model: str, strategy_name: str,
                              min_trades_per_year: int = 20) -> pd.DataFrame:
    """Same as :func:`state_performance_table` but one row per (state,
    year) cell - the granularity ``flag_insufficient`` is actually meant
    to catch."""
    j = _join_labels_and_strategy(labels, strategy, model)
    j = j.assign(year=j.index.year)
    rows = []
    for (state, year), g in j.groupby(["state", "year"]):
        rows.append({"state": int(state), "year": int(year),
                    **_strategy_score(g, strategy_name, min_trades_per_year)})
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    df.insert(1, "strategy", strategy_name)
    return df


def state_performance_by_fold(labels: pd.DataFrame, strategy: pd.DataFrame,
                              model: str, strategy_name: str,
                              min_trades_per_year: int = 20) -> pd.DataFrame:
    """Same as :func:`state_performance_table` but one row per (state,
    fold) cell."""
    l = labels[labels["model"] == model].set_index("date")[["fold", "state"]]
    s = strategy.set_index("day") if "day" in strategy.columns else strategy
    j = s.join(l, how="inner")
    rows = []
    for (state, fold), g in j.groupby(["state", "fold"]):
        rows.append({"state": int(state), "fold": int(fold),
                    **_strategy_score(g, strategy_name, min_trades_per_year)})
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    df.insert(1, "strategy", strategy_name)
    return df


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pred = pd.read_parquet(REGIME_DIR / "regime_labels_main_k3.parquet")
    desc = pd.read_parquet(REGIME_DIR / "regime_labels_descriptive_k3.parquet")
    for model in ("hmm", "kmeans"):
        print(model, state_agreement(pred, desc, model))
        print(state_durations(pred, model))
        print(empirical_transition_matrix(pred, model).round(3))


if __name__ == "__main__":
    main()
