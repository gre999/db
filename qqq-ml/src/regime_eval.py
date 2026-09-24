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
from src.models import regimes as G
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


# --------------------------------------------------------------------------
# 3. Significance: state vs full-sample mean, and a random-labeling baseline
# --------------------------------------------------------------------------
def state_vs_full_sample_test(labels: pd.DataFrame, strategy: pd.DataFrame,
                              model: str, lags: int | None = None) -> pd.DataFrame:
    """Newey-West test of each state's mean ``bps_return`` against the
    **full-sample mean** (the config's primary comparison; this is the
    "config.comparison.primary" test).

    Implemented as an OLS of ``bps_return`` on a single state-``i`` dummy
    (i.e. state i vs every other day), which gives an exact, closed-form
    HAC t-stat/p-value; the "vs state i's own complement" coefficient
    ``beta`` is then rescaled to "vs full sample" by ``(1 - w_i)`` where
    ``w_i`` is state i's share of days - algebraically,
    ``full_sample_mean = w_i * state_i_mean + (1-w_i) * rest_mean``, so
    ``state_i_mean - full_sample_mean = (1-w_i) * (state_i_mean -
    rest_mean) = (1-w_i) * beta``. The t-statistic is identical either way
    (the rescaling cancels), so this is not an approximation - only the
    point estimate of the gap is rescaled, not its significance.
    """
    j = _join_labels_and_strategy(labels, strategy, model)
    y = j["bps_return"].to_numpy()
    n = len(y)
    full_mean = float(y.mean())
    lag = lags if lags is not None else int(np.floor(n ** (1 / 3)))

    rows = []
    for state in sorted(j["state"].unique()):
        d = (j["state"] == state).astype(float).to_numpy()
        w = float(d.mean())
        Z = np.column_stack([np.ones(n), d])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        se = np.sqrt(np.diag(newey_west_cov(Z, resid, lag)))
        t = beta[1] / se[1]
        p = float(2 * stats.norm.sf(abs(t)))
        state_mean = full_mean + (1 - w) * beta[1]
        rows.append({"state": int(state), "n": int(d.sum()),
                    "state_mean_bps": state_mean, "full_sample_mean_bps": full_mean,
                    "diff_vs_full_sample": float((1 - w) * beta[1]),
                    "se": float((1 - w) * se[1]), "t": float(t), "p": p})
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    return df


def label_block_shuffle_null(labels: pd.DataFrame, strategy: pd.DataFrame,
                             model: str, block_size: int = 20,
                             n_reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Random-labeling baseline: block-shuffle the STATE LABELS (the
    ``bps_return`` sequence is never touched, so its own serial correlation
    is preserved) and recompute "state i mean - full-sample mean" each
    time. Reports where the *actually observed* gap falls in this null
    distribution - a state whose real gap isn't clearly outside the range
    a random (but still block-persistent) labeling would produce isn't
    doing anything a random state assignment couldn't also produce.
    """
    j = _join_labels_and_strategy(labels, strategy, model)
    y = j["bps_return"].to_numpy()
    s = j["state"].to_numpy()
    n = len(y)
    full_mean = float(y.mean())
    states = sorted(j["state"].unique())

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(n - block_size + 1)
    null_diffs = {st: np.empty(n_reps) for st in states}
    for i in range(n_reps):
        idx = np.concatenate([np.arange(b, b + block_size)
                              for b in rng.choice(starts, n_blocks)])[:n]
        s_shuf = s[idx]
        for st in states:
            mask = s_shuf == st
            null_diffs[st][i] = (y[mask].mean() if mask.any() else np.nan) - full_mean

    rows = []
    for st in states:
        observed = float(y[s == st].mean() - full_mean)
        null = null_diffs[st][~np.isnan(null_diffs[st])]
        p = float(np.mean(np.abs(null) >= abs(observed))) if len(null) else np.nan
        rows.append({"state": int(st), "observed_diff": observed,
                    "null_mean": float(null.mean()) if len(null) else np.nan,
                    "null_std": float(null.std(ddof=0)) if len(null) else np.nan,
                    "p_vs_random_labeling": p})
    df = pd.DataFrame(rows)
    df.insert(0, "model", model)
    return df


# --------------------------------------------------------------------------
# 4. Tradability: trade only in training-positive states
# --------------------------------------------------------------------------
def _daily_eval(ret_bps: pd.Series) -> dict:
    """Annualized return/vol/Sharpe and max drawdown of a daily bps-return
    series (0 on no-trade days is a valid, intended input - that is how a
    tradability filter's "sit out" days are represented)."""
    r = (ret_bps / 1e4).to_numpy()
    n = len(r)
    equity = np.cumprod(1 + r)
    ann_return = float(equity[-1] ** (252 / n) - 1) if n else np.nan
    sd = r.std(ddof=0)
    ann_vol = float(sd * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    dd = equity / np.maximum.accumulate(equity) - 1
    return {"n": n, "ann_return": ann_return, "ann_vol": ann_vol,
           "sharpe": sharpe, "max_drawdown": float(dd.min())}


def _positive_states_from_training(train_state: np.ndarray, train_bps: np.ndarray
                                   ) -> set[int]:
    states = set(np.unique(train_state).tolist())
    return {int(s) for s in states if train_bps[train_state == s].mean() > 0}


def tradability_test(state_features: pd.DataFrame, strategy: pd.DataFrame,
                     model_family: str, splitter: WalkForwardSplit = WalkForwardSplit(),
                     k_choices=(2, 3), min_state_frac: float = 0.15,
                     n_init: int = 10) -> dict:
    """Per fold: fit the **per-fold coverage-selected** k (not the fixed
    k=3 used for cross-fold-aggregated tables - no cross-fold alignment is
    needed here, each fold's trade/no-trade decision stands on its own),
    find which states had a positive mean ``bps_return`` **in that fold's
    training window** (using the strategy's own historical daily results
    joined to the training dates), then in the test window only count a
    day's actual return if that day's state was training-positive -
    otherwise the day contributes 0 (sit out). Compared against trading
    every OOS day (no filter).

    Returns ``{"filtered": pd.Series, "unfiltered": pd.Series,
    "positive_states_by_fold": dict, "eval_filtered": dict,
    "eval_unfiltered": dict}``, all on the OOS date index.
    """
    s = strategy.set_index("day") if "day" in strategy.columns else strategy
    filtered_rows, unfiltered_rows, pos_states = [], [], {}

    for f in splitter.folds(state_features.index):
        Xtr = state_features.iloc[f.train_idx]
        Xte = state_features.iloc[f.test_idx]
        tr_bps = s["bps_return"].reindex(Xtr.index)
        te_bps = s["bps_return"].reindex(Xte.index)
        ok_tr = tr_bps.notna()
        if ok_tr.sum() < 30 or Xte.empty:
            continue

        if model_family == "hmm":
            fit = G.fit_hmm_fold_main(Xtr, k_choices, min_state_frac, n_init)
            Ztr = fit["scaler"].transform(Xtr.to_numpy())
            train_state = G.forward_filter(fit["model"], Ztr)[0].argmax(axis=1)
            train_state = fit["mapping"][train_state]
            test_state, _ = G.filter_hmm_oos(fit, Xte)
        elif model_family == "kmeans":
            fit = G.fit_kmeans_fold_main(Xtr, k_choices, min_state_frac, n_init)
            train_state = fit["mapping"][fit["model"].labels_]
            test_state = G.predict_kmeans(fit, Xte)
        else:
            raise ValueError(model_family)

        pos = _positive_states_from_training(train_state[ok_tr.to_numpy()],
                                             tr_bps[ok_tr].to_numpy())
        pos_states[f.number] = pos
        keep = np.isin(test_state, list(pos)) if pos else np.zeros(len(Xte), bool)
        te_ret = te_bps.fillna(0.0).to_numpy()
        filtered_rows.append(pd.Series(np.where(keep, te_ret, 0.0), index=Xte.index))
        unfiltered_rows.append(pd.Series(te_ret, index=Xte.index))

    filtered = pd.concat(filtered_rows).sort_index()
    unfiltered = pd.concat(unfiltered_rows).sort_index()
    return {"filtered": filtered, "unfiltered": unfiltered,
           "positive_states_by_fold": pos_states,
           "eval_filtered": _daily_eval(filtered),
           "eval_unfiltered": _daily_eval(unfiltered)}


def tradability_test_vol_tercile(strategy: pd.DataFrame, fold_windows: pd.DataFrame,
                                 pred: pd.DataFrame | None = None,
                                 pred_name: str = "week4_models",
                                 model: str = "har_resid_xgb", target: str = "rv",
                                 min_train_days: int = 50) -> dict:
    """:func:`tradability_test`'s counterpart for the vol-tercile baseline:
    same expanding pre-test-start window as :func:`vol_tercile_labels`,
    but the training window's OWN tercile bucketing is also computed (not
    just the test window's) so "positive in training" can be evaluated.
    """
    if pred is None:
        pred = load_predictions(pred_name)
        pred = pred[(pred["model"] == model) & (pred["target"] == target)]
    p = pred.set_index("date").sort_index() if "date" in pred.columns else pred.sort_index()
    s = strategy.set_index("day") if "day" in strategy.columns else strategy

    filtered_rows, unfiltered_rows, pos_states = [], [], {}
    for _, fb in fold_windows.iterrows():
        test_start, test_end = pd.Timestamp(fb["test_start"]), pd.Timestamp(fb["test_end"])
        tr = p.loc[p.index < test_start]
        te = p.loc[(p.index >= test_start) & (p.index <= test_end)]
        if len(tr) < min_train_days or te.empty:
            continue
        q1, q2 = tr["y_pred_var"].quantile([1 / 3, 2 / 3])
        train_state = pd.cut(tr["y_pred_var"], bins=[-np.inf, q1, q2, np.inf],
                             labels=[0, 1, 2]).astype(int)
        test_state = pd.cut(te["y_pred_var"], bins=[-np.inf, q1, q2, np.inf],
                            labels=[0, 1, 2]).astype(int)

        tr_bps = s["bps_return"].reindex(tr.index)
        te_bps = s["bps_return"].reindex(te.index)
        ok_tr = tr_bps.notna()
        if ok_tr.sum() < 30:
            continue
        pos = _positive_states_from_training(train_state[ok_tr].to_numpy(),
                                             tr_bps[ok_tr].to_numpy())
        pos_states[int(fb["fold"])] = pos
        keep = test_state.isin(pos).to_numpy() if pos else np.zeros(len(te), bool)
        te_ret = te_bps.fillna(0.0).to_numpy()
        filtered_rows.append(pd.Series(np.where(keep, te_ret, 0.0), index=te.index))
        unfiltered_rows.append(pd.Series(te_ret, index=te.index))

    filtered = pd.concat(filtered_rows).sort_index()
    unfiltered = pd.concat(unfiltered_rows).sort_index()
    return {"filtered": filtered, "unfiltered": unfiltered,
           "positive_states_by_fold": pos_states,
           "eval_filtered": _daily_eval(filtered),
           "eval_unfiltered": _daily_eval(unfiltered)}


def tradability_common_dates(results: dict[str, dict]) -> tuple[pd.Index, pd.DataFrame]:
    """Restricts a set of :func:`tradability_test`/:func:`tradability_test_
    vol_tercile` outputs (keyed by model name, e.g. ``{"hmm": ..., "kmeans":
    ..., "vol_tercile": ...}``) to the date range every source actually
    covers, and recomputes :func:`_daily_eval` there for both the filtered
    and unfiltered series.

    Without this, "unfiltered" Sharpe/drawdown differ across sources not
    because of anything about the state labels but because vol_tercile's
    date range is a strict subset of HMM/KMeans's (fold 1's test window has
    no prior predictions to set a tercile cutoff from - see
    :func:`vol_tercile_labels` - so it is dropped there but not for
    HMM/KMeans). "Unfiltered" is the same trading rule (trade every day)
    applied over a different sample window in that case, which is not a
    valid three-way comparison. Returns the common date index (so the
    caller can report its range/length) and one eval row per model.
    """
    common = None
    for out in results.values():
        idx = out["unfiltered"].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    rows = []
    for model, out in results.items():
        f_eval = _daily_eval(out["filtered"].loc[common])
        u_eval = _daily_eval(out["unfiltered"].loc[common])
        rows.append({"model": model,
                    **{f"filtered_{k}": v for k, v in f_eval.items()},
                    **{f"unfiltered_{k}": v for k, v in u_eval.items()}})
    return common, pd.DataFrame(rows)


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
