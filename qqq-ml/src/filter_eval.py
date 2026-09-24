"""Week 8 (phase 4, week 1): filtered-performance evaluation.

Reads config/week08_filter.toml (success criteria, random-null method) and
already-computed outputs - the saved logistic filter predictions
(``src.models.filter``, ``data/processed/predictions/week08_filter.
parquet``) and ORB's own daily results (``data/processed/strategies/
orb_daily.parquet``). Never refits the filter, ORB, or the HMM regime
model - only evaluates.

Every comparison uses the exact same convention as
``src.regime_eval``'s tradability test: a full-calendar daily
``bps_return`` series over the OOS window, doji days and filtered-out
signal days both contributing 0 - so "filtered" and "unfiltered" differ
only in which signal days trade, never in sample coverage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loader as dl
from src import regime_eval as E
from src.models import filter as FL
from src.models import regimes as G

log = logging.getLogger(__name__)


def _oos_range(preds: pd.DataFrame, orb: pd.DataFrame) -> pd.DatetimeIndex:
    s = orb.set_index("day") if "day" in orb.columns else orb
    oos_dates = pd.DatetimeIndex(sorted(preds["date"].unique()))
    return s.index[(s.index >= oos_dates.min()) & (s.index <= oos_dates.max())]


def logistic_filter_series(preds: pd.DataFrame, orb: pd.DataFrame) -> dict:
    """filtered/unfiltered daily bps_return series over the calendar range
    the filter's test folds cover (doji days = 0 in both)."""
    s = orb.set_index("day") if "day" in orb.columns else orb
    full_range = _oos_range(preds, orb)
    unfiltered = s["bps_return"].reindex(full_range).fillna(0.0)

    signal_dates = pd.DatetimeIndex(preds["date"])
    keep_dates = pd.DatetimeIndex(preds.loc[preds["keep"], "date"])
    dropped = signal_dates.difference(keep_dates)
    filtered = unfiltered.copy()
    filtered.loc[dropped] = 0.0
    return {"filtered": filtered, "unfiltered": unfiltered}


def random_filter_null(preds: pd.DataFrame, orb: pd.DataFrame,
                       n_reps: int = 1000, seed: int = 0) -> np.ndarray:
    """config [significance.random_filter_null]: per fold, draw (without
    replacement) the same number of days that fold's rule actually kept,
    uniformly from that fold's non-doji ORB-signal test days; concatenate
    all folds' draws into one full-OOS-period Sharpe per repetition."""
    base = logistic_filter_series(preds, orb)["unfiltered"]
    rng = np.random.default_rng(seed)
    fold_dates = {f: g["date"].to_numpy() for f, g in preds.groupby("fold")}
    fold_n_keep = preds.groupby("fold")["keep"].sum().to_dict()

    sharpes = np.empty(n_reps)
    for i in range(n_reps):
        filtered = base.copy()
        for f, dates in fold_dates.items():
            n_keep = int(fold_n_keep[f])
            keep = rng.choice(dates, size=n_keep, replace=False)
            drop = np.setdiff1d(dates, keep)
            filtered.loc[pd.DatetimeIndex(drop)] = 0.0
        sharpes[i] = E._daily_eval(filtered)["sharpe"]
    return sharpes


def hmm_state_rule_series(orb: pd.DataFrame,
                          state_features: pd.DataFrame | None = None) -> dict:
    """Phase-3 comparison: HMM training-positive-state tradability rule
    (src.regime_eval.tradability_test), its own OOS window - aligned onto
    a common date range with the other sources by the caller
    (tradability_common_dates)."""
    if state_features is None:
        state_features = G.load_state_features()
    out = E.tradability_test(state_features, orb, "hmm")
    return {"filtered": out["filtered"], "unfiltered": out["unfiltered"]}


def compare_sources(preds: pd.DataFrame, orb: pd.DataFrame,
                    state_features: pd.DataFrame | None = None
                    ) -> tuple[pd.Index, pd.DataFrame]:
    """logistic filter vs the phase-3 HMM state rule, both restricted to
    the date range they actually share (src.regime_eval.
    tradability_common_dates) - "unfiltered" then means the exact same
    thing in both rows."""
    results = {
        "logistic_filter": logistic_filter_series(preds, orb),
        "hmm_state_rule": hmm_state_rule_series(orb, state_features),
    }
    return E.tradability_common_dates(results)


def evaluate_success(preds: pd.DataFrame, orb: pd.DataFrame,
                     min_trades_per_year: int = FL.MIN_TRADES_PER_YEAR,
                     n_reps: int = 1000, seed: int = 0) -> dict:
    """config [significance].success_requires_all_of, evaluated."""
    series = logistic_filter_series(preds, orb)
    ev_f = E._daily_eval(series["filtered"])
    ev_u = E._daily_eval(series["unfiltered"])
    null = random_filter_null(preds, orb, n_reps=n_reps, seed=seed)
    p95 = float(np.percentile(null, 95))

    rows = []
    for f, g in preds.groupby("fold"):
        span_days = (g["date"].max() - g["date"].min()).days + 1
        n_years = max(span_days / 365.25, 1e-9)
        n_keep = int(g["keep"].sum())
        rows.append({"fold": f, "n_keep": n_keep,
                    "trades_per_year": n_keep / n_years})
    by_fold = pd.DataFrame(rows)
    min_tpy_ok = bool((by_fold["trades_per_year"] >= min_trades_per_year).all())

    return {
        "filtered_sharpe": ev_f["sharpe"], "unfiltered_sharpe": ev_u["sharpe"],
        "filtered_max_drawdown": ev_f["max_drawdown"],
        "unfiltered_max_drawdown": ev_u["max_drawdown"],
        "filtered_ann_return": ev_f["ann_return"],
        "unfiltered_ann_return": ev_u["ann_return"],
        "random_null_p95": p95, "random_null_mean": float(null.mean()),
        "beats_unfiltered": bool(ev_f["sharpe"] > ev_u["sharpe"]),
        "beats_random_p95": bool(ev_f["sharpe"] > p95),
        "min_trades_per_year_ok": min_tpy_ok,
        "by_fold_trades_per_year": by_fold,
        "success": bool(ev_f["sharpe"] > ev_u["sharpe"] and ev_f["sharpe"] > p95
                        and min_tpy_ok),
    }


def threshold_curve(preds: pd.DataFrame, orb: pd.DataFrame,
                    retention_grid=FL.RETENTION_GRID) -> pd.DataFrame:
    """Descriptive only: for every retention level in the grid, what would
    trades/year and Sharpe have been on the TEST folds' already-fitted
    probabilities (not the validation-selected one per fold) - shows how
    sensitive the outcome is to the threshold, never used to pick anything."""
    base = logistic_filter_series(preds, orb)["unfiltered"]
    full_range = base.index
    n_years = max(full_range.year.nunique(), 1)
    signal_dates = pd.DatetimeIndex(preds["date"])

    rows = []
    for retention in retention_grid:
        keep_parts = []
        for _, g in preds.groupby("fold"):
            thr = g["y_pred_proba"].quantile(1 - retention)
            keep_parts.append(g.loc[g["y_pred_proba"] >= thr, "date"])
        keep_dates = pd.DatetimeIndex(pd.concat(keep_parts)) if keep_parts else pd.DatetimeIndex([])
        dropped = signal_dates.difference(keep_dates)
        filtered = base.copy()
        filtered.loc[dropped] = 0.0
        ev = E._daily_eval(filtered)
        rows.append({"retention": retention, "n_trades": len(keep_dates),
                    "trades_per_year": len(keep_dates) / n_years,
                    "sharpe": ev["sharpe"], "max_drawdown": ev["max_drawdown"]})
    return pd.DataFrame(rows)


def missed_big_wins(preds: pd.DataFrame, orb: pd.DataFrame,
                    r_threshold: float = 5.0) -> dict:
    """Of ORB's own big-win days (r_net >= r_threshold) inside the filter's
    OOS window, how many did the actual fold-chosen filter drop."""
    s = orb.set_index("day") if "day" in orb.columns else orb
    full_range = _oos_range(preds, orb)
    big = s.index[(s["traded"]) & (s["r_net"] >= r_threshold) &
                  s.index.isin(full_range)]
    p = preds.set_index("date")
    covered = big.intersection(p.index)
    dropped = covered[~p.loc[covered, "keep"]]
    return {"n_big_win_days_in_window": len(big),
           "n_covered_by_predictions": len(covered),
           "n_dropped": len(dropped),
           "frac_dropped": len(dropped) / len(covered) if len(covered) else np.nan,
           "dropped_dates": list(dropped)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from src.models.base import load_predictions
    preds = load_predictions("week08_filter")
    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")
    result = evaluate_success(preds, orb)
    for k, v in result.items():
        if k != "by_fold_trades_per_year":
            log.info("%s: %s", k, v)


if __name__ == "__main__":
    main()
