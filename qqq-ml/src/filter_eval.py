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

from src import backtest as BT
from src import data_loader as dl
from src import regime_eval as E
from src import rules as R
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


# --------------------------------------------------------------------------
# Pre-conclusion checks (requested before the report's finding is finalized)
# --------------------------------------------------------------------------
def block_bootstrap_significance(preds: pd.DataFrame, orb: pd.DataFrame,
                                 block_size: int = 20, n_boot: int = 2000,
                                 seed: int = 0) -> dict:
    """Moving-block-bootstrap p-value for filtered Sharpe - unfiltered
    Sharpe != 0 (``src.backtest.block_bootstrap_sharpe_diff``, reused as-is
    - both series already share an index and serial dependence structure,
    exactly what that function assumes)."""
    series = logistic_filter_series(preds, orb)
    obs, p = BT.block_bootstrap_sharpe_diff(
        series["filtered"] / 1e4, series["unfiltered"] / 1e4,
        block_size=block_size, n_boot=n_boot, seed=seed)
    return {"observed_sharpe_diff": obs, "p_value": p, "block_size": block_size,
           "n_boot": n_boot}


def random_filter_p_value(preds: pd.DataFrame, orb: pd.DataFrame,
                          null: np.ndarray | None = None,
                          n_reps: int = 1000, seed: int = 0) -> dict:
    """Exact rank of the actually-observed filtered Sharpe inside the
    random-filter null (config [significance.random_filter_null]) - the
    single number ``beats_random_p95`` in :func:`evaluate_success`
    summarizes as a threshold check."""
    if null is None:
        null = random_filter_null(preds, orb, n_reps=n_reps, seed=seed)
    observed = E._daily_eval(logistic_filter_series(preds, orb)["filtered"])["sharpe"]
    n_at_or_above = int(np.sum(null >= observed))
    return {"observed_sharpe": observed, "n_reps": len(null),
           "rank_from_top": n_at_or_above + 1,
           "p_value": n_at_or_above / len(null)}


def yearly_breakdown(preds: pd.DataFrame, orb: pd.DataFrame) -> pd.DataFrame:
    """Filtered vs unfiltered Sharpe by calendar year - checks the overall
    improvement isn't concentrated in one or two years."""
    series = logistic_filter_series(preds, orb)
    rows = []
    for y, idx in series["unfiltered"].groupby(series["unfiltered"].index.year).groups.items():
        f_ev = E._daily_eval(series["filtered"].loc[idx])
        u_ev = E._daily_eval(series["unfiltered"].loc[idx])
        rows.append({"year": y, "n_days": len(idx),
                    "filtered_sharpe": f_ev["sharpe"], "unfiltered_sharpe": u_ev["sharpe"],
                    "filtered_ann_return": f_ev["ann_return"],
                    "unfiltered_ann_return": u_ev["ann_return"]})
    return pd.DataFrame(rows)


def fold_breakdown(preds: pd.DataFrame, orb: pd.DataFrame) -> pd.DataFrame:
    """Filtered vs unfiltered Sharpe by test fold - same check as
    :func:`yearly_breakdown` at fold granularity."""
    series = logistic_filter_series(preds, orb)
    rows = []
    for f, g in preds.groupby("fold"):
        fold_dates = pd.DatetimeIndex(sorted(g["date"].unique()))
        full_range = series["unfiltered"].index[
            (series["unfiltered"].index >= fold_dates.min()) &
            (series["unfiltered"].index <= fold_dates.max())]
        f_ev = E._daily_eval(series["filtered"].loc[full_range])
        u_ev = E._daily_eval(series["unfiltered"].loc[full_range])
        rows.append({"fold": f, "n_days": len(full_range),
                    "filtered_sharpe": f_ev["sharpe"], "unfiltered_sharpe": u_ev["sharpe"]})
    return pd.DataFrame(rows)


def cost_sensitivity(preds: pd.DataFrame, orb_variants: dict[str, pd.DataFrame] | None = None,
                     minute: pd.DataFrame | None = None) -> pd.DataFrame:
    """Re-evaluates the ALREADY-CHOSEN keep/drop decisions (unchanged -
    this never refits or reselects a threshold) against ORB re-run at
    different cost assumptions: gross (no cost at all) and 2x slippage.
    Isolates whether the filtered-vs-unfiltered Sharpe gap survives
    without any cost-avoidance-by-trading-less effect.

    ``orb_variants`` lets callers (tests) inject already-computed ORB runs
    instead of re-running ``src.rules.run_orb`` from 1-min bars (slow, and
    needs real data) - same DI pattern as
    ``src.features_open.load_extras``'s ``pred``/``hmm_labels`` params."""
    if orb_variants is None:
        if minute is None:
            minute = dl.load_minute_rth()
        configs = {
            "main_cost": R.ORBConfig(),
            "gross_no_cost": R.ORBConfig(cost_per_share=0.0),
            "2x_slippage": R.ORBConfig(cost_per_share=R.MAIN_COMMISSION + 2 * R.MAIN_SLIPPAGE),
        }
        orb_variants = {name: R.run_orb(minute, cfg=cfg) for name, cfg in configs.items()}

    rows = []
    for name, orb_variant in orb_variants.items():
        series = logistic_filter_series(preds, orb_variant)
        f_ev = E._daily_eval(series["filtered"])
        u_ev = E._daily_eval(series["unfiltered"])
        rows.append({"cost_config": name,
                    "filtered_sharpe": f_ev["sharpe"], "unfiltered_sharpe": u_ev["sharpe"],
                    "filtered_ann_return": f_ev["ann_return"],
                    "unfiltered_ann_return": u_ev["ann_return"],
                    "sharpe_gap": f_ev["sharpe"] - u_ev["sharpe"]})
    return pd.DataFrame(rows)


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
