"""Week 8 (phase 4, week 1): trade-day filter, logistic regression baseline.

Reads ``config/week08_filter.toml`` for the threshold-selection rule
(never picks a threshold by looking at test-fold performance - see the
module's own docstring section there). Reads the open_5m feature matrix
(``src/features_open.py``) and ORB's own daily results
(``data/processed/strategies/orb_daily.parquet``) - never refits ORB or
the feature pipeline itself.

Per fold (``src.validation.WalkForwardSplit`` defaults):

1. Split the training window into a model-fit window (first ~2 years) and
   a validation window (last ~1 year), calendar-based
   (``test_start - 1 year``).
2. Fit ``Pipeline(StandardScaler -> LogisticRegression(class_weight=
   "balanced"))`` on the model-fit window's ORB-signal (``traded==True``)
   days only - doji/invalid-session days are dropped from the modeling set
   entirely (:func:`build_labels`), never coded as a negative label.
3. Score the validation window with this model. For each retention rate in
   ``RETENTION_GRID``, the candidate threshold is this model's validation-
   window probability quantile giving that retention; the validation-
   window cost-adjusted Sharpe (:func:`_daily_eval` on the full-calendar
   ``bps_return`` series, doji days contributing 0 either way - same
   convention as ``src.regime_eval``'s tradability test) picks the best
   retention among those with >= ``min_trades_per_year`` trades/year.
   100% retention (no filter) is always a candidate.
4. Apply this exact model and threshold, unchanged, to the test fold.

The test fold never influences model fitting, scaling, or threshold
selection. Outputs one row per (test-fold, signal day):
``date, fold, model, target, y_true, y_pred_proba, threshold,
retention_selected, keep`` - saved to
``data/processed/predictions/week08_filter.parquet``, read back with
``src.models.base.load_predictions("week08_filter")``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src import data_loader as dl
from src import regime_eval as E
from src.features_open import OPEN_FEATURE_COLUMNS
from src.models.base import PRED_DIR
from src.validation import WalkForwardSplit

log = logging.getLogger(__name__)
RETENTION_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
MIN_TRADES_PER_YEAR = 20


def build_labels(orb: pd.DataFrame) -> pd.DataFrame:
    """Binary label (r_net > 0) + continuous r_net, ORB-signal
    (``traded==True``) days only - doji/invalid-session days have nothing
    for the filter to decide and are dropped, not coded as a negative
    (config/week08_filter.toml [label])."""
    s = orb.set_index("day") if "day" in orb.columns else orb
    tr = s[s["traded"]]
    return pd.DataFrame({"label": (tr["r_net"] > 0).astype(int),
                         "r_net": tr["r_net"]}, index=tr.index)


def _validation_score(bps_full: pd.Series, kept_dates: pd.DatetimeIndex,
                      signal_dates: pd.DatetimeIndex,
                      min_trades_per_year: int) -> tuple[float, float]:
    """Sharpe of the validation window with only ``kept_dates`` (a subset
    of ``signal_dates``) trading, every other day (doji, and dropped
    signal days) contributing 0 - same full-calendar convention as
    :func:`src.regime_eval._daily_eval`'s callers. Returns (sharpe,
    trades_per_year)."""
    dropped = signal_dates.difference(kept_dates)
    filtered = bps_full.copy()
    filtered.loc[dropped] = 0.0
    n_years = max(bps_full.index.year.nunique(), 1)
    sharpe = E._daily_eval(filtered)["sharpe"]
    return sharpe, len(kept_dates) / n_years


def _fit_fold(Xy: pd.DataFrame, cols: list[str], f, s: pd.DataFrame,
             X_index: pd.DatetimeIndex, retention_grid, min_trades_per_year: int):
    """One fold's model-fit + threshold-selection - the inner loop body
    shared by :func:`fit_filter_folds` (needs the test-fold predictions)
    and :func:`coefficient_table` (needs the fitted pipeline) so there is
    exactly one implementation of the nested split. Returns ``None`` if the
    fold is skipped (too little fit/validation data), else ``(pipe, best,
    test_rows)`` where ``test_rows`` is ``Xy`` restricted to this fold's
    test dates."""
    train_dates = X_index[f.train_idx]
    test_dates = X_index[f.test_idx]
    validation_start = f.test_start - pd.DateOffset(years=1)
    fit_dates = train_dates[train_dates < validation_start]
    val_dates = train_dates[train_dates >= validation_start]

    fit_rows = Xy.loc[Xy.index.isin(fit_dates)]
    val_rows = Xy.loc[Xy.index.isin(val_dates)]
    if len(fit_rows) < 30 or len(val_rows) < 30:
        return None

    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced",
                                                max_iter=1000))])
    pipe.fit(fit_rows[cols], fit_rows["label"])

    val_proba = pd.Series(pipe.predict_proba(val_rows[cols])[:, 1],
                          index=val_rows.index)
    val_signal_dates = val_rows.index
    bps_val_full = s["bps_return"].reindex(val_dates).fillna(0.0)

    best = None
    for retention in retention_grid:
        threshold = float(val_proba.quantile(1 - retention))
        kept = val_signal_dates[val_proba >= threshold]
        sharpe, tpy = _validation_score(bps_val_full, kept,
                                        val_signal_dates,
                                        min_trades_per_year)
        if tpy < min_trades_per_year:
            continue
        if best is None or sharpe > best["sharpe"] or \
                (sharpe == best["sharpe"] and retention > best["retention"]):
            best = {"retention": retention, "threshold": threshold,
                   "sharpe": sharpe}
    if best is None:      # shouldn't happen (100% always qualifies if
        best = {"retention": 1.0,   # ORB itself clears the bar), but
                "threshold": float(val_proba.min()), "sharpe": np.nan}

    test_rows = Xy.loc[Xy.index.isin(test_dates)]
    return pipe, best, test_rows


def fit_filter_folds(X: pd.DataFrame, orb: pd.DataFrame,
                     retention_grid=RETENTION_GRID,
                     min_trades_per_year: int = MIN_TRADES_PER_YEAR,
                     splitter: WalkForwardSplit = WalkForwardSplit()
                     ) -> pd.DataFrame:
    """The full per-fold nested fit + threshold-selection procedure - see
    the module docstring. ``X`` is the open_5m feature matrix (only
    ``OPEN_FEATURE_COLUMNS`` are used - the 2019+-only robustness extras
    are never part of this main-analysis fit)."""
    s = orb.set_index("day") if "day" in orb.columns else orb
    labels = build_labels(s)
    cols = list(OPEN_FEATURE_COLUMNS)
    Xy = X[cols].join(labels, how="inner").dropna(subset=cols)

    rows = []
    for f in splitter.folds(X.index):
        out = _fit_fold(Xy, cols, f, s, X.index, retention_grid, min_trades_per_year)
        if out is None:
            continue
        pipe, best, test_rows = out
        if test_rows.empty:
            continue
        test_proba = pd.Series(pipe.predict_proba(test_rows[cols])[:, 1],
                               index=test_rows.index)
        for d in test_rows.index:
            p = float(test_proba.loc[d])
            rows.append({"date": d, "fold": f.number, "model": "logistic",
                        "target": "net_r_positive",
                        "y_true": int(test_rows.loc[d, "label"]),
                        "y_pred_proba": p, "threshold": best["threshold"],
                        "retention_selected": best["retention"],
                        "keep": p >= best["threshold"]})
        log.info("fold %d: retention=%.0f%% threshold=%.4f (val sharpe=%.3f)",
                 f.number, best["retention"] * 100, best["threshold"],
                 best["sharpe"])
    return pd.DataFrame(rows)


def coefficient_table(X: pd.DataFrame, orb: pd.DataFrame,
                      retention_grid=RETENTION_GRID,
                      min_trades_per_year: int = MIN_TRADES_PER_YEAR,
                      splitter: WalkForwardSplit = WalkForwardSplit()
                      ) -> pd.DataFrame:
    """Per-fold standardized logistic coefficients (one row per fold, one
    column per feature) - the exact deployed model from :func:`_fit_fold`,
    not a re-fit, so this matches :func:`fit_filter_folds`'s predictions
    fold for fold. Coefficients are on the standardized (StandardScaler
    output) scale, so directly comparable across features within a fold."""
    s = orb.set_index("day") if "day" in orb.columns else orb
    labels = build_labels(s)
    cols = list(OPEN_FEATURE_COLUMNS)
    Xy = X[cols].join(labels, how="inner").dropna(subset=cols)

    rows = []
    for f in splitter.folds(X.index):
        out = _fit_fold(Xy, cols, f, s, X.index, retention_grid, min_trades_per_year)
        if out is None:
            continue
        pipe, best, _ = out
        coef = pipe.named_steps["clf"].coef_[0]
        row = {"fold": f.number, "intercept": float(pipe.named_steps["clf"].intercept_[0]),
              "retention_selected": best["retention"]}
        row.update({c: float(v) for c, v in zip(cols, coef)})
        rows.append(row)
    return pd.DataFrame(rows)


def auc(preds: pd.DataFrame) -> float:
    return float(roc_auc_score(preds["y_true"], preds["y_pred_proba"]))


def decile_calibration(preds: pd.DataFrame, orb: pd.DataFrame, n_bins: int = 10
                       ) -> pd.DataFrame:
    """Predicted-probability decile -> actual win rate and mean net R
    (calibration diagnostic, not used by the threshold rule itself)."""
    s = orb.set_index("day") if "day" in orb.columns else orb
    j = preds.set_index("date").join(s[["r_net"]], how="left")
    j["decile"] = pd.qcut(j["y_pred_proba"], n_bins, labels=False, duplicates="drop")
    out = j.groupby("decile").agg(
        n=("y_true", "size"), proba_mean=("y_pred_proba", "mean"),
        win_rate=("y_true", "mean"), mean_r_net=("r_net", "mean"))
    return out.reset_index()


def build_and_save(processed_dir: Path = dl.PROCESSED_DIR) -> Path:
    X = pd.read_parquet(Path(processed_dir) / "features_open_5m.parquet")
    orb = pd.read_parquet(Path(processed_dir) / "strategies" / "orb_daily.parquet")
    preds = fit_filter_folds(X, orb)
    p = PRED_DIR / "week08_filter.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(p, index=False)
    return p


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = build_and_save()
    log.info("wrote %s", p)


if __name__ == "__main__":
    main()
