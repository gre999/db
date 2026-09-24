"""Trade-day filter: logistic (week 8), tree models and a regression
baseline (week 9).

Reads ``config/week08_filter.toml`` / ``config/week09_filter.toml`` for
the threshold-selection rule (never picks a threshold by looking at
test-fold performance - see those files' own docstring sections). Reads
the open_5m feature matrix (``src/features_open.py``) and ORB's own daily
results (``data/processed/strategies/orb_daily.parquet``) - never refits
ORB or the feature pipeline itself.

Per fold (``src.validation.WalkForwardSplit`` defaults), model-agnostic
(:func:`_fit_fold`, shared by every model below):

1. Split the training window into a model-fit window (first ~2 years) and
   a validation window (last ~1 year), calendar-based
   (``test_start - 1 year``).
2. Fit the model on the model-fit window's ORB-signal (``traded==True``)
   days only - doji/invalid-session days are dropped from the modeling set
   entirely (:func:`build_labels`), never coded as a negative label.
3. Score the validation window with this model. For each retention rate in
   ``RETENTION_GRID``, the candidate threshold is this model's validation-
   window score quantile giving that retention (probability for a
   classifier, raw predicted value for the regression baseline); the
   validation-window cost-adjusted Sharpe (:func:`_daily_eval` on the
   full-calendar ``bps_return`` series, doji days contributing 0 either
   way - same convention as ``src.regime_eval``'s tradability test) picks
   the best retention among those with >= ``min_trades_per_year``
   trades/year. 100% retention (no filter) is always a candidate.
4. Apply this exact model and threshold, unchanged, to the test fold.

The test fold never influences model fitting, scaling, or threshold
selection. ``fit_filter_folds``/``fit_filter_folds_generic`` output one row
per (test-fold, signal day): ``date, fold, model, target, y_true,
y_pred_proba, threshold, retention_selected, keep`` (``y_pred_proba`` is a
probability for classifiers, the raw predicted r_net for the regression
baseline - same column name for interface consistency).
"""
from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import data_loader as dl
from src import regime_eval as E
from src.features_open import OPEN_FEATURE_COLUMNS
from src.models.base import PRED_DIR
from src.validation import WalkForwardSplit

log = logging.getLogger(__name__)
RETENTION_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
MIN_TRADES_PER_YEAR = 20

ModelFactory = Callable[[pd.Series | None], object]


# --------------------------------------------------------------------------
# Model factories (config/week09_filter.toml [tree_models], [regression_
# baseline] - fixed hyperparameters, no per-fold tuning; each takes the
# model-fit window's y so scale_pos_weight can be computed per fold)
# --------------------------------------------------------------------------
def make_logistic(y: pd.Series | None = None) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced",
                                                max_iter=1000))])


def make_random_forest(y: pd.Series | None = None, max_depth: int = 4,
                       min_samples_leaf: int = 20, n_estimators: int = 300,
                       random_state: int = 0,
                       max_features: str = "sqrt") -> RandomForestClassifier:
    return RandomForestClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight="balanced", n_estimators=n_estimators,
        random_state=random_state, max_features=max_features, n_jobs=-1)


def make_xgboost(y: pd.Series | None = None, max_depth: int = 2,
                 learning_rate: float = 0.05, min_child_weight: int = 10,
                 n_estimators: int = 200, subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 0) -> XGBClassifier:
    """Fixed hyperparameters (config/week09_filter.toml [tree_models.
    xgboost]) - amended before running to drop early stopping (the fit
    window's last 6 months held only ~30 winning trades, too noisy to
    early-stop against, and would have cost a quarter of the training
    data): fixed n_estimators=200 on the full model-fit window instead.
    scale_pos_weight = n_negative/n_positive of the model-fit window
    itself (the only data this model ever sees for scale_pos_weight or
    fitting)."""
    spw = 1.0
    if y is not None:
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
    return XGBClassifier(
        max_depth=max_depth, learning_rate=learning_rate,
        min_child_weight=min_child_weight, n_estimators=n_estimators,
        subsample=subsample, colsample_bytree=colsample_bytree,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=random_state,
        scale_pos_weight=spw, n_jobs=4)


def make_huber(y: pd.Series | None = None) -> Pipeline:
    """sklearn defaults (epsilon=1.35, alpha=1e-4), not tuned - secondary/
    regression baseline (config/week09_filter.toml [regression_baseline])."""
    return Pipeline([("scale", StandardScaler()), ("reg", HuberRegressor())])


MODEL_FACTORIES: dict[str, tuple[ModelFactory, str]] = {
    "logistic": (make_logistic, "label"),
    "random_forest": (make_random_forest, "label"),
    "xgboost": (make_xgboost, "label"),
    "xgboost_depth3": (partial(make_xgboost, max_depth=3), "label"),
    "huber": (make_huber, "r_net"),
}


def _score(model, X) -> np.ndarray:
    """Ranking score: predicted P(win) for a classifier, raw predicted
    r_net for a regressor - the retention-grid quantile-threshold logic
    downstream is identical either way (both just rank days by value)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X), dtype=float)


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


def _prep_xy(X: pd.DataFrame, orb: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    s = orb.set_index("day") if "day" in orb.columns else orb
    labels = build_labels(s)
    return X[cols].join(labels, how="inner").dropna(subset=cols)


def _fit_fold(Xy: pd.DataFrame, cols: list[str], f, s: pd.DataFrame,
             X_index: pd.DatetimeIndex, retention_grid, min_trades_per_year: int,
             model_factory: ModelFactory = make_logistic, y_col: str = "label"):
    """One fold's model-fit + threshold-selection - the inner loop body
    shared by every model (:func:`fit_filter_folds_generic`,
    :func:`coefficient_table`, :func:`fold_overfitting_table`,
    :func:`permutation_importance_table`) so there is exactly one
    implementation of the nested split, whatever the model. Returns
    ``None`` if the fold is skipped (too little fit/validation data), else
    ``(model, best, fit_rows, test_rows)`` where ``fit_rows``/``test_rows``
    are ``Xy`` restricted to this fold's model-fit/test dates."""
    train_dates = X_index[f.train_idx]
    test_dates = X_index[f.test_idx]
    validation_start = f.test_start - pd.DateOffset(years=1)
    fit_dates = train_dates[train_dates < validation_start]
    val_dates = train_dates[train_dates >= validation_start]

    fit_rows = Xy.loc[Xy.index.isin(fit_dates)]
    val_rows = Xy.loc[Xy.index.isin(val_dates)]
    if len(fit_rows) < 30 or len(val_rows) < 30:
        return None

    model = model_factory(fit_rows[y_col])
    model.fit(fit_rows[cols], fit_rows[y_col])

    val_score = pd.Series(_score(model, val_rows[cols]), index=val_rows.index)
    val_signal_dates = val_rows.index
    bps_val_full = s["bps_return"].reindex(val_dates).fillna(0.0)

    best = None
    for retention in retention_grid:
        threshold = float(val_score.quantile(1 - retention))
        kept = val_signal_dates[val_score >= threshold]
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
                "threshold": float(val_score.min()), "sharpe": np.nan}

    test_rows = Xy.loc[Xy.index.isin(test_dates)]
    return model, best, fit_rows, test_rows


def fit_filter_folds_generic(X: pd.DataFrame, orb: pd.DataFrame,
                             model_factory: ModelFactory, model_name: str,
                             y_col: str = "label",
                             target_name: str = "net_r_positive",
                             feature_cols: list[str] | None = None,
                             retention_grid=RETENTION_GRID,
                             min_trades_per_year: int = MIN_TRADES_PER_YEAR,
                             splitter: WalkForwardSplit = WalkForwardSplit()
                             ) -> pd.DataFrame:
    """The full per-fold nested fit + threshold-selection procedure for any
    model - see the module docstring. ``feature_cols`` defaults to
    ``OPEN_FEATURE_COLUMNS`` (the main-analysis feature set); pass the
    2019+-only robustness extras explicitly (config/week09_filter.toml
    [subsample_2019]) to include them, restricting ``X``/``orb`` to the
    2019+ subsample first so both the with/without-extras runs share the
    same date range."""
    cols = list(feature_cols) if feature_cols is not None else list(OPEN_FEATURE_COLUMNS)
    Xy = _prep_xy(X, orb, cols)
    s = orb.set_index("day") if "day" in orb.columns else orb

    rows = []
    for f in splitter.folds(X.index):
        out = _fit_fold(Xy, cols, f, s, X.index, retention_grid,
                        min_trades_per_year, model_factory, y_col)
        if out is None:
            continue
        model, best, _, test_rows = out
        if test_rows.empty:
            continue
        test_score = pd.Series(_score(model, test_rows[cols]), index=test_rows.index)
        for d in test_rows.index:
            p = float(test_score.loc[d])
            rows.append({"date": d, "fold": f.number, "model": model_name,
                        "target": target_name,
                        "y_true": int(test_rows.loc[d, "label"]),
                        "y_pred_proba": p, "threshold": best["threshold"],
                        "retention_selected": best["retention"],
                        "keep": p >= best["threshold"]})
        log.info("%s fold %d: retention=%.0f%% threshold=%.4f (val sharpe=%.3f)",
                 model_name, f.number, best["retention"] * 100, best["threshold"],
                 best["sharpe"])
    return pd.DataFrame(rows)


def fit_filter_folds(X: pd.DataFrame, orb: pd.DataFrame,
                     retention_grid=RETENTION_GRID,
                     min_trades_per_year: int = MIN_TRADES_PER_YEAR,
                     splitter: WalkForwardSplit = WalkForwardSplit()
                     ) -> pd.DataFrame:
    """Week 8's logistic baseline - thin wrapper over
    :func:`fit_filter_folds_generic`, kept for backward compatibility."""
    return fit_filter_folds_generic(X, orb, make_logistic, "logistic",
                                    y_col="label", retention_grid=retention_grid,
                                    min_trades_per_year=min_trades_per_year,
                                    splitter=splitter)


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
    cols = list(OPEN_FEATURE_COLUMNS)
    Xy = _prep_xy(X, orb, cols)
    s = orb.set_index("day") if "day" in orb.columns else orb

    rows = []
    for f in splitter.folds(X.index):
        out = _fit_fold(Xy, cols, f, s, X.index, retention_grid,
                        min_trades_per_year, make_logistic, "label")
        if out is None:
            continue
        model, best, _, _ = out
        coef = model.named_steps["clf"].coef_[0]
        row = {"fold": f.number,
              "intercept": float(model.named_steps["clf"].intercept_[0]),
              "retention_selected": best["retention"]}
        row.update({c: float(v) for c, v in zip(cols, coef)})
        rows.append(row)
    return pd.DataFrame(rows)


def fold_overfitting_table(X: pd.DataFrame, orb: pd.DataFrame,
                           model_factory: ModelFactory, model_name: str,
                           retention_grid=RETENTION_GRID,
                           min_trades_per_year: int = MIN_TRADES_PER_YEAR,
                           splitter: WalkForwardSplit = WalkForwardSplit()
                           ) -> pd.DataFrame:
    """Fit-window (in-sample) vs test-fold (OOS) AUC per fold - config
    [diagnostics.overfitting_check]: check this FIRST, before trusting a
    tree model that looks much better than logistic on point estimates.
    Classifiers only (``y_col="label"`` always - AUC needs a binary
    target)."""
    cols = list(OPEN_FEATURE_COLUMNS)
    Xy = _prep_xy(X, orb, cols)
    s = orb.set_index("day") if "day" in orb.columns else orb

    rows = []
    for f in splitter.folds(X.index):
        out = _fit_fold(Xy, cols, f, s, X.index, retention_grid,
                        min_trades_per_year, model_factory, "label")
        if out is None:
            continue
        model, _, fit_rows, test_rows = out
        if test_rows.empty or fit_rows["label"].nunique() < 2 \
                or test_rows["label"].nunique() < 2:
            continue
        fit_score = _score(model, fit_rows[cols])
        test_score = _score(model, test_rows[cols])
        rows.append({"fold": f.number, "model": model_name,
                    "fit_auc": float(roc_auc_score(fit_rows["label"], fit_score)),
                    "test_auc": float(roc_auc_score(test_rows["label"], test_score)),
                    "n_fit": len(fit_rows), "n_test": len(test_rows)})
    return pd.DataFrame(rows)


def permutation_importance_table(X: pd.DataFrame, orb: pd.DataFrame,
                                 model_factory: ModelFactory, model_name: str,
                                 scoring: str = "roc_auc", n_repeats: int = 20,
                                 random_state: int = 0,
                                 retention_grid=RETENTION_GRID,
                                 min_trades_per_year: int = MIN_TRADES_PER_YEAR,
                                 splitter: WalkForwardSplit = WalkForwardSplit()
                                 ) -> pd.DataFrame:
    """Test-fold permutation importance (config [diagnostics.
    permutation_importance]) - a post-hoc diagnostic on the already-
    deployed per-fold model, never used to select or filter features.
    Classifiers only."""
    cols = list(OPEN_FEATURE_COLUMNS)
    Xy = _prep_xy(X, orb, cols)
    s = orb.set_index("day") if "day" in orb.columns else orb

    rows = []
    for f in splitter.folds(X.index):
        out = _fit_fold(Xy, cols, f, s, X.index, retention_grid,
                        min_trades_per_year, model_factory, "label")
        if out is None:
            continue
        model, _, _, test_rows = out
        if test_rows.empty or test_rows["label"].nunique() < 2:
            continue
        r = permutation_importance(model, test_rows[cols], test_rows["label"],
                                   scoring=scoring, n_repeats=n_repeats,
                                   random_state=random_state)
        row = {"fold": f.number, "model": model_name}
        row.update({c: float(v) for c, v in zip(cols, r.importances_mean)})
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
    """Week 8's logistic baseline only - kept for backward compatibility.
    See :func:`build_and_save_model` for week 9's tree/regression models."""
    X = pd.read_parquet(Path(processed_dir) / "features_open_5m.parquet")
    orb = pd.read_parquet(Path(processed_dir) / "strategies" / "orb_daily.parquet")
    preds = fit_filter_folds(X, orb)
    p = PRED_DIR / "week08_filter.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(p, index=False)
    return p


def build_and_save_model(model_key: str, out_name: str | None = None,
                         processed_dir: Path = dl.PROCESSED_DIR) -> Path:
    """Any model in :data:`MODEL_FACTORIES`, saved to
    ``data/processed/predictions/{out_name or 'week09_filter_' + model_key}
    .parquet``."""
    factory, y_col = MODEL_FACTORIES[model_key]
    X = pd.read_parquet(Path(processed_dir) / "features_open_5m.parquet")
    orb = pd.read_parquet(Path(processed_dir) / "strategies" / "orb_daily.parquet")
    preds = fit_filter_folds_generic(X, orb, factory, model_key, y_col=y_col)
    name = out_name or f"week09_filter_{model_key}"
    p = PRED_DIR / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(p, index=False)
    return p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="logistic", choices=list(MODEL_FACTORIES))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.model == "logistic":
        out = build_and_save()
    else:
        out = build_and_save_model(args.model)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
