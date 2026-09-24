"""Week 11 (phase 5, week 1): flattened-sequence baselines for tasks A/B,
and the primary CNN-vs-baseline comparison test.

Every model here fits on each fold's MODEL-FIT WINDOW ONLY (the first
~2 years of the 3-year training window, excluding the last
``validation_years``) - the same window ``src.models.cnn`` trains its
weights on, never the full 3-year training window - so a "CNN vs best
baseline" AUC comparison is between models that saw identical training
data. No threshold selection or early stopping here: every baseline's
hyperparameters are fixed in advance (config/week11_dl.toml
[baselines]), so there is nothing the validation window would be used
for on the baseline side.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.filter import make_xgboost
from src.validation import WalkForwardSplit

log = logging.getLogger(__name__)


def flatten(tensor: np.ndarray) -> np.ndarray:
    return tensor.reshape(tensor.shape[0], -1)


def make_logistic_l2(y: pd.Series | None = None) -> Pipeline:
    """L2-penalized (sklearn default) logistic regression, standardized -
    same factory-signature convention (unused ``y`` arg) as
    ``src.models.filter``'s model factories, for interface consistency."""
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", class_weight="balanced",
                                                max_iter=1000))])


def fit_predict_folds(X: pd.DataFrame, y: pd.Series, model_factory, model_name: str,
                      target_name: str, validation_years: int = 1,
                      splitter: WalkForwardSplit = WalkForwardSplit(),
                      min_fit_rows: int = 30) -> pd.DataFrame:
    """Per fold: fit ``model_factory(y_fit)`` on the model-fit window only,
    predict probabilities on the test window. ``X`` is indexed by day
    (any flattened feature matrix - a flattened sequence tensor or
    week 8's hand-crafted OPEN_FEATURE_COLUMNS); ``y`` shares that index.
    Returns the same ``date, fold, model, target, y_true, y_pred_proba``
    schema as ``src.models.filter.fit_filter_folds_generic``'s output, so
    ``src.models.filter.auc`` works on it unchanged."""
    days = X.index
    rows = []
    for f in splitter.folds(days):
        train_dates = days[f.train_idx]
        test_dates = days[f.test_idx]
        validation_start = f.test_start - pd.DateOffset(years=validation_years)
        fit_dates = train_dates[train_dates < validation_start]

        Xfit, yfit = X.loc[fit_dates], y.loc[fit_dates]
        Xtest, ytest = X.loc[test_dates], y.loc[test_dates]
        if len(Xfit) < min_fit_rows or Xtest.empty or yfit.nunique() < 2:
            continue

        model = model_factory(yfit)
        model.fit(Xfit, yfit)
        proba = model.predict_proba(Xtest)[:, 1]
        for d, p, t in zip(test_dates, proba, ytest):
            rows.append({"date": d, "fold": f.number, "model": model_name,
                        "target": target_name, "y_true": int(t),
                        "y_pred_proba": float(p)})
        log.info("%s fold %d: n_fit=%d n_test=%d", model_name, f.number,
                 len(Xfit), len(Xtest))
    return pd.DataFrame(rows)


def decile_win_rate(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Predicted-probability decile -> actual win rate (label==1 rate) -
    task-agnostic version of ``src.models.filter.decile_calibration`` that
    doesn't assume an ``r_net`` column exists to join against."""
    j = preds.copy()
    j["decile"] = pd.qcut(j["y_pred_proba"], n_bins, labels=False, duplicates="drop")
    out = j.groupby("decile").agg(n=("y_true", "size"),
                                  proba_mean=("y_pred_proba", "mean"),
                                  win_rate=("y_true", "mean"))
    return out.reset_index()
