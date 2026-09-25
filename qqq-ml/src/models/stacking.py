"""Week 12 (phase 5, week 2): incremental-information check for task A
(config/week12_dl.toml [incremental_info]) - does the CNN's score carry
information the hand-crafted features don't already have.

Three sub-checks, all on already-fitted/already-frozen models (never
refits the CNN or the hand-feature logistic):

1. Spearman correlation between the CNN's test-fold score and each of a
   few named hand-crafted features (:func:`spearman_by_fold`).
2. OLS R² of the CNN's logit on all 14 OPEN_FEATURE_COLUMNS
   (:func:`r_squared_per_fold`, :func:`r_squared_pooled_standardized`).
3. Stacking (:func:`fit_stacking_fold`): a 2-input (hand-feature logit,
   CNN logit) logistic regression, fit ONLY on the validation window (the
   CNN's own model-fit-window scores are in-sample for the CNN, so only
   the validation window is a fair OOS-for-the-CNN training signal for a
   second-stage model), applied unchanged to the test window. The CNN
   coefficient's significance uses a hand-rolled Wald test
   (:func:`wald_test`) on an UNREGULARIZED logistic fit (sklearn
   ``LogisticRegression(penalty=None)``) - cross-checked against
   statsmodels in tests/test_stacking.py (dev-only dependency,
   requirements-dev.txt; no runtime dependency here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def wald_test(X: np.ndarray, y: np.ndarray) -> dict:
    """Unregularized logistic regression (MLE) + a classical Wald test on
    every coefficient (intercept first). Asymptotic covariance
    ``(X'WX)^-1``, ``W = diag(p_i(1-p_i))`` at the fitted probabilities -
    the standard formula for logistic-regression MLE inference, computed
    by hand rather than via an external package (cross-checked against
    statsmodels.Logit in tests/test_stacking.py)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = LogisticRegression(penalty=None, max_iter=2000).fit(X, y)
    p_hat = model.predict_proba(X)[:, 1]
    w = p_hat * (1 - p_hat)
    Xd = np.column_stack([np.ones(len(X)), X])
    fisher = (Xd * w[:, None]).T @ Xd
    cov = np.linalg.inv(fisher)
    se = np.sqrt(np.diag(cov))
    coef = np.concatenate([model.intercept_, model.coef_[0]])
    z = coef / se
    pval = 2 * stats.norm.sf(np.abs(z))
    return {"coef": coef, "se": se, "z": z, "p": pval, "cov": cov, "model": model}


def fit_stacking_fold(hand_val_proba: np.ndarray, cnn_val_proba: np.ndarray,
                      y_val: np.ndarray, hand_test_proba: np.ndarray,
                      cnn_test_proba: np.ndarray) -> dict:
    """One fold's stacking model: fit on the VALIDATION window's (hand
    logit, CNN logit) pair, applied to the test window, never refit. See
    the module docstring for why the validation window (not the model-fit
    window) is the only valid training data for this second-stage model."""
    X_val = np.column_stack([logit(hand_val_proba), logit(cnn_val_proba)])
    X_test = np.column_stack([logit(hand_test_proba), logit(cnn_test_proba)])
    wald = wald_test(X_val, y_val)
    model = wald["model"]
    test_proba = model.predict_proba(X_test)[:, 1]
    return {"test_proba": test_proba, "wald": wald,
           "intercept": float(wald["coef"][0]),
           "hand_coef": float(wald["coef"][1]), "hand_se": float(wald["se"][1]),
           "hand_p": float(wald["p"][1]),
           "cnn_coef": float(wald["coef"][2]), "cnn_se": float(wald["se"][2]),
           "cnn_p": float(wald["p"][2])}


def spearman_by_fold(preds: pd.DataFrame, features: pd.DataFrame,
                     feature_names: list[str]) -> pd.DataFrame:
    """Per fold, Spearman correlation between the CNN's test-fold
    probability and each named feature's own test-fold value. Spearman is
    rank-based, so probability vs logit gives identical correlations."""
    j = preds.set_index("date").join(features[feature_names], how="inner")
    rows = []
    for f, g in j.groupby("fold"):
        row = {"fold": f, "n": len(g)}
        for name in feature_names:
            row[name] = g["y_pred_proba"].corr(g[name], method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def _r_squared(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS y ~ X (intercept added internally). Returns R²/adjusted R²."""
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
    return {"n": n, "k": k, "r2": r2, "adj_r2": adj_r2}


def r_squared_per_fold(preds: pd.DataFrame, features: pd.DataFrame,
                       feature_names: list[str]) -> pd.DataFrame:
    """Config [incremental_info.r_squared] PRIMARY: one OLS per fold
    (CNN logit ~ all named features), not pooled - each fold's CNN is a
    separately-trained model with its own logit scale/calibration."""
    j = preds.set_index("date").join(features[feature_names], how="inner")
    rows = []
    for f, g in j.groupby("fold"):
        y = logit(g["y_pred_proba"].to_numpy())
        X = g[feature_names].to_numpy()
        out = _r_squared(y, X)
        out["fold"] = f
        rows.append(out)
    return pd.DataFrame(rows)[["fold", "n", "k", "r2", "adj_r2"]]


def r_squared_pooled_standardized(preds: pd.DataFrame, features: pd.DataFrame,
                                  feature_names: list[str]) -> dict:
    """Config [incremental_info.r_squared] SUPPLEMENTARY: z-score each
    fold's CNN logit WITHIN that fold first (removes the fold-to-fold
    scale/calibration difference), then pool all folds into one OLS."""
    j = preds.set_index("date").join(features[feature_names], how="inner")
    j = j.copy()
    logits = logit(j["y_pred_proba"].to_numpy())
    j = j.assign(_logit=logits)
    j["_std_logit"] = j.groupby("fold")["_logit"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    y = j["_std_logit"].to_numpy()
    X = j[feature_names].to_numpy()
    return _r_squared(y, X)
