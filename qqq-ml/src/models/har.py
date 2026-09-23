"""Baseline RV forecasters: random walk and HAR (Corsi 2009).

Both forecast log RV of target day t from information known at t-1's close
(the ``prev_close`` feature matrix):

* :class:`RandomWalk` - ``log RV_hat(t) = log RV(t-1)``; its variance forecast
  is RV(t-1) itself, so no bias correction is applied (``resid_var_ = 0``).
* :class:`HARModel` - OLS of ``log RV(t)`` on a constant and the log of the
  1-, 5- and 22-session average RV (``har_logrv_1d/5d/22d``). Coefficients
  come with Newey-West (HAC) standard errors because the residuals are
  serially correlated; ``resid_var_`` is the training residual variance used
  for the ``exp(s2/2)`` correction.

Run the week-3 experiment (both targets, all folds, saves predictions)::

    python -m src.models.har
"""
from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.metrics import evaluate
from src.models.base import (HAR_LOG, Forecaster, load_dataset,
                             run_walk_forward, save_predictions)
from src.validation import WalkForwardSplit


class RandomWalk(Forecaster):
    """Tomorrow's RV equals today's: ``log RV_hat(t) = log RV(t-1)``."""

    def __init__(self, feature: str = "har_logrv_1d"):
        self.feature = feature

    @property
    def features(self) -> list[str]:
        return [self.feature]

    def fit(self, X, y=None):
        self.resid_var_ = 0.0
        self.n_train_ = len(X)
        return self

    def predict(self, X) -> np.ndarray:
        return self._X(X)[:, 0]


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance of OLS coefficients (Bartlett kernel)."""
    xe = X * resid[:, None]
    S = xe.T @ xe
    for k in range(1, lags + 1):
        g = xe[k:].T @ xe[:-k]
        S += (1 - k / (lags + 1)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv


class HARModel(Forecaster):
    """HAR in logs: ``log RV(t) = b0 + bd·logRV_d + bw·logRV_w + bm·logRV_m``.

    Args:
        regressors: regressor columns (default: 1/5/22-session log RV).
        nw_lags: Newey-West lags for the standard errors (only affects
            ``coef_table``, not the forecasts).
    """

    def __init__(self, regressors: list[str] | None = None, nw_lags: int = 5):
        self.regressors = regressors
        self.nw_lags = nw_lags

    @property
    def features(self) -> list[str]:
        return list(self.regressors) if self.regressors is not None \
            else list(HAR_LOG)

    def fit(self, X, y):
        Z = np.column_stack([np.ones(len(X)), self._X(X)])
        yv = np.asarray(y, float)
        beta, *_ = np.linalg.lstsq(Z, yv, rcond=None)
        resid = yv - Z @ beta
        n, k = Z.shape
        self.coef_ = pd.Series(beta, index=["const"] + list(self.features))
        self.resid_var_ = float(resid @ resid / (n - k))
        self.se_ = pd.Series(np.sqrt(np.diag(
            newey_west_cov(Z, resid, self.nw_lags))), index=self.coef_.index)
        self.r2_ = float(1 - resid @ resid / np.sum((yv - yv.mean()) ** 2))
        self.n_train_ = n
        return self

    def predict(self, X) -> np.ndarray:
        return self.coef_.iloc[0] + self._X(X) @ self.coef_.iloc[1:].to_numpy()

    def coef_table(self) -> pd.DataFrame:
        t = pd.DataFrame({"param": self.coef_.index, "value": self.coef_.values,
                          "se": self.se_.values})
        return pd.concat([t, pd.DataFrame([{"param": "r2_train",
                                            "value": self.r2_}])])


def baseline_models() -> dict[str, Forecaster]:
    return {"rw": RandomWalk(), "har": HARModel()}


def run(splitter: WalkForwardSplit = WalkForwardSplit(),
        targets=("rv", "rv_on")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Week-3 experiment: RW and HAR, walk-forward, both RV variants."""
    preds, coefs = [], []
    for tgt in targets:
        X, y = load_dataset(tgt)
        p, c = run_walk_forward(baseline_models(), X, y, splitter, tgt)
        preds.append(p)
        coefs.append(c)
    pred = pd.concat(preds, ignore_index=True)
    coef = pd.concat(coefs, ignore_index=True)
    save_predictions(pred, "har_baselines",
                     extra={"splitter": asdict(splitter),
                            "features": HAR_LOG,
                            "note": "row t = target day; features known at "
                                    "t-1 close; y = log RV of day t"},
                     coefficients=coef)
    return pred, coef


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    pred, _ = run()
    tbl = evaluate(pred, by="fold")
    with pd.option_context("display.width", 120, "display.max_rows", 100):
        print(tbl.xs("all", level="fold").round(4))


if __name__ == "__main__":
    main()
