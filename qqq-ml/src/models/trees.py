"""Tree ensembles for log RV: XGBoost and random forest.

Both follow the :class:`~src.models.base.Forecaster` contract (fit on log
RV, predict log RV, expose ``resid_var_`` for the ``exp(s2/2)`` correction)
and tune themselves **inside the training slice they are given**:

1. The last ``val_years`` of the training slice become a validation block;
   the earlier part is the inner training block. The test slice of the
   walk-forward fold is never passed to ``fit``.
2. Every grid point is fitted on the inner block and scored by RMSE (log)
   on the validation block. XGBoost additionally early-stops on it, which
   sets the number of trees.
3. The winner is refitted on the whole training slice (XGBoost with the
   early-stopped number of trees).
4. ``resid_var_`` = mean squared validation residual of the winner. In-sample
   tree residuals are far too small (trees fit the training data almost
   perfectly), so the held-out validation block gives an honest variance
   for the bias correction; it is still inside the training slice.

``best_params_`` and ``grid_results_`` record what was chosen per fold.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.base import HAR_LOG, Forecaster

# HAR logs + every week-2 prev_close/calendar feature + week-4 jump features.
# Level versions of the HAR terms are left out: trees are invariant to the
# monotone log transform, so they would only duplicate columns.
TREE_FEATURES = HAR_LOG + [
    "rskew_1d", "rkurt_1d", "volume_rel_20d", "close_loc_1d",
    "day_of_week", "is_half_day", "prev_is_half_day",
    "vix_1d", "vxn_1d",
    "max_abs_ret5_1d", "jump_rv_bv_1d", "vix_chg_1d",
]

XGB_GRID = {"max_depth": [2, 3, 4, 6], "learning_rate": [0.02, 0.05, 0.1]}
RF_GRID = {"max_depth": [4, 8, None], "min_samples_leaf": [5, 20],
           "max_features": [0.33, 0.66]}


def inner_split(index: pd.DatetimeIndex, val_years: int = 1
                ) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (inner-train, validation): validation = last N years."""
    cut = index[-1] - pd.DateOffset(years=val_years)
    va = np.asarray(index > cut)
    return ~va, va


def _grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in product(*grid.values())]


class _TunedTree(Forecaster):
    """Shared tuning loop; subclasses build the estimator."""

    def __init__(self, feature_list: list[str] | None = None,
                 grid: dict | None = None, val_years: int = 1,
                 random_state: int = 0):
        self.feature_list = feature_list
        self.grid = grid
        self.val_years = val_years
        self.random_state = random_state

    @property
    def features(self) -> list[str]:
        return list(self.feature_list) if self.feature_list is not None \
            else list(TREE_FEATURES)

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, float), index=X.index)
        tr, va = inner_split(pd.DatetimeIndex(X.index), self.val_years)
        Xtr, Xva = self._X(X[tr]), self._X(X[va])
        ytr, yva = y[tr].to_numpy(), y[va].to_numpy()
        results = []
        for params in _grid(self.grid or self.default_grid):
            est, extra = self._fit_inner(params, Xtr, ytr, Xva, yva)
            err = yva - est.predict(Xva)
            results.append({**params, **extra,
                            "val_rmse": float(np.sqrt(np.mean(err ** 2))),
                            "val_mse": float(np.mean(err ** 2))})
        res = pd.DataFrame(results)
        best = res.loc[res["val_rmse"].idxmin()].to_dict()
        self.grid_results_ = res
        self.best_params_ = best
        self.resid_var_ = float(best["val_mse"])
        self.model_ = self._fit_final(best, self._X(X), y.to_numpy())
        self.n_train_ = len(X)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(self._X(X))


class XGBModel(_TunedTree):
    """XGBoost regressor on log RV, tuned per fold with early stopping."""

    default_grid = XGB_GRID
    max_estimators = 2000
    early_stopping_rounds = 50
    fixed = {"subsample": 0.8, "colsample_bytree": 0.8,
             "min_child_weight": 5, "reg_lambda": 1.0,
             "objective": "reg:squarederror", "tree_method": "hist"}

    def _make(self, params, n_estimators, early_stop):
        p = {k: v for k, v in params.items()
             if k in ("max_depth", "learning_rate")}
        return XGBRegressor(n_estimators=n_estimators, random_state=self.random_state,
                            n_jobs=4, eval_metric="rmse",
                            early_stopping_rounds=early_stop,
                            **{**self.fixed, **p,
                               "max_depth": int(p["max_depth"])})

    def _fit_inner(self, params, Xtr, ytr, Xva, yva):
        est = self._make(params, self.max_estimators, self.early_stopping_rounds)
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return est, {"n_estimators": int(est.best_iteration) + 1}

    def _fit_final(self, best, X, y):
        est = self._make(best, int(best["n_estimators"]), None)
        return est.fit(X, y, verbose=False)


class RFModel(_TunedTree):
    """Random forest on log RV, tuned per fold on the validation block."""

    default_grid = RF_GRID
    n_estimators = 300

    def _make(self, params):
        md = params["max_depth"]
        md = None if md is None or (isinstance(md, float) and np.isnan(md)) \
            else int(md)
        return RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=md,
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=float(params["max_features"]),
            random_state=self.random_state, n_jobs=-1)

    def _fit_inner(self, params, Xtr, ytr, Xva, yva):
        return self._make(params).fit(Xtr, ytr), {}

    def _fit_final(self, best, X, y):
        return self._make(best).fit(X, y)


def permutation_importance(model: Forecaster, X: pd.DataFrame, y: pd.Series,
                           n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Increase in log-scale RMSE when each feature is shuffled in ``X``.

    Computed on the rows given (the fold's test slice, or a sub-slice of it);
    the model is not refitted. Correlated features (e.g. the three HAR terms,
    VIX/VXN) share credit, so each one alone can look less important than the
    group.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    base = np.sqrt(np.mean((y - model.predict(X)) ** 2))
    rows = []
    for c in model.features:
        inc = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            inc.append(np.sqrt(np.mean((y - model.predict(Xp)) ** 2)) - base)
        rows.append({"feature": c, "importance": float(np.mean(inc)),
                     "importance_std": float(np.std(inc)),
                     "base_rmse": float(base)})
    return pd.DataFrame(rows)
