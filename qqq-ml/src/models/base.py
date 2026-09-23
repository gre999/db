"""Common interface, datasets and walk-forward runner for RV forecasters.

Every model (random walk, HAR, later XGBoost ...) follows the sklearn
estimator API and this contract:

* ``fit(X_train, y_train)`` - ``X`` is a feature DataFrame, ``y`` is
  log RV of the target day. Must set ``resid_var_`` (variance of the
  training residuals on the log scale, used for the lognormal bias
  correction) and may expose ``coef_table()``.
* ``predict(X)`` - returns log-RV forecasts (numpy array).

Row ``t`` of every dataset is the **target day**: features come from the
``prev_close`` matrix (known at t-1's close) and the label is RV of day t,
known at t's close. "Forecast RV(t+1) with information up to t" is the same
thing shifted by one row.

Out-of-sample predictions are saved under ``data/processed/predictions/``
with one row per (date, model, target):
``date, fold, model, target, y_true_log, y_pred_log, y_true_var,
y_pred_var, resid_var``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from src import data_loader as dl
from src import features as F
from src.metrics import to_variance
from src.validation import WalkForwardSplit

PRED_DIR = dl.PROCESSED_DIR / "predictions"
HAR_LOG = ["har_logrv_1d", "har_logrv_5d", "har_logrv_22d"]
TARGETS = {"rv": "log_rv", "rv_on": "log_rv_on"}


class Forecaster(BaseEstimator, RegressorMixin):
    """Base class: subclasses implement ``fit`` / ``predict`` on log RV."""

    features: list[str] = []

    def coef_table(self) -> pd.DataFrame:
        """param / value / se rows (empty for models without parameters)."""
        return pd.DataFrame(columns=["param", "value", "se"])

    def _X(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in X]
        if missing:
            raise KeyError(f"{type(self).__name__} needs columns {missing}")
        return X[self.features].to_numpy(float)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def check_label_timing(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex,
                       cutoff: str = "prev_close") -> None:
    """Raise unless every label is known strictly after its features."""
    lt = F.label_time(dates)
    ft = F.cutoff_time(dates, cutoff, calendar)
    bad = dates[~(lt > ft)]
    if len(bad):
        raise AssertionError(f"label not after feature cutoff on {list(bad[:5])}")


def make_dataset(X: pd.DataFrame, targets: pd.DataFrame, target: str = "rv",
                 har_override: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Align a prev_close feature matrix with a log-RV label.

    Args:
        X: ``features_prev_close`` matrix (index = target day).
        targets: ``targets_daily`` frame (``log_rv``, ``log_rv_on``,
            ``valid``, ``label_time``).
        target: ``"rv"`` (intraday, main) or ``"rv_on"`` (+ overnight²).
        har_override: HAR columns computed from the matching RV variant;
            required for ``"rv_on"`` so regressors and label use the same RV.

    Returns:
        (X, y_log) on the same index; y is NaN on invalid sessions.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {list(TARGETS)}")
    X = X.copy()
    if har_override is not None:
        X[har_override.columns] = har_override.reindex(X.index)
    elif target != "rv":
        raise ValueError("rv_on needs HAR features built from rv_on")
    y = targets[TARGETS[target]].where(targets["valid"].astype(bool))
    y = y.reindex(X.index).rename("y_log")
    ok = X[HAR_LOG].notna().all(axis=1) & y.notna()
    check_label_timing(X.index[ok], X.index)
    return X, y


def load_dataset(target: str = "rv", processed_dir: Path = dl.PROCESSED_DIR
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Read the week-2 matrices and build (X, y_log) for ``target``."""
    P = Path(processed_dir)
    X = pd.read_parquet(P / "features_prev_close.parquet")
    y = pd.read_parquet(P / "targets_daily.parquet")
    har = None
    if target == "rv_on":
        md = F.MarketData.from_processed(P, cboe_dir=None)
        cfg = F.FeatureConfig(rv_include_overnight=True)
        har = F.build_feature_matrix(md, "prev_close", cfg,
                                     names=HAR_LOG + ["har_rv_1d"])
    return make_dataset(X, y, target, har)


# --------------------------------------------------------------------------
# Walk-forward runner
# --------------------------------------------------------------------------
def run_walk_forward(models: dict[str, Forecaster], X: pd.DataFrame,
                     y_log: pd.Series, splitter: WalkForwardSplit,
                     target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every model on each fold's training slice, predict its test slice.

    All models share one sample: the dates where every model's features and
    the label are present, so their errors are directly comparable.

    Returns:
        (predictions, coefficients). Coefficients are long format:
        ``target, model, fold, train_start, train_end, n_train, param,
        value, se`` (plus ``resid_var`` rows).
    """
    need = sorted({c for m in models.values() for c in m.features})
    ok = X[need].notna().all(axis=1) & y_log.notna()
    Xs, ys = X.loc[ok], y_log.loc[ok]
    preds, coefs = [], []
    for f in splitter.folds(Xs.index):
        Xtr, ytr = Xs.iloc[f.train_idx], ys.iloc[f.train_idx]
        Xte, yte = Xs.iloc[f.test_idx], ys.iloc[f.test_idx]
        for name, proto in models.items():
            m = clone(proto).fit(Xtr, ytr)
            p_log = m.predict(Xte)
            preds.append(pd.DataFrame({
                "date": Xte.index, "fold": f.number, "model": name,
                "target": target,
                "y_true_log": yte.to_numpy(), "y_pred_log": p_log,
                "y_true_var": np.exp(yte.to_numpy()),
                "y_pred_var": to_variance(p_log, m.resid_var_),
                "resid_var": m.resid_var_,
            }))
            ct = m.coef_table()
            ct = pd.concat([ct, pd.DataFrame([{"param": "resid_var",
                                               "value": m.resid_var_}])])
            coefs.append(ct.assign(
                target=target, model=name, fold=f.number,
                train_start=f.train_start, train_end=f.train_end,
                n_train=len(f.train_idx)))
    pred = pd.concat(preds, ignore_index=True)
    coef = pd.concat(coefs, ignore_index=True)[
        ["target", "model", "fold", "train_start", "train_end", "n_train",
         "param", "value", "se"]]
    coef[["value", "se"]] = coef[["value", "se"]].astype(float)
    return pred, coef


def save_predictions(pred: pd.DataFrame, name: str, extra: dict | None = None,
                     pred_dir: Path = PRED_DIR, **frames: pd.DataFrame) -> Path:
    """Write ``<name>.parquet`` (+ ``<name>_<key>.parquet`` and meta json).

    Refuses to save if any (model, target, date) appears twice or a date
    belongs to more than one fold.
    """
    validate_predictions(pred)
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{name}.parquet"
    pred.to_parquet(path, index=False)
    for k, df in frames.items():
        df.to_parquet(pred_dir / f"{name}_{k}.parquet", index=False)
    meta = {"name": name, "saved_at": pd.Timestamp.now(tz=dl.ET).isoformat(),
            "rows": len(pred), "models": sorted(pred["model"].unique()),
            "targets": sorted(pred["target"].unique()), **(extra or {})}
    (pred_dir / f"{name}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def validate_predictions(pred: pd.DataFrame) -> None:
    """Unique (model, target, date); each date in exactly one fold."""
    if pred.duplicated(["model", "target", "date"]).any():
        raise ValueError("duplicate (model, target, date) predictions")
    folds_per_date = pred.groupby("date")["fold"].nunique()
    if (folds_per_date > 1).any():
        raise ValueError("a date appears in more than one fold")


def load_predictions(name: str = "har_baselines",
                     pred_dir: Path = PRED_DIR) -> pd.DataFrame:
    """Read a saved prediction file (e.g. for weeks 4-5)."""
    return pd.read_parquet(Path(pred_dir) / f"{name}.parquet")
