"""Walk-forward splitting and leak-free sklearn pipelines.

Splits are made on the time axis of the samples (trading days), never by
shuffling. Default: a rolling window of 3 years of training followed by 1
year of testing, moved forward 1 year per fold, e.g. with samples starting
2015-01-23 (first test year = first Jan 1 with a full 3-year history)::

    fold 1: train 2016-2018 | test 2019
    fold 2: train 2017-2019 | test 2020
    ...

Pass ``first_test_start="2018-01-01"`` to start earlier with a slightly
shorter first training window.

Every preprocessing step (imputation, scaling, feature selection) must sit
inside the sklearn ``Pipeline`` so it is fitted on the training slice of
each fold only; :func:`walk_forward_predict` clones and fits the whole
pipeline per fold.

``embargo_days`` drops the last N training sessions before each test
period. Use it when a label spans several sessions (e.g. 5-day RV) so the
final training labels do not overlap the test period.

Example::

    wf = WalkForwardSplit(train_years=3, test_years=1, step_years=1)
    print(wf.describe(X.index))
    pipe = make_pipeline(LinearRegression())
    preds, folds = walk_forward_predict(pipe, X, y, wf)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold (positions into the sample index)."""

    number: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardSplit:
    """Rolling (or expanding) walk-forward splitter over trading days.

    Args:
        train_years: length of the training window (calendar years).
        test_years: length of each test window.
        step_years: how far the window moves per fold.
        first_test_start: start of the first test period. Default: the
            first Jan 1 at least ``train_years`` after the first sample.
        expanding: if True the training window always starts at the first
            sample (anchored) instead of rolling.
        embargo_days: training sessions dropped right before each test
            period.
        min_test_days: a final, partial test period shorter than this is
            dropped.

    Compatible with sklearn's ``cv=`` argument through :meth:`split`.
    """

    train_years: int = 3
    test_years: int = 1
    step_years: int = 1
    first_test_start: str | pd.Timestamp | None = None
    expanding: bool = False
    embargo_days: int = 0
    min_test_days: int = 20

    def _first_test(self, dates: pd.DatetimeIndex) -> pd.Timestamp:
        if self.first_test_start is not None:
            return pd.Timestamp(self.first_test_start)
        earliest = dates[0] + pd.DateOffset(years=self.train_years)
        jan1 = pd.Timestamp(year=earliest.year, month=1, day=1)
        return jan1 if jan1 >= earliest else jan1 + pd.DateOffset(years=1)

    def folds(self, dates) -> list[Fold]:
        """Build the folds for a sorted, unique DatetimeIndex of samples."""
        dates = pd.DatetimeIndex(dates)
        if not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValueError("dates must be sorted and unique (time order)")
        out = []
        test_start = self._first_test(dates)
        k = 0
        while test_start <= dates[-1]:
            test_end = test_start + pd.DateOffset(years=self.test_years)
            train_start = dates[0] if self.expanding else \
                test_start - pd.DateOffset(years=self.train_years)
            tr = np.flatnonzero((dates >= train_start) & (dates < test_start))
            if self.embargo_days:
                tr = tr[:-self.embargo_days] if len(tr) > self.embargo_days \
                    else tr[:0]
            te = np.flatnonzero((dates >= test_start) & (dates < test_end))
            if len(te) >= self.min_test_days and len(tr):
                k += 1
                out.append(Fold(k, tr, te, dates[tr[0]], dates[tr[-1]],
                                dates[te[0]], dates[te[-1]]))
            test_start = test_start + pd.DateOffset(years=self.step_years)
        return out

    def split(self, X, y=None, groups=None):
        """sklearn-style generator of (train_idx, test_idx)."""
        for f in self.folds(_index_of(X)):
            yield f.train_idx, f.test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.folds(_index_of(X)))

    def describe(self, dates) -> pd.DataFrame:
        """Table of fold date ranges and sizes, for manual checking."""
        cols = ["fold", "train_start", "train_end", "n_train", "test_start",
                "test_end", "n_test"]
        return pd.DataFrame([{
            "fold": f.number,
            "train_start": f.train_start.date(), "train_end": f.train_end.date(),
            "n_train": len(f.train_idx),
            "test_start": f.test_start.date(), "test_end": f.test_end.date(),
            "n_test": len(f.test_idx),
        } for f in self.folds(dates)], columns=cols).set_index("fold")


def _index_of(X) -> pd.DatetimeIndex:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return pd.DatetimeIndex(X.index)
    return pd.DatetimeIndex(X)


def make_pipeline(model, impute: str | None = "median", scale: bool = True,
                  selector=None) -> Pipeline:
    """Pipeline: [imputer] -> [scaler] -> [selector] -> model.

    Everything before the model is fitted inside each fold's training slice
    when used with :func:`walk_forward_predict` (or any sklearn CV).
    """
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy=impute)))
    if scale:
        steps.append(("scale", StandardScaler()))
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", model))
    return Pipeline(steps)


def walk_forward_predict(pipeline, X: pd.DataFrame, y: pd.Series,
                         splitter: WalkForwardSplit,
                         drop_na_target: bool = True
                         ) -> tuple[pd.DataFrame, list[Pipeline]]:
    """Fit a fresh clone of ``pipeline`` per fold, predict its test slice.

    Rows with a missing target are dropped from training (and, if
    ``drop_na_target``, from testing). Returns out-of-sample predictions
    (``fold``, ``y_true``, ``y_pred`` indexed by date) and the fitted
    pipelines, one per fold.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    preds, fitted = [], []
    for f in splitter.folds(X.index):
        tr, te = X.index[f.train_idx], X.index[f.test_idx]
        ytr = y.loc[tr].dropna()
        model = clone(pipeline).fit(X.loc[ytr.index], ytr)
        te_rows = y.loc[te].dropna().index if drop_na_target else te
        preds.append(pd.DataFrame({"fold": f.number,
                                   "y_true": y.loc[te_rows],
                                   "y_pred": model.predict(X.loc[te_rows])},
                                  index=te_rows))
        fitted.append(model)
    return pd.concat(preds), fitted
