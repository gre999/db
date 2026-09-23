"""HAR-plus-residual models: HAR extrapolates, the tree only corrects locally.

Week-4 finding: XGBoost/RF collapse to a near-flat prediction once RV moves
past the range seen in training (the 2020 COVID slice) because trees cannot
extrapolate. Here the tree predicts HAR's residual (``y - HAR.predict(X)``)
instead of ``y`` directly, so the extreme move is still carried by HAR's
linear extrapolation; the tree only has to learn a local correction on top
of it.

Both stages fit on the same training slice: the tree's own inner
train/validation split (see :mod:`src.models.trees`) stays entirely inside
this slice, never touching the fold's test slice. HAR is refit on the whole
training slice *before* the residual target is built, so the tree tunes on
the residual it will actually have to predict at deployment time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.models.base import Forecaster
from src.models.har import HARModel
from src.models.trees import RFModel, XGBModel


class HARResidual(Forecaster):
    """``predict(X) = har.predict(X) + tree.predict(X)``, tree fit on HAR's residual.

    Args:
        har: a :class:`HARModel` prototype (plain HAR or HAR-X); cloned and
            refit on the whole training slice.
        tree: a tuned-tree prototype (:class:`~src.models.trees.XGBModel` or
            :class:`~src.models.trees.RFModel`); cloned and fit on
            ``y - har.predict(X)`` with its own inner tuning split.
    """

    def __init__(self, har: HARModel | None = None, tree=None):
        self.har = har
        self.tree = tree

    @property
    def features(self) -> list[str]:
        return sorted(set(self.har.features) | set(self.tree.features))

    def fit(self, X, y):
        har = clone(self.har).fit(X, y)
        resid = pd.Series(np.asarray(y, float) - har.predict(X), index=X.index)
        tree = clone(self.tree).fit(X, resid)
        self.har_, self.tree_ = har, tree
        # the tree's own validation residual *is* the combined model's
        # validation residual: y - (har.predict + tree.predict)
        #   = (y - har.predict) - tree.predict = resid - tree.predict
        self.resid_var_ = tree.resid_var_
        if hasattr(tree, "best_params_"):
            self.best_params_ = tree.best_params_
            self.grid_results_ = tree.grid_results_
        self.n_train_ = len(X)
        return self

    def predict(self, X) -> np.ndarray:
        return self.har_.predict(X) + self.tree_.predict(X)

    def coef_table(self) -> pd.DataFrame:
        return self.har_.coef_table()


def har_residual_xgb_model() -> HARResidual:
    """HAR carries the trend/extrapolation; XGBoost corrects its residual."""
    return HARResidual(har=HARModel(), tree=XGBModel())


def har_residual_rf_model() -> HARResidual:
    """HAR carries the trend/extrapolation; random forest corrects its residual."""
    return HARResidual(har=HARModel(), tree=RFModel())
