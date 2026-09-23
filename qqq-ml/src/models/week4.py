"""Week 4: HAR-X, XGBoost, random forest and a HAR+residual-XGB model vs the
week-3 HAR baseline.

* Target: log RV of day t (intraday RV, the main target), features from the
  ``prev_close`` matrix (known at t-1's close).
* Same walk-forward folds as week 3; each model is refitted per fold and the
  tree models tune only inside the fold's training slice (see
  :mod:`src.models.trees`).
* HAR and random-walk forecasts are *read* from ``har_baselines.parquet``,
  never retrained.
* ``har_resid_xgb`` (:mod:`src.models.residual`) lets HAR carry the
  extrapolation and has XGBoost correct only its residual, so it isn't
  capped at the top of the training range the way plain XGBoost is.

Outputs in ``data/processed/predictions/`` (same format as
``har_baselines.parquet``)::

    week4_models.parquet               harx / xgb / rf / har_resid_xgb predictions
    week4_models_params.parquet        chosen hyper-parameters per fold
    week4_models_grid.parquet          every grid point's validation score
    week4_models_coefficients.parquet  HAR-X coefficients per fold
    week4_models_importance.parquet    test-slice permutation importance

Usage::

    python -m src.models.week4
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

import pandas as pd

from src import data_loader as dl
from src import metrics as M
from src.models.base import load_dataset, run_walk_forward, save_predictions
from src.models.har import HARModel, harx_model
from src.models.residual import HARResidual
from src.models.trees import (TREE_FEATURES, RFModel, XGBModel,
                              permutation_importance)
from src.validation import WalkForwardSplit

log = logging.getLogger(__name__)
NAME = "week4_models"
MIN_SLICE_ROWS = 15


def week4_models() -> dict:
    return {"harx": harx_model(), "xgb": XGBModel(), "rf": RFModel(),
            "har_resid_xgb": HARResidual(har=HARModel(), tree=XGBModel())}


def run(splitter: WalkForwardSplit = WalkForwardSplit(), target: str = "rv",
        n_repeats: int = 10, models: dict | None = None, X=None, y=None,
        rv_calendar: pd.Series | None = None, save: bool = True) -> dict:
    """Run the week-4 models; returns and (optionally) saves every output.

    ``X``, ``y`` and ``rv_calendar`` default to the processed data; tests
    pass synthetic ones.
    """
    if X is None:
        X, y = load_dataset(target)
        missing = [c for c in TREE_FEATURES if c not in X]
        if missing:
            raise KeyError(f"features {missing} missing - rerun "
                           "`python -m src.features` (week-4 jump features)")
    if rv_calendar is None:
        t = pd.read_parquet(dl.PROCESSED_DIR / "targets_daily.parquet")
        rv_calendar = t["rv"].where(t["valid"].astype(bool))
    models = models or week4_models()
    slices = M.eval_slices(X.index, rv_calendar)

    params, grids, imps = [], [], []

    def on_fit(f, name, m, Xte, yte):
        if hasattr(m, "best_params_"):
            params.append({"model": name, "fold": f.number,
                           "test_year": f.test_start.year, **m.best_params_})
            grids.append(m.grid_results_.assign(model=name, fold=f.number))
        sl = slices.reindex(Xte.index)
        for sname in ["all"] + list(slices.columns):
            mask = pd.Series(True, index=Xte.index) if sname == "all" \
                else sl[sname]
            if mask.sum() < MIN_SLICE_ROWS:
                continue
            imp = permutation_importance(m, Xte[mask], yte[mask], n_repeats,
                                         seed=f.number)
            imps.append(imp.assign(model=name, fold=f.number, slice=sname,
                                   n=int(mask.sum())))
        log.info("fold %s %s done", f.number, name)

    pred, coef = run_walk_forward(models, X, y, splitter, target, on_fit=on_fit)
    out = {
        "pred": pred,
        "coefficients": coef[coef["model"] == "harx"],
        "params": pd.DataFrame(params),
        "grid": pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(),
        "importance": pd.concat(imps, ignore_index=True),
    }
    if save:
        save_predictions(pred, NAME,
                         extra={"splitter": asdict(splitter), "target": target,
                                "tree_features": TREE_FEATURES,
                                "eval_config": M.load_eval_config()},
                         coefficients=out["coefficients"],
                         params=out["params"], grid=out["grid"],
                         importance=out["importance"])
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = run()
    print(M.evaluate(out["pred"]).xs("all", level="fold").round(4))


if __name__ == "__main__":
    main()
