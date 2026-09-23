"""HAR / random-walk tests: estimation, fold isolation, alignment, outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src import features as F
from src.metrics import evaluate
from src.models.base import (HAR_LOG, check_label_timing, make_dataset,
                             run_walk_forward, save_predictions,
                             validate_predictions)
from src.models.har import HARModel, RandomWalk, baseline_models
from src.validation import WalkForwardSplit
from tests import synth


@pytest.fixture(scope="module")
def md():
    return synth.make_market_data()


@pytest.fixture(scope="module")
def datasets(md):
    X = F.build_feature_matrix(md, "prev_close")
    y = F.build_targets(md)
    har_on = F.build_feature_matrix(
        md, "prev_close", F.FeatureConfig(rv_include_overnight=True),
        names=HAR_LOG)
    return {"rv": make_dataset(X, y, "rv"),
            "rv_on": make_dataset(X, y, "rv_on", har_on),
            "targets": y}


@pytest.fixture(scope="module")
def results(datasets):
    out = {}
    for t in ("rv", "rv_on"):
        X, y = datasets[t]
        out[t] = run_walk_forward(baseline_models(), X, y, WalkForwardSplit(), t)
    return out


# ------------------------------------------------------------- estimation
def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n, b = 4000, np.array([-1.0, 0.35, 0.3, 0.2])
    lr = np.full(n, -10.0)
    for t in range(22, n):
        d, w, m = lr[t - 1], lr[t - 5:t].mean(), lr[t - 22:t].mean()
        lr[t] = b[0] + b[1] * d + b[2] * w + b[3] * m + rng.normal(0, 0.5)
    s = pd.Series(lr)
    X = pd.DataFrame({"har_logrv_1d": s.shift(1),
                      "har_logrv_5d": s.shift(1).rolling(5).mean(),
                      "har_logrv_22d": s.shift(1).rolling(22).mean()}).iloc[100:]
    m = HARModel().fit(X, s.iloc[100:])
    np.testing.assert_allclose(m.coef_.to_numpy(), b, atol=0.12)
    assert m.resid_var_ == pytest.approx(0.25, rel=0.1)
    assert (m.se_ > 0).all()


def test_random_walk_and_clone(datasets):
    X, _ = datasets["rv"]
    rw = RandomWalk().fit(X, None)
    np.testing.assert_array_equal(rw.predict(X), X["har_logrv_1d"].to_numpy())
    assert rw.resid_var_ == 0
    c = clone(HARModel(nw_lags=3))
    assert c.features == HAR_LOG and c.nw_lags == 3


# ------------------------------------------------------------- alignment
@pytest.mark.parametrize("target,col", [("rv", "log_rv"),
                                        ("rv_on", "log_rv_on")])
def test_feature_label_alignment(datasets, target, col):
    """har_logrv_1d on day t must be the label of the previous session."""
    X, y = datasets[target]
    tg = datasets["targets"]
    ok = y.notna() & X["har_logrv_1d"].notna()
    d = X.index[ok][300]
    prev = X.index[X.index.get_loc(d) - 1]
    assert y[d] == tg.loc[d, col]
    assert X.loc[d, "har_logrv_1d"] == pytest.approx(tg.loc[prev, col])


def test_label_after_features(datasets):
    X, y = datasets["rv"]
    dates = X.index[y.notna()][1:]
    check_label_timing(dates, X.index)
    assert (F.label_time(dates) > F.cutoff_time(dates, "prev_close",
                                                X.index)).all()


# ------------------------------------------------------------- fold isolation
def test_coefficients_ignore_test_segment(datasets, results):
    """Replacing a fold's test rows must not change that fold's coefficients."""
    X, y = datasets["rv"]
    _, coef = results["rv"]
    ok = X[HAR_LOG].notna().all(axis=1) & y.notna()
    wf = WalkForwardSplit()
    rng = np.random.default_rng(5)
    for f in wf.folds(X.index[ok]):
        test_dates = X.index[ok][f.test_idx]
        X2, y2 = X.copy(), y.copy()
        X2.loc[test_dates, HAR_LOG] = rng.normal(-9, 3, (len(test_dates), 3))
        y2.loc[test_dates] = rng.normal(-9, 3, len(test_dates))
        _, coef2 = run_walk_forward({"har": HARModel()}, X2, y2, wf, "rv")
        a = coef[(coef["model"] == "har") & (coef["fold"] == f.number)]
        b = coef2[coef2["fold"] == f.number]
        np.testing.assert_allclose(a["value"].to_numpy(), b["value"].to_numpy())


# ------------------------------------------------------------- outputs
def test_prediction_frame_integrity(results, datasets):
    pred = pd.concat([results["rv"][0], results["rv_on"][0]])
    validate_predictions(pred)
    for t in ("rv", "rv_on"):
        p = pred[pred["target"] == t]
        assert not p.duplicated(["model", "date"]).any()
        assert p.groupby("date")["fold"].nunique().max() == 1
        _, y = datasets[t]
        np.testing.assert_allclose(p["y_true_log"], y.loc[p["date"]].to_numpy())
        assert set(p["model"]) == {"rw", "har"}
        # both models evaluated on exactly the same dates
        d = p.groupby("model")["date"].apply(frozenset)
        assert d["rw"] == d["har"]
    assert pd.DatetimeIndex(pred["date"]).min() >= pd.Timestamp("2019-01-01")


def test_save_rejects_duplicates(results, tmp_path):
    pred = results["rv"][0]
    path = save_predictions(pred, "t", pred_dir=tmp_path,
                            coefficients=results["rv"][1])
    assert pd.read_parquet(path).shape == pred.shape
    assert (tmp_path / "t_coefficients.parquet").exists()
    with pytest.raises(ValueError):
        save_predictions(pd.concat([pred, pred.iloc[:1]]), "t2",
                         pred_dir=tmp_path)
    bad = pred.copy()
    bad.loc[bad.index[0], "fold"] = 99
    with pytest.raises(ValueError):
        validate_predictions(bad)


def test_bias_correction_uses_training_residuals(results):
    pred, coef = results["rv"]
    h = pred[pred["model"] == "har"]
    rv_ = coef[(coef["model"] == "har") & (coef["param"] == "resid_var")]
    for f, s2 in zip(rv_["fold"], rv_["value"]):
        p = h[h["fold"] == f]
        np.testing.assert_allclose(p["y_pred_var"],
                                   np.exp(p["y_pred_log"] + s2 / 2))
    r = pred[pred["model"] == "rw"]
    np.testing.assert_allclose(r["y_pred_var"], np.exp(r["y_pred_log"]))


def test_har_beats_random_walk(results):
    """Synthetic log-vol is persistent but noisy: HAR must beat RW."""
    for t in ("rv", "rv_on"):
        tbl = evaluate(results[t][0]).xs("all", level="fold").loc[t]
        for m in ("rmse_log", "mae_log", "qlike"):
            assert tbl.loc["har", m] < tbl.loc["rw", m], (t, m)
