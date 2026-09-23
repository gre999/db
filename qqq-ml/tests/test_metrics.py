"""Metric tests with hand-computed examples."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics as M


def test_qlike_hand_example():
    # RV / RV_hat = 1, 2, 2  ->  0, 1 - ln 2, 1 - ln 2
    rv = [1.0, 2.0, 4.0]
    rv_hat = [1.0, 1.0, 2.0]
    np.testing.assert_allclose(M.qlike_loss(rv, rv_hat),
                               [0.0, 1 - np.log(2), 1 - np.log(2)])
    assert M.qlike(rv, rv_hat) == pytest.approx(2 * (1 - np.log(2)) / 3)


def test_qlike_properties():
    rng = np.random.default_rng(0)
    rv = rng.lognormal(size=100)
    assert M.qlike(rv, rv) == 0
    assert (M.qlike_loss(rv, rv * rng.lognormal(size=100)) >= 0).all()
    # Asymmetric: under-predicting by 2x costs more than over-predicting 2x.
    assert M.qlike([2.0], [1.0]) > M.qlike([1.0], [2.0])
    with pytest.raises(ValueError):
        M.qlike([1.0, 0.0], [1.0, 1.0])


def test_rmse_mae_hand_example():
    assert M.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert M.mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_bias_correction_matches_lognormal_mean():
    mu, s2 = -9.5, 0.4
    assert M.to_variance(mu, s2) == pytest.approx(np.exp(mu + s2 / 2))
    draws = np.exp(np.random.default_rng(1).normal(mu, np.sqrt(s2), 400_000))
    assert draws.mean() == pytest.approx(M.to_variance(mu, s2), rel=0.01)
    assert M.to_variance(mu, 0.0) == pytest.approx(np.exp(mu))


def _pred_frame():
    rng = np.random.default_rng(2)
    rows = []
    for fold, year in [(1, 2019), (2, 2020)]:
        dates = pd.bdate_range(f"{year}-01-02", periods=50)
        y = rng.normal(-9, 1, 50)
        for model, noise in [("good", 0.2), ("bad", 1.0)]:
            p = y + rng.normal(0, noise, 50)
            rows.append(pd.DataFrame({
                "date": dates, "fold": fold, "model": model, "target": "rv",
                "y_true_log": y, "y_pred_log": p,
                "y_true_var": np.exp(y), "y_pred_var": np.exp(p)}))
    return pd.concat(rows, ignore_index=True)


def test_evaluate_folds_and_overall():
    pred = _pred_frame()
    t = M.evaluate(pred, by="fold")
    assert t.loc[("rv", "good", "all"), "n"] == 100
    assert t.loc[("rv", "good", 1), "n"] == 50
    sub = pred[pred["model"] == "good"]
    assert t.loc[("rv", "good", "all"), "rmse_log"] == pytest.approx(
        M.rmse(sub["y_true_log"], sub["y_pred_log"]))
    y = M.evaluate(pred, by="year")
    assert ("rv", "bad", 2020) in y.index
    rel = M.relative_to(t, "bad")
    assert (rel.xs("good", level="model") < 1).all().all()


def test_diebold_mariano():
    pred = _pred_frame()
    g = pred[pred["model"] == "good"]
    b = pred[pred["model"] == "bad"]
    la = (g["y_true_log"] - g["y_pred_log"]).to_numpy() ** 2
    lb = (b["y_true_log"] - b["y_pred_log"]).to_numpy() ** 2
    dm, p = M.diebold_mariano(la, lb)
    assert dm < 0 and p < 0.01


def test_acf_and_ljung_box():
    rng = np.random.default_rng(3)
    wn = rng.normal(size=2000)
    assert np.abs(M.acf(wn, 10)).max() < 0.1
    assert M.ljung_box(wn, 10)[1] > 0.01
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.6 * ar[i - 1] + wn[i]
    assert M.acf(ar, 1)[0] == pytest.approx(0.6, abs=0.05)
    assert M.ljung_box(ar, 10)[1] < 1e-6


def test_eval_config_and_slices():
    cfg = M.load_eval_config()
    assert set(cfg["periods"]) == {"covid_2020", "tariff_2025"}
    for p in cfg["periods"].values():
        assert pd.Timestamp(p["start"]) < pd.Timestamp(p["end"])
    days = pd.bdate_range("2020-01-02", "2020-06-30")
    rv = pd.Series(1.0, index=days)
    rv[pd.Timestamp("2020-03-16")] = 5.0          # 5x the trailing mean
    rv[pd.Timestamp("2020-03-17")] = 2.1          # trailing mean now > 1.05
    sl = M.eval_slices(days, rv, cfg)
    assert sl.loc["2020-02-24", "covid_2020"] and sl.loc["2020-04-30", "covid_2020"]
    assert not sl.loc["2020-02-21", "covid_2020"]
    assert not sl.loc["2020-05-01", "covid_2020"]
    assert sl["jump_day"].sum() == 1 and sl.loc["2020-03-16", "jump_day"]
    assert not sl["tariff_2025"].any()
    # First `window` sessions have no trailing mean -> never a jump day.
    assert not sl["jump_day"].iloc[:17].any()


def test_slice_scores():
    pred = _pred_frame()
    dates = pd.DatetimeIndex(pred["date"].unique())
    sl = pd.DataFrame({"s": dates < dates[10]}, index=dates)
    t = M.slice_scores(pred, sl)
    assert t.loc[("rv", "good", "all"), "n"] == 100
    assert t.loc[("rv", "good", "s"), "n"] == int((pred.loc[pred["model"] == "good", "date"] < dates[10]).sum())
