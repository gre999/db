"""Week 5: turn an RV forecast into a position.

Every calibration step here (the intraday-RV -> close-to-close scale ratio,
the variance-risk-premium threshold) is fit **per fold, on that fold's
training window only** - never on the fold's test window, and never on
data from another fold. Callers get the fold windows from
:func:`fold_windows`, which reads them off an already-saved walk-forward
run instead of re-deriving folds (every model in a run shares the same
splitter, so any one of them - HAR, random walk - has the same windows).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import PRED_DIR

TRADING_DAYS = 252


def fold_windows(coef_name: str = "har_baselines", model: str = "har",
                 target: str = "rv", pred_dir: Path = PRED_DIR) -> pd.DataFrame:
    """(fold, train_start, train_end) read off a saved run's coefficients file."""
    c = pd.read_parquet(Path(pred_dir) / f"{coef_name}_coefficients.parquet")
    c = c[(c["target"] == target) & (c["model"] == model)]
    if c.empty:
        raise ValueError(f"no rows for target={target!r} model={model!r} "
                         f"in {coef_name}_coefficients.parquet")
    return (c.drop_duplicates("fold")[["fold", "train_start", "train_end"]]
            .sort_values("fold").reset_index(drop=True))


def _map_by_fold(value_fn, index_kind: str, series_a: pd.Series,
                 series_b: pd.Series | None, folds: pd.DataFrame) -> pd.Series:
    out = {}
    for f in folds.itertuples():
        m = (series_a.index >= f.train_start) & (series_a.index <= f.train_end)
        out[f.fold] = value_fn(series_a[m], series_b[m] if series_b is not None
                               else None)
    return pd.Series(out)


def scale_ratios(rv: pd.Series, rv_on: pd.Series, folds: pd.DataFrame
                 ) -> pd.Series:
    """Per-fold ratio E[rv_on] / E[rv], estimated on the training window only.

    Intraday RV (``rv``) understates close-to-close risk because it misses
    the overnight gap; ``rv_on = rv + overnight_return**2`` is the project's
    close-to-close realized-variance proxy (see ``src/features.py``). This
    ratio rescales a model's intraday-RV variance *forecast* into a
    close-to-close variance forecast. ``rv``/``rv_on`` must be the actual
    (realized) series - a fold's ratio only ever looks at dates inside that
    fold's own ``[train_start, train_end]``.
    """
    def ratio(r, ron):
        r, ron = r.dropna(), ron.dropna()
        idx = r.index.intersection(ron.index)
        return float(ron.loc[idx].mean() / r.loc[idx].mean())
    return _map_by_fold(ratio, "fold", rv, rv_on, folds).rename("scale_ratio")


def cc_variance_forecast(pred: pd.DataFrame, ratios: pd.Series) -> pd.Series:
    """Calibrated close-to-close variance forecast.

    ``pred`` (indexed by date) needs ``fold`` and ``y_pred_var`` (a model's
    intraday-RV variance forecast, already bias-corrected - see
    ``src/models/base.py::run_walk_forward``); returns
    ``ratios[fold] * y_pred_var``.
    """
    missing = set(pred["fold"].unique()) - set(ratios.index)
    if missing:
        raise KeyError(f"no scale ratio for folds {sorted(missing)}")
    return (pred["y_pred_var"] * pred["fold"].map(ratios)).rename("var_cc_hat")


def vol_target_weight(var_cc: pd.Series, vol_target: float = 0.15,
                      leverage_cap: float = 1.5) -> pd.Series:
    """``w = min(vol_target / annualized forecast vol, leverage_cap)``.

    ``var_cc`` is a *daily* close-to-close variance forecast; annualized
    here as ``TRADING_DAYS * var_cc``.
    """
    ann_vol = np.sqrt(TRADING_DAYS * var_cc.clip(lower=1e-12))
    return (vol_target / ann_vol).clip(upper=leverage_cap).rename("weight")


def smooth_variance_forecast(var_cc: pd.Series, window: int = 20) -> pd.Series:
    """Trailing mean of a model's own daily variance forecast, ending at t.

    Diagnostic for "does 20-day historical vol win because it's smoother, or
    because it has different information?" - averaging a model's own past
    forecasts stays causal (each input was already known at its own date),
    so this isolates the effect of reaction speed alone, holding the
    forecast's information source fixed.
    """
    return var_cc.rolling(window, min_periods=window).mean().rename("var_cc_smoothed")


def historical_vol_weight(ret_cc: pd.Series, window: int = 20,
                          vol_target: float = 0.15,
                          leverage_cap: float = 1.5) -> pd.Series:
    """Trailing-realized-vol target: sessions t-window..t-1, excluding t.

    The weight at index t is "decided using information available at
    (t-1)'s close" - the same convention every model source uses (see
    ``src/backtest.py``'s module docstring) - so it must never include
    ``ret_cc[t]`` itself: that return is the one this weight is then used
    to earn. No fold calibration needed - ``ret_cc`` is already the
    close-to-close return, so the rolling variance is already on the
    close-to-close scale.
    """
    var = ret_cc.shift(1).rolling(window, min_periods=window).var(ddof=0)
    return vol_target_weight(var, vol_target, leverage_cap)


def buy_and_hold_weight(index: pd.Index) -> pd.Series:
    """Constant fully-invested weight (the "no adjustment" source)."""
    return pd.Series(1.0, index=index, name="weight")


def variance_risk_premium(vxn_close: pd.Series, var_cc_hat: pd.Series
                          ) -> pd.Series:
    """Implied annualized variance (from VXN) minus the forecast's.

    ``var_cc_hat`` is the *daily* close-to-close variance forecast (e.g.
    from :func:`cc_variance_forecast`); it is annualized here before
    comparing to VXN, which already quotes an annualized implied vol. VXN
    is a ~30-calendar-day-forward measure while the forecast is a 1-day-
    ahead one annualized by x252 - the two are on the same *scale* (annual
    variance) but not exactly the same *period*; treat this premium as a
    directional signal, not an exact risk-neutral variance swap payoff.
    """
    implied = (vxn_close / 100.0) ** 2
    return (implied - TRADING_DAYS * var_cc_hat).rename("premium")


def premium_thresholds(premium: pd.Series, folds: pd.DataFrame,
                       q: float = 0.25) -> pd.Series:
    """Per-fold threshold: the ``q``-quantile of the premium, training window only."""
    def thr(p, _):
        return float(p.dropna().quantile(q))
    return _map_by_fold(thr, "fold", premium, None, folds
                        ).rename("premium_threshold")


def strategy_n_weight(base_weight: pd.Series, premium: pd.Series,
                      fold: pd.Series, thresholds: pd.Series,
                      derate: float = 0.6) -> pd.Series:
    """Derate ``base_weight`` when the variance risk premium is compressed.

    When ``premium < thresholds[fold]`` (the options market is pricing
    less of a cushion above the model's forecast than it typically does in
    that fold's training window - historically a sign of complacency ahead
    of vol spikes), scale the position down by ``derate``; otherwise leave
    it unchanged.
    """
    thr = fold.map(thresholds)
    tilt = np.where(premium < thr, derate, 1.0)
    return (base_weight * tilt).rename("weight")
