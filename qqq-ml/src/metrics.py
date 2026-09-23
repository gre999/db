"""Forecast evaluation for realized-variance models.

Conventions (used by every model in the project):

* Models forecast **log RV**. RMSE and MAE are computed on that log scale.
* QLIKE is computed on the **variance** scale:
  ``QLIKE = RV / RV_hat - log(RV / RV_hat) - 1`` (0 for a perfect forecast,
  always >= 0; robust to noise in the RV proxy, Patton 2011).
* Turning a log forecast into a variance forecast needs a bias correction:
  if ``log RV = mu + e`` with ``e ~ N(0, s2)`` then ``E[RV] = exp(mu + s2/2)``.
  ``s2`` must be the residual variance of the *training* slice of the same
  fold (never of the test slice) - see :func:`to_variance`.

The prediction frame expected by :func:`evaluate` has columns
``date, fold, model, target, y_true_log, y_pred_log, y_true_var,
y_pred_var``.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true, y_pred) -> float:
    e = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(y_true, y_pred) -> float:
    e = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.mean(np.abs(e)))


def qlike_loss(rv, rv_hat) -> np.ndarray:
    """Per-observation QLIKE on the variance scale (both inputs > 0)."""
    rv = np.asarray(rv, float)
    rv_hat = np.asarray(rv_hat, float)
    if (rv <= 0).any() or (rv_hat <= 0).any():
        raise ValueError("QLIKE needs strictly positive variances")
    ratio = rv / rv_hat
    return ratio - np.log(ratio) - 1.0


def qlike(rv, rv_hat) -> float:
    """Mean QLIKE: mean(RV/RV_hat - log(RV/RV_hat) - 1)."""
    return float(np.mean(qlike_loss(rv, rv_hat)))


def to_variance(pred_log, resid_var: float) -> np.ndarray:
    """Log-scale forecast -> variance forecast with lognormal bias correction.

    ``exp(pred_log + resid_var / 2)``; ``resid_var`` is the training-slice
    residual variance of the fold (0 disables the correction).
    """
    return np.exp(np.asarray(pred_log, float) + resid_var / 2.0)


def score(df: pd.DataFrame) -> pd.Series:
    """RMSE / MAE (log) and QLIKE (variance) for one prediction frame."""
    return pd.Series({
        "n": len(df),
        "rmse_log": rmse(df["y_true_log"], df["y_pred_log"]),
        "mae_log": mae(df["y_true_log"], df["y_pred_log"]),
        "qlike": qlike(df["y_true_var"], df["y_pred_var"]),
    })


def evaluate(pred: pd.DataFrame, by: str | list[str] = "fold",
             keys: tuple[str, ...] = ("target", "model")) -> pd.DataFrame:
    """Scores per ``keys`` x ``by`` group plus an overall ``all`` row.

    ``by`` is a column of ``pred`` (e.g. ``"fold"``) or ``"year"`` (derived
    from ``date``). The overall row pools every out-of-sample observation.
    """
    p = pred.copy()
    if by == "year":
        p["year"] = pd.DatetimeIndex(p["date"]).year
    by = [by] if isinstance(by, str) else list(by)
    keys = list(keys)
    per = p.groupby(keys + by).apply(score, include_groups=False)
    allr = p.groupby(keys).apply(score, include_groups=False)
    for b in by:
        allr[b] = "all"
    allr = allr.set_index(by, append=True)
    out = pd.concat([per, allr]).sort_index()
    out["n"] = out["n"].astype(int)
    return out


def relative_to(table: pd.DataFrame, base_model: str,
                metrics=("rmse_log", "mae_log", "qlike")) -> pd.DataFrame:
    """Each model's metric divided by ``base_model``'s (< 1 = better)."""
    t = table.reset_index()
    idx = [c for c in t.columns if c not in ("model", "n", *metrics)]
    base = t[t["model"] == base_model].set_index(idx)[list(metrics)]
    rel = t.set_index(idx + ["model"])[list(metrics)].div(base, axis=0)
    return rel


def diebold_mariano(loss_a, loss_b, lag: int | None = None) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    ``d = loss_a - loss_b``; negative mean -> model A better. Uses a
    Newey-West long-run variance with ``lag`` (default: n^(1/3)).
    Returns (DM statistic, two-sided p-value, normal approximation).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lag = int(np.floor(n ** (1 / 3))) if lag is None else lag
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    return float(dm), float(2 * stats.norm.sf(abs(dm)))


def acf(x, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelations for lags 1..nlags (NaNs removed)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)] - np.nanmean(x)
    den = x @ x
    return np.array([(x[k:] @ x[:-k]) / den for k in range(1, nlags + 1)])


def ljung_box(x, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value for no autocorrelation up to ``lags``."""
    x = np.asarray(x, float)
    n = int((~np.isnan(x)).sum())
    r = acf(x, lags)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


# --------------------------------------------------------------------------
# Evaluation slices (config/evaluation.toml)
# --------------------------------------------------------------------------
EVAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "evaluation.toml"


def load_eval_config(path: Path = EVAL_CONFIG) -> dict:
    """Read the fixed evaluation periods and the jump-day rule."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def jump_days(rv: pd.Series, multiple: float, window: int = 22,
              min_frac: float = 0.8) -> pd.Series:
    """Ex-post jump flag: RV(t) > multiple x mean RV of sessions t-window..t-1.

    ``rv`` is a variance series on the full session calendar (NaN on
    invalid sessions). Evaluation filter only - it uses day t's own RV.
    """
    base = rv.shift(1).rolling(window,
                               min_periods=int(np.ceil(window * min_frac))).mean()
    return (rv > multiple * base).where(rv.notna() & base.notna())


def eval_slices(dates, rv: pd.Series, cfg: dict | None = None) -> pd.DataFrame:
    """Boolean slice columns for ``dates``: one per period + ``jump_day``.

    Args:
        dates: dates to label (e.g. the out-of-sample prediction dates).
        rv: variance series on the full session calendar, used for the
            jump rule's trailing mean.
        cfg: parsed ``evaluation.toml`` (default: the project file).
    """
    cfg = cfg or load_eval_config()
    d = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=d)
    for key, p in cfg["periods"].items():
        out[key] = (d >= pd.Timestamp(p["start"])) & (d <= pd.Timestamp(p["end"]))
    j = cfg["jump_days"]
    jd = jump_days(rv, j["multiple"], j["window"])
    out["jump_day"] = jd.reindex(d).fillna(False).astype(bool).to_numpy()
    return out


def slice_scores(pred: pd.DataFrame, slices: pd.DataFrame,
                 keys: tuple[str, ...] = ("target", "model")) -> pd.DataFrame:
    """Scores on every slice column (rows where it is True) plus ``all``."""
    parts = []
    s = slices.reindex(pd.DatetimeIndex(pred["date"])).to_numpy()
    for i, col in enumerate(["all"] + list(slices.columns)):
        mask = np.ones(len(pred), bool) if col == "all" else s[:, i - 1]
        sub = pred.loc[mask]
        if len(sub):
            parts.append(sub.groupby(list(keys)).apply(score, include_groups=False)
                         .assign(slice=col))
    out = pd.concat(parts).set_index("slice", append=True)
    out["n"] = out["n"].astype(int)
    return out
