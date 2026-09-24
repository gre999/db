"""Week 11 (phase 5, week 1): CNN-vs-baseline evaluation.

The primary comparison (config/week11_dl.toml [primary_comparison]): for
each task, the CNN's test-fold AUC minus the best baseline's test-fold
AUC, tested with a moving-block bootstrap over TRADING DAYS (not returns -
:func:`block_bootstrap_auc_diff` is the AUC-difference analogue of
``src.backtest.block_bootstrap_sharpe_diff``, same centered-null design).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def block_bootstrap_auc_diff(y_true: pd.Series, p_a: pd.Series, p_b: pd.Series,
                             block_size: int = 20, n_boot: int = 2000,
                             seed: int = 0) -> dict:
    """Moving-block-bootstrap p-value for AUC(a) - AUC(b) != 0, resampled
    in blocks of TRADING DAYS.

    ``y_true``/``p_a``/``p_b`` share a day index (inner-joined here);
    each bootstrap replicate resamples blocks of DAYS (preserving each
    day's own (label, p_a, p_b) triple and the serial dependence across
    nearby days), recomputes AUC(a) and AUC(b) on the resampled labels/
    scores, and takes their difference. The bootstrap distribution is
    centered at its own mean (a null of "no difference"); the two-sided
    p-value is the share of centered bootstrap diffs at least as far from
    0 as the observed diff - identical design to
    ``src.backtest.block_bootstrap_sharpe_diff``, applied to AUC instead
    of Sharpe.
    """
    df = pd.DataFrame({"y": y_true, "a": p_a, "b": p_b}).dropna()
    df = df.sort_index()
    y = df["y"].to_numpy()
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    n = len(df)

    def _auc_diff(yy, aa, bb):
        if len(np.unique(yy)) < 2:
            return np.nan
        return roc_auc_score(yy, aa) - roc_auc_score(yy, bb)

    obs = _auc_diff(y, a, b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(max(n - block_size + 1, 1))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block_size)
                              for s in rng.choice(starts, n_blocks)])[:n]
        diffs[i] = _auc_diff(y[idx], a[idx], b[idx])
    valid = ~np.isnan(diffs)
    diffs = diffs[valid]
    centered = diffs - diffs.mean()
    p = float(np.mean(np.abs(centered) >= abs(obs))) if len(diffs) else np.nan
    return {"observed_auc_diff": float(obs), "p_value": p, "n_days": n,
           "block_size": block_size, "n_boot": int(valid.sum())}
