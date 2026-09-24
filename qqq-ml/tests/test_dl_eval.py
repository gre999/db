"""Tests for src/dl_eval.py: the AUC-difference block bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import dl_eval as DE


def test_block_bootstrap_auc_diff_detects_a_planted_large_effect():
    rng = np.random.default_rng(5)
    n = 1000
    days = pd.bdate_range("2019-01-01", periods=n)
    y = pd.Series((rng.random(n) < 0.5).astype(int), index=days)
    # a: strongly informative score; b: pure noise
    p_a = pd.Series(np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n)),
                    index=days)
    p_b = pd.Series(rng.uniform(0, 1, n), index=days)

    out = DE.block_bootstrap_auc_diff(y, p_a, p_b, block_size=20, n_boot=500, seed=1)
    assert out["observed_auc_diff"] > 0.3
    assert out["p_value"] < 0.05


def test_block_bootstrap_auc_diff_identical_scores_gives_zero_diff_and_p1():
    rng = np.random.default_rng(6)
    n = 500
    days = pd.bdate_range("2019-01-01", periods=n)
    y = pd.Series((rng.random(n) < 0.5).astype(int), index=days)
    p = pd.Series(rng.uniform(0, 1, n), index=days)

    out = DE.block_bootstrap_auc_diff(y, p, p, block_size=20, n_boot=200, seed=1)
    assert out["observed_auc_diff"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)


def test_block_bootstrap_auc_diff_inner_joins_on_date():
    rng = np.random.default_rng(7)
    days_a = pd.bdate_range("2019-01-01", periods=300)
    days_b = days_a[50:250]   # a strict subset
    y = pd.Series((rng.random(300) < 0.5).astype(int), index=days_a)
    p_a = pd.Series(rng.uniform(0, 1, 300), index=days_a)
    p_b = pd.Series(rng.uniform(0, 1, 200), index=days_b)

    out = DE.block_bootstrap_auc_diff(y, p_a, p_b, n_boot=50, seed=0)
    assert out["n_days"] == 200
