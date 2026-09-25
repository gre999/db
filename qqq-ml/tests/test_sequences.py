"""Tests for src/sequences.py: minute-sequence tensors for tasks A/B."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import rules as R
from src import sequences as SEQ

ET = "America/New_York"


def _synthetic_minute(n_days: int = 30, bars_per_day: int = 40, seed: int = 0) -> pd.DataFrame:
    """Multi-day synthetic 1-min RTH bars, enough days for a 20-day
    trailing volume history and enough bars/day for window_len<=30 tests."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for day in days:
        ts0 = (day + pd.Timedelta(minutes=9 * 60 + 30)).tz_localize(ET)
        price = 100.0 + rng.normal(0, 1)
        for i in range(bars_per_day):
            ret = rng.normal(0, 0.001)
            o = price
            c = price * (1 + ret)
            hi = max(o, c) + abs(rng.normal(0, 0.01))
            lo = min(o, c) - abs(rng.normal(0, 0.01))
            vol = float(rng.uniform(1000, 5000))
            ts = ts0 + pd.Timedelta(minutes=i)
            rows.append({"ts": ts, "bar_end": ts + pd.Timedelta(minutes=1), "day": day,
                        "open": o, "high": hi, "low": lo, "close": c, "volume": vol})
            price = c
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def minute():
    return _synthetic_minute()


# --------------------------------------------------------------- shape/basic
def test_build_channel_tensor_shape_and_no_nan(minute):
    tensor, days = SEQ.build_channel_tensor(minute, window_len=5)
    assert tensor.shape[1:] == (5, SEQ.N_CHANNELS)
    assert not np.isnan(tensor).any()
    # first 20 trading days can't have a full trailing-20-day history
    assert days.min() >= minute["day"].sort_values().unique()[20]


def test_channels_match_manual_calc_for_one_day(minute):
    tensor, days = SEQ.build_channel_tensor(minute, window_len=5)
    day = days[5]
    g = minute[minute["day"] == day].sort_values("ts").reset_index(drop=True)
    w = g.iloc[:5]

    manual_log_ret = np.log(w["close"].to_numpy() / w["open"].iloc[0])
    row = tensor[list(days).index(day)]
    assert np.allclose(row[:, 0], manual_log_ret)

    manual_vwap = R.cumulative_vwap(w)
    manual_cum_vwap_dist = np.log(w["close"].to_numpy() / manual_vwap)
    assert np.allclose(row[:, 3], manual_cum_vwap_dist)

    cummax_h = np.maximum.accumulate(w["high"].to_numpy())
    cummin_l = np.minimum.accumulate(w["low"].to_numpy())
    manual_range_pos = (w["close"].to_numpy() - cummin_l) / (cummax_h - cummin_l)
    assert np.allclose(row[:, 2], manual_range_pos)


def test_rel_volume_channel_matches_manual_trailing_average(minute):
    tensor, days = SEQ.build_channel_tensor(minute, window_len=5)
    day = days[3]   # an early-but-valid day
    hist = SEQ.same_minute_volume_history(minute, window_len=5, trailing_days=20)
    manual_avg = hist.loc[day].to_numpy()
    g = minute[minute["day"] == day].sort_values("ts").reset_index(drop=True)
    manual_rel_vol = g["volume"].iloc[:5].to_numpy() / manual_avg
    row = tensor[list(days).index(day)]
    assert np.allclose(row[:, 1], manual_rel_vol)


# ---------------------------------------------------------- leakage checks
def test_truncation_tensor_unchanged_when_later_bars_removed(minute):
    """Dropping every bar strictly after the window (same day) must not
    change the tensor - _day_tensor only ever reads the first window_len
    rows."""
    window_len = 5
    tensor, days = SEQ.build_channel_tensor(minute, window_len=window_len)

    m = minute.sort_values("ts")
    keep = m.groupby("day").cumcount() < (window_len + 2)
    truncated = m[keep]
    tensor2, days2 = SEQ.build_channel_tensor(truncated, window_len=window_len)

    common = days.intersection(days2)
    assert len(common) > 0
    for d in common:
        row1 = tensor[list(days).index(d)]
        row2 = tensor2[list(days2).index(d)]
        assert np.allclose(row1, row2)


def test_mutation_same_day_and_later_days_leaves_tensor_unchanged(minute):
    """Week-11 required check: multiply-by-3-plus-1000 every bar AT OR
    AFTER the window's own cutoff - both later the SAME day and every
    LATER day (the rel_volume channel reads prior days' history, so a bug
    there would only show up by mutating later days too, not just the
    same day) - the tensor for days before the mutation boundary must be
    bit-for-bit unchanged."""
    window_len = 5
    tensor, days = SEQ.build_channel_tensor(minute, window_len=window_len)

    target_day = days[len(days) // 2]
    m2 = minute.copy()
    same_day_after = (m2["day"] == target_day) & (
        m2.groupby("day").cumcount() >= window_len)
    later_days = m2["day"] > target_day
    mask = same_day_after | later_days
    for col in ("open", "high", "low", "close"):
        m2.loc[mask, col] = m2.loc[mask, col] * 3 + 1000
    m2.loc[mask, "volume"] = m2.loc[mask, "volume"] * 3 + 1000

    tensor2, days2 = SEQ.build_channel_tensor(m2, window_len=window_len)
    # days up to and including target_day must be identical
    idx1 = list(days)
    idx2 = list(days2)
    for d in days[days <= target_day]:
        if d not in idx2:
            continue
        assert np.allclose(tensor[idx1.index(d)], tensor2[idx2.index(d)]), d


# --------------------------------------------------------------- task B labels
def test_task_b_labels_direction_and_net_bps_match_manual_calc(minute):
    n_minutes = 15
    cost_per_share = 0.0045
    labels = SEQ.build_task_b_labels(minute, n_minutes, cost_per_share)
    day = labels.index[10]
    g = minute[minute["day"] == day].sort_values("ts").reset_index(drop=True)
    entry = float(g["open"].iloc[n_minutes])
    exit_ = float(g["close"].iloc[-1])
    raw_bps = (exit_ - entry) / entry * 1e4
    cost_bps = 2 * cost_per_share / entry * 1e4
    manual_net = abs(raw_bps) - cost_bps
    assert labels.loc[day, "net_bps"] == pytest.approx(manual_net)
    assert labels.loc[day, "label"] == int(raw_bps > 0)
    assert labels.loc[day, "net_bps"] > 0   # every retained row must clear cost
    assert labels.loc[day, "raw_bps"] == pytest.approx(raw_bps)
    assert labels.loc[day, "cost_bps"] == pytest.approx(cost_bps)
    # net_bps is always the BEST-CASE (correct-direction) return
    assert labels.loc[day, "net_bps"] == pytest.approx(
        abs(labels.loc[day, "raw_bps"]) - labels.loc[day, "cost_bps"])


def test_task_b_labels_excludes_subcost_days():
    """A day whose entry-to-close move doesn't clear round-trip cost in
    either direction must be dropped entirely, not coded as a negative
    label - same treatment as ORB's own doji days."""
    day = pd.Timestamp("2023-01-02")
    ts0 = (day + pd.Timedelta(minutes=9 * 60 + 30)).tz_localize(ET)
    rows = []
    price = 100.0
    for i in range(20):
        # tiny, ~0 drift - well under any plausible round-trip cost in bps
        c = price * (1 + 0.0000001)
        rows.append({"ts": ts0 + pd.Timedelta(minutes=i),
                    "bar_end": ts0 + pd.Timedelta(minutes=i + 1), "day": day,
                    "open": price, "high": max(price, c), "low": min(price, c),
                    "close": c, "volume": 1000.0})
        price = c
    minute = pd.DataFrame(rows)
    labels = SEQ.build_task_b_labels(minute, n_minutes=10, cost_per_share=0.0045)
    assert day not in labels.index


def test_task_b_label_timing_is_after_the_decision_window(minute):
    n_minutes = 15
    labels = SEQ.build_task_b_labels(minute, n_minutes, 0.0045)
    cutoff = pd.Timedelta(minutes=9 * 60 + 30 + n_minutes)
    for day, row in labels.iterrows():
        day_start = pd.Timestamp(day).tz_localize(ET)
        assert row["entry_time"] >= day_start + cutoff
        assert row["exit_time"] > row["entry_time"]


# --------------------------------------------------------------- standardize
def test_standardize_uses_only_fit_days():
    # needs more valid (post-20-day-warmup) days than the shared fixture
    # has, so there's a real non-fit day to perturb
    minute = _synthetic_minute(n_days=45, bars_per_day=10, seed=7)
    tensor, days = SEQ.build_channel_tensor(minute, window_len=5)
    assert len(days) >= 15
    fit_days = days[:10]
    std_tensor, mean, std = SEQ.standardize(tensor, days, fit_days)

    fit_mask = days.isin(fit_days)
    manual_mean = tensor[fit_mask].reshape(-1, SEQ.N_CHANNELS).mean(axis=0)
    manual_std = tensor[fit_mask].reshape(-1, SEQ.N_CHANNELS).std(axis=0)
    assert np.allclose(mean, manual_mean)
    assert np.allclose(std, np.where(manual_std > 0, manual_std, 1.0))
    assert np.allclose(std_tensor, (tensor - mean) / std)

    # perturbing a NON-fit day must not change mean/std
    tensor2 = tensor.copy()
    tensor2[-1] += 1000.0
    _, mean2, std2 = SEQ.standardize(tensor2, days, fit_days)
    assert np.allclose(mean, mean2)
    assert np.allclose(std, std2)


# ------------------------------------------------------------ save/load
def test_save_and_load_tensor_round_trips(minute, tmp_path):
    tensor, days = SEQ.build_channel_tensor(minute, window_len=5)
    SEQ.save_tensor("task_a_test", tensor, days, out_dir=tmp_path)
    tensor2, days2 = SEQ.load_tensor("task_a_test", out_dir=tmp_path)
    assert np.allclose(tensor, tensor2)
    assert list(pd.DatetimeIndex(days2).normalize()) == list(pd.DatetimeIndex(days).normalize())
