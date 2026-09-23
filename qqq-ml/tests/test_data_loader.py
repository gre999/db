"""Tests for src.data_loader: time zones, sessions, filling, no look-ahead."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from tests import conftest as c


def _rth(built):
    return dl.load_minute_rth(processed_dir=built["out"])


# ---------------------------------------------------------------- time zone
def test_utc_to_et_handles_dst(built):
    """09:30 ET is 13:30 UTC in summer time and 14:30 UTC in winter."""
    rth = _rth(built)
    assert str(rth["ts"].dt.tz) == dl.ET
    first = rth.groupby("day")["ts"].min()
    assert (first.dt.strftime("%H:%M") == "09:30").all()
    utc = first.dt.tz_convert("UTC").dt.strftime("%H:%M")
    assert utc[pd.Timestamp("2023-11-03")] == "13:30"   # before DST ends
    assert utc[pd.Timestamp("2023-11-06")] == "14:30"   # after DST ends


def test_raw_reader_dedups_overlapping_files(built):
    bars, stats = dl.read_raw_minute(built["raw"])
    assert stats["dedup_dropped"] > 0          # weekly files overlap
    assert stats["dedup_conflicts"] == 0
    assert bars["ts"].is_unique and bars["ts"].is_monotonic_increasing


# ------------------------------------------------------------ sessions
@pytest.mark.parametrize("hhmm,half,expected", [
    ("09:29", False, "pre"), ("09:30", False, "rth"), ("15:59", False, "rth"),
    ("16:00", False, "post"), ("04:00", False, "pre"),
    ("12:59", True, "rth"), ("13:00", True, "post"),
])
def test_label_sessions(hhmm, half, expected):
    day = c.HALF_DAY if half else "2023-11-15"
    ts = pd.Timestamp(f"{day} {hhmm}", tz=dl.ET)
    out = dl.label_sessions(pd.DataFrame({"ts": [ts]}))
    assert out["session"].iloc[0] == expected
    assert bool(out["is_half_day"].iloc[0]) is half


def test_rth_only_regular_hours(built):
    rth = _rth(built)
    minute = rth["ts"].dt.hour * 60 + rth["ts"].dt.minute
    assert minute.min() == 9 * 60 + 30
    full = rth[~rth["is_half_day"]]
    assert (full.groupby("day").size() == 390).all()
    assert (full["ts"].dt.strftime("%H:%M").max()) == "15:59"


def test_extended_hours_kept_separately(built):
    ext = dl.load_extended(processed_dir=built["out"])
    assert set(ext["session"]) == {"pre", "post"}
    rth_keys = set(_rth(built)["ts"])
    assert not rth_keys & set(ext["ts"])


# ------------------------------------------------------------ calendar
def test_half_day_rules():
    hd = dl.nyse_half_days(2014, 2026)
    for d in ["2014-12-24", "2017-07-03", "2019-11-29", "2023-11-24",
              "2024-07-03", "2024-12-24", "2025-07-03", "2026-12-24"]:
        assert date.fromisoformat(d) in hd, d
    # Friday Jul 3 / Dec 24 are full holidays, not half days.
    for d in ["2015-07-03", "2020-07-03", "2021-12-24", "2026-07-03"]:
        assert date.fromisoformat(d) not in hd, d


def test_half_day_session(built):
    rth = _rth(built)
    hd = rth[rth["day"] == pd.Timestamp(c.HALF_DAY)]
    assert len(hd) == 210 and hd["is_half_day"].all()
    assert hd["ts"].max().strftime("%H:%M") == "12:59"


def test_exclude_half_days_option(built):
    d = dl.load_daily(exclude_half_days=True, processed_dir=built["out"])
    assert pd.Timestamp(c.HALF_DAY) not in d.index
    d_all = dl.load_daily(processed_dir=built["out"])
    assert pd.Timestamp(c.HALF_DAY) in d_all.index


def test_holiday_and_partial_day(built):
    d = dl.load_daily(processed_dir=built["out"])
    assert pd.Timestamp(c.HOLIDAY) not in d.index
    assert pd.Timestamp(c.PARTIAL_DAY) not in d.index
    assert built["meta"]["dropped_incomplete_days"] == [c.PARTIAL_DAY]
    assert built["meta"]["missing_days_vs_ibkr_daily"] == []
    assert c.SHORT_EXT_DAY in built["meta"]["short_extended_sessions"]


# ------------------------------------------------------------ filling
def test_missing_minutes_forward_filled(built, raw_minutes):
    rth = _rth(built).set_index("ts")
    miss = pd.to_datetime(c.MISSING).tz_localize(dl.ET)
    rows = rth.loc[miss]
    assert (rows["fill_source"] == "filled").all()
    assert (rows["volume"] == 0).all() and rows["is_filled"].all()
    # Filled with the last close strictly BEFORE the gap, never after.
    before = raw_minutes[raw_minutes["ts"] < miss[0]]["close"].iloc[-1]
    assert np.allclose(rows[["open", "high", "low", "close"]], before)


def test_fill_never_uses_future_prices():
    """Changing prices after a gap must not change the filled values."""
    ts = pd.date_range("2023-11-15 09:30", periods=5, freq="1min", tz=dl.ET)
    base = pd.DataFrame({"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": [10.0, 11.0, 12.0, 13.0, 14.0],
                         "volume": 1.0, "vwap": 1.0, "bar_count": 1})
    gap = base.drop(index=2)
    a = dl.fill_rth_minutes(dl.label_sessions(gap))
    b_in = gap.copy()
    b_in.loc[b_in["ts"] > ts[2], "close"] = 999.0
    b = dl.fill_rth_minutes(dl.label_sessions(b_in))
    row_a = a[a["ts"] == ts[2]].iloc[0]
    row_b = b[b["ts"] == ts[2]].iloc[0]
    assert row_a["close"] == row_b["close"] == 11.0


def test_ibkr_zero_bar_counted_as_filled(built):
    rth = _rth(built).set_index("ts")
    row = rth.loc[pd.Timestamp(c.ZERO_BAR, tz=dl.ET)]
    assert row["fill_source"] == "ibkr_zero" and row["is_filled"]


def test_daily_fill_ratio(built):
    d = dl.load_daily(processed_dir=built["out"])
    day = d.loc[pd.Timestamp("2023-11-15")]
    assert day["missing_ratio"] == pytest.approx(3 / 390)
    assert day["fill_ratio"] == pytest.approx(3 / 390)
    assert d.loc[pd.Timestamp("2023-11-16"), "fill_ratio"] == \
        pytest.approx(1 / 390)


# ------------------------------------------------------------ anomalies
def test_spike_flagged_not_removed(built):
    an = dl.load_anomalies(processed_dir=built["out"])
    t = pd.Timestamp(c.SPIKE, tz=dl.ET)
    assert t in set(an["ts"])
    assert "abs_return" in an.set_index("ts").loc[t, "flags"]
    assert t in set(_rth(built)["ts"])       # still in the data


def test_anomaly_sigma_is_past_only(built):
    """Altering later bars must not change an earlier bar's sigma."""
    rth = _rth(built)
    cfg = dl.CleanConfig(anomaly_abs_return=0.0)   # flag everything
    a = dl.flag_anomalies(rth, cfg).set_index("ts")["sigma"]
    cut = rth["ts"].iloc[len(rth) // 2]
    rth2 = rth.copy()
    rth2.loc[rth2["ts"] > cut, "close"] *= 1.5
    b = dl.flag_anomalies(rth2, cfg).set_index("ts")["sigma"]
    early = a.index[a.index <= cut]
    pd.testing.assert_series_equal(a.loc[early], b.loc[early])


# ------------------------------------------------------------ dividends
def test_dividend_detected(built):
    di = pd.read_parquet(built["out"] / "daily_ibkr.parquet")
    assert di["is_ex_div"].sum() == 1
    ex = pd.Timestamp(c.EX_DIV_DAY)
    assert di.loc[ex, "is_ex_div"]
    assert di.loc[ex, "dividend"] == pytest.approx(c.DIVIDEND, abs=0.01)


def test_overnight_return_adds_back_dividend(built):
    d = dl.load_daily(processed_dir=built["out"])
    ex = pd.Timestamp(c.EX_DIV_DAY)
    i = d.index.get_loc(ex)
    prev_close = d["close"].iloc[i - 1]
    raw_gap = d.loc[ex, "open"] / prev_close - 1
    assert d.loc[ex, "ret_overnight"] == pytest.approx(
        raw_gap + d.loc[ex, "dividend"] / prev_close)
    assert d.loc[ex, "ret_overnight"] > raw_gap


def test_no_return_across_missing_session(built):
    d = dl.load_daily(processed_dir=built["out"])
    assert d["prev_day_missing"].iloc[0]
    assert np.isnan(d["ret_cc"].iloc[0])
    # Thanksgiving is a holiday, not a hole: the 24th has a valid return.
    assert not d.loc[pd.Timestamp(c.HALF_DAY), "prev_day_missing"]


def test_minute_close_matches_daily(built):
    d = dl.load_daily(processed_dir=built["out"])
    assert d["close_vs_ibkr"].abs().max() < 1e-9
    assert not d["split_suspect"].any()


# ------------------------------------------------------------ aggregation
def test_5min_aggregation(built):
    rth = _rth(built)
    b5 = dl.load_5min(processed_dir=built["out"])
    t0 = pd.Timestamp("2023-11-15 09:30", tz=dl.ET)
    one = rth[(rth["ts"] >= t0) & (rth["ts"] < t0 + pd.Timedelta(minutes=5))]
    bar = b5.set_index("ts").loc[t0]
    assert bar["open"] == one["open"].iloc[0]
    assert bar["close"] == one["close"].iloc[-1]
    assert bar["high"] == one["high"].max()
    assert bar["low"] == one["low"].min()
    assert bar["volume"] == one["volume"].sum()
    assert bar["bar_end"] == t0 + pd.Timedelta(minutes=5)
    full = b5[~b5["is_half_day"]]
    assert (full.groupby("day").size() == 78).all()
    assert (b5["n_minutes"] == 5).all()


def test_cache_skips_rebuild(built):
    meta = dl.build_all(built["raw"], built["out"])
    assert meta["built_at"] == built["meta"]["built_at"]
