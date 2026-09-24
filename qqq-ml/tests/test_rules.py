"""ORB and VWAP-trend rule tests: entry/stop/target logic, timing, costs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import rules as R

DAY = pd.Timestamp("2023-06-01")
ET = "America/New_York"


def _bars(day: pd.Timestamp, rows: list[dict], start_min: int = 0) -> pd.DataFrame:
    """rows: list of {open, high, low, close, volume}; one per RTH minute
    starting at 09:30 + start_min."""
    ts0 = (day + pd.Timedelta(minutes=9 * 60 + 30 + start_min)).tz_localize(ET)
    out = []
    for i, r in enumerate(rows):
        ts = ts0 + pd.Timedelta(minutes=i)
        out.append({"ts": ts, "bar_end": ts + pd.Timedelta(minutes=1),
                   "day": day, **r})
    return pd.DataFrame(out)


def _flat_bars(day, n, price=100.0, volume=1000.0, start_min=0):
    return _bars(day, [{"open": price, "high": price, "low": price,
                        "close": price, "volume": volume}] * n,
                start_min=start_min)


# --------------------------------------------------------------------------
# ORB
# --------------------------------------------------------------------------
def test_orb_green_first_bar_goes_long_with_stop_at_range_low():
    rows = [{"open": 100.0, "high": 101.5, "low": 99.5, "close": 101.0,
             "volume": 1000}] * 5           # green range bar (5 x 1-min)
    rows += [{"open": 101.0, "high": 101.2, "low": 100.8, "close": 101.0,
              "volume": 1000}] * 50          # quiet afterwards -> EOD exit
    bars = _bars(DAY, rows)
    rec = R.orb_day(bars)
    assert rec["traded"] and rec["direction"] == 1
    assert rec["entry_price"] == pytest.approx(101.0)       # bar 6's open
    assert rec["stop_price"] == pytest.approx(99.5)         # range low
    assert rec["target_price"] == pytest.approx(101.0 + 10 * 1.5)
    assert rec["exit_reason"] == "eod"


def test_orb_red_first_bar_goes_short_with_stop_at_range_high():
    rows = [{"open": 100.0, "high": 100.5, "low": 98.0, "close": 98.5,
             "volume": 1000}] * 5            # red range bar
    rows += [{"open": 98.5, "high": 98.7, "low": 98.3, "close": 98.5,
              "volume": 1000}] * 50
    bars = _bars(DAY, rows)
    rec = R.orb_day(bars)
    assert rec["traded"] and rec["direction"] == -1
    assert rec["entry_price"] == pytest.approx(98.5)
    assert rec["stop_price"] == pytest.approx(100.5)         # range high
    assert rec["target_price"] == pytest.approx(98.5 - 10 * 2.0)


def test_orb_doji_no_trade():
    rows = [{"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
             "volume": 1000}] * 5
    rows += [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0,
              "volume": 1000}] * 50
    rec = R.orb_day(_bars(DAY, rows))
    assert not rec["traded"] and rec["reason"] == "doji"


def test_orb_same_bar_both_hit_counts_as_stop():
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 1000}] * 5            # green, risk = 1.0, target = +10
    # entry bar's range spans both stop (99.0) and target (110.5) -> stop
    rows += [{"open": 100.5, "high": 111.0, "low": 90.0, "close": 100.5,
              "volume": 1000}]
    rows += [{"open": 100.5, "high": 100.6, "low": 100.4, "close": 100.5,
              "volume": 1000}] * 20
    rec = R.orb_day(_bars(DAY, rows))
    assert rec["exit_reason"] == "stop"
    assert rec["exit_price"] == pytest.approx(99.0)


def test_orb_intraday_time_order_earlier_stop_beats_later_target():
    """A later bar that would (in isolation) hit target must not override
    an earlier bar that already hit stop - resolution must walk forward in
    time and stop at the first trigger."""
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 1000}] * 5            # green range, stop = 99.0
    rows += [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0,
              "volume": 1000}] * 3           # entry (risk=1, target=110) + 2 quiet
    rows += [{"open": 100.0, "high": 100.1, "low": 98.5, "close": 99.0,
              "volume": 1000}]               # hits stop (98.5 <= 99.0)
    rows += [{"open": 99.0, "high": 120.0, "low": 98.9, "close": 110.0,
              "volume": 1000}]               # would hit target, must not be reached
    bars = _bars(DAY, rows)
    rec = R.orb_day(bars)
    assert rec["exit_reason"] == "stop"
    assert rec["exit_time"] == bars.iloc[8]["ts"]


def test_orb_target_hit_before_stop_in_time():
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 1000}] * 5            # green range, stop = 99.0
    rows += [{"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0,
              "volume": 1000}]               # entry bar: risk=1, target=110, quiet
    rows += [{"open": 100.0, "high": 111.0, "low": 99.5, "close": 110.0,
              "volume": 1000}]               # hits target (111 >= 110), not stop
    rows += [{"open": 110.0, "high": 110.1, "low": 50.0, "close": 110.0,
              "volume": 1000}]               # would hit stop, must not be reached
    bars = _bars(DAY, rows)
    rec = R.orb_day(bars)
    assert rec["exit_reason"] == "target"
    assert rec["exit_price"] == pytest.approx(110.0)
    assert rec["exit_time"] == bars.iloc[6]["ts"]


def test_orb_eod_exit_uses_last_bar_close():
    rows = [{"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.2,
             "volume": 1000}] * 5
    rows += [{"open": 100.2, "high": 100.3, "low": 100.1, "close": 100.2,
              "volume": 1000}] * 385         # full day, nothing triggers
    rows[-1] = {"open": 100.2, "high": 100.3, "low": 100.1, "close": 103.0,
               "volume": 1000}
    bars = _bars(DAY, rows)
    rec = R.orb_day(bars)
    assert rec["exit_reason"] == "eod"
    assert rec["exit_price"] == pytest.approx(103.0)
    assert rec["exit_time"] == bars.iloc[-1]["bar_end"]


def test_orb_cost_reduces_r_net_by_two_sides():
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
             "volume": 1000}] * 5            # green range, stop = 99.0
    rows += [{"open": 100.0, "high": 111.0, "low": 99.5, "close": 110.0,
              "volume": 1000}]               # entry=100 -> risk=1, target=110, hit immediately
    cfg = R.ORBConfig(cost_per_share=0.01)
    rec = R.orb_day(_bars(DAY, rows), cfg)
    assert rec["r_gross"] == pytest.approx(10.0)
    assert rec["r_net"] == pytest.approx(10.0 - 2 * 0.01 / 1.0)


def test_orb_cross_day_independence():
    """Perturbing one day's bars must not move another day's trade record -
    the same look-ahead-style check as assert_output_at_t_unaffected_by_
    input_at_t, applied at the day-panel level since orb_day/vwap_day take
    whole sessions, not scalar Series positions."""
    days = pd.bdate_range("2023-06-01", periods=3)
    rows = ([{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
              "volume": 1000}] * 5
           + [{"open": 100.5, "high": 100.6, "low": 100.4, "close": 100.5,
               "volume": 1000}] * 50)
    panel = pd.concat([_bars(d, rows) for d in days], ignore_index=True)
    out = R.run_orb(panel, mask_invalid_sessions=False)

    panel2 = panel.copy()
    mid = panel2["day"] == days[1]
    panel2.loc[mid, "high"] += 50            # blow up only the middle day
    out2 = R.run_orb(panel2, mask_invalid_sessions=False)

    pd.testing.assert_series_equal(out.loc[days[0]], out2.loc[days[0]])
    pd.testing.assert_series_equal(out.loc[days[2]], out2.loc[days[2]])
    assert out.loc[days[1], "exit_reason"] != out2.loc[days[1], "exit_reason"]


# --------------------------------------------------------------------------
# VWAP trend
# --------------------------------------------------------------------------
def test_vwap_signal_acts_on_next_bar_open_not_same_bar():
    # price crosses above vwap at bar index 2's close; entry must be at
    # bar 3's open, not bar 2's own open or close.
    rows = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
             "volume": 100}] * 2
    rows += [{"open": 100.0, "high": 101.0, "low": 100.0, "close": 101.0,
              "volume": 100}]                # bar 2: closes above vwap
    rows += [{"open": 101.5, "high": 101.5, "low": 101.5, "close": 101.5,
              "volume": 100}]                # bar 3: entry happens here
    rows += [{"open": 101.5, "high": 101.5, "low": 101.5, "close": 101.5,
              "volume": 100}] * 300
    bars = _bars(DAY, rows)
    rec = R.vwap_day(bars)
    assert rec["first_entry_time"] == bars.iloc[3]["ts"]
    assert rec["first_entry_price"] == pytest.approx(101.5)


def test_vwap_reversal_cost_is_two_sides_per_segment():
    # forces exactly 2 segments: long from bar 1, reversal to short later,
    # then forced flatten - segments = 2 -> cost = cfg.cost_per_share * 2 * 2
    rows = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
             "volume": 100}]
    rows += [{"open": 100.0, "high": 101.0, "low": 100.0, "close": 101.0,
              "volume": 100}]                 # close above vwap -> long
    rows += [{"open": 101.0, "high": 101.0, "low": 99.0, "close": 99.0,
              "volume": 100}] * 3              # drags close below vwap
    rows += [{"open": 99.0, "high": 99.0, "low": 99.0, "close": 99.0,
              "volume": 100}] * 5
    cfg = R.VWAPConfig(cost_per_share=0.01)
    rec = R.vwap_day(_bars(DAY, rows), cfg)
    assert rec["n_segments"] == 2
    assert rec["cost"] == pytest.approx(0.01 * 2 * 2)


def test_vwap_forced_flatten_at_session_close():
    rows = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
             "volume": 100}]
    rows += [{"open": 100.0, "high": 101.0, "low": 100.0, "close": 101.0,
              "volume": 100}]
    rows += [{"open": 101.5, "high": 101.5, "low": 101.5, "close": 101.5,
              "volume": 100}] * 30
    rows[-1] = {"open": 101.5, "high": 101.5, "low": 101.5, "close": 104.0,
               "volume": 100}
    bars = _bars(DAY, rows)
    rec = R.vwap_day(bars)
    assert rec["final_exit_time"] == bars.iloc[-1]["bar_end"]
    assert rec["final_exit_price"] == pytest.approx(104.0)


def test_cumulative_vwap_matches_manual_typical_price_calc():
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 100},
           {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 200},
           {"open": 101.5, "high": 101.5, "low": 100.5, "close": 101.0, "volume": 50}]
    bars = _bars(DAY, rows)
    vwap = R.cumulative_vwap(bars)

    tp = [(101.0 + 99.0 + 100.5) / 3, (102.0 + 100.0 + 101.5) / 3,
         (101.5 + 100.5 + 101.0) / 3]
    vol = [100, 200, 50]
    manual = []
    cum_pv = cum_v = 0.0
    for t, v in zip(tp, vol):
        cum_pv += t * v
        cum_v += v
        manual.append(cum_pv / cum_v)
    assert vwap == pytest.approx(manual)


def test_vwap_cross_day_independence():
    days = pd.bdate_range("2023-06-01", periods=3)
    rows = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
             "volume": 100}]
    rows += [{"open": 100.0, "high": 101.0, "low": 100.0, "close": 101.0,
              "volume": 100}]
    rows += [{"open": 101.5, "high": 101.5, "low": 101.5, "close": 101.5,
              "volume": 100}] * 300
    panel = pd.concat([_bars(d, rows) for d in days], ignore_index=True)
    out = R.run_vwap(panel, mask_invalid_sessions=False)

    panel2 = panel.copy()
    mid = panel2["day"] == days[1]
    panel2.loc[mid, "close"] = panel2.loc[mid, "close"] - 200
    out2 = R.run_vwap(panel2, mask_invalid_sessions=False)

    pd.testing.assert_series_equal(out.loc[days[0]], out2.loc[days[0]])
    pd.testing.assert_series_equal(out.loc[days[2]], out2.loc[days[2]])


# --------------------------------------------------------------------------
# Validity masking
# --------------------------------------------------------------------------
def test_apply_validity_mask_no_cross_day_leakage():
    """Same check as assert_output_at_t_unaffected_by_input_at_t (Week 5),
    applied directly: flipping one day's validity flag must only ever
    change that day's own row. ``valid`` is boolean, so the generic helper
    (built for numeric perturbation) doesn't apply as-is here."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    out = pd.DataFrame({"traded": [True] * 10, "reason": [None] * 10,
                        "bps_return": [1.0] * 10, "r_net": [0.5] * 10},
                       index=idx)
    valid = pd.Series(True, index=idx)
    masked = R.apply_validity_mask(out, valid)

    valid2 = valid.copy()
    valid2.iloc[3] = False
    masked2 = R.apply_validity_mask(out, valid2)

    pd.testing.assert_frame_equal(masked.drop(index=idx[3]),
                                  masked2.drop(index=idx[3]))
    assert masked["traded"].iloc[3] and not masked2["traded"].iloc[3]
