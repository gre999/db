"""Feature tests: RV definitions, alignment, and look-ahead (truncation) checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl
from src import features as F
from tests import synth


@pytest.fixture(scope="module")
def data() -> F.MarketData:
    return synth.make_market_data()


@pytest.fixture(scope="module")
def full(data):
    return {c: F.build_feature_matrix(data, c) for c in F.CUTOFFS}


def _pick_days(data, n=10, seed=7) -> list[pd.Timestamp]:
    """Random days plus the planted edge cases (after warm-up)."""
    cal = data.calendar
    rng = np.random.default_rng(seed)
    rand = list(cal[rng.choice(np.arange(40, len(cal)), n, replace=False)])
    nxt = lambda d: cal[cal.searchsorted(d, side="right")]  # noqa: E731
    edge = [nxt(synth.HOLE_DAY), nxt(synth.BAD_DAY),
            nxt(synth.CBOE_ONLY), synth.VIX_GAP[-1], nxt(synth.VIX_GAP[-1]),
            data.dividends.index[3], pd.Timestamp("2017-11-24"),  # half day
            pd.Timestamp("2017-11-27")]                           # after it
    return sorted(set(rand + edge))


# ------------------------------------------------------------- truncation
@pytest.mark.parametrize("cutoff", F.CUTOFFS)
def test_truncation_features_unchanged(data, full, cutoff):
    """Delete everything after each day's cutoff; the day's row must not move.

    Runs every feature available at ``cutoff`` on ~18 days, including the
    days right after a missing session, an invalid session, a Cboe-only
    date, a VIX gap, an ex-dividend date and a half day.
    """
    X = full[cutoff]
    assert len(X.columns) == len(F.available_features(cutoff, data))
    for d in _pick_days(data):
        t_cut = F.cutoff_time(pd.DatetimeIndex([d]), cutoff, data.calendar)[0]
        Xt = F.build_feature_matrix(data.truncate(t_cut), cutoff)
        a, b = X.loc[d], Xt.loc[d]
        diff = [c for c in X.columns
                if not (np.isclose(a[c], b[c], rtol=0, atol=1e-12)
                        or (np.isnan(a[c]) and np.isnan(b[c])))]
        assert not diff, f"{d.date()} cutoff={cutoff}: changed {diff}"


def test_truncation_catches_a_leaky_feature(data):
    """Negative control: a feature using day t's close must fail the check."""
    name = "_leaky_same_day_close"
    F.register(name, "prev_close", "LEAKY: close of t")(
        lambda ctx: ctx.day_close)
    try:
        d = data.calendar[300]
        t_cut = F.cutoff_time(pd.DatetimeIndex([d]), "prev_close",
                              data.calendar)[0]
        full = F.build_feature_matrix(data, "prev_close", names=[name])
        trunc = F.build_feature_matrix(data.truncate(t_cut), "prev_close",
                                       names=[name])
        assert not np.isnan(full.loc[d, name])
        assert np.isnan(trunc.loc[d, name])      # the check would flag it
    finally:
        del F.FEATURES[name]


def test_later_cutoff_features_excluded(full):
    assert "overnight_gap" not in full["prev_close"]
    assert "overnight_gap" in full["open"]
    assert set(full["calendar"]) == {"day_of_week", "is_half_day",
                                     "prev_is_half_day"}
    with pytest.raises(ValueError):
        F.build_feature_matrix(synth.make_market_data(), "prev_close",
                               names=["overnight_gap"])


def test_extras_respect_cutoff(data):
    s = pd.Series(1.0, index=data.calendar)
    X = F.build_feature_matrix(data, "open", names=["day_of_week"],
                               extras={"stage2_pred": (s, "prev_close")})
    assert (X["stage2_pred"] == 1).all()
    with pytest.raises(ValueError):
        F.build_feature_matrix(data, "prev_close",
                               extras={"x": (s, "open_5m")})


# ------------------------------------------------------------- labels
def test_label_time_after_every_feature_cutoff(data):
    y = F.build_targets(data)
    for spec in F.available_features("open_5m", data):
        t = F.cutoff_time(y.index[1:], spec.cutoff, data.calendar)
        assert (y["label_time"].iloc[1:].to_numpy() > t).all(), spec.name
    assert (y["label_time"].dt.strftime("%H:%M")
            .isin(["16:00", "13:00"])).all()


# ------------------------------------------------------------- RV
def test_realized_variance_definition(data):
    b = data.bars5
    d = data.calendar[100]
    day = b[b["day"] == d].sort_values("ts")
    px = np.r_[day["open"].iloc[0], day["close"].to_numpy()]
    manual = np.sum(np.diff(np.log(px)) ** 2)
    rv = F.realized_variance(b)
    assert rv.loc[d, "rv"] == pytest.approx(manual)
    assert rv.loc[d, "log_rv"] == pytest.approx(np.log(manual))
    assert rv.loc[d, "n_returns"] == 78


def test_realized_variance_with_overnight(data):
    ctx = F.FeatureContext(data)
    on = ctx.overnight
    rv0 = F.realized_variance(data.bars5)
    rv1 = F.realized_variance(data.bars5, include_overnight=True, overnight=on)
    d = data.calendar[100]
    assert rv1.loc[d, "rv"] == pytest.approx(rv0.loc[d, "rv"] + on[d] ** 2)
    # No overnight return across a missing session.
    after_hole = data.calendar[data.calendar.searchsorted(synth.HOLE_DAY) + 1]
    assert np.isnan(on[after_hole]) and np.isnan(rv1.loc[after_hole, "rv"])


def test_overnight_adds_dividend(data):
    ctx = F.FeatureContext(data)
    ex = data.dividends.index[2]
    prev = data.calendar[data.calendar.searchsorted(ex) - 1]
    expect = np.log((ctx.day_open[ex] + 0.3) / ctx.day_close[prev])
    assert ctx.overnight[ex] == pytest.approx(expect)


def test_invalid_session_is_nan(data, full):
    X = full["prev_close"]
    nxt = data.calendar[data.calendar.searchsorted(synth.BAD_DAY) + 1]
    assert np.isnan(X.loc[nxt, "har_rv_1d"])
    assert not np.isnan(X.loc[nxt, "har_rv_5d"])    # window tolerates gaps


# ------------------------------------------------------------- features
def test_har_windows(data, full):
    X = full["prev_close"]
    ctx = F.FeatureContext(data)
    rv = ctx.rv
    i = 200
    d = data.calendar[i]
    assert X.loc[d, "har_rv_1d"] == pytest.approx(rv.iloc[i - 1])
    assert X.loc[d, "har_rv_5d"] == pytest.approx(rv.iloc[i - 5:i].mean())
    assert X.loc[d, "har_rv_22d"] == pytest.approx(rv.iloc[i - 22:i].mean())
    assert X.loc[d, "har_logrv_5d"] == pytest.approx(
        np.log(rv.iloc[i - 5:i].mean()))


def test_volume_close_loc_dow(data, full):
    X = full["prev_close"]
    ctx = F.FeatureContext(data)
    i = 200
    d, p = data.calendar[i], data.calendar[i - 1]
    v = ctx.day_volume
    assert X.loc[d, "volume_rel_20d"] == pytest.approx(
        v.iloc[i - 1] / v.iloc[i - 20:i].mean())
    loc = (ctx.day_close[p] - ctx.day_low[p]) / (ctx.day_high[p] - ctx.day_low[p])
    assert X.loc[d, "close_loc_1d"] == pytest.approx(loc)
    assert 0 <= X["close_loc_1d"].min() and X["close_loc_1d"].max() <= 1
    assert X.loc[d, "day_of_week"] == d.dayofweek


def test_realized_moments_reasonable(full):
    X = full["prev_close"]
    assert X["rkurt_1d"].median() > 3          # t(5) returns: fat tails
    assert abs(X["rskew_1d"].median()) < 0.5


def test_vix_alignment(data, full):
    X = full["prev_close"]
    vi = data.vol_index
    mon = pd.Timestamp("2016-03-28")            # after Good Friday Cboe row
    assert X.loc[mon, "vix_1d"] == vi.loc[synth.CBOE_ONLY, "vix"]
    # Last VIX before the gap is > max_stale_days old by the end -> NaN.
    assert np.isnan(X.loc[synth.VIX_GAP[-1], "vix_1d"])
    a = dl.align_prev_close(vi["vix"], data.calendar)
    ok = a["source_date"].notna()
    assert (a.loc[ok, "source_date"] < a.index[ok]).all()


def test_read_cboe_csv(tmp_path):
    vi = synth.make_vol_index().iloc[:50]
    p = tmp_path / "VIX_History.csv"
    synth.write_cboe_csv(p, vi, "vix")
    df = dl.read_cboe_daily(p)
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert df.index[0] == vi.index[0] and df["close"].iloc[5] == \
        pytest.approx(vi["vix"].iloc[5])


def test_from_processed_pipeline(built, tmp_path):
    """End-to-end on week-1 synthetic processed data + Cboe files."""
    cboe = tmp_path / "cboe"
    cboe.mkdir()
    vi = synth.make_vol_index()
    synth.write_cboe_csv(cboe / "VIX_History.csv", vi, "vix")
    synth.write_cboe_csv(cboe / "VXN_History.csv", vi, "vxn")
    md = F.MarketData.from_processed(built["out"], cboe)
    X = F.build_feature_matrix(md, "open")
    assert "vix_1d" in X and "overnight_gap" in X
    ex = pd.Timestamp("2023-11-20")                   # planted dividend
    assert md.dividends.loc[ex] == pytest.approx(0.60, abs=0.01)


# ------------------------------------------------------------- jump features
def test_jump_features(data, full):
    X = full["prev_close"]
    ctx = F.FeatureContext(data)
    i = 200
    d, p = data.calendar[i], data.calendar[i - 1]
    b = data.bars5[data.bars5["day"] == p].sort_values("ts")
    px = np.r_[b["open"].iloc[0], b["close"].to_numpy()]
    r = np.diff(np.log(px))
    n = len(r)
    bv = np.pi / 2 * n / (n - 1) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    rv = np.sum(r ** 2)
    assert X.loc[d, "max_abs_ret5_1d"] == pytest.approx(np.abs(r).max())
    assert X.loc[d, "jump_rv_bv_1d"] == pytest.approx(max(rv - bv, 0.0))
    assert (X["jump_rv_bv_1d"].dropna() >= 0).all()
    vi = data.vol_index["vix"]
    src = vi.index[vi.index < d]
    assert X.loc[d, "vix_chg_1d"] == pytest.approx(
        vi.loc[src[-1]] - vi.loc[src[-2]])
    # BV is jump-robust: on no-jump synthetic data it tracks RV closely.
    js = ctx.jump_stats
    assert (js["bv"] / (js["jump"] + js["bv"])).median() > 0.8
