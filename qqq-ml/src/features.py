"""Daily features for predicting trading day t, with explicit information cutoffs.

Every feature is registered with a **cutoff**: the latest moment whose
information it may use, relative to the target day t.

============  ==========================================================
cutoff        information allowed (all times US/Eastern)
============  ==========================================================
calendar      exchange schedule only (known in advance): day of week,
              half-day flags
prev_close    everything up to the previous session's close, including
              the Cboe index close (16:15; 13:15 on half days)
open          plus day t's opening print (first 5-min bar's *open* only)
open_5m       plus day t's first 5-min bar (09:30-09:35) - the trade-
              filter features (stage 4, e.g. opening 5-min relative volume)
============  ==========================================================

``build_feature_matrix(data, cutoff="prev_close")`` returns only the
features whose cutoff is at or before the requested one, so matrices for
different decision times are built separately and can never mix.

How leakage is prevented structurally:

* All inputs live in :class:`MarketData`, which can be truncated at any
  timestamp (:meth:`MarketData.truncate`). A bar is kept only once it has
  ended; the bar in progress keeps only its open. The tests rebuild the
  features from truncated data and require identical values.
* Day-level quantities (RV, close, volume, ...) are computed per session
  and then *lagged along the exchange calendar*, never along "the previous
  available row", so a missing session cannot silently shift data.
* A session's quality flag (fill ratio, completeness) is only known at its
  close, so it masks only end-of-day quantities, never day t's open.

Extending: decorate a function ``f(ctx) -> pd.Series`` (indexed by target
day) with :func:`register`, e.g.::

    @register("open5m_rel_volume", "open_5m", "first 5-min volume / ...")
    def _f(ctx): ...

Every registered feature is automatically covered by
``tests/test_features.py::test_truncation_features_unchanged`` (rebuilds
it from data truncated at its own cutoff and requires an identical value)
and ``test_label_time_after_every_feature_cutoff`` - no extra test needed
per feature. This is the project's strongest look-ahead safety net; when a
new feature's information doesn't fit naturally into a ``FeatureContext``
quantity (e.g. it needs data from outside ``MarketData``), prefer building
it here over a bespoke helper elsewhere in ``src/`` - a rolling window
outside this registry (as in ``src/strategies.py``) needs its own,
hand-written causality test instead, and that class of check is easy to
get wrong (see ``historical_vol_weight``'s history in
``reports/week05_review_notes.md``).

Values produced outside this module (e.g. stage-2 model forecasts) can be
passed to :func:`build_feature_matrix` via ``extras`` with their cutoff;
their producer is responsible for their own information timing.

Usage::

    python -m src.features          # writes data/processed/features_*.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src import data_loader as dl

log = logging.getLogger(__name__)

CUTOFFS = ("calendar", "prev_close", "open", "open_5m")
CUTOFF_DOC = {
    "calendar": "exchange calendar only (known in advance)",
    "prev_close": "t-1 close (16:15 ET incl. Cboe close; 13:15 on half days)",
    "open": "t open print, 09:30 ET (first 5-min bar's open only)",
    "open_5m": "t first 5-min bar, 09:35 ET",
}
BARS_PER_DAY = {False: 78, True: 42}       # 5-min RTH bars: full / half day
HAR_WINDOWS = (1, 5, 22)                   # sessions: day / week / month


def cutoff_rank(cutoff: str) -> int:
    if cutoff not in CUTOFFS:
        raise ValueError(f"unknown cutoff {cutoff!r}; expected one of {CUTOFFS}")
    return CUTOFFS.index(cutoff)


def cutoff_time(days: pd.DatetimeIndex, cutoff: str,
                calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Latest timestamp (ET) whose information a feature may use for ``days``.

    ``calendar`` is the full exchange schedule; the previous session of t is
    looked up in it.
    """
    days = pd.DatetimeIndex(days)
    cal = pd.DatetimeIndex(calendar)
    cutoff_rank(cutoff)
    if cutoff in ("open", "open_5m"):
        mins = 9 * 60 + 30 + (5 if cutoff == "open_5m" else 0)
        return (days + pd.Timedelta(minutes=mins)).tz_localize(dl.ET)
    pos = cal.searchsorted(days) - 1
    if (pos < 0).any():
        raise ValueError("calendar must include the session before each day")
    prev = cal[pos]
    if cutoff == "calendar":
        return prev.tz_localize(dl.ET)                  # prev day 00:00
    half = dl._half_day_mask(prev)
    close = np.where(half, dl.HALF_CLOSE_MIN, dl.RTH_CLOSE_MIN) + 15
    return (prev + pd.to_timedelta(close, unit="m")).tz_localize(dl.ET)


def label_time(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """When a same-day target (e.g. RV of day t) becomes known: t's close."""
    days = pd.DatetimeIndex(days)
    close = np.where(dl._half_day_mask(days), dl.HALF_CLOSE_MIN,
                     dl.RTH_CLOSE_MIN)
    return (days + pd.to_timedelta(close, unit="m")).tz_localize(dl.ET)


def cboe_known_time(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """When a Cboe daily close becomes known: 16:15 ET, 13:15 on half days."""
    dates = pd.DatetimeIndex(dates)
    mins = np.where(dl._half_day_mask(dates), dl.HALF_CLOSE_MIN,
                    dl.RTH_CLOSE_MIN) + 15
    return (dates + pd.to_timedelta(mins, unit="m")).tz_localize(dl.ET)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass
class MarketData:
    """All raw inputs to the features, truncatable in time.

    Attributes:
        bars5: 5-min RTH bars (``ts``, ``bar_end``, OHLCV, ``n_filled``,
            ``n_minutes``, ``day``) as written by ``data_loader``.
        calendar: exchange sessions (naive dates). Known in advance, so it
            is never truncated.
        dividends: cash dividend per ex-date (USD), indexed by date.
        vol_index: Cboe closes by date (columns e.g. ``vix``, ``vxn``) or
            None.
    """

    bars5: pd.DataFrame
    calendar: pd.DatetimeIndex
    dividends: pd.Series
    vol_index: pd.DataFrame | None = None

    @classmethod
    def from_processed(cls, processed_dir: Path = dl.PROCESSED_DIR,
                       cboe_dir: Path | None = dl.CBOE_DIR) -> "MarketData":
        """Load from ``data/processed`` (and Cboe files if ``cboe_dir``)."""
        bars5 = dl.load_5min(processed_dir=processed_dir)
        di = pd.read_parquet(Path(processed_dir) / dl.OUTPUTS["daily_ibkr"])
        days = pd.DatetimeIndex(bars5["day"].unique()).sort_values()
        cal = di.index[(di.index >= days[0]) & (di.index <= days[-1])]
        cal = cal.union(days)
        divs = di.loc[di["is_ex_div"], "dividend"]
        vi = dl.load_cboe(cboe_dir=cboe_dir) if cboe_dir is not None else None
        return cls(bars5, cal, divs, vi)

    def truncate(self, t_cut: pd.Timestamp) -> "MarketData":
        """Data as it would have been known at ``t_cut`` (tz-aware ET).

        Finished bars (``bar_end <= t_cut``) are kept; a bar in progress
        keeps only its open; Cboe closes count as known at 16:15 ET of
        their date (13:15 on half days); dividends are kept up to
        ``t_cut``'s date.
        """
        b = self.bars5
        done = b[b["bar_end"] <= t_cut]
        live = b[(b["ts"] <= t_cut) & (b["bar_end"] > t_cut)].copy()
        live[["high", "low", "close", "vwap"]] = np.nan
        live[["volume", "n_filled", "n_minutes"]] = np.nan
        bars = pd.concat([done, live]).sort_values("ts").reset_index(drop=True)
        cut_day = t_cut.tz_localize(None).normalize()
        divs = self.dividends[self.dividends.index <= cut_day]
        vi = None
        if self.vol_index is not None:
            known = cboe_known_time(self.vol_index.index) <= t_cut
            vi = self.vol_index[known]
        return MarketData(bars, self.calendar, divs, vi)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature parameters (saved next to every matrix)."""

    rv_include_overnight: bool = False   # HAR inputs: intraday RV or + gap^2
    volume_window: int = 20
    min_periods_frac: float = 0.8        # rolling windows tolerate gaps
    max_fill_ratio: float = 0.05         # sessions above -> NaN quantities
    vix_max_stale_days: int = 5


# --------------------------------------------------------------------------
# Day-level quantities
# --------------------------------------------------------------------------
def intraday_returns(bars5: pd.DataFrame) -> pd.DataFrame:
    """5-min log returns within each session (never across sessions).

    The first return of a session is close/open of its first bar, so the
    returns cover open to close. Returns ``day``, ``ts`` and ``r``.
    """
    b = bars5.sort_values("ts")
    prev = b.groupby("day")["close"].shift(1)
    prev = prev.fillna(b["open"].where(b.groupby("day").cumcount() == 0))
    r = np.log(b["close"]) - np.log(prev)
    return pd.DataFrame({"day": b["day"].to_numpy(), "ts": b["ts"].to_numpy(),
                         "r": r.to_numpy()})


def realized_variance(bars5: pd.DataFrame, include_overnight: bool = False,
                      overnight: pd.Series | None = None) -> pd.DataFrame:
    """Daily realized variance from 5-min returns.

    ``rv`` = sum of squared intraday 5-min log returns (open to close of
    the session). With ``include_overnight`` the squared overnight log
    return (previous close to open, dividend added back) is added; pass it
    as ``overnight`` (indexed by day, NaN when the previous session is
    missing - the sum is then NaN too).

    Returns:
        DataFrame indexed by day with ``rv``, ``log_rv`` and ``n_returns``.
    """
    r = intraday_returns(bars5)
    g = r.groupby("day")["r"]
    rv = g.apply(lambda x: (x ** 2).sum(min_count=1))
    out = pd.DataFrame({"rv": rv, "n_returns": g.count()})
    if include_overnight:
        if overnight is None:
            raise ValueError("include_overnight=True needs `overnight`")
        out["rv"] = out["rv"] + overnight.reindex(out.index) ** 2
    out["log_rv"] = np.log(out["rv"].where(out["rv"] > 0))
    out.index.name = "day"
    return out


class FeatureContext:
    """Lazily computed day-level quantities on the exchange calendar.

    Quantities are indexed by session date ``s`` and describe that session
    (so they are known at s's close unless stated). Features turn them into
    values for target day t with :meth:`lag` / :meth:`rolling_mean`.
    """

    def __init__(self, data: MarketData, cfg: FeatureConfig = FeatureConfig()):
        self.data = data
        self.cfg = cfg
        self.cal = pd.DatetimeIndex(data.calendar).sort_values()

    # --- helpers ---------------------------------------------------------
    def on_cal(self, s: pd.Series) -> pd.Series:
        return s.reindex(self.cal)

    def lag(self, s: pd.Series, k: int = 1) -> pd.Series:
        """Value from k sessions before t along the exchange calendar."""
        return self.on_cal(s).shift(k)

    def rolling_mean(self, s: pd.Series, window: int) -> pd.Series:
        """Mean over sessions t-window .. t-1 (excludes t)."""
        mp = max(1, int(np.ceil(window * self.cfg.min_periods_frac)))
        return self.on_cal(s).shift(1).rolling(window, min_periods=mp).mean()

    # --- raw per-session quantities -------------------------------------
    @cached_property
    def _g(self):
        return self.data.bars5.groupby("day")

    @cached_property
    def valid(self) -> pd.Series:
        """Session complete and fill ratio <= max_fill_ratio (known at close)."""
        g = self._g
        n = g["close"].count()
        half = pd.Series(dl._half_day_mask(n.index), index=n.index)
        expected = half.map(BARS_PER_DAY)
        fill = g["n_filled"].sum() / g["n_minutes"].sum()
        ok = (n == expected) & (fill <= self.cfg.max_fill_ratio)
        return self.on_cal(ok).fillna(False).astype(bool)

    def _eod(self, s: pd.Series) -> pd.Series:
        """End-of-day quantity, NaN on invalid sessions."""
        return self.on_cal(s).where(self.valid)

    @cached_property
    def day_open(self) -> pd.Series:
        """Session open (first 5-min bar's open) - known at 09:30 of s."""
        return self.on_cal(self._g["open"].first())

    @cached_property
    def day_close(self) -> pd.Series:
        return self._eod(self._g["close"].last())

    @cached_property
    def day_high(self) -> pd.Series:
        return self._eod(self._g["high"].max())

    @cached_property
    def day_low(self) -> pd.Series:
        return self._eod(self._g["low"].min())

    @cached_property
    def day_volume(self) -> pd.Series:
        return self._eod(self._g["volume"].sum())

    @cached_property
    def first_bar(self) -> pd.DataFrame:
        """Session s's first 5-min bar (09:30-09:35 ET) OHLCV - known at
        s's 09:35, the "open_5m" cutoff. Not ``_eod``-masked (like
        :attr:`day_open`, a bar's own completeness doesn't depend on how
        the rest of the session later turns out) - matches
        ``src/rules.py``'s ORB range bar exactly (same 5-min aggregation,
        verified to match numerically day for day)."""
        g = self._g
        return pd.DataFrame({
            "open": self.on_cal(g["open"].first()),
            "high": self.on_cal(g["high"].first()),
            "low": self.on_cal(g["low"].first()),
            "close": self.on_cal(g["close"].first()),
            "volume": self.on_cal(g["volume"].first()),
        })

    @cached_property
    def true_range(self) -> pd.Series:
        """Session s's true range: max(high-low, |high-prev_close|,
        |low-prev_close|) - known at s's close."""
        prev_close = self.day_close.shift(1)
        return pd.concat([
            self.day_high - self.day_low,
            (self.day_high - prev_close).abs(),
            (self.day_low - prev_close).abs(),
        ], axis=1).max(axis=1)

    @cached_property
    def dividend(self) -> pd.Series:
        return self.data.dividends.reindex(self.cal).fillna(0.0)

    @cached_property
    def overnight(self) -> pd.Series:
        """log((open_s + div_s) / close_{s-1}); known at 09:30 of s.

        NaN when the previous calendar session is missing or invalid.
        """
        prev_close = self.day_close.shift(1)
        return np.log((self.day_open + self.dividend) / prev_close)

    @cached_property
    def cc_return(self) -> pd.Series:
        """log((close_s + div_s) / close_{s-1}): the full close-to-close
        return (intraday + overnight), known at s's close.

        NaN when session s or s-1 is missing/invalid.
        """
        prev_close = self.day_close.shift(1)
        return np.log((self.day_close + self.dividend) / prev_close)

    @cached_property
    def _returns(self) -> pd.DataFrame:
        return intraday_returns(self.data.bars5)

    @cached_property
    def rv_intraday(self) -> pd.Series:
        return self._eod(realized_variance(self.data.bars5)["rv"])

    @cached_property
    def rv(self) -> pd.Series:
        """RV used by the HAR features (variant set by the config)."""
        if self.cfg.rv_include_overnight:
            return self.rv_intraday + self.overnight ** 2
        return self.rv_intraday

    @cached_property
    def realized_moments(self) -> pd.DataFrame:
        """Realized skewness and kurtosis of 5-min returns per session.

        rskew = sqrt(N) * sum r^3 / RV^1.5,  rkurt = N * sum r^4 / RV^2
        (Amaya et al. 2015); rkurt is about 3 for normal returns.
        """
        r = self._returns
        g = r.assign(r2=r["r"] ** 2, r3=r["r"] ** 3, r4=r["r"] ** 4) \
            .groupby("day")
        n = g["r"].count()
        rv = g["r2"].sum(min_count=1)
        rv = rv.where(rv > 0)
        return pd.DataFrame({
            "rskew": np.sqrt(n) * g["r3"].sum(min_count=1) / rv ** 1.5,
            "rkurt": n * g["r4"].sum(min_count=1) / rv ** 2,
        })

    @cached_property
    def jump_stats(self) -> pd.DataFrame:
        """Per-session jump measures from 5-min returns.

        * ``max_abs_r``: largest |5-min log return| of the session.
        * ``bv``: bipower variation (pi/2)·N/(N-1)·Σ|r_i||r_{i-1}|, a
          jump-robust estimate of the continuous part of RV
          (Barndorff-Nielsen & Shephard 2004).
        * ``jump``: max(RV - BV, 0), the jump component.
        """
        r = self._returns.sort_values("ts")
        a = r["r"].abs()
        prod = a * a.groupby(r["day"]).shift(1)
        g = r.assign(a=a, prod=prod, r2=r["r"] ** 2).groupby("day")
        n = g["r"].count()
        rv = g["r2"].sum(min_count=1)
        bv = (np.pi / 2) * (n / (n - 1)) * g["prod"].sum(min_count=1)
        return pd.DataFrame({"max_abs_r": g["a"].max(), "bv": bv,
                             "jump": (rv - bv).clip(lower=0)})

    def vol_index_prev(self, name: str, diff: bool = False) -> pd.Series:
        """Last Cboe close on/before t-1 (or its change vs the one before)."""
        vi = self.data.vol_index
        if vi is None or name not in vi:
            raise KeyError(f"vol index {name!r} not loaded")
        s = vi[name].dropna()
        if diff:
            s = s.diff()
        a = dl.align_prev_close(s, self.cal, self.cfg.vix_max_stale_days)
        return a["value"]


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    cutoff: str
    definition: str
    func: Callable[[FeatureContext], pd.Series] = field(repr=False)
    needs: tuple[str, ...] = ()     # optional inputs, e.g. ("vix",)


FEATURES: dict[str, FeatureSpec] = {}


def register(name: str, cutoff: str, definition: str,
             needs: tuple[str, ...] = ()):
    """Decorator adding a feature function to the registry."""
    cutoff_rank(cutoff)

    def deco(func):
        if name in FEATURES:
            raise ValueError(f"feature {name!r} already registered")
        FEATURES[name] = FeatureSpec(name, cutoff, definition, func, needs)
        return func
    return deco


def _har(k: int):
    def f(ctx: FeatureContext) -> pd.Series:
        return ctx.lag(ctx.rv) if k == 1 else ctx.rolling_mean(ctx.rv, k)
    return f


def _log_har(k: int):
    def f(ctx: FeatureContext) -> pd.Series:
        v = _har(k)(ctx)
        return np.log(v.where(v > 0))
    return f


for _k in HAR_WINDOWS:
    register(f"har_rv_{_k}d", "prev_close",
             f"mean RV over sessions t-{_k}..t-1" if _k > 1
             else "RV of session t-1")(_har(_k))
    register(f"har_logrv_{_k}d", "prev_close",
             f"log of har_rv_{_k}d")(_log_har(_k))


@register("rskew_1d", "prev_close",
          "realized skewness of t-1 5-min returns: sqrt(N)·Σr³/RV^1.5")
def _rskew(ctx):
    return ctx.lag(ctx.realized_moments["rskew"].where(ctx.valid))


@register("rkurt_1d", "prev_close",
          "realized kurtosis of t-1 5-min returns: N·Σr⁴/RV²")
def _rkurt(ctx):
    return ctx.lag(ctx.realized_moments["rkurt"].where(ctx.valid))


@register("volume_rel_20d", "prev_close",
          "RTH volume of t-1 / mean RTH volume of t-20..t-1")
def _vol_rel(ctx):
    w = ctx.cfg.volume_window
    mp = max(1, int(np.ceil(w * ctx.cfg.min_periods_frac)))
    avg = ctx.on_cal(ctx.day_volume).rolling(w, min_periods=mp).mean()
    return ctx.lag(ctx.day_volume / avg)


@register("close_loc_1d", "prev_close",
          "(C-L)/(H-L) of session t-1 (0.5 if H==L)")
def _close_loc(ctx):
    rng = ctx.day_high - ctx.day_low
    loc = (ctx.day_close - ctx.day_low) / rng.where(rng > 0)
    loc = loc.mask((rng == 0), 0.5)
    return ctx.lag(loc)


@register("overnight_gap", "open",
          "log((open_t + dividend_t) / close_{t-1}); known only at t's 09:30 "
          "open")
def _gap(ctx):
    return ctx.overnight


@register("overnight_gap_1d", "prev_close",
          "t-1's own overnight gap (session t-2 close to t-1 open), lagged "
          "into row t - the predictive-label counterpart of overnight_gap, "
          "which is only known at t's own 09:30 open and can't be used to "
          "predict t")
def _gap_1d(ctx):
    return ctx.lag(ctx.overnight)


@register("day_of_week", "calendar", "weekday of t (0=Mon .. 4=Fri)")
def _dow(ctx):
    return pd.Series(ctx.cal.dayofweek, index=ctx.cal, dtype=float)


@register("is_half_day", "calendar", "1 if t is a 13:00 early close")
def _half(ctx):
    return pd.Series(dl._half_day_mask(ctx.cal), index=ctx.cal, dtype=float)


@register("prev_is_half_day", "calendar",
          "1 if t-1 was an early close (its RV covers fewer bars)")
def _prev_half(ctx):
    return _half(ctx).shift(1)


@register("vix_1d", "prev_close",
          "VIX close of the last Cboe date on/before session t-1",
          needs=("vix",))
def _vix(ctx):
    return ctx.vol_index_prev("vix")


@register("vxn_1d", "prev_close",
          "VXN close of the last Cboe date on/before session t-1",
          needs=("vxn",))
def _vxn(ctx):
    return ctx.vol_index_prev("vxn")


# --- jump features (week 4) -----------------------------------------------
@register("max_abs_ret5_1d", "prev_close",
          "largest |5-min log return| of session t-1")
def _max_abs_r(ctx):
    return ctx.lag(ctx.jump_stats["max_abs_r"].where(ctx.valid))


@register("jump_rv_bv_1d", "prev_close",
          "jump component of t-1: max(RV - BV, 0), BV = bipower variation")
def _jump(ctx):
    return ctx.lag(ctx.jump_stats["jump"].where(ctx.valid))


@register("vix_chg_1d", "prev_close",
          "VIX change: last Cboe close on/before session t-1 minus the "
          "Cboe close before that", needs=("vix",))
def _vix_chg(ctx):
    return ctx.vol_index_prev("vix", diff=True)


# --- close-to-close historical vol (week 5) -------------------------------
@register("hist_cc_var_20d", "prev_close",
          "trailing 20-session close-to-close variance: mean of squared "
          "full-day (intraday + overnight, dividend-adjusted) log returns "
          "over sessions t-20..t-1")
def _hist_cc_var_20d(ctx):
    return ctx.rolling_mean(ctx.cc_return ** 2, 20)


# --- pre-open trade-filter features (week 8, phase 4) ---------------------
@register("ret_cc_1d", "prev_close",
          "t-1's own close-to-close (intraday + overnight) log return, "
          "lagged into row t")
def _ret_cc_1d(ctx):
    return ctx.lag(ctx.cc_return)


@register("atr_20d", "prev_close",
          "trailing 20-session average true range: mean of "
          "max(high-low, |high-prev_close|, |low-prev_close|) over "
          "sessions t-20..t-1")
def _atr_20d(ctx):
    return ctx.rolling_mean(ctx.true_range, 20)


@register("gap_atr_ratio", "open",
          "day t's overnight gap / atr_20d - a scale-free gap size "
          "relative to the recent daily range")
def _gap_atr_ratio(ctx):
    return ctx.overnight / _atr_20d(ctx)


# --- first-5-min-bar trade-filter features (week 8, phase 4) --------------
@register("open5m_direction", "open_5m",
          "sign of t's first 5-min bar (close - open): +1/-1, 0 on a "
          "doji (close == open) - matches src/rules.py orb_day()'s own "
          "direction rule exactly")
def _open5m_direction(ctx):
    fb = ctx.first_bar
    return np.sign(fb["close"] - fb["open"])


@register("open5m_body_ratio", "open_5m",
          "|close-open| / (high-low) of t's first 5-min bar - how much of "
          "the opening range the candle's body fills")
def _open5m_body_ratio(ctx):
    fb = ctx.first_bar
    rng = fb["high"] - fb["low"]
    return ((fb["close"] - fb["open"]).abs() / rng).where(rng > 0)


@register("open5m_range_rel", "open_5m",
          "(high-low) of t's first 5-min bar / mean first-5-min (high-low) "
          "of sessions t-20..t-1")
def _open5m_range_rel(ctx):
    fb = ctx.first_bar
    rng = fb["high"] - fb["low"]
    return rng / ctx.rolling_mean(rng, 20)


@register("open5m_rel_volume", "open_5m",
          "volume of t's first 5-min bar / mean first-5-min volume of "
          "sessions t-20..t-1")
def _open5m_rel_volume(ctx):
    vol = ctx.first_bar["volume"]
    return vol / ctx.rolling_mean(vol, 20)


@register("open5m_gap_agree", "open_5m",
          "1 if t's first-5-min direction (open5m_direction) has the same "
          "sign as t's overnight gap, 0 if opposite signs, NaN if either "
          "is exactly zero (doji or zero gap)")
def _open5m_gap_agree(ctx):
    d = _open5m_direction(ctx)
    g = np.sign(ctx.overnight)
    same = (d == g) & (d != 0) & (g != 0)
    undefined = (d == 0) | (g == 0)
    return pd.Series(np.where(undefined, np.nan, same.astype(float)), index=d.index)


# --------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------
def available_features(cutoff: str, data: MarketData | None = None
                       ) -> list[FeatureSpec]:
    """Registered features usable at ``cutoff`` (and whose inputs exist)."""
    r = cutoff_rank(cutoff)
    vi = set(data.vol_index.columns) if data is not None and \
        data.vol_index is not None else set()
    return [s for s in FEATURES.values() if cutoff_rank(s.cutoff) <= r
            and (data is None or set(s.needs) <= vi)]


def build_feature_matrix(data: MarketData, cutoff: str = "prev_close",
                         cfg: FeatureConfig = FeatureConfig(),
                         names: list[str] | None = None,
                         extras: dict[str, tuple[pd.Series, str]] | None = None
                         ) -> pd.DataFrame:
    """Feature matrix for decisions made at ``cutoff`` on each target day.

    Rows: every session of ``data.calendar`` (warm-up rows contain NaN).
    Columns: registered features with cutoff <= ``cutoff`` (or ``names``),
    plus ``extras`` given as ``{name: (series_by_target_day, cutoff)}``.
    Requesting a feature whose cutoff is later than ``cutoff`` raises.
    """
    ctx = FeatureContext(data, cfg)
    specs = available_features(cutoff, data)
    if names is not None:
        by_name = {s.name: s for s in specs}
        bad = [n for n in names if n not in by_name]
        if bad:
            raise ValueError(f"not available at cutoff {cutoff!r}: {bad}")
        specs = [by_name[n] for n in names]
    cols = {s.name: s.func(ctx).reindex(ctx.cal).astype(float) for s in specs}
    for name, (series, c) in (extras or {}).items():
        if cutoff_rank(c) > cutoff_rank(cutoff):
            raise ValueError(f"extra {name!r} has cutoff {c!r} > {cutoff!r}")
        cols[name] = series.reindex(ctx.cal).astype(float)
    X = pd.DataFrame(cols, index=ctx.cal)
    X.index.name = "date"
    X.attrs["cutoff"] = cutoff
    return X


# --------------------------------------------------------------------------
# Descriptive (same-day, post-hoc) state features - week 6
# --------------------------------------------------------------------------
DESCRIPTIVE_COLUMNS = ("log_rv_desc", "close_loc_desc", "volume_rel_desc",
                      "overnight_gap_desc")
# predictive (decision-safe, prev_close cutoff) counterparts, same order
PREDICTIVE_STATE_COLUMNS = ("har_logrv_1d", "close_loc_1d", "volume_rel_20d",
                           "overnight_gap_1d")


def build_descriptive_matrix(data: MarketData, cfg: FeatureConfig = FeatureConfig(),
                             volume_window: int | None = None) -> pd.DataFrame:
    """Same-day versions of the four state features, for *labeling only*.

    Every column here uses day t's own realized values - explicitly **not**
    decision-safe (do not feed these into a strategy or a fold-fit model;
    see ``build_feature_matrix(..., cutoff="prev_close")`` and the
    ``*_1d``/``*_20d`` columns there for the predictive counterparts a
    strategy is actually allowed to use). Column names carry a ``_desc``
    suffix so the two matrices can never be mixed up by accident.

    * ``log_rv_desc``: log of day t's own intraday realized variance.
    * ``close_loc_desc``: (close-low)/(high-low) of day t's own session
      (0.5 if high == low).
    * ``volume_rel_desc``: day t's own RTH volume / the mean RTH volume of
      the 20 sessions t-19..t (inclusive of t - there is no look-ahead
      concern here since this label is never used for a decision).
    * ``overnight_gap_desc``: day t's own overnight gap (same value as the
      ``overnight_gap`` feature, cutoff "open" - included here under a
      ``_desc`` name for a single consistent descriptive matrix).
    """
    ctx = FeatureContext(data, cfg)
    w = volume_window or cfg.volume_window
    mp = max(1, int(np.ceil(w * cfg.min_periods_frac)))

    log_rv = np.log(ctx.rv_intraday.where(ctx.rv_intraday > 0))
    rng = ctx.day_high - ctx.day_low
    close_loc = ((ctx.day_close - ctx.day_low) / rng.where(rng > 0)).mask(
        rng == 0, 0.5)
    vol_avg = ctx.on_cal(ctx.day_volume).rolling(w, min_periods=mp).mean()
    volume_rel = ctx.day_volume / vol_avg

    X = pd.DataFrame({"log_rv_desc": log_rv, "close_loc_desc": close_loc,
                      "volume_rel_desc": volume_rel,
                      "overnight_gap_desc": ctx.overnight},
                     index=ctx.cal).astype(float)
    X.index.name = "date"
    X.attrs["cutoff"] = "descriptive (same-day, not decision-safe)"
    return X


def build_targets(data: MarketData, cfg: FeatureConfig = FeatureConfig()
                  ) -> pd.DataFrame:
    """Same-day RV targets for day t and the time they become known.

    ``rv`` (intraday), ``rv_on`` (+ overnight squared), their logs, and
    ``label_time`` = t's RTH close. Not used for modelling this week; the
    tests use it to check every feature is known before its label.
    """
    ctx = FeatureContext(data, cfg)
    rv = ctx.rv_intraday
    rv_on = rv + ctx.overnight ** 2
    y = pd.DataFrame({
        "rv": rv, "log_rv": np.log(rv.where(rv > 0)),
        "rv_on": rv_on, "log_rv_on": np.log(rv_on.where(rv_on > 0)),
        "overnight": ctx.overnight,
        "valid": ctx.valid,
    }, index=ctx.cal)
    y["label_time"] = label_time(ctx.cal)
    y.index.name = "date"
    return y


def feature_table(cutoff: str = "open_5m") -> pd.DataFrame:
    """Name / cutoff / definition of every registered feature."""
    return pd.DataFrame([{"name": s.name, "cutoff": s.cutoff,
                          "cutoff_meaning": CUTOFF_DOC[s.cutoff],
                          "definition": s.definition}
                         for s in available_features(cutoff)])


def build_and_save(processed_dir: Path = dl.PROCESSED_DIR,
                   cboe_dir: Path | None = dl.CBOE_DIR,
                   cfg: FeatureConfig = FeatureConfig()) -> dict[str, Path]:
    """Build the prev_close and open matrices + targets and write parquet.

    Rows are restricted to sessions that have minute data.
    """
    data = MarketData.from_processed(processed_dir, cboe_dir)
    days = pd.DatetimeIndex(data.bars5["day"].unique())
    out = {}
    for c in ("prev_close", "open"):
        X = build_feature_matrix(data, c, cfg)
        X = X.loc[X.index.isin(days)]
        p = Path(processed_dir) / f"features_{c}.parquet"
        X.to_parquet(p)
        out[c] = p
    y = build_targets(data, cfg)
    y = y.loc[y.index.isin(days)]
    out["targets"] = Path(processed_dir) / "targets_daily.parquet"
    y.to_parquet(out["targets"])
    meta = {"config": asdict(cfg), "cutoffs": CUTOFF_DOC,
            "features": feature_table().to_dict(orient="records"),
            "vol_index": None if data.vol_index is None
            else list(data.vol_index.columns),
            "built_at": pd.Timestamp.now(tz=dl.ET).isoformat()}
    out["meta"] = Path(processed_dir) / "features_meta.json"
    out["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build daily feature matrices.")
    p.add_argument("--no-cboe", action="store_true",
                   help="build without VIX/VXN features")
    p.add_argument("--rv-overnight", action="store_true",
                   help="HAR features use RV incl. overnight return squared")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = replace(FeatureConfig(), rv_include_overnight=args.rv_overnight)
    paths = build_and_save(cboe_dir=None if args.no_cboe else dl.CBOE_DIR,
                           cfg=cfg)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
