"""Load, clean and aggregate QQQ bars downloaded from IBKR.

Input (``data/raw/ibkr/``, produced by ``scripts/download_ibkr.py``, never
modified):

* ``minute/QQQ_1min_TRADES_*.csv`` - 1-min TRADES bars, all sessions, UTC,
  timestamp = bar start, prices NOT dividend-adjusted, weekly files overlap.
* ``daily/QQQ_1day_TRADES.csv``        - daily bars, split-adjusted only.
* ``daily/QQQ_1day_ADJUSTED_LAST.csv`` - daily bars, split + dividend adjusted.

Output (``data/processed/``, all parquet, timestamps tz-aware US/Eastern):

* ``minute_all_sessions.parquet`` - deduplicated raw minutes + session label
* ``minute_rth.parquet``          - 09:30-16:00 (13:00 on half days) 1-min
  grid, missing minutes filled, ``is_filled`` / ``fill_source`` flags
* ``minute_ext.parquet``          - pre-market and after-hours, unfilled
* ``bars_5min.parquet``           - 5-min RTH bars built from ``minute_rth``
* ``daily.parquet``               - daily RTH bars from minutes, merged with
  IBKR daily prices, dividend info and per-day quality stats
* ``daily_ibkr.parquet``          - IBKR daily raw + adjusted + dividends
* ``anomalies.parquet``           - flagged minutes (never deleted)
* ``build_meta.json``             - counts, config and data-quality notes

Look-ahead rules enforced here (the project's hard line):

* Every bar carries ``bar_end``; a bar may only be used at or after
  ``bar_end``. The 09:30 1-min bar is known at 09:31, the 09:30 5-min bar at
  09:35.
* Missing minutes are filled with the last price seen *strictly before*
  that minute (any session). Never back-filled.
* The anomaly sigma is a rolling std of *past* returns only (shifted by 1).
* ``ADJUSTED_LAST`` is back-adjusted: its price *levels* change whenever a
  later dividend is paid, so they embed future information. Only use its
  returns (``ret_cc_adj``); use raw prices for anything level-based.
* Dividends are applied on the ex-date, when the amount is already public.
* Half days come from exchange calendar rules that are known in advance.
* A session that was still in progress when the data was downloaded is
  dropped rather than treated as a short day.

Usage::

    python -m src.data_loader            # build everything (uses cache)
    python -m src.data_loader --force    # rebuild from raw

    from src.data_loader import load_minute_rth, load_5min, load_daily
    df = load_5min(exclude_half_days=True)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ET = "America/New_York"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ibkr"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Session boundaries in minutes after midnight ET (bar start times).
PRE_OPEN_MIN = 4 * 60           # 04:00
RTH_OPEN_MIN = 9 * 60 + 30      # 09:30
RTH_CLOSE_MIN = 16 * 60         # 16:00
HALF_CLOSE_MIN = 13 * 60        # 13:00
POST_CLOSE_MIN = 20 * 60        # 20:00
HALF_POST_CLOSE_MIN = 17 * 60   # 17:00

PRICE_COLS = ["open", "high", "low", "close"]
BAR_COLS = PRICE_COLS + ["volume", "vwap", "bar_count"]


@dataclass(frozen=True)
class CleanConfig:
    """Tunable cleaning parameters (saved into ``build_meta.json``)."""

    anomaly_abs_return: float = 0.01   # |1-min log return| above this
    anomaly_sigma: float = 8.0         # |return| above k * rolling std
    anomaly_vol_window: int = 390      # rolling window (1-min RTH bars)
    anomaly_min_periods: int = 60
    anomaly_bar_range: float = 0.01    # (high - low) / close above this
    dividend_min_amount: float = 0.01  # USD; smaller implied dividends = noise
    split_check_tolerance: float = 0.02  # minute vs daily close mismatch


# --------------------------------------------------------------------------
# Exchange calendar
# --------------------------------------------------------------------------
def _thanksgiving(year: int) -> date:
    """Fourth Thursday of November."""
    nov1 = date(year, 11, 1)
    first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + timedelta(weeks=3)


def nyse_half_days(start_year: int, end_year: int) -> set[date]:
    """Return scheduled NYSE/Nasdaq early-close (13:00 ET) days.

    Rules (stable since at least 2010):

    * the day after Thanksgiving;
    * July 3 when it falls Mon-Thu (a Friday July 3 is the observed holiday);
    * December 24 when it falls Mon-Thu (a Friday Dec 24 is the observed
      Christmas holiday; weekends have no session).

    Irregular closures (e.g. national days of mourning) are full closures
    and simply have no data, so they need no special handling here.
    """
    days: set[date] = set()
    for y in range(start_year, end_year + 1):
        days.add(_thanksgiving(y) + timedelta(days=1))
        for d in (date(y, 7, 3), date(y, 12, 24)):
            if d.weekday() <= 3:
                days.add(d)
    return days


def _half_day_mask(days: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: which session dates (naive, normalized) are half days."""
    days = pd.DatetimeIndex(days)
    if len(days) == 0:
        return np.zeros(0, dtype=bool)
    hd = nyse_half_days(days.min().year, days.max().year)
    return np.isin(days.date, list(hd))


def rth_close_minute(is_half_day: np.ndarray | bool) -> np.ndarray:
    """Minute-of-day at which RTH ends (exclusive): 16:00 or 13:00."""
    return np.where(is_half_day, HALF_CLOSE_MIN, RTH_CLOSE_MIN)


# --------------------------------------------------------------------------
# Raw readers
# --------------------------------------------------------------------------
def read_raw_minute(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Read every weekly minute file, deduplicate, convert to US/Eastern.

    Weekly files overlap on purpose; when a timestamp appears in more than
    one file the copy from the later file is kept. Timestamps with
    *different* values across files are counted as ``dedup_conflicts``.

    Returns:
        (bars, stats). ``bars`` has a tz-aware ET ``ts`` column (bar start)
        plus open/high/low/close/volume/vwap/bar_count, sorted by ``ts``.
    """
    files = sorted((Path(raw_dir) / "minute").glob("QQQ_1min_TRADES_*.csv"))
    if not files:
        raise FileNotFoundError(f"No minute files under {raw_dir}/minute")
    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df["_file"] = i
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"average": "vwap", "barCount": "bar_count"})
    raw["ts"] = pd.to_datetime(raw["date"], utc=True).dt.tz_convert(ET)
    raw = raw.sort_values(["ts", "_file"], kind="stable")

    n_rows = len(raw)
    distinct = raw.drop_duplicates(["ts"] + PRICE_COLS + ["volume"])
    n_conflicts = int(distinct["ts"].duplicated().sum())
    bars = raw.drop_duplicates("ts", keep="last")
    bars = bars[["ts"] + BAR_COLS].reset_index(drop=True)
    bars["volume"] = bars["volume"].astype("float64")
    bars["bar_count"] = bars["bar_count"].astype("int64")
    stats = {
        "minute_files": len(files),
        "minute_rows_raw": n_rows,
        "minute_rows_dedup": len(bars),
        "dedup_dropped": n_rows - len(bars),
        "dedup_conflicts": n_conflicts,
    }
    return bars, stats


def read_raw_daily(raw_dir: Path = RAW_DIR,
                   cfg: CleanConfig = CleanConfig()) -> pd.DataFrame:
    """Read IBKR daily TRADES + ADJUSTED_LAST and derive dividends.

    IBKR's TRADES daily bars are already split-adjusted; ADJUSTED_LAST is
    additionally dividend-adjusted (backwards, multiplicative). On an
    ex-date t the back-adjustment factor jumps:
    ``factor[t-1] / factor[t] = 1 - D / close_raw[t-1]``, which recovers
    the cash dividend ``D``.

    Returns:
        DataFrame indexed by naive session ``date`` with ``*_raw`` and
        ``*_adj`` OHLC, ``volume``, ``adj_factor``, ``dividend`` (USD, 0 on
        non ex-dates), ``is_ex_div`` and ``ret_cc_adj`` (total return).
    """
    d = Path(raw_dir) / "daily"
    raw = pd.read_csv(d / "QQQ_1day_TRADES.csv", parse_dates=["date"])
    adj = pd.read_csv(d / "QQQ_1day_ADJUSTED_LAST.csv", parse_dates=["date"])
    raw = raw.set_index("date")[PRICE_COLS + ["volume"]]
    adj = adj.set_index("date")[PRICE_COLS]
    df = raw.add_suffix("_raw").join(adj.add_suffix("_adj"), how="inner")
    df = df.rename(columns={"volume_raw": "volume"})
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"

    df["adj_factor"] = df["close_adj"] / df["close_raw"]
    ratio = df["adj_factor"].shift(1) / df["adj_factor"]
    implied = df["close_raw"].shift(1) * (1.0 - ratio)
    df["is_ex_div"] = (implied > cfg.dividend_min_amount).fillna(False)
    df["dividend"] = implied.where(df["is_ex_div"], 0.0).fillna(0.0)
    df["ret_cc_adj"] = df["close_adj"].pct_change()
    df["is_half_day"] = _half_day_mask(df.index)
    return df


# --------------------------------------------------------------------------
# Cleaning steps
# --------------------------------------------------------------------------
def label_sessions(bars: pd.DataFrame) -> pd.DataFrame:
    """Add ``day`` (naive session date), ``session`` and ``is_half_day``.

    ``session`` is ``pre`` (04:00-09:29), ``rth`` (09:30 to 15:59, or 12:59
    on half days) or ``post`` (everything after). Bar times are bar starts.
    """
    out = bars.copy()
    ts = out["ts"]
    out["day"] = ts.dt.tz_localize(None).dt.normalize()
    minute = ts.dt.hour * 60 + ts.dt.minute
    out["is_half_day"] = _half_day_mask(out["day"])
    close = rth_close_minute(out["is_half_day"].to_numpy())
    out["session"] = np.select(
        [minute < RTH_OPEN_MIN, minute < close], ["pre", "rth"], "post")
    return out


def drop_incomplete_last_day(bars: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Drop the final session if RTH had not finished when data was pulled.

    Data downloaded during market hours contains a partial day that would
    otherwise look like a short (and misleading) session.
    """
    if bars.empty:
        return bars, []
    last_day = bars["day"].max()
    sel = bars[(bars["day"] == last_day) & (bars["session"] == "rth")]
    half = bool(_half_day_mask([last_day])[0])
    last_rth_bar = last_day + pd.Timedelta(
        minutes=int(rth_close_minute(half)) - 1)
    if sel.empty or sel["ts"].max().tz_localize(None) < last_rth_bar:
        return bars[bars["day"] != last_day].reset_index(drop=True), \
            [str(last_day.date())]
    return bars, []


def build_rth_grid(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Every expected RTH bar start (tz-aware ET) for the given sessions."""
    days = pd.DatetimeIndex(days).sort_values()
    n = np.where(_half_day_mask(days),
                 HALF_CLOSE_MIN - RTH_OPEN_MIN, RTH_CLOSE_MIN - RTH_OPEN_MIN)
    day_rep = np.repeat(days.to_numpy(), n)
    offsets = np.concatenate([np.arange(k) for k in n]) if len(n) else []
    local = (day_rep + np.timedelta64(RTH_OPEN_MIN, "m")
             + np.asarray(offsets, dtype="timedelta64[m]"))
    return pd.DatetimeIndex(local).tz_localize(ET)


def fill_rth_minutes(bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex RTH bars onto the full minute grid and fill gaps causally.

    * ``fill_source = "ibkr"``      real bar with at least one trade
    * ``fill_source = "ibkr_zero"`` bar IBKR sent with no trades
      (volume 0, price carried forward by IBKR)
    * ``fill_source = "filled"``    bar missing from IBKR; OHLC and vwap set
      to the last close seen strictly before it (any session, possibly the
      previous day), volume 0

    ``is_filled`` is True for the last two. Sessions with no RTH bars at
    all are not created here (they are reported as missing days instead).
    """
    rth = bars[bars["session"] == "rth"]
    days = pd.DatetimeIndex(rth["day"].unique())
    grid = build_rth_grid(days)
    out = rth.set_index("ts")[BAR_COLS].reindex(grid)
    out.index.name = "ts"
    out = out.reset_index()

    missing = out["close"].isna().to_numpy()
    src = np.where(missing, "filled",
                   np.where(out["bar_count"].fillna(0).to_numpy() > 0,
                            "ibkr", "ibkr_zero"))
    if missing.any():
        known = bars[["ts", "close"]].rename(columns={"close": "prev_close"})
        gaps = out.loc[missing, ["ts"]]
        prev = pd.merge_asof(gaps, known, on="ts", direction="backward",
                             allow_exact_matches=False)
        fill_px = prev["prev_close"].to_numpy()
        for c in PRICE_COLS + ["vwap"]:
            out.loc[missing, c] = fill_px
        out.loc[missing, "volume"] = 0.0
        out.loc[missing, "bar_count"] = 0
    out["bar_count"] = out["bar_count"].astype("int64")
    out["fill_source"] = src
    out["is_filled"] = src != "ibkr"
    out["day"] = out["ts"].dt.tz_localize(None).dt.normalize()
    out["is_half_day"] = _half_day_mask(out["day"])
    out["bar_end"] = out["ts"] + pd.Timedelta(minutes=1)
    return out


def flag_anomalies(rth: pd.DataFrame,
                   cfg: CleanConfig = CleanConfig()) -> pd.DataFrame:
    """Flag suspicious RTH minutes. Nothing is removed.

    Checks per bar:

    * ``abs_return``: |log(close / prev close)| > ``anomaly_abs_return``
      (within the same day only; the overnight gap is not a 1-min return)
    * ``sigma_return``: |return| > ``anomaly_sigma`` x rolling std of the
      previous ``anomaly_vol_window`` returns (past-only, no look-ahead)
    * ``bar_range``: (high - low) / close > ``anomaly_bar_range``
    * ``ohlc_invalid``: high < max(open, close), low > min(open, close),
      or any price <= 0

    Returns:
        One row per flagged bar with ``flags`` (";"-joined), ``ret``,
        ``sigma`` and the bar's OHLCV for manual review.
    """
    df = rth[["ts", "day"] + BAR_COLS + ["is_filled"]].copy()
    logc = np.log(df["close"])
    df["ret"] = logc.groupby(df["day"]).diff()
    df["sigma"] = (df["ret"].shift(1)
                   .rolling(cfg.anomaly_vol_window,
                            min_periods=cfg.anomaly_min_periods).std())
    df["prev_close"] = df.groupby("day")["close"].shift(1)
    absr = df["ret"].abs()
    checks = {
        "abs_return": absr > cfg.anomaly_abs_return,
        "sigma_return": absr > cfg.anomaly_sigma * df["sigma"],
        "bar_range": (df["high"] - df["low"]) / df["close"]
        > cfg.anomaly_bar_range,
        "ohlc_invalid": (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df[PRICE_COLS] <= 0).any(axis=1),
    }
    flags = pd.DataFrame({k: v.fillna(False) for k, v in checks.items()})
    any_flag = flags.any(axis=1)
    out = df.loc[any_flag].copy()
    out["flags"] = flags.loc[any_flag].apply(
        lambda r: ";".join(k for k, v in r.items() if v), axis=1)
    cols = ["ts", "day", "flags", "ret", "sigma", "prev_close"] + BAR_COLS \
        + ["is_filled"]
    return out[cols].reset_index(drop=True)


def aggregate_bars(rth: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """Aggregate 1-min RTH bars into N-minute bars (labelled by bar start).

    Bars never span two sessions. ``vwap`` is volume-weighted from the
    1-min vwaps (falls back to close when volume is 0). ``bar_end`` marks
    when the bar becomes known.
    """
    df = rth.copy()
    df["bucket"] = df["ts"].dt.floor(f"{minutes}min")
    df["pv"] = df["vwap"] * df["volume"]
    g = df.groupby("bucket", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "pv": g["pv"].sum(),
        "n_minutes": g["close"].size(),
        "n_filled": g["is_filled"].sum(),
        "day": g["day"].first(),
        "is_half_day": g["is_half_day"].first(),
    })
    out["vwap"] = np.where(out["volume"] > 0,
                           out["pv"] / out["volume"].where(out["volume"] > 0),
                           out["close"])
    out = out.drop(columns="pv")
    out.index.name = "ts"
    out = out.reset_index()
    out["bar_end"] = out["ts"] + pd.Timedelta(minutes=minutes)
    return out


def aggregate_daily(rth: pd.DataFrame, ext: pd.DataFrame,
                    daily_ibkr: pd.DataFrame, anomalies: pd.DataFrame,
                    cfg: CleanConfig = CleanConfig()) -> pd.DataFrame:
    """Build daily RTH bars from minutes and attach quality + dividend info.

    Columns include raw OHLCV/vwap from minutes, ``fill_ratio`` (share of
    RTH minutes not traded, IBKR-zero or missing), ``missing_ratio`` (only
    minutes absent from IBKR), anomaly count, pre/post volume, the matching
    IBKR daily close (raw and adjusted), ``dividend``, ``is_ex_div``, and
    dividend-aware returns:

    * ``ret_intraday``  = close / open - 1 (same session, no adjustment)
    * ``ret_overnight`` = (open + dividend) / prev close - 1
    * ``ret_cc``        = (close + dividend) / prev close - 1

    ``prev close`` is the previous row's close; ``gap_days`` > 1 trading
    days apart is flagged via ``prev_day_missing`` and the returns are set
    to NaN there (never compute a return across a hole in the data).
    """
    g = rth.assign(pv=rth["vwap"] * rth["volume"]).groupby("day")
    d = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "vwap": g["pv"].sum() / g["volume"].sum().replace(0, np.nan),
        "n_minutes": g["close"].size(),
        "fill_ratio": g["is_filled"].mean(),
        "missing_ratio": g["fill_source"].apply(lambda s: (s == "filled")
                                                .mean()),
        "is_half_day": g["is_half_day"].first(),
    })
    d.index.name = "date"
    d["n_anomalies"] = anomalies.groupby("day").size().reindex(d.index) \
        .fillna(0).astype(int)
    ev = ext.groupby(["day", "session"])["volume"].sum().unstack()
    for s in ("pre", "post"):
        d[f"volume_{s}"] = (ev[s] if s in ev else pd.Series(dtype=float)) \
            .reindex(d.index).fillna(0.0)

    di = daily_ibkr[["close_raw", "close_adj", "dividend", "is_ex_div",
                     "volume"]].rename(columns={"volume": "volume_ibkr"})
    d = d.join(di, how="left")
    d["in_ibkr_daily"] = d["close_raw"].notna()
    d["close_vs_ibkr"] = d["close"] / d["close_raw"] - 1
    d["split_suspect"] = d["close_vs_ibkr"].abs() > cfg.split_check_tolerance
    d["dividend"] = d["dividend"].fillna(0.0)
    d["is_ex_div"] = d["is_ex_div"].fillna(False).astype(bool)

    # Previous trading session according to the IBKR daily calendar.
    cal = daily_ibkr.index
    pos = cal.searchsorted(d.index)
    prev_expected = pd.DatetimeIndex(
        [cal[p - 1] if 0 < p <= len(cal) else pd.NaT for p in pos])
    prev_actual = pd.Series(d.index, index=d.index).shift(1)
    d["prev_day_missing"] = (prev_actual.to_numpy() != prev_expected.to_numpy())
    d.iloc[0, d.columns.get_loc("prev_day_missing")] = True

    prev_close = d["close"].shift(1).where(~d["prev_day_missing"])
    d["ret_intraday"] = d["close"] / d["open"] - 1
    d["ret_overnight"] = (d["open"] + d["dividend"]) / prev_close - 1
    d["ret_cc"] = (d["close"] + d["dividend"]) / prev_close - 1
    return d


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
OUTPUTS = {
    "all": "minute_all_sessions.parquet",
    "rth": "minute_rth.parquet",
    "ext": "minute_ext.parquet",
    "bars5": "bars_5min.parquet",
    "daily": "daily.parquet",
    "daily_ibkr": "daily_ibkr.parquet",
    "anomalies": "anomalies.parquet",
}


def _raw_mtime(raw_dir: Path) -> float:
    files = list(Path(raw_dir).rglob("*.csv"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def _is_fresh(out_dir: Path, raw_dir: Path) -> bool:
    paths = [out_dir / f for f in OUTPUTS.values()] + [out_dir /
                                                        "build_meta.json"]
    if not all(p.exists() for p in paths):
        return False
    return min(p.stat().st_mtime for p in paths) >= _raw_mtime(raw_dir)


def build_all(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR,
              cfg: CleanConfig = CleanConfig(), force: bool = False) -> dict:
    """Run the whole cleaning pipeline and write every output to parquet.

    Skips work when all outputs exist and are newer than every raw file,
    unless ``force``. Returns the build metadata dict.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "build_meta.json"
    if not force and _is_fresh(out_dir, raw_dir):
        log.info("processed data is up to date; use force=True to rebuild")
        return json.loads(meta_path.read_text())

    log.info("reading raw minute files ...")
    bars, stats = read_raw_minute(raw_dir)
    bars = label_sessions(bars)
    bars, dropped_days = drop_incomplete_last_day(bars)
    bars.to_parquet(out_dir / OUTPUTS["all"], index=False)

    log.info("reading daily files ...")
    daily_ibkr = read_raw_daily(raw_dir, cfg)
    daily_ibkr.to_parquet(out_dir / OUTPUTS["daily_ibkr"])

    log.info("building RTH grid and filling gaps ...")
    rth = fill_rth_minutes(bars)
    rth.to_parquet(out_dir / OUTPUTS["rth"], index=False)
    ext = bars[bars["session"] != "rth"].reset_index(drop=True)
    ext = ext.assign(bar_end=ext["ts"] + pd.Timedelta(minutes=1))
    ext.to_parquet(out_dir / OUTPUTS["ext"], index=False)

    log.info("flagging anomalies ...")
    anomalies = flag_anomalies(rth, cfg)
    anomalies.to_parquet(out_dir / OUTPUTS["anomalies"], index=False)

    log.info("aggregating 5-min and daily bars ...")
    aggregate_bars(rth, 5).to_parquet(out_dir / OUTPUTS["bars5"], index=False)
    daily = aggregate_daily(rth, ext, daily_ibkr, anomalies, cfg)
    daily.to_parquet(out_dir / OUTPUTS["daily"])

    # Cross-checks between the minute data and the IBKR daily calendar.
    lo, hi = daily.index.min(), daily.index.max()
    ibkr_days = daily_ibkr.loc[lo:hi].index
    missing_days = ibkr_days.difference(daily.index)
    extra_days = daily.index.difference(daily_ibkr.index)
    last_bar = ext.groupby("day")["ts"].max()
    expected_end = pd.Series(
        np.where(_half_day_mask(last_bar.index), HALF_POST_CLOSE_MIN,
                 POST_CLOSE_MIN) - 1, index=last_bar.index)
    last_min = last_bar.dt.hour * 60 + last_bar.dt.minute
    short_ext = last_min[last_min < expected_end]
    hd_in_range = sorted(d for d in nyse_half_days(lo.year, hi.year)
                         if lo.date() <= d <= hi.date())

    stats.update({
        "built_at": pd.Timestamp.now(tz=ET).isoformat(),
        "config": asdict(cfg),
        "minute_range": [str(bars["ts"].min()), str(bars["ts"].max())],
        "dropped_incomplete_days": dropped_days,
        "sessions": int(len(daily)),
        "rth_minutes": int(len(rth)),
        "rth_minutes_filled": int(rth["is_filled"].sum()),
        "rth_minutes_missing": int((rth["fill_source"] == "filled").sum()),
        "rth_minutes_ibkr_zero": int((rth["fill_source"] == "ibkr_zero")
                                     .sum()),
        "ext_minutes": int(len(ext)),
        "anomalies": int(len(anomalies)),
        "half_days": [str(d) for d in hd_in_range],
        "half_days_without_data": [str(d) for d in hd_in_range
                                   if pd.Timestamp(d) not in daily.index],
        "missing_days_vs_ibkr_daily": [str(d.date()) for d in missing_days],
        "extra_days_not_in_ibkr_daily": [str(d.date()) for d in extra_days],
        "short_extended_sessions": {str(k.date()): f"{v // 60:02d}:{v % 60:02d}"
                                    for k, v in short_ext.items()},
        "split_suspect_days": [str(d.date()) for d in
                               daily.index[daily["split_suspect"]]],
        "ex_div_days": int(daily["is_ex_div"].sum()),
    })
    meta_path.write_text(json.dumps(stats, indent=2))
    log.info("done: %s sessions, %s anomalies", stats["sessions"],
             stats["anomalies"])
    return stats


# --------------------------------------------------------------------------
# External daily data (Cboe volatility indices)
# --------------------------------------------------------------------------
CBOE_DIR = PROJECT_ROOT / "data" / "raw" / "cboe"
CBOE_CLOSE_TIME = pd.Timedelta(hours=16, minutes=15)  # index close, ET


def read_cboe_daily(path: Path) -> pd.DataFrame:
    """Read a Cboe index history CSV (e.g. ``VIX_History.csv``).

    Expected layout (Cboe "daily prices" download): a ``DATE`` column in
    ``MM/DD/YYYY`` plus ``OPEN, HIGH, LOW, CLOSE``. Header case and extra
    leading lines are tolerated.

    Returns:
        DataFrame indexed by naive ``date`` with lower-case
        open/high/low/close, sorted, duplicates dropped (last kept).
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    skip = next(i for i, ln in enumerate(lines) if ln.upper().startswith("DATE"))
    df = pd.read_csv(path, skiprows=skip, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    return df[cols].astype(float)


def align_prev_close(series: pd.Series, calendar: pd.DatetimeIndex,
                     max_stale_days: int = 5) -> pd.DataFrame:
    """For each trading day t, take the last value dated strictly before t.

    This is the value known at the previous close, so it is safe to use as
    a feature for day t. Cboe and the exchange calendars differ on a few
    days (e.g. Cboe publishes on some days the stock market is closed), so
    the lookup is by date, not by row position.

    Returns:
        DataFrame indexed by ``calendar`` with ``value``, ``source_date``
        and ``stale_days`` (calendar days between source_date and t). Values
        older than ``max_stale_days`` are set to NaN.
    """
    s = series.dropna().sort_index()
    cal = pd.DatetimeIndex(calendar)
    pos = s.index.searchsorted(cal, side="left") - 1   # strictly before t
    ok = pos >= 0
    src = pd.DatetimeIndex(s.index[np.maximum(pos, 0)]).where(ok)
    val = np.where(ok, s.to_numpy()[np.maximum(pos, 0)], np.nan)
    out = pd.DataFrame({"value": val, "source_date": src}, index=cal)
    out["stale_days"] = (out.index - out["source_date"]).dt.days
    out.loc[out["stale_days"] > max_stale_days, "value"] = np.nan
    return out


def load_cboe(names: tuple[str, ...] = ("VIX", "VXN"),
              cboe_dir: Path = CBOE_DIR) -> pd.DataFrame:
    """Close prices of Cboe indices, one column per name (lower case).

    Reads ``<cboe_dir>/<NAME>_History.csv``. Missing files raise
    ``FileNotFoundError`` so absent inputs never pass silently.
    """
    cols = {}
    for n in names:
        cols[n.lower()] = read_cboe_daily(Path(cboe_dir) / f"{n}_History.csv")[
            "close"]
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------
# Loaders for downstream code
# --------------------------------------------------------------------------
def _load(name: str, processed_dir: Path, exclude_half_days: bool,
          max_fill_ratio: float | None = None) -> pd.DataFrame:
    """Read one processed file, optionally dropping half / low-quality days.

    ``max_fill_ratio`` drops whole sessions whose share of filled RTH
    minutes (see ``daily.fill_ratio``) is above the threshold. The filter is
    decided per day from that day's own data quality, so it does not leak
    future information into earlier days.
    """
    processed_dir = Path(processed_dir)
    path = processed_dir / OUTPUTS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run "
                                "`python -m src.data_loader` first")
    df = pd.read_parquet(path)
    if exclude_half_days:
        df = df[~df["is_half_day"].astype(bool)]
    if max_fill_ratio is not None:
        if name == "daily":
            df = df[df["fill_ratio"] <= max_fill_ratio]
        else:
            fr = pd.read_parquet(processed_dir / OUTPUTS["daily"],
                                 columns=["fill_ratio"])["fill_ratio"]
            bad = fr.index[fr > max_fill_ratio]
            df = df[~df["day"].isin(bad)]
    return df


def load_minute_rth(exclude_half_days: bool = False,
                    max_fill_ratio: float | None = None,
                    processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """1-min RTH bars on a complete grid (see ``fill_rth_minutes``)."""
    return _load("rth", processed_dir, exclude_half_days, max_fill_ratio)


def load_extended(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Pre-market and after-hours 1-min bars (unfilled, ``session`` column)."""
    return _load("ext", processed_dir, False)


def load_5min(exclude_half_days: bool = False,
              max_fill_ratio: float | None = None,
              processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """5-min RTH bars (``ts`` = bar start, ``bar_end`` = when known)."""
    return _load("bars5", processed_dir, exclude_half_days, max_fill_ratio)


def load_daily(exclude_half_days: bool = False,
               max_fill_ratio: float | None = None,
               processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Daily RTH bars from minutes plus quality, IBKR and dividend columns."""
    return _load("daily", processed_dir, exclude_half_days, max_fill_ratio)


def load_anomalies(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Flagged RTH minutes for manual review."""
    return _load("anomalies", processed_dir, False)


def main() -> None:
    p = argparse.ArgumentParser(description="Build data/processed from raw.")
    p.add_argument("--force", action="store_true", help="ignore cache")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    meta = build_all(args.raw_dir, args.out_dir, force=args.force)
    for k in ("sessions", "rth_minutes_filled", "rth_minutes_missing",
              "anomalies", "dropped_incomplete_days",
              "missing_days_vs_ibkr_daily"):
        print(f"{k}: {meta.get(k)}")


if __name__ == "__main__":
    main()
