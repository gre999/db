"""Multi-year synthetic 5-min bars + Cboe closes for feature/validation tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_loader as dl
from src.features import MarketData

START, END = "2014-12-22", "2020-12-31"
HOLE_DAY = pd.Timestamp("2016-03-15")       # in calendar, no bars at all
BAD_DAY = pd.Timestamp("2017-06-20")        # 40% filled bars -> invalid
CBOE_ONLY = pd.Timestamp("2016-03-25")      # Good Friday: Cboe row, no session
VIX_GAP = pd.date_range("2018-02-05", "2018-02-16", freq="B")  # no VIX rows


def calendar() -> pd.DatetimeIndex:
    days = pd.bdate_range(START, END)
    holidays = pd.to_datetime([
        "2014-12-25", "2015-01-01", "2015-12-25", "2016-01-01", "2016-03-25",
        "2016-12-26", "2017-01-02", "2017-12-25", "2018-01-01", "2018-12-25",
        "2019-01-01", "2019-12-25", "2020-01-01", "2020-12-25"])
    return days.difference(holidays)


def make_bars5(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = calendar()
    half = dl._half_day_mask(cal)
    n = np.where(half, 42, 78)
    # Persistent daily volatility (log AR(1)) so HAR-type features vary.
    lv = np.zeros(len(cal))
    for i in range(1, len(cal)):
        lv[i] = 0.97 * lv[i - 1] + 0.25 * rng.normal()
    sig = 0.0012 * np.exp(lv)
    day = np.repeat(cal.to_numpy(), n)
    k = np.concatenate([np.arange(m) for m in n])
    r = rng.standard_t(5, size=len(day)) * np.repeat(sig, n) / np.sqrt(5 / 3)
    gaps = rng.normal(0, 0.004, len(cal))
    first = k == 0
    r_all = r + np.where(first, np.repeat(gaps, n), 0.0)
    logp = np.log(100) + np.cumsum(r_all)
    close = np.exp(logp)
    open_ = np.exp(logp - r)            # first bar opens after the gap
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 3e-4, len(day))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 3e-4, len(day))))
    ts = pd.DatetimeIndex(day + (570 + 5 * k).astype("timedelta64[m]")) \
        .tz_localize(dl.ET)
    b = pd.DataFrame({"ts": ts, "open": open_, "high": hi, "low": lo,
                      "close": close,
                      "volume": rng.integers(2e5, 6e5, len(day)).astype(float),
                      "n_minutes": 5, "n_filled": 0, "day": day,
                      "is_half_day": np.repeat(half, n)})
    b["vwap"] = (b["high"] + b["low"]) / 2
    b["bar_end"] = b["ts"] + pd.Timedelta(minutes=5)
    b = b[b["day"] != HOLE_DAY]
    b.loc[b["day"] == BAD_DAY, "n_filled"] = 2          # 40% of minutes
    return b.reset_index(drop=True)


def make_dividends(bars5: pd.DataFrame) -> pd.Series:
    days = pd.DatetimeIndex(bars5["day"].unique())
    third_fri = pd.date_range(START, END, freq="WOM-3FRI")
    ex = [days[days.searchsorted(d)] for d in third_fri[2::3]
          if days.searchsorted(d) < len(days)]
    return pd.Series(0.3, index=pd.DatetimeIndex(ex), name="dividend")


def make_vol_index(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = calendar().union([CBOE_ONLY]).difference(VIX_GAP)
    vix = 15 + np.cumsum(rng.normal(0, 0.5, len(dates))).clip(-5, 40)
    vxn = vix * 1.2 * np.exp(rng.normal(0, 0.05, len(dates)))
    return pd.DataFrame({"vix": vix, "vxn": vxn}, index=dates)


def make_market_data() -> MarketData:
    b = make_bars5()
    return MarketData(b, calendar(), make_dividends(b), make_vol_index())


def write_cboe_csv(path, df: pd.DataFrame, col: str) -> None:
    """Write in Cboe's download layout: DATE (MM/DD/YYYY),OPEN,HIGH,LOW,CLOSE."""
    out = pd.DataFrame({"DATE": df.index.strftime("%m/%d/%Y"),
                        "OPEN": df[col], "HIGH": df[col] + 1,
                        "LOW": df[col] - 1, "CLOSE": df[col]})
    out.to_csv(path, index=False)
