"""Synthetic IBKR-format raw data with known, planted defects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ET = "America/New_York"

# Planted facts the tests check against.
FIRST_DAY = "2023-10-30"
LAST_FULL_DAY = "2023-11-30"
PARTIAL_DAY = "2023-12-01"          # downloaded mid-session -> must be dropped
HOLIDAY = "2023-11-23"              # Thanksgiving, no session
HALF_DAY = "2023-11-24"             # 13:00 close
EX_DIV_DAY = "2023-11-20"
DIVIDEND = 0.60
MISSING = ["2023-11-15 10:00", "2023-11-15 10:01", "2023-11-15 10:02"]
ZERO_BAR = "2023-11-16 11:00"       # IBKR bar with no trades
SPIKE = "2023-11-17 14:00"          # +3% one-minute spike
SHORT_EXT_DAY = "2023-11-09"        # after-hours stops at 19:29


def _sessions() -> pd.DatetimeIndex:
    days = pd.bdate_range(FIRST_DAY, PARTIAL_DAY)
    return days[days != pd.Timestamp(HOLIDAY)]


def make_minutes(seed: int = 0) -> pd.DataFrame:
    """All-session 1-min bars in ET with the planted defects applied."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 380.0
    for day in _sessions():
        half = day == pd.Timestamp(HALF_DAY)
        end_min = 17 * 60 if half else 20 * 60
        if day == pd.Timestamp(SHORT_EXT_DAY):
            end_min = 19 * 60 + 30
        if day == pd.Timestamp(PARTIAL_DAY):
            end_min = 10 * 60 + 16
        if day == pd.Timestamp(EX_DIV_DAY):
            price -= DIVIDEND
        for m in range(4 * 60, end_min):
            ts = day + pd.Timedelta(minutes=m)
            o = price
            price = price * (1 + rng.normal(0, 0.0004))
            c = price
            rows.append((ts, o, max(o, c) + 0.01, min(o, c) - 0.01, c,
                         float(rng.integers(100, 5000)), (o + c) / 2,
                         int(rng.integers(1, 50))))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "volume", "average", "barCount"])
    df["ts"] = df["ts"].dt.tz_localize(ET)

    loc = df["ts"].dt.tz_localize(None)
    df = df[~loc.isin(pd.to_datetime(MISSING))].reset_index(drop=True)
    loc = df["ts"].dt.tz_localize(None)

    z = loc == pd.Timestamp(ZERO_BAR)
    prev_close = df["close"].shift(1)[z].iloc[0]
    df.loc[z, ["open", "high", "low", "close", "average"]] = prev_close
    df.loc[z, ["volume", "barCount"]] = [0.0, 0]

    s = loc == pd.Timestamp(SPIKE)
    base = df.loc[s, "open"].iloc[0]
    df.loc[s, ["close", "high"]] = [base * 1.03, base * 1.03 + 0.01]
    return df


def write_raw(root: Path, minutes: pd.DataFrame) -> Path:
    """Write minutes as overlapping weekly IBKR files + both daily files."""
    raw = root / "data" / "raw" / "ibkr"
    (raw / "minute").mkdir(parents=True)
    (raw / "daily").mkdir(parents=True)
    out = minutes.copy()
    out["date"] = out["ts"].dt.tz_convert("UTC").map(lambda t: t.isoformat())
    cols = ["date", "open", "high", "low", "close", "volume", "average",
            "barCount"]
    ends = pd.date_range("2023-11-04", "2023-12-02", freq="7D", tz=ET)
    for end in ends:
        sel = out[(out["ts"] >= end - pd.Timedelta(days=8)) & (out["ts"] < end)]
        sel[cols].to_csv(raw / "minute" / f"QQQ_1min_TRADES_{end:%Y-%m-%d}.csv",
                         index=False)

    loc = minutes["ts"].dt.tz_localize(None)
    m = loc.dt.hour * 60 + loc.dt.minute
    day = loc.dt.normalize()
    close = np.where(day == pd.Timestamp(HALF_DAY), 13 * 60, 16 * 60)
    rth = minutes[(m >= 570) & (m < close)].assign(date=day)
    rth = rth[rth["date"] < pd.Timestamp(PARTIAL_DAY)]
    g = rth.groupby("date")
    daily = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                          "low": g["low"].min(), "close": g["close"].last(),
                          "volume": g["volume"].sum(),
                          "average": g["average"].mean(),
                          "barCount": g["barCount"].sum()})
    daily.index = daily.index.strftime("%Y-%m-%d")
    daily.index.name = "date"
    daily.to_csv(raw / "daily" / "QQQ_1day_TRADES.csv")

    adj = daily.copy()
    ex = pd.Timestamp(EX_DIV_DAY).strftime("%Y-%m-%d")
    prev_close = daily["close"].shift(1).loc[ex]
    factor = np.where(daily.index < ex, 1 - DIVIDEND / prev_close, 1.0)
    for c in ["open", "high", "low", "close", "average"]:
        adj[c] = (adj[c] * factor).round(4)
    adj.to_csv(raw / "daily" / "QQQ_1day_ADJUSTED_LAST.csv")
    return raw


@pytest.fixture(scope="session")
def raw_minutes() -> pd.DataFrame:
    return make_minutes()


@pytest.fixture(scope="session")
def built(tmp_path_factory, raw_minutes):
    """Run the full pipeline once on the synthetic raw data."""
    from src.data_loader import build_all

    root = tmp_path_factory.mktemp("proj")
    raw = write_raw(root, raw_minutes)
    out = root / "data" / "processed"
    meta = build_all(raw, out, force=True)
    return {"raw": raw, "out": out, "meta": meta}
