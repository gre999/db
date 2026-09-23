"""Inspect raw data files in data/raw/ without modifying them.

Prints, for each file: size, first/last rows, inferred columns, timestamp
range, bar spacing, and which clock times appear (to spot pre/post-market
bars and the timezone). Read-only: never writes to data/raw/.

Usage:
    python scripts/inspect_raw.py [path]   # default: data/raw
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

TEXT_EXT = {".csv", ".txt", ".tsv"}
PARQUET_EXT = {".parquet", ".pq"}


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def raw_lines(path: Path, n: int = 5) -> tuple[list[str], list[str]]:
    """Return the first and last n raw text lines of a file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = [line.rstrip("\n") for _, line in zip(range(n), f)]
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", errors="replace").splitlines()[-n:]
    return head, tail


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in PARQUET_EXT:
        return pd.read_parquet(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        sep, has_header = ",", True
    return pd.read_csv(path, sep=sep, header=0 if has_header else None)


def find_timestamp(df: pd.DataFrame) -> pd.Series | None:
    """Best-effort: build a timestamp series from date/time-like columns."""
    cols = [str(c) for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for key in ("timestamp", "datetime", "date_time", "time", "date", "t"):
        if key in lower:
            c = lower[key]
            if key == "date" and "time" in lower:
                s = df[c].astype(str) + " " + df[lower["time"]].astype(str)
            else:
                s = df[c]
            break
    else:
        s = df.iloc[:, 0]
        if pd.api.types.is_integer(df.columns[0]) and df.shape[1] > 1:
            second = df.iloc[:, 1].astype(str)
            if second.str.match(r"^\d{1,2}:\d{2}").all():
                s = s.astype(str) + " " + second
    try:
        if pd.api.types.is_numeric_dtype(s):
            unit = "ms" if s.iloc[0] > 1e11 else "s"
            if s.iloc[0] > 1e17:
                unit = "ns"
            return pd.to_datetime(s, unit=unit, utc=True)
        return pd.to_datetime(s, format="mixed")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not parse timestamps: {exc}")
        return None


def inspect(path: Path) -> None:
    print("=" * 78)
    print(f"FILE: {path}")
    print(f"SIZE: {human_size(path.stat().st_size)}")
    if path.suffix.lower() in TEXT_EXT:
        head, tail = raw_lines(path)
        print("RAW HEAD:")
        for line in head:
            print(f"  {line}")
        print("RAW TAIL:")
        for line in tail:
            print(f"  {line}")

    try:
        df = read_table(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not read as table: {exc}")
        return

    print(f"ROWS: {len(df):,}   COLUMNS: {list(df.columns)}")
    print("DTYPES:")
    print("  " + df.dtypes.to_string().replace("\n", "\n  "))
    print("NULLS PER COLUMN:")
    print("  " + df.isna().sum().to_string().replace("\n", "\n  "))

    ts = find_timestamp(df)
    if ts is None:
        return
    ts = pd.Series(ts).reset_index(drop=True)
    print(f"TIMESTAMP DTYPE: {ts.dtype}   (tz-aware: {ts.dt.tz is not None})")
    print(f"RANGE: {ts.min()}  ->  {ts.max()}")
    print(f"MONOTONIC INCREASING: {ts.is_monotonic_increasing}   "
          f"DUPLICATES: {ts.duplicated().sum():,}")

    diffs = ts.diff().dropna()
    if len(diffs):
        print("MOST COMMON BAR SPACING:")
        print("  " + diffs.value_counts().head(5).to_string().replace("\n", "\n  "))

    if diffs.median() < pd.Timedelta(days=1):
        days = ts.dt.normalize()
        print(f"DISTINCT DAYS: {days.nunique():,}")
        per_day = ts.groupby(days)
        first = per_day.min().dt.strftime("%H:%M")
        last = per_day.max().dt.strftime("%H:%M")
        print("DAILY FIRST-BAR TIME (top 5):")
        print("  " + first.value_counts().head(5).to_string().replace("\n", "\n  "))
        print("DAILY LAST-BAR TIME (top 5):")
        print("  " + last.value_counts().head(5).to_string().replace("\n", "\n  "))
        counts = per_day.size()
        print("BARS PER DAY: " + counts.describe().round(1).to_string()
              .replace("\n", ", "))
        hours = ts.dt.hour.value_counts().sort_index()
        print("BARS BY CLOCK HOUR (as stored):")
        print("  " + hours.to_string().replace("\n", "\n  "))
        if ts.dt.tz is not None:
            et = ts.dt.tz_convert("America/New_York")
            et_days = et.dt.normalize()
            print("DAILY FIRST/LAST BAR IN US/EASTERN (top 3):")
            print("  first: " + et.groupby(et_days).min().dt.strftime("%H:%M")
                  .value_counts().head(3).to_dict().__repr__())
            print("  last:  " + et.groupby(et_days).max().dt.strftime("%H:%M")
                  .value_counts().head(3).to_dict().__repr__())
        # Sample one known half-day (13:00 ET close) to see how it looks.
        loc = ts.dt.tz_convert("America/New_York") if ts.dt.tz is not None else ts
        loc_days = loc.dt.normalize()
        for d in ("2023-11-24", "2022-11-25", "2021-11-26", "2019-11-29"):
            sel = loc[loc_days == pd.Timestamp(d, tz=loc.dt.tz)]
            if len(sel):
                print(f"HALF-DAY CHECK {d}: {len(sel)} bars, "
                      f"{sel.min():%H:%M} -> {sel.max():%H:%M}")
                break


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    if not root.exists():
        sys.exit(f"Path not found: {root.resolve()}")
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and not p.name.startswith("."))
    print(f"Found {len(files)} file(s) under {root.resolve()}")
    for p in files:
        print(f"  {human_size(p.stat().st_size):>10}  {p.relative_to(root)}")
    for p in files:
        if p.suffix.lower() in TEXT_EXT | PARQUET_EXT:
            inspect(p)
        else:
            print("=" * 78)
            print(f"SKIPPED (unknown format): {p}")


if __name__ == "__main__":
    main()
