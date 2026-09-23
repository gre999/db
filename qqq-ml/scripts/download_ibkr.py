"""Download QQQ historical bars from Interactive Brokers into data/raw/.

Requires TWS or IB Gateway running and logged in, with the API enabled
(Configure -> API -> Settings -> "Enable ActiveX and Socket Clients").

What gets downloaded (all raw, never modified afterwards):

* ``data/raw/ibkr/minute/QQQ_1min_TRADES_<week_end>.csv``
  1-minute TRADES bars, *including* pre/post-market (useRTH=False), one file
  per calendar week. Each request asks for "8 D" ending Saturday 00:00 ET, so
  consecutive files overlap by a day or so on purpose (no gaps); the loader
  drops duplicate timestamps. Prices are NOT adjusted for dividends.
* ``data/raw/ibkr/daily/QQQ_1day_TRADES.csv``        unadjusted daily bars
* ``data/raw/ibkr/daily/QQQ_1day_ADJUSTED_LAST.csv`` split+dividend adjusted
* ``data/raw/ibkr/download_meta.json``               request parameters

Timestamps: minute bars are stored as UTC ISO strings and mark the bar
*start* (IBKR convention: the 09:30 ET bar covers 09:30:00-09:30:59).

The minute download is resumable: finished weeks are skipped on re-run, so
it is safe to stop with Ctrl+C and start again.

Usage:
    python scripts/download_ibkr.py --port 7496 --start 2015-01-01
    python scripts/download_ibkr.py --port 7496 --daily-only
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
SYMBOL = "QQQ"
OUT_DIR = Path("data/raw/ibkr")


def week_ends(start: date, today: date) -> list[datetime]:
    """Return weekly request end points (Saturday 00:00 ET), newest first.

    Requests use a duration slightly longer than a week, so consecutive
    windows overlap instead of leaving gaps. The last window ends at the
    Saturday after ``today`` (the current, possibly incomplete, week).
    """
    first_sat = start + timedelta(days=(5 - start.weekday()) % 7 or 7)
    ends = []
    d = first_sat
    last_sat = today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    while d <= last_sat:
        ends.append(datetime(d.year, d.month, d.day, tzinfo=ET))
        d += timedelta(days=7)
    return ends[::-1]


def bars_to_df(bars):
    from ib_async import util

    df = util.df(bars)
    if df is None or df.empty:
        return None
    return df


def download_daily(ib, contract) -> None:
    out = OUT_DIR / "daily"
    out.mkdir(parents=True, exist_ok=True)
    for what in ("TRADES", "ADJUSTED_LAST"):
        print(f"[daily] requesting {what} ...")
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="30 Y",
            barSizeSetting="1 day", whatToShow=what, useRTH=True,
            formatDate=1, timeout=180,
        )
        df = bars_to_df(bars)
        if df is None:
            print(f"[daily] {what}: no data returned")
            continue
        path = out / f"{SYMBOL}_1day_{what}.csv"
        df.to_csv(path, index=False)
        print(f"[daily] {what}: {len(df):,} rows "
              f"{df['date'].iloc[0]} -> {df['date'].iloc[-1]}  -> {path}")


class LinkMonitor:
    """Track whether TWS is connected to IBKR's servers.

    TWS reports error 1100 when it loses its link to IBKR, and 1101/1102
    when the link is restored. While the link is down every historical
    request just times out, so we wait instead of burning requests.
    """

    def __init__(self, ib):
        self.ib = ib
        self.link_up = True
        ib.errorEvent += self._on_error

    def _on_error(self, req_id, code, msg, *args):
        if code == 1100:
            self.link_up = False
        elif code in (1101, 1102):
            self.link_up = True

    def wait(self, pause: float) -> None:
        """Block until the socket to TWS and TWS's link to IBKR are up."""
        announced = False
        while not self.ib.isConnected() or not self.link_up:
            if not self.ib.isConnected():
                raise SystemExit("Lost connection to TWS itself. Restart TWS "
                                 "and re-run; finished weeks are kept.")
            if not announced:
                print("  ... TWS lost its link to IBKR, waiting to reconnect")
                announced = True
            self.ib.sleep(pause)
        if announced:
            print("  ... link restored")


def download_minute(ib, contract, start: date, pause: float,
                    head: datetime | None = None) -> None:
    """Download 1-min bars week by week, newest first; resumable.

    An empty reply is ambiguous: it can mean "no data that week" or a
    timeout (e.g. TWS lost its connection to IBKR, error 1100). Empty weeks
    are therefore retried with back-off and never written to disk, so a
    later re-run picks them up again. Weeks that end before IBKR's earliest
    available bar (``head``) are skipped instead of requested.
    """
    out = OUT_DIR / "minute"
    out.mkdir(parents=True, exist_ok=True)
    today = datetime.now(ET).date()
    if head is not None:
        start = max(start, head.astimezone(ET).date())
    ends = week_ends(start, today)
    print(f"[minute] {len(ends)} weekly windows from {start} to {today}")
    failed = []
    link = LinkMonitor(ib)
    for i, end in enumerate(ends, 1):
        path = out / f"{SYMBOL}_1min_TRADES_{end:%Y-%m-%d}.csv"
        is_current_week = end.date() > today
        if path.exists() and not is_current_week:
            continue
        df = None
        for attempt in range(4):
            link.wait(pause)
            try:
                bars = ib.reqHistoricalData(
                    contract, endDateTime=end, durationStr="8 D",
                    barSizeSetting="1 min", whatToShow="TRADES",
                    useRTH=False, formatDate=2, timeout=120,
                )
                df = bars_to_df(bars)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {end:%Y-%m-%d} attempt {attempt + 1}: {exc}")
            if df is not None:
                break
            backoff = pause * 10 * (attempt + 1)
            print(f"  ! week ending {end:%Y-%m-%d} empty/timeout "
                  f"(attempt {attempt + 1}/4), retrying in {backoff:.0f}s")
            ib.sleep(backoff)

        if df is None:
            failed.append(end.date())
            print(f"[minute] {i}/{len(ends)} week ending {end:%Y-%m-%d}: "
                  f"FAILED, not saved (re-run later)")
        else:
            # formatDate=2 -> tz-aware UTC datetimes; store as ISO UTC.
            df["date"] = df["date"].map(lambda t: t.astimezone(timezone.utc)
                                        .isoformat())
            df.to_csv(path, index=False)
            print(f"[minute] {i}/{len(ends)} week ending {end:%Y-%m-%d}: "
                  f"{len(df):,} bars")
        ib.sleep(pause)

    if failed:
        print(f"[minute] {len(failed)} week(s) not downloaded: "
              f"{', '.join(map(str, failed))}")
        print("[minute] Run the same command again to retry them.")
    else:
        print("[minute] all weeks downloaded.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7496,
                   help="TWS live 7496, TWS paper 7497, "
                        "Gateway live 4001, Gateway paper 4002")
    p.add_argument("--client-id", type=int, default=17)
    p.add_argument("--start", default="2015-01-01",
                   help="earliest date for minute bars (YYYY-MM-DD)")
    p.add_argument("--pause", type=float, default=3.0,
                   help="seconds between minute requests (IBKR pacing)")
    p.add_argument("--daily-only", action="store_true")
    p.add_argument("--minute-only", action="store_true")
    args = p.parse_args()

    from ib_async import IB, Stock

    ib = IB()
    print(f"Connecting to {args.host}:{args.port} ...")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    try:
        contract = Stock(SYMBOL, "SMART", "USD", primaryExchange="NASDAQ")
        ib.qualifyContracts(contract)
        print(f"Contract: {contract}")
        head = ib.reqHeadTimeStamp(contract, whatToShow="TRADES",
                                   useRTH=False, formatDate=2)
        print(f"Earliest TRADES data IBKR reports: {head}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "symbol": SYMBOL,
            "contract": str(contract),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "head_timestamp": str(head),
            "minute": {"barSize": "1 min", "whatToShow": "TRADES",
                       "useRTH": False, "timestamp": "UTC ISO, bar start",
                       "adjusted": "no (raw traded prices)"},
            "daily": {"barSize": "1 day", "whatToShow": ["TRADES",
                                                          "ADJUSTED_LAST"],
                      "useRTH": True,
                      "timestamp": "session date (ET)"},
        }
        (OUT_DIR / "download_meta.json").write_text(json.dumps(meta, indent=2))

        if not args.minute_only:
            download_daily(ib, contract)
        if not args.daily_only:
            download_minute(ib, contract, date.fromisoformat(args.start),
                            args.pause,
                            head if isinstance(head, datetime) else None)
    finally:
        ib.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
