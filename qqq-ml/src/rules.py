"""Week 6: rule-based intraday strategies (ORB, VWAP trend).

Both are pure execution rules from Zarattini & Aziz - ORB is their "5-min
opening-range **momentum**" strategy (direction set by the first 5-min
bar's own candle, *not* the traditional breakout-of-range-high/low ORB),
VWAP trend is the QuantConnect reproduction of their always-in-the-market
VWAP-cross strategy. Each produces one row per session: whether a trade
happened, direction, entry/exit timestamps and prices, and (ORB only) an
R multiple. No state/regime classification happens here (that is
``src/models/regimes.py``) and no cross-state evaluation (next week).

Look-ahead discipline:

* ORB's entry is the **second** 5-min bar's open (09:35) - the first bar
  (09:30-09:35) only supplies direction and stop, it is never traded on
  itself. Stop/target are checked 1-min bar by 1-min bar, walking forward
  in time; a bar's OHLC is only used once the bar has fully formed
  (matches ``src/data_loader.py``'s ``bar_end`` rule elsewhere).
* VWAP's signal is read off a bar's **close** and only acted on at the
  **next** bar's open - a reversal, or the initial entry, is never
  executed at the same bar whose close triggered it.

Two cost/sizing modes:

* Main (default): $0.0035/share commission + $0.001/share slippage, one
  unit (1 share) of exposure per day - the output used for every day's
  cross-strategy comparison in phase 3.
* Replication: $0.0005/share commission, no slippage (matches the papers'
  assumptions), 1%-of-equity risk per trade with a 4x leverage cap - run
  once over each paper's own sample window to sanity-check against their
  reported Sharpe/drawdown (see :func:`replicate`).

Usage::

    python -m src.rules
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loader as dl

log = logging.getLogger(__name__)

MAIN_COMMISSION = 0.0035
MAIN_SLIPPAGE = 0.001
REPL_COMMISSION = 0.0005
REPL_SLIPPAGE = 0.0

ORB_PAPER_WINDOW = ("2016-01-01", "2023-02-28")
ORB_PAPER_SHARPE, ORB_PAPER_MAX_DD = 1.13, 0.22
VWAP_PAPER_WINDOW = ("2018-01-01", "2023-09-30")
VWAP_PAPER_SHARPE, VWAP_PAPER_MAX_DD = 2.1, 0.094


@dataclass(frozen=True)
class ORBConfig:
    range_minutes: int = 5
    doji_threshold: float = 0.0     # |close/open - 1| <= this -> no trade
    target_r: float = 10.0
    cost_per_share: float = MAIN_COMMISSION + MAIN_SLIPPAGE


@dataclass(frozen=True)
class VWAPConfig:
    cost_per_share: float = MAIN_COMMISSION + MAIN_SLIPPAGE


# --------------------------------------------------------------------------
# ORB: 5-minute opening-range momentum
# --------------------------------------------------------------------------
def _orb_no_trade(day, reason: str, **extra) -> dict:
    base = {"day": day, "traded": False, "reason": reason, "direction": 0,
           "range_open": None, "range_close": None, "range_high": None,
           "range_low": None, "entry_time": pd.NaT, "entry_price": np.nan,
           "stop_price": np.nan, "target_price": np.nan, "exit_time": pd.NaT,
           "exit_price": np.nan, "exit_reason": None, "r_gross": np.nan,
           "r_net": np.nan, "bps_return": 0.0}
    base.update(extra)
    return base


def orb_day(bars: pd.DataFrame, cfg: ORBConfig = ORBConfig()) -> dict:
    """One session's 1-min RTH bars (sorted by ``ts``) -> an ORB trade record.

    ``bars`` must cover the full session, columns ``ts, bar_end, open,
    high, low, close``.
    """
    n = cfg.range_minutes
    day = bars["day"].iloc[0] if len(bars) else None
    if len(bars) <= n:
        return _orb_no_trade(day, "too_few_bars")

    rng = bars.iloc[:n]
    o, c = float(rng["open"].iloc[0]), float(rng["close"].iloc[-1])
    hi, lo = float(rng["high"].max()), float(rng["low"].min())
    if o == 0 or abs(c / o - 1) <= cfg.doji_threshold:
        return _orb_no_trade(day, "doji", range_open=o, range_close=c,
                             range_high=hi, range_low=lo)

    direction = 1 if c > o else -1
    entry_bar = bars.iloc[n]
    entry_price = float(entry_bar["open"])
    entry_time = entry_bar["ts"]
    stop_price = lo if direction == 1 else hi
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return _orb_no_trade(day, "zero_risk", range_open=o, range_close=c,
                             range_high=hi, range_low=lo)
    target_price = entry_price + direction * cfg.target_r * risk

    exit_price = exit_time = exit_reason = None
    for _, b in bars.iloc[n:].iterrows():
        if direction == 1:
            hit_stop = b["low"] <= stop_price
            hit_target = b["high"] >= target_price
        else:
            hit_stop = b["high"] >= stop_price
            hit_target = b["low"] <= target_price
        if hit_stop:            # same-bar both -> stop, per spec
            exit_price, exit_time, exit_reason = stop_price, b["ts"], "stop"
            break
        if hit_target:
            exit_price, exit_time, exit_reason = target_price, b["ts"], "target"
            break
    if exit_price is None:
        last = bars.iloc[-1]
        exit_price = float(last["close"])
        exit_time = last["bar_end"]
        exit_reason = "eod"

    r_gross = direction * (exit_price - entry_price) / risk
    cost_r = (2 * cfg.cost_per_share) / risk
    r_net = r_gross - cost_r
    bps_return = (direction * (exit_price - entry_price)
                 - 2 * cfg.cost_per_share) / entry_price * 1e4
    return {"day": day, "traded": True, "reason": None, "direction": direction,
           "range_open": o, "range_close": c, "range_high": hi, "range_low": lo,
           "entry_time": entry_time, "entry_price": entry_price,
           "stop_price": stop_price, "target_price": target_price,
           "exit_time": exit_time, "exit_price": float(exit_price),
           "exit_reason": exit_reason, "r_gross": float(r_gross),
           "r_net": float(r_net), "bps_return": float(bps_return)}


# --------------------------------------------------------------------------
# VWAP trend: always-in-the-market VWAP cross
# --------------------------------------------------------------------------
def _vwap_no_trade(day, reason: str) -> dict:
    return {"day": day, "traded": False, "reason": reason, "n_segments": 0,
           "first_entry_time": pd.NaT, "first_entry_price": np.nan,
           "final_exit_time": pd.NaT, "final_exit_price": np.nan,
           "gross_pnl": np.nan, "cost": 0.0, "bps_return": 0.0}


def vwap_day(bars: pd.DataFrame, cfg: VWAPConfig = VWAPConfig()) -> dict:
    """One session's 1-min RTH bars -> a VWAP-trend day record.

    Signal at bar i's close is acted on at bar i+1's open (initial entry
    or a reversal); the position is flattened at the session's own final
    close. ``bps_return`` aggregates every segment's P&L that day at unit
    (1-share) exposure.
    """
    day = bars["day"].iloc[0] if len(bars) else None
    if len(bars) < 2:
        return _vwap_no_trade(day, "too_few_bars")

    tp = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    cum_pv = (tp * bars["volume"]).cumsum()
    cum_v = bars["volume"].cumsum()
    vwap = (cum_pv / cum_v.replace(0, np.nan)).to_numpy()
    close = bars["close"].to_numpy()
    ts = bars["ts"].to_numpy()
    opens = bars["open"].to_numpy()

    sign = np.sign(close - vwap)
    sign = pd.Series(sign).replace(0, np.nan).ffill().fillna(0).to_numpy()

    segments = []     # (entry_time, entry_price, exit_time, exit_price, dir)
    pos_dir = 0
    entry_time = entry_price = None
    n = len(bars)
    for i in range(n - 1):        # signal at close[i] -> act at open[i+1]
        sig = sign[i]
        if sig == 0:
            continue
        if pos_dir == 0:
            pos_dir = sig
            entry_time, entry_price = ts[i + 1], opens[i + 1]
        elif sig != pos_dir:
            xt, xp = ts[i + 1], opens[i + 1]
            segments.append((entry_time, entry_price, xt, xp, pos_dir))
            pos_dir, entry_time, entry_price = sig, xt, xp
    if pos_dir != 0:
        last = bars.iloc[-1]
        segments.append((entry_time, entry_price, last["bar_end"],
                         float(last["close"]), pos_dir))

    if not segments:
        return _vwap_no_trade(day, "no_signal")

    gross_pnl = sum(d * (xp - ep) for (_, ep, _, xp, d) in segments)
    cost = cfg.cost_per_share * 2 * len(segments)
    first_entry_price = segments[0][1]
    bps_return = (gross_pnl - cost) / first_entry_price * 1e4
    return {"day": day, "traded": True, "reason": None,
           "n_segments": len(segments),
           "first_entry_time": segments[0][0],
           "first_entry_price": float(first_entry_price),
           "final_exit_time": segments[-1][2],
           "final_exit_price": float(segments[-1][3]),
           "gross_pnl": float(gross_pnl), "cost": float(cost),
           "bps_return": float(bps_return)}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _valid_days(processed_dir: Path) -> pd.Series:
    t = pd.read_parquet(Path(processed_dir) / "targets_daily.parquet",
                        columns=["valid"])
    return t["valid"].astype(bool)


def run_orb(minute: pd.DataFrame | None = None, cfg: ORBConfig = ORBConfig(),
           processed_dir: Path = dl.PROCESSED_DIR,
           mask_invalid_sessions: bool = True) -> pd.DataFrame:
    minute = dl.load_minute_rth(processed_dir=processed_dir) if minute is None else minute
    rows = [orb_day(g.sort_values("ts").reset_index(drop=True), cfg)
           for _, g in minute.groupby("day", sort=True)]
    out = pd.DataFrame(rows).set_index("day").sort_index()
    if mask_invalid_sessions:
        out = _mask_invalid(out, processed_dir)
    return out


def run_vwap(minute: pd.DataFrame | None = None, cfg: VWAPConfig = VWAPConfig(),
            processed_dir: Path = dl.PROCESSED_DIR,
            mask_invalid_sessions: bool = True) -> pd.DataFrame:
    minute = dl.load_minute_rth(processed_dir=processed_dir) if minute is None else minute
    rows = [vwap_day(g.sort_values("ts").reset_index(drop=True), cfg)
           for _, g in minute.groupby("day", sort=True)]
    out = pd.DataFrame(rows).set_index("day").sort_index()
    if mask_invalid_sessions:
        out = _mask_invalid(out, processed_dir)
    return out


def apply_validity_mask(out: pd.DataFrame, valid: pd.Series) -> pd.DataFrame:
    """Blank out rows on sessions ``valid`` marks False (incomplete/low-quality).

    Pure function of ``out`` and ``valid`` (day-indexed) - no file I/O - so
    it can't leak one day's validity flag into another day's row; see
    ``tests/test_rules.py::test_apply_validity_mask_no_cross_day_leakage``.
    """
    out = out.copy()
    bad = ~valid.reindex(out.index).fillna(False)
    if bad.any():
        out.loc[bad, "traded"] = False
        out.loc[bad, "reason"] = "invalid_session"
        out.loc[bad, "bps_return"] = 0.0
        num_cols = [c for c in out.columns if out[c].dtype.kind in "fc"
                   and c != "bps_return"]
        out.loc[bad, num_cols] = np.nan
    return out


def _mask_invalid(out: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    return apply_validity_mask(out, _valid_days(processed_dir))


# --------------------------------------------------------------------------
# Replication mode: risk-based sizing, paper cost assumptions, paper window
# --------------------------------------------------------------------------
def _evaluate_equity(equity: pd.Series, start_equity: float) -> dict:
    prev = equity.shift(1).fillna(start_equity)
    ret = equity / prev - 1.0
    ann_return = float((equity.iloc[-1] / start_equity) ** (252 / len(equity)) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    return {"n": len(equity), "ann_return": ann_return, "ann_vol": ann_vol,
           "sharpe": sharpe, "max_drawdown": float(dd.min())}


def replicate_orb(orb_daily: pd.DataFrame, start_equity: float = 100_000.0,
                  risk_frac: float = 0.01, leverage_cap: float = 4.0,
                  cost_per_share: float = REPL_COMMISSION + REPL_SLIPPAGE,
                  window: tuple[str, str] = ORB_PAPER_WINDOW) -> pd.DataFrame:
    """Risk-1%-of-equity, 4x-leverage-cap replay of already-computed ORB
    trades (reuses entry/stop/exit prices - direction and price levels are
    cost/sizing-independent) over the paper's own sample window."""
    df = orb_daily.loc[window[0]:window[1]]
    equity, rows = start_equity, []
    for day, r in df.iterrows():
        if not r["traded"] or pd.isna(r["entry_price"]):
            rows.append({"day": day, "equity": equity, "pnl": 0.0, "shares": 0})
            continue
        risk_per_share = abs(r["entry_price"] - r["stop_price"])
        shares = np.floor(equity * risk_frac / risk_per_share) if risk_per_share > 0 else 0.0
        shares = min(shares, np.floor(equity * leverage_cap / r["entry_price"]))
        gross = shares * r["direction"] * (r["exit_price"] - r["entry_price"])
        cost = shares * 2 * cost_per_share
        equity += gross - cost
        rows.append({"day": day, "equity": equity, "pnl": gross - cost,
                    "shares": shares})
    return pd.DataFrame(rows).set_index("day")


def replicate_vwap(vwap_daily: pd.DataFrame, start_equity: float = 100_000.0,
                   cost_per_share: float = REPL_COMMISSION + REPL_SLIPPAGE,
                   window: tuple[str, str] = VWAP_PAPER_WINDOW) -> pd.DataFrame:
    """100%-equity, 1x (no leverage) replay of already-computed VWAP-trend
    trades over the paper's own sample window. Share count is set once at
    the day's open from that day's *starting* equity and held fixed for
    every reversal that day - never recomputed intraday off a moving
    equity figure (that compounding is what caused the first replication
    attempt's equity to blow up to ~$19M average - see
    reports/week06_regimes.md)."""
    df = vwap_daily.loc[window[0]:window[1]]
    equity, rows = start_equity, []
    for day, r in df.iterrows():
        if not r["traded"] or pd.isna(r["first_entry_price"]):
            rows.append({"day": day, "equity": equity, "pnl": 0.0, "shares": 0})
            continue
        shares = np.floor(equity / r["first_entry_price"])
        gross = shares * r["gross_pnl"]
        cost = shares * 2 * r["n_segments"] * cost_per_share
        equity += gross - cost
        rows.append({"day": day, "equity": equity, "pnl": gross - cost,
                    "shares": shares})
    return pd.DataFrame(rows).set_index("day")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path,
                   default=dl.PROCESSED_DIR / "strategies")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    minute = dl.load_minute_rth()
    orb = run_orb(minute)
    vwap = run_vwap(minute)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orb.reset_index().to_parquet(args.out_dir / "orb_daily.parquet", index=False)
    vwap.reset_index().to_parquet(args.out_dir / "vwap_daily.parquet", index=False)

    traded = orb[orb["traded"]]
    win_rate = (traded["r_net"] > 0).mean()
    log.info("ORB: %d/%d days traded, win rate %.1f%%, mean r_net %.3f",
            len(traded), len(orb), 100 * win_rate, traded["r_net"].mean())
    log.info("VWAP: %d/%d days traded, mean bps_return %.2f",
            int(vwap["traded"].sum()), len(vwap), vwap["bps_return"].mean())


if __name__ == "__main__":
    main()
