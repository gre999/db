"""Week 11 (phase 5, week 1): minute-sequence tensors for the CNN and its
flattened-sequence baselines.

One tensor per session, shape ``(window_len, 4)`` - the first
``window_len`` 1-min RTH bars from the open, 4 channels
(``CHANNEL_NAMES``), every channel computed causally (never reads a bar at
or after the window's own last bar, and the volume channel only ever
reads PRIOR sessions, never anything from the current session beyond the
window). Two tasks (config/week11_dl.toml):

* Task A: ``window_len=5`` (ORB's own opening range), label = ORB net R >
  0 (:func:`src.models.filter.build_labels`, reused unchanged).
* Task B: ``window_len=N`` (15 or 30), label = whether entering at bar N's
  open in the direction of the eventual close gives a net-of-cost profit
  (:func:`build_task_b_labels`) - sessions where even the best-case
  direction doesn't clear the round-trip cost are dropped from the
  modeling set entirely, the same "nothing to decide" treatment
  config/week08_filter.toml gives ORB's own doji days.

Standardization is always fit on a caller-supplied set of days (a fold's
model-fit window) and applied to every day - never fit on the whole
sample, matching every prior week's rule.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loader as dl
from src import rules as R

log = logging.getLogger(__name__)
SEQ_DIR = dl.PROCESSED_DIR / "sequences"

CHANNEL_NAMES = ("log_ret_from_open", "rel_volume_same_minute",
                 "cum_range_position", "cum_vwap_distance")
N_CHANNELS = len(CHANNEL_NAMES)


def same_minute_volume_history(minute: pd.DataFrame, window_len: int,
                               trailing_days: int = 20) -> pd.DataFrame:
    """Trailing ``trailing_days``-session mean volume at each minute-of-day
    index 0..window_len-1, using only sessions STRICTLY BEFORE the current
    one (NaN until ``trailing_days`` prior sessions exist). Index: day;
    columns: minute index within the window."""
    m = minute[["day", "ts", "volume"]].sort_values(["day", "ts"]).copy()
    m["minute_idx"] = m.groupby("day").cumcount()
    m = m[m["minute_idx"] < window_len]
    pivot = m.pivot(index="day", columns="minute_idx", values="volume").sort_index()
    return pivot.shift(1).rolling(trailing_days, min_periods=trailing_days).mean()


def _day_tensor(bars_day: pd.DataFrame, window_len: int,
                rel_vol_avg: np.ndarray) -> np.ndarray | None:
    """One session's tensor, or None if it has fewer than ``window_len``
    bars. ``bars_day`` must be sorted by ``ts``; ``rel_vol_avg`` is that
    session's precomputed trailing same-minute volume average (length
    ``window_len``, from :func:`same_minute_volume_history`)."""
    if len(bars_day) < window_len:
        return None
    w = bars_day.iloc[:window_len]
    close = w["close"].to_numpy(dtype=float)
    high = w["high"].to_numpy(dtype=float)
    low = w["low"].to_numpy(dtype=float)
    volume = w["volume"].to_numpy(dtype=float)
    day_open = float(w["open"].iloc[0])

    log_ret = np.log(close / day_open)

    rel_volume = np.divide(volume, rel_vol_avg,
                           out=np.full(window_len, np.nan), where=rel_vol_avg > 0)

    cummax_high = np.maximum.accumulate(high)
    cummin_low = np.minimum.accumulate(low)
    rng = cummax_high - cummin_low
    cum_range_position = np.where(rng > 0, (close - cummin_low) / np.where(rng > 0, rng, 1.0), 0.5)

    vwap = R.cumulative_vwap(w)
    cum_vwap_distance = np.log(close / vwap)

    return np.column_stack([log_ret, rel_volume, cum_range_position, cum_vwap_distance])


def build_channel_tensor(minute: pd.DataFrame, window_len: int,
                         trailing_days: int = 20
                         ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Every session's tensor, stacked. Returns ``(array of shape (n_days,
    window_len, N_CHANNELS), day_index)`` - only sessions with a full
    ``trailing_days``-session volume history AND >= ``window_len`` bars
    are included (both requirements matching the causal-only guarantee
    every other channel already has by construction)."""
    vol_hist = same_minute_volume_history(minute, window_len, trailing_days)
    rows, days = [], []
    for day, g in minute.sort_values("ts").groupby("day"):
        if day not in vol_hist.index or vol_hist.loc[day].isna().any():
            continue
        t = _day_tensor(g, window_len, vol_hist.loc[day].to_numpy(dtype=float))
        if t is None:
            continue
        rows.append(t)
        days.append(day)
    if not rows:
        return np.empty((0, window_len, N_CHANNELS)), pd.DatetimeIndex([])
    return np.stack(rows), pd.DatetimeIndex(days)


def build_task_b_labels(minute: pd.DataFrame, n_minutes: int,
                        cost_per_share: float) -> pd.DataFrame:
    """Binary label (1 = the eventual close-direction, net of round-trip
    cost, was up) + continuous net return in bps, entering at bar
    ``n_minutes``'s OPEN (the bar starting at 09:30+N - the first bar not
    part of the decision window, same "decide on bars 0..k-1, execute at
    bar k's open" convention as ORB's own entry rule) and exiting at the
    session's own final close.

    Sessions where even the best-case direction doesn't clear the
    round-trip cost are dropped entirely (``net_bps <= 0`` for both
    directions is only possible when ``abs(raw_bps) <= cost_bps``) - the
    same "nothing to decide" treatment as ORB's own doji days
    (config/week08_filter.toml [label]), not coded as a negative label.
    """
    cost_bps_per_share = cost_per_share  # applied in bps terms below
    rows = []
    for day, g in minute.sort_values("ts").groupby("day"):
        g = g.reset_index(drop=True)
        if len(g) <= n_minutes:
            continue
        entry_price = float(g["open"].iloc[n_minutes])
        exit_price = float(g["close"].iloc[-1])
        if entry_price <= 0:
            continue
        raw_bps = (exit_price - entry_price) / entry_price * 1e4
        cost_bps = 2 * cost_bps_per_share / entry_price * 1e4
        net_bps = abs(raw_bps) - cost_bps
        if net_bps <= 0:
            continue
        rows.append({"day": day, "label": int(raw_bps > 0), "net_bps": net_bps,
                    "entry_time": g["ts"].iloc[n_minutes],
                    "exit_time": g["bar_end"].iloc[-1]})
    if not rows:
        return pd.DataFrame(columns=["day", "label", "net_bps", "entry_time",
                                     "exit_time"]).set_index("day")
    return pd.DataFrame(rows).set_index("day")


def standardize(tensor: np.ndarray, day_index: pd.DatetimeIndex,
                fit_days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-score per channel, mean/std fit ONLY on ``fit_days`` (a fold's
    model-fit window) - applied to every day in ``tensor``. Returns
    ``(standardized_tensor, mean, std)``, ``mean``/``std`` shape
    ``(N_CHANNELS,)`` for reuse on the validation/test windows of the same
    fold."""
    fit_mask = day_index.isin(fit_days)
    fit_rows = tensor[fit_mask].reshape(-1, tensor.shape[-1])
    mean = np.nanmean(fit_rows, axis=0)
    std = np.nanstd(fit_rows, axis=0)
    std = np.where(std > 0, std, 1.0)
    return (tensor - mean) / std, mean, std


def save_tensor(name: str, tensor: np.ndarray, days: pd.DatetimeIndex,
                out_dir: Path = SEQ_DIR) -> Path:
    """Raw (unstandardized) tensor + its day index - standardization is
    always fold-specific (:func:`standardize`), so nothing pre-standardized
    is ever cached to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.npz"
    np.savez_compressed(p, tensor=tensor,
                        days=days.values.astype("datetime64[ns]"))
    return p


def load_tensor(name: str, out_dir: Path = SEQ_DIR) -> tuple[np.ndarray, pd.DatetimeIndex]:
    data = np.load(out_dir / f"{name}.npz")
    return data["tensor"], pd.DatetimeIndex(data["days"])


def build_and_save_all(processed_dir: Path = dl.PROCESSED_DIR) -> dict[str, Path]:
    minute = dl.load_minute_rth(processed_dir=processed_dir)
    orb = pd.read_parquet(Path(processed_dir) / "strategies" / "orb_daily.parquet")

    out = {}
    tA, daysA = build_channel_tensor(minute, window_len=5)
    from src.models.filter import build_labels
    labels_a = build_labels(orb)
    common_a = daysA.intersection(labels_a.index)
    mask_a = daysA.isin(common_a)
    out["task_a_tensor"] = save_tensor("task_a", tA[mask_a], daysA[mask_a])
    out["task_a_labels"] = SEQ_DIR / "task_a_labels.parquet"
    labels_a.loc[common_a].reset_index().rename(columns={"index": "day"}) \
        .to_parquet(out["task_a_labels"], index=False)
    log.info("task A: %d days", len(common_a))

    cost_per_share = R.MAIN_COMMISSION + R.MAIN_SLIPPAGE
    for n in (15, 30):
        tB, daysB = build_channel_tensor(minute, window_len=n)
        labels_b = build_task_b_labels(minute, n, cost_per_share)
        common = daysB.intersection(labels_b.index)
        mask = daysB.isin(common)
        out[f"task_b{n}_tensor"] = save_tensor(f"task_b{n}", tB[mask], daysB[mask])
        out[f"task_b{n}_labels"] = SEQ_DIR / f"task_b{n}_labels.parquet"
        labels_b.loc[common].reset_index().rename(columns={"index": "day"}) \
            .to_parquet(out[f"task_b{n}_labels"], index=False)
        log.info("task B (N=%d): %d days", n, len(common))
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = build_and_save_all()
    for k, v in out.items():
        log.info("wrote %s -> %s", k, v)


if __name__ == "__main__":
    main()
