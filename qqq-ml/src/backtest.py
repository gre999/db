"""Week 5: execute a position series, apply costs, evaluate.

Reads forecasts from ``data/processed/predictions/`` (never refits a model
- see ``src/models/har.py`` / ``src/models/week4.py`` for that) and prices
from ``data/processed/daily.parquet``. Position sizing itself lives in
:mod:`src.strategies`; this module is execution (timing, no-trade band,
turnover cost) and evaluation.

Timeline convention: a forecast/position row indexed by date ``d`` was
computed from information known at close(d-1) - true for every model in
this project (see ``src/models/base.py``'s label-timing check) and for the
20-day trailing-vol and buy-and-hold sources by construction. Under the
main ("close") timing that position is implemented right away, at
close(d-1), and earns the close(d-1)->close(d) return, ``ret_cc``. Under
the "next_open" robustness variant it is implemented one leg later, at
open(d): the overnight leg close(d-1)->open(d) is earned by whatever was
already held (yesterday's decision), and only the intraday leg open(d)->
close(d) earns the new weight.

Usage::

    python -m src.backtest
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import data_loader as dl
from src import strategies as S
from src.models.base import load_predictions
from src.models.har import newey_west_cov

log = logging.getLogger(__name__)
OUT_DIR = dl.PROCESSED_DIR / "backtests"
NAME = "week5_backtests"

# name -> (prediction file, model) for the tree/HAR sources; "hist20",
# "buy_hold" and "perfect" are computed directly from prices/actual RV.
MODEL_SOURCES = {
    "har": ("har_baselines", "har"),
    "har_resid_xgb": ("week4_models", "har_resid_xgb"),
}
SOURCES = ["buy_hold", "hist20", "har", "har_resid_xgb", "perfect"]


@dataclass(frozen=True)
class BacktestConfig:
    vol_target: float = 0.15
    leverage_cap: float = 1.5
    timing: str = "close"           # "close" or "next_open"
    cost_bps: float = 1.0
    band: float = 0.0               # no-trade band, in weight units
    hist_window: int = 20
    premium_q: float = 0.25
    derate: float = 0.6


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_inputs(target: str = "rv") -> dict:
    """Everything needed to build weights and returns, independent of ``cfg``."""
    daily = dl.load_daily()[["ret_cc", "ret_overnight", "ret_intraday"]]
    t = pd.read_parquet(dl.PROCESSED_DIR / "targets_daily.parquet")
    rv = t["rv"].where(t["valid"].astype(bool))
    rv_on = t["rv_on"].where(t["valid"].astype(bool))
    vxn = pd.read_parquet(dl.PROCESSED_DIR / "features_prev_close.parquet",
                          columns=["vxn_1d"])["vxn_1d"]
    folds = S.fold_windows(target=target)
    ratios = S.scale_ratios(rv, rv_on, folds)

    preds = {}
    for name, (pred_name, model) in MODEL_SOURCES.items():
        p = load_predictions(pred_name)
        p = p[(p["model"] == model) & (p["target"] == target)]
        preds[name] = p.set_index("date").sort_index()

    common = None
    for p in preds.values():
        common = p.index if common is None else common.intersection(p.index)
    return {"daily": daily, "rv": rv, "rv_on": rv_on, "vxn": vxn,
            "folds": folds, "ratios": ratios, "preds": preds,
            "common_index": common.sort_values()}


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------
def build_weight(name: str, inputs: dict, cfg: BacktestConfig) -> pd.Series:
    """Target weight series for one source, restricted to ``common_index``."""
    idx = inputs["common_index"]
    if name == "buy_hold":
        w = S.buy_and_hold_weight(idx)
    elif name == "hist20":
        w = S.historical_vol_weight(inputs["daily"]["ret_cc"], cfg.hist_window,
                                    cfg.vol_target, cfg.leverage_cap).reindex(idx)
    elif name == "perfect":
        w = S.vol_target_weight(inputs["rv_on"].reindex(idx), cfg.vol_target,
                                cfg.leverage_cap)
    elif name in MODEL_SOURCES:
        p = inputs["preds"][name].reindex(idx)
        var_cc = S.cc_variance_forecast(p, inputs["ratios"])
        w = S.vol_target_weight(var_cc, cfg.vol_target, cfg.leverage_cap)
    else:
        raise ValueError(f"unknown source {name!r}")
    return w.rename(name)


def build_strategy_n_weight(name: str, inputs: dict, cfg: BacktestConfig
                            ) -> tuple[pd.Series, pd.Series]:
    """(base weight, premium-tilted weight) for a model source (strategy N)."""
    if name not in MODEL_SOURCES:
        raise ValueError("strategy N needs a model variance forecast")
    idx = inputs["common_index"]
    p = inputs["preds"][name].reindex(idx)
    var_cc = S.cc_variance_forecast(p, inputs["ratios"])
    base = S.vol_target_weight(var_cc, cfg.vol_target, cfg.leverage_cap)
    premium = S.variance_risk_premium(inputs["vxn"].reindex(idx), var_cc)
    thr = S.premium_thresholds(premium, inputs["folds"], cfg.premium_q)
    tilted = S.strategy_n_weight(base, premium, p["fold"], thr, cfg.derate)
    return base.rename(name), tilted.rename(f"{name}_stratN")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def run_backtest(target_weight: pd.Series, daily: pd.DataFrame, *,
                 timing: str = "close", cost_bps: float = 1.0,
                 band: float = 0.0, w0: float = 0.0) -> pd.DataFrame:
    """Simulate holding ``target_weight`` with a no-trade band and linear cost.

    ``target_weight`` indexed by date d = the weight decided from
    information available at close(d-1) (see module docstring). Returns a
    frame indexed by date with ``weight_target``, ``weight_held`` (what
    actually earned that day's return), ``turnover``, ``cost``,
    ``gross_return``, ``net_return``, ``equity`` (cumulative product from 1).
    """
    idx = target_weight.dropna().index
    if timing not in ("close", "next_open"):
        raise ValueError(f"timing must be 'close' or 'next_open', got {timing!r}")
    need = ["ret_cc"] if timing == "close" else ["ret_overnight", "ret_intraday"]
    d = daily.reindex(idx)
    bad = d[need].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"missing {need} on {list(idx[bad][:5])}")

    held = w0
    rows = []
    for dt, tgt in target_weight.loc[idx].items():
        new_held = tgt if abs(tgt - held) >= band else held
        turnover = abs(new_held - held)
        cost = cost_bps * 1e-4 * turnover
        if timing == "close":
            gross = new_held * d.at[dt, "ret_cc"]
        else:
            gross = held * d.at[dt, "ret_overnight"] + \
                new_held * d.at[dt, "ret_intraday"]
        rows.append((dt, tgt, new_held, turnover, cost, gross, gross - cost))
        held = new_held
    out = pd.DataFrame(rows, columns=["date", "weight_target", "weight_held",
                                      "turnover", "cost", "gross_return",
                                      "net_return"]).set_index("date")
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(bt: pd.DataFrame, vol_target: float = 0.15) -> pd.Series:
    """Annualized return/vol/Sharpe, max drawdown, Calmar, leverage, turnover."""
    r = bt["net_return"]
    n = len(r)
    ann_return = float((1.0 + r).prod() ** (S.TRADING_DAYS / n) - 1.0)
    ann_vol = float(r.std(ddof=0) * np.sqrt(S.TRADING_DAYS))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    dd = bt["equity"] / bt["equity"].cummax() - 1.0
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    return pd.Series({
        "n": n, "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_drawdown": max_dd, "calmar": calmar,
        "avg_leverage": float(bt["weight_held"].mean()),
        "avg_turnover": float(bt["turnover"].mean()),
        "tracking_error": ann_vol - vol_target,
    })


def block_bootstrap_sharpe_diff(ret_a: pd.Series, ret_b: pd.Series,
                                block_size: int = 20, n_boot: int = 2000,
                                seed: int = 0) -> tuple[float, float]:
    """Moving-block-bootstrap p-value for Sharpe(a) - Sharpe(b) != 0.

    Resamples blocks of the *paired* (a, b) daily-return series, preserving
    both the cross-sectional pairing and the serial dependence within a
    block (vol clustering). The bootstrap distribution is centered at its
    own mean to build a null of "no difference"; the two-sided p-value is
    the share of centered bootstrap diffs at least as far from 0 as the
    observed diff.
    """
    a, b = ret_a.align(ret_b, join="inner")
    x = np.column_stack([a.to_numpy(), b.to_numpy()])
    n = len(x)

    def sharpe(col: np.ndarray) -> float:
        s = col.std(ddof=0)
        return float(col.mean() / s * np.sqrt(S.TRADING_DAYS)) if s > 0 else np.nan

    obs = sharpe(x[:, 0]) - sharpe(x[:, 1])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.arange(n - block_size + 1)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block_size)
                              for s in rng.choice(starts, n_blocks)])[:n]
        rs = x[idx]
        diffs[i] = sharpe(rs[:, 0]) - sharpe(rs[:, 1])
    centered = diffs - diffs.mean()
    p = float(np.mean(np.abs(centered) >= abs(obs)))
    return float(obs), p


def regress_weight_diff_on_return(weight_a: pd.Series, weight_b: pd.Series,
                                  ret: pd.Series, lags: int | None = None
                                  ) -> dict:
    """OLS of the same-day return on ``weight_a - weight_b``, Newey-West HAC se.

    Return-timing vs risk-timing diagnostic: if source A is more cautious
    than B specifically on days that turn out to have worse returns (and
    more aggressive on days with better returns), the weight difference has
    a significant *positive* slope on the return that weight earns - the
    source's edge is about which days it's positioned for, not just how
    precisely it hits a volatility target. A source that only differs from
    B by noise around a shared risk target would have a slope of ~0.
    """
    diff = (weight_a - weight_b).dropna()
    idx = diff.index.intersection(ret.dropna().index)
    x, y = diff.loc[idx].to_numpy(), ret.loc[idx].to_numpy()
    n = len(x)
    lags = int(np.floor(n ** (1 / 3))) if lags is None else lags
    Z = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    se = np.sqrt(np.diag(newey_west_cov(Z, resid, lags)))
    tstat = beta / se
    pval = 2 * stats.norm.sf(np.abs(tstat))
    return {"n": n, "intercept": float(beta[0]), "slope": float(beta[1]),
           "slope_se": float(se[1]), "slope_t": float(tstat[1]),
           "slope_p": float(pval[1])}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(cfg: BacktestConfig = BacktestConfig(), inputs: dict | None = None,
       save: bool = True) -> dict:
    """Run every source under ``cfg`` (+ the strategy-N pair), evaluate all."""
    inputs = inputs or load_inputs()
    weights = {name: build_weight(name, inputs, cfg) for name in SOURCES}
    base_n, tilt_n = build_strategy_n_weight("har_resid_xgb", inputs, cfg)
    weights["har_resid_xgb_stratN"] = tilt_n

    bts, scores = {}, []
    for name, w in weights.items():
        bt = run_backtest(w, inputs["daily"], timing=cfg.timing,
                          cost_bps=cfg.cost_bps, band=cfg.band)
        bts[name] = bt
        scores.append({"source": name, **evaluate(bt, cfg.vol_target).to_dict()})
    scores = pd.DataFrame(scores).set_index("source")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, bt in bts.items():
            bt.reset_index().to_parquet(OUT_DIR / f"{NAME}_{name}.parquet",
                                        index=False)
        scores.reset_index().to_parquet(OUT_DIR / f"{NAME}_scores.parquet",
                                        index=False)
        (OUT_DIR / f"{NAME}_meta.json").write_text(json.dumps(
            {"config": asdict(cfg), "sources": list(weights),
             "n_days": len(inputs["common_index"]),
             "saved_at": pd.Timestamp.now(tz=dl.ET).isoformat()},
            indent=2, default=str), encoding="utf-8")
    return {"weights": weights, "backtests": bts, "scores": scores,
           "inputs": inputs}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = run()
    with pd.option_context("display.width", 120):
        print(out["scores"].round(4))


if __name__ == "__main__":
    main()
