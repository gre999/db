"""Generate reports/week05_position.md (+ figures).

Reads model forecasts from ``data/processed/predictions/`` and the
main-config backtest from ``data/processed/backtests/`` (run
``python -m src.backtest`` first). Every robustness variant here (cost,
no-trade band, execution timing, vol-target x leverage-cap grid, the
optional HAR+residual-RF source) is cheap and recomputed on the fly.

Usage:
    python scripts/make_week05_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from make_week01_report import (INK_2, MUTED, _date_axis,  # noqa: E402
                                _md_table, _save)
from src import backtest as B  # noqa: E402
from src import metrics as M  # noqa: E402
from src import strategies as S  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

REPORTS = ROOT / "reports"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, PURPLE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c23b6b", "#7b4fd1")
SOURCES = ["buy_hold", "hist20", "har", "har_resid_xgb", "perfect"]
LABEL = {"buy_hold": "Buy & hold", "hist20": "20d hist. vol", "har": "HAR",
         "har_resid_xgb": "HAR+Resid", "perfect": "Perfect foresight",
         "har_resid_xgb_stratN": "HAR+Resid+VRP", "har_resid_rf": "HAR+Resid RF"}
COLOR = {"buy_hold": MUTED, "hist20": YELLOW, "har": BLUE,
         "har_resid_xgb": MAGENTA, "perfect": AQUA,
         "har_resid_xgb_stratN": PURPLE, "har_resid_rf": ORANGE}
CRISIS_SLICES = ["covid_2020", "tariff_2025"]
QLIKE = {"har": 0.1974, "har_resid_xgb": 0.1827}  # from week04_xgb.md, target=rv


def source_var_cc(name: str, inputs: dict, cfg: B.BacktestConfig) -> pd.Series:
    """The (uncapped) daily close-to-close variance forecast behind a weight."""
    idx = inputs["common_index"]
    if name == "hist20":
        return inputs["daily"]["ret_cc"].rolling(
            cfg.hist_window, min_periods=cfg.hist_window).var(ddof=0).reindex(idx)
    if name == "perfect":
        return inputs["rv_on"].reindex(idx)
    if name in B.MODEL_SOURCES:
        p = inputs["preds"][name].reindex(idx)
        return S.cc_variance_forecast(p, inputs["ratios"])
    raise ValueError(name)


def source_qlike(name: str, inputs: dict, cfg: B.BacktestConfig) -> float:
    if name == "buy_hold":
        return float("nan")
    idx = inputs["common_index"]
    rv_on = inputs["rv_on"].reindex(idx)
    var_cc = source_var_cc(name, inputs, cfg)
    ok = var_cc.notna() & rv_on.notna() & (var_cc > 0) & (rv_on > 0)
    return M.qlike(rv_on[ok], var_cc[ok])


def fig_equity_drawdown(bts: dict[str, pd.DataFrame], names: list[str]) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name in names:
        bt = bts[name]
        ax1.plot(bt.index, bt["equity"], color=COLOR[name], linewidth=1.4,
                 label=LABEL[name])
        dd = bt["equity"] / bt["equity"].cummax() - 1.0
        ax2.plot(bt.index, dd * 100, color=COLOR[name], linewidth=1.2)
    ax1.set_yscale("log")
    ax1.set_title("Equity (log scale, starts at 1)")
    ax1.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax2.set_title("Drawdown")
    ax2.set_ylabel("%")
    ax2.axhline(0, color=INK_2, linewidth=0.8)
    _date_axis(ax2)
    fig.tight_layout()
    return _save(fig, "w05_equity_drawdown.png")


def fig_tracking(bts: dict[str, pd.DataFrame], names: list[str],
                 vol_target: float, window: int = 60) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for name in names:
        r = bts[name]["net_return"]
        rv = r.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252) * 100
        ax.plot(rv.index, rv, color=COLOR[name], linewidth=1.3, label=LABEL[name])
    ax.axhline(vol_target * 100, color=INK_2, linewidth=1.0, linestyle="--",
              label=f"{vol_target:.0%} target")
    ax.set_title(f"Rolling {window}-day realized annualized vol vs target")
    ax.set_ylabel("Annualized vol, %")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    _date_axis(ax)
    return _save(fig, "w05_tracking.png")


def main() -> None:
    inputs = B.load_inputs()
    cfg = B.BacktestConfig()
    out = B.run(cfg, inputs=inputs, save=True)
    bts, scores = out["backtests"], out["scores"]

    # ---- two-layer summary: forecast quality (QLIKE) x strategy outcome
    two_layer = pd.DataFrame({
        "QLIKE": {n: source_qlike(n, inputs, cfg) for n in SOURCES},
    }).join(scores[["ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar",
                    "avg_leverage", "avg_turnover", "tracking_error"]])
    two_layer = two_layer.loc[SOURCES].round(4)
    two_layer.index = [LABEL[n] for n in two_layer.index]
    two_layer["QLIKE"] = two_layer["QLIKE"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.4f}")

    # ---- yearly Sharpe
    rows = []
    for name in SOURCES:
        bt = bts[name].copy()
        bt["year"] = bt.index.year
        for y, g in bt.groupby("year"):
            rows.append({"source": name, "year": y,
                        "sharpe": B.evaluate(g, cfg.vol_target)["sharpe"]})
    yearly = pd.DataFrame(rows).pivot(index="year", columns="source",
                                      values="sharpe")[SOURCES]
    yearly.columns = [LABEL[c] for c in yearly.columns]
    yearly = yearly.round(2).reset_index()
    # iterrows()/to_numpy() upcasts a mixed int+float row to float64 - stringify
    # the int column first so "year" doesn't render as "2019.0" in _md_table.
    yearly["year"] = yearly["year"].astype(int).astype(str)

    # ---- crisis slices
    t = pd.read_parquet(ROOT / "data" / "processed" / "targets_daily.parquet")
    rv = t["rv"].where(t["valid"].astype(bool))
    dates = pd.DatetimeIndex(sorted(inputs["common_index"]))
    slices = M.eval_slices(dates, rv)
    srows = []
    for name in SOURCES:
        bt = bts[name]
        for sname in CRISIS_SLICES:
            mask = slices.reindex(bt.index)[sname].to_numpy()
            sub = bt.loc[mask]
            if len(sub) < 5:
                continue
            ev = B.evaluate(sub, cfg.vol_target)
            srows.append({"source": LABEL[name], "slice": sname, "n": len(sub),
                         "sharpe": ev["sharpe"], "max_drawdown": ev["max_drawdown"],
                         "ann_vol": ev["ann_vol"]})
    crisis = pd.DataFrame(srows)
    crisis_sharpe = crisis.pivot(index="slice", columns="source",
                                 values="sharpe")[[LABEL[n] for n in SOURCES]].round(2)
    crisis_dd = crisis.pivot(index="slice", columns="source",
                             values="max_drawdown")[[LABEL[n] for n in SOURCES]].round(3)

    # ---- strategy N
    base_n, tilt_n = B.build_strategy_n_weight("har_resid_xgb", inputs, cfg)
    bt_tilt = B.run_backtest(tilt_n, inputs["daily"], timing=cfg.timing,
                             cost_bps=cfg.cost_bps, band=cfg.band)
    bts["har_resid_xgb_stratN"] = bt_tilt
    ev_base = B.evaluate(bts["har_resid_xgb"], cfg.vol_target)
    ev_tilt = B.evaluate(bt_tilt, cfg.vol_target)
    stratn_tbl = pd.DataFrame({LABEL["har_resid_xgb"]: ev_base,
                               LABEL["har_resid_xgb_stratN"]: ev_tilt}).T[
        ["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_leverage",
         "avg_turnover"]].round(4).reset_index(names="source")

    # ---- Sharpe significance (block bootstrap)
    pairs = [("har_resid_xgb", "har"), ("har_resid_xgb", "hist20"),
             ("har_resid_xgb", "buy_hold"),
             ("har_resid_xgb_stratN", "har_resid_xgb")]
    dm_rows = []
    for a, b in pairs:
        obs, p = B.block_bootstrap_sharpe_diff(bts[a]["net_return"],
                                               bts[b]["net_return"],
                                               block_size=20, n_boot=3000, seed=0)
        dm_rows.append({"A": LABEL.get(a, a), "B": LABEL.get(b, b),
                        "sharpe(A)-sharpe(B)": round(obs, 4), "p-value": round(p, 4)})
    dm_tbl = pd.DataFrame(dm_rows)

    # ---- robustness: cost / band / timing
    variants = {
        "main (close, 1bp, no band)": B.BacktestConfig(),
        "cost = 3bp": B.BacktestConfig(cost_bps=3.0),
        "5% no-trade band": B.BacktestConfig(band=0.05),
        "timing = next-day open": B.BacktestConfig(timing="next_open"),
    }
    rob_rows = []
    for vname, vcfg in variants.items():
        vout = B.run(vcfg, inputs=inputs, save=False)
        s = vout["scores"]
        rob_rows.append({"variant": vname,
                         "hist20 sharpe": s.loc["hist20", "sharpe"],
                         "har sharpe": s.loc["har", "sharpe"],
                         "har_resid_xgb sharpe": s.loc["har_resid_xgb", "sharpe"],
                         "har_resid_xgb turnover": s.loc["har_resid_xgb",
                                                         "avg_turnover"]})
    rob_tbl = pd.DataFrame(rob_rows).round(4)

    # ---- robustness: vol target x leverage cap grid
    grid_rows = []
    for vt in (0.10, 0.15, 0.20):
        for cap in (1.0, 1.5, 2.0):
            gcfg = B.BacktestConfig(vol_target=vt, leverage_cap=cap)
            gout = B.run(gcfg, inputs=inputs, save=False)
            s = gout["scores"]
            grid_rows.append({"vol_target": vt, "leverage_cap": cap,
                              "hist20": s.loc["hist20", "sharpe"],
                              "har": s.loc["har", "sharpe"],
                              "har_resid_xgb": s.loc["har_resid_xgb", "sharpe"]})
    grid = pd.DataFrame(grid_rows)
    grid["resid_beats_har"] = grid["har_resid_xgb"] > grid["har"]
    grid["resid_beats_hist20"] = grid["har_resid_xgb"] > grid["hist20"]
    n_beats_har = int(grid["resid_beats_har"].sum())
    n_beats_hist20 = int(grid["resid_beats_hist20"].sum())
    grid = grid.round(3)

    # ---- optional: HAR+residual random forest
    rf_path = B.dl.PROCESSED_DIR / "predictions" / "week5_har_resid_rf.parquet"
    rf_section = ""
    if rf_path.exists():
        rf_pred = load_predictions("week5_har_resid_rf")
        rf_pred = rf_pred[(rf_pred["model"] == "har_resid_rf") &
                          (rf_pred["target"] == "rv")].set_index("date").sort_index()
        idx = inputs["common_index"].intersection(rf_pred.index)
        rf_pred = rf_pred.reindex(idx)
        var_cc_rf = S.cc_variance_forecast(rf_pred, inputs["ratios"])
        w_rf = S.vol_target_weight(var_cc_rf, cfg.vol_target, cfg.leverage_cap)
        bt_rf = B.run_backtest(w_rf, inputs["daily"], timing=cfg.timing,
                               cost_bps=cfg.cost_bps, band=cfg.band)
        ev_rf = B.evaluate(bt_rf, cfg.vol_target)
        rv_on_rf = inputs["rv_on"].reindex(idx)
        ok = var_cc_rf.notna() & rv_on_rf.notna() & (var_cc_rf > 0) & (rv_on_rf > 0)
        qlike_rf = M.qlike(rv_on_rf[ok], var_cc_rf[ok])
        obs_rf, p_rf = B.block_bootstrap_sharpe_diff(
            bt_rf["net_return"], bts["har_resid_xgb"]["net_return"],
            block_size=20, n_boot=3000, seed=0)
        rf_tbl = pd.DataFrame({
            "HAR+Resid XGB": {"QLIKE": QLIKE["har_resid_xgb"], **ev_base.to_dict()},
            "HAR+Resid RF": {"QLIKE": round(qlike_rf, 4), **ev_rf.to_dict()},
        }).T[["QLIKE", "ann_return", "ann_vol", "sharpe", "max_drawdown",
             "avg_turnover"]].round(4).reset_index(names="source")
        rf_section = f"""
## 八、選做：HAR+殘差隨機森林

{_md_table(rf_tbl)}

隨機森林版對 HAR+殘差XGB 的夏普差異：sharpe(RF)−sharpe(XGB) = {obs_rf:+.4f}，
block bootstrap p = {p_rf:.4f}。
"""

    # ---- smoothing diagnostic: reaction speed vs information content
    p_resid = inputs["preds"]["har_resid_xgb"].reindex(inputs["common_index"])
    var_cc_resid = S.cc_variance_forecast(p_resid, inputs["ratios"])
    smooth_rows = []
    for window in (10, 20, 40):
        var_sm = S.smooth_variance_forecast(var_cc_resid, window)
        w_sm = S.vol_target_weight(var_sm, cfg.vol_target, cfg.leverage_cap)
        bt_sm = B.run_backtest(w_sm, inputs["daily"], timing=cfg.timing,
                               cost_bps=cfg.cost_bps, band=cfg.band)
        bts[f"har_resid_xgb_smooth{window}"] = bt_sm
        ev_sm = B.evaluate(bt_sm, cfg.vol_target)
        rv_on = inputs["rv_on"].reindex(var_sm.index)
        ok = var_sm.notna() & rv_on.notna() & (var_sm > 0) & (rv_on > 0)
        qlike_sm = M.qlike(rv_on[ok], var_sm[ok])
        _, p_vs_hist20 = B.block_bootstrap_sharpe_diff(
            bt_sm["net_return"], bts["hist20"]["net_return"],
            block_size=20, n_boot=3000, seed=0)
        _, p_vs_unsmoothed = B.block_bootstrap_sharpe_diff(
            bt_sm["net_return"], bts["har_resid_xgb"]["net_return"],
            block_size=20, n_boot=3000, seed=0)
        smooth_rows.append({"window": window, "QLIKE": round(qlike_sm, 4),
                           "sharpe": round(ev_sm["sharpe"], 4),
                           "avg_turnover": round(ev_sm["avg_turnover"], 4),
                           "p vs hist20": round(p_vs_hist20, 4),
                           "p vs unsmoothed": round(p_vs_unsmoothed, 4)})
    smooth_tbl = pd.DataFrame(smooth_rows)
    smooth_tbl["window"] = smooth_tbl["window"].astype(str)

    # ---- overnight diagnostic: does predicting rv_on natively close the gap?
    t_all = pd.read_parquet(ROOT / "data" / "processed" / "targets_daily.parquet")
    t_all = t_all[t_all["valid"].astype(bool)]
    on_share = (t_all["overnight"] ** 2) / t_all["rv_on"]
    on_share_yearly = on_share.groupby(t_all.index.year).mean()
    on_share_monthly = on_share.groupby([t_all.index.year, t_all.index.month]).mean()

    har_on = load_predictions("har_baselines")
    har_on = har_on[(har_on["model"] == "har") &
                    (har_on["target"] == "rv_on")].set_index("date").sort_index()
    resid_on = load_predictions("week5_har_resid_xgb_rvon")
    resid_on = resid_on[(resid_on["model"] == "har_resid_xgb") &
                        (resid_on["target"] == "rv_on")
                        ].set_index("date").sort_index()
    idx_on = har_on.index.intersection(resid_on.index).intersection(
        inputs["common_index"])

    on_rows = []
    for name, p in [("har_rvon", har_on), ("har_resid_xgb_rvon", resid_on)]:
        var_cc_on = p["y_pred_var"].reindex(idx_on)
        w_on = S.vol_target_weight(var_cc_on, cfg.vol_target, cfg.leverage_cap)
        bt_on = B.run_backtest(w_on, inputs["daily"], timing=cfg.timing,
                               cost_bps=cfg.cost_bps, band=cfg.band)
        bts[name] = bt_on
        ev_on = B.evaluate(bt_on, cfg.vol_target)
        rv_on_actual = inputs["rv_on"].reindex(idx_on)
        ok = var_cc_on.notna() & rv_on_actual.notna() & \
            (var_cc_on > 0) & (rv_on_actual > 0)
        qlike_on = M.qlike(rv_on_actual[ok], var_cc_on[ok])
        _, p_vs_hist20_on = B.block_bootstrap_sharpe_diff(
            bt_on["net_return"], bts["hist20"]["net_return"].reindex(idx_on),
            block_size=20, n_boot=3000, seed=0)
        _, p_vs_rv_target = B.block_bootstrap_sharpe_diff(
            bt_on["net_return"], bts["har_resid_xgb"]["net_return"].reindex(idx_on),
            block_size=20, n_boot=3000, seed=0)
        on_rows.append({"source": name, "QLIKE": round(qlike_on, 4),
                       "sharpe": round(ev_on["sharpe"], 4),
                       "max_drawdown": round(ev_on["max_drawdown"], 4),
                       "avg_turnover": round(ev_on["avg_turnover"], 4),
                       "p vs hist20": round(p_vs_hist20_on, 4),
                       "p vs rv-target": round(p_vs_rv_target, 4)})
    on_tbl = pd.DataFrame(on_rows)
    qlike_har_rv = two_layer.loc[LABEL["har"], "QLIKE"]
    qlike_resid_rv = two_layer.loc[LABEL["har_resid_xgb"], "QLIKE"]

    # ---- feature diagnostic: add trailing cc variance as a feature instead
    ccfeat_path = B.dl.PROCESSED_DIR / "predictions" / \
        "week5_har_resid_xgb_ccfeat.parquet"
    ccfeat_section = ""
    if ccfeat_path.exists():
        p_cc = load_predictions("week5_har_resid_xgb_ccfeat")
        p_cc = p_cc[(p_cc["model"] == "har_resid_xgb_ccfeat") &
                   (p_cc["target"] == "rv")].set_index("date").sort_index()
        idx_cc = p_cc.index.intersection(inputs["common_index"])
        p_cc = p_cc.reindex(idx_cc)
        var_cc_feat = S.cc_variance_forecast(p_cc, inputs["ratios"])
        w_cc = S.vol_target_weight(var_cc_feat, cfg.vol_target, cfg.leverage_cap)
        bt_cc = B.run_backtest(w_cc, inputs["daily"], timing=cfg.timing,
                               cost_bps=cfg.cost_bps, band=cfg.band)
        bts["har_resid_xgb_ccfeat"] = bt_cc
        ev_cc = B.evaluate(bt_cc, cfg.vol_target)
        rv_on_cc = inputs["rv_on"].reindex(idx_cc)
        ok = var_cc_feat.notna() & rv_on_cc.notna() & \
            (var_cc_feat > 0) & (rv_on_cc > 0)
        qlike_cc = M.qlike(rv_on_cc[ok], var_cc_feat[ok])
        _, p_cc_vs_hist20 = B.block_bootstrap_sharpe_diff(
            bt_cc["net_return"], bts["hist20"]["net_return"].reindex(idx_cc),
            block_size=20, n_boot=3000, seed=0)
        _, p_cc_vs_base = B.block_bootstrap_sharpe_diff(
            bt_cc["net_return"], bts["har_resid_xgb"]["net_return"].reindex(idx_cc),
            block_size=20, n_boot=3000, seed=0)
        imp_cc = pd.read_parquet(B.dl.PROCESSED_DIR / "predictions" /
                                 "week5_har_resid_xgb_ccfeat_importance.parquet")
        imp_all_cc = imp_cc[imp_cc["slice"] == "all"].groupby(
            "feature")["importance"].mean().sort_values(ascending=False)
        cc_rank = int(imp_all_cc.rank(ascending=False)["hist_cc_var_20d"])
        cc_n = len(imp_all_cc)
        cc_tbl = pd.DataFrame([{
            "source": "har_resid_xgb_ccfeat", "QLIKE": round(qlike_cc, 4),
            "sharpe": round(ev_cc["sharpe"], 4),
            "max_drawdown": round(ev_cc["max_drawdown"], 4),
            "avg_turnover": round(ev_cc["avg_turnover"], 4),
            "p vs hist20": round(p_cc_vs_hist20, 4),
            "p vs base (no cc feature)": round(p_cc_vs_base, 4)}])
        ccfeat_section = f"""
## 十二、特徵診斷：把過去 20 日收盤到收盤變異數當新特徵

`src/features.py::hist_cc_var_20d`：跟 HAR 三項寫法完全平行（`ctx.rolling_mean`），
過去 20 個交易日（t-20..t-1）收盤到收盤（含隔夜、含股利）報酬平方的平均，`prev_close` cutoff，
自動通過第五節提到的時點正確性測試。加進 HAR+殘差XGB 的樹特徵集（15 個變成 16 個），其他都不變。
這節同樣不牽涉 `historical_vol_weight`，結果獨立成立：

{_md_table(cc_tbl)}

夏普 {cc_tbl.iloc[0]['sharpe']} 跟原本沒加這個特徵的版本（{ev_base['sharpe']:.4f}）幾乎一樣
（p = {cc_tbl.iloc[0]['p vs base (no cc feature)']}，不顯著）。permutation importance 顯示樹
幾乎沒用這個特徵：`hist_cc_var_20d` 在 {cc_n} 個特徵裡重要度排第 {cc_rank} 名（`har_logrv_1d`、
`har_logrv_5d` 仍然主導）——不是特徵沒被試到，是樹在殘差目標上判斷它加不了什麼邊際訊號，可能因為
跟 `har_logrv_5d/22d` 高度相關、資訊已經被吃掉了。**結論：加這個特徵對 HAR+殘差XGB 自己的表現
沒有顯著影響**（原本「能不能追上 hist20」的框架已經不成立，見結論的更正說明）。
"""

    # ---- return-timing vs risk-timing diagnostic
    ret_cc_all = inputs["daily"]["ret_cc"]
    timing_rows = []
    for name in ["har", "har_resid_xgb"]:
        res = B.regress_weight_diff_on_return(
            bts["hist20"]["weight_target"], bts[name]["weight_target"], ret_cc_all)
        timing_rows.append({"A": "20d hist. vol", "B": LABEL[name],
                           "n": res["n"], "slope": round(res["slope"], 5),
                           "slope_se": round(res["slope_se"], 5),
                           "slope_t": round(res["slope_t"], 2),
                           "slope_p": round(res["slope_p"], 4)})
    timing_tbl = pd.DataFrame(timing_rows)

    diff_hist_har = (bts["hist20"]["weight_target"] -
                     bts["har"]["weight_target"]).dropna()
    idx_t = diff_hist_har.index.intersection(ret_cc_all.dropna().index)
    tercile = pd.qcut(diff_hist_har.loc[idx_t], 3,
                      labels=["hist20 much more cautious", "similar",
                             "hist20 much more aggressive"])
    tercile_ret = ret_cc_all.loc[idx_t].groupby(tercile, observed=True).agg(
        ["mean", "count"]).round(6)
    tercile_ret = tercile_ret.reset_index(names="hist20 vs HAR weight")

    track_cmp = scores.loc[["hist20", "har", "har_resid_xgb"],
                          ["ann_vol", "tracking_error", "sharpe"]].round(4)
    track_cmp.index = [LABEL[n] for n in track_cmp.index]
    track_cmp = track_cmp.reset_index(names="source")

    timing_section = f"""
## 十三、報酬擇時 vs 風險預測——以及一個 bug 的發現與修正

這節原本的動機（第十二節結尾）是驗證「模型/估計風險」跟「報酬擇時」哪個能解釋 20 日歷史波動
顯著贏 HAR。兩個檢查做下去，第二個檢查本身意外揪出一個前視偏誤（look-ahead bug），推翻了
「顯著贏」這個前提，所以這節如實記錄發現與修正的過程，而不是原本設想的「哪個假說對」。

**(a) 追蹤誤差**：如果 20 日歷史波動贏在「風險控制更準」，它的年化波動應該比 HAR 系列更貼近
15% 目標。實際上相反：

{_md_table(track_cmp)}

HAR 的追蹤誤差（{track_cmp.loc[track_cmp['source']==LABEL['har'],'tracking_error'].iloc[0]:+.4f}）
比 20 日歷史波動（{track_cmp.loc[track_cmp['source']==LABEL['hist20'],'tracking_error'].iloc[0]:+.4f}）
更接近 0——HAR 系列把實際波動控制得更準。這個方向的觀察不受下面的 bug 影響（風險控制準不準，
跟賺不賺錢是兩回事，本來就可能背離），只是修正後兩者夏普已經沒有顯著差異，這裡不再需要拿它來
解釋一個「贏」。

**(b) 部位差異對「當天報酬」迴歸——這一步發現了 bug**：把 20 日歷史波動的部位減去 HAR（或
HAR+殘差XGB）的部位，對兩者共用的當天實際收盤到收盤報酬（`ret_cc`）做 OLS，Newey-West 標準誤
（`src/backtest.py::regress_weight_diff_on_return`）。第一次跑出來斜率顯著為正（p≈0.006，
20 日歷史波動比 HAR 保守的日子報酬顯著更差），看起來是個乾淨的「報酬擇時」故事。但這個結果不
合理地強——检查 `src/strategies.py::historical_vol_weight` 的實作才發現：20 日滾動視窗**沒有
排除當天自己**，用 `ret_cc.rolling(20).var()` 而不是 `ret_cc.shift(1).rolling(20).var()`。
也就是說 t 日的權重用到了 t 日自己的收盤到收盤報酬，而這個權重接著在回測裡被拿去賺的正是 t 日
的報酬——用還沒發生（決策當下）的資訊決定當天的部位。因為部位差 = hist20 部位 − HAR 部位，而
hist20 部位本身就摻了一點 t 日報酬的資訊，這個迴歸自然會測到「顯著」的斜率，但那不是報酬擇時
能力，是資料洩漏。加一個 `.shift(1)` 修好之後（測試見
`tests/test_strategies.py::test_historical_vol_weight_excludes_the_same_day_return`），
重跑同一個迴歸：

{_md_table(timing_tbl)}

斜率不再顯著（p ≈ {timing_tbl.iloc[0]['slope_p']:.2f}）。三等分依然列出來，但現在只是描述性的、
不代表因果或可預測性：

{_md_table(tercile_ret)}

**修正後的結論：沒有證據支持「20 日歷史波動贏在報酬擇時」，因為 hist20 修正 bug 之後根本沒有
顯著贏。** 上一輪的「動量/槓桿效應」敘事整個是這個 bug 的產物——這是本週最重要的一次自我修正，
過程和影響已經寫進 `week05_review_notes.md` 最前面。
"""

    figs = {"eq": fig_equity_drawdown(bts, SOURCES),
            "track": fig_tracking(bts, SOURCES, cfg.vol_target)}

    resid_vs_har_p = dm_tbl.loc[
        (dm_tbl["A"] == LABEL["har_resid_xgb"]) & (dm_tbl["B"] == LABEL["har"]),
        "p-value"].iloc[0]
    resid_vs_hist20_p = dm_tbl.loc[
        (dm_tbl["A"] == LABEL["har_resid_xgb"]) & (dm_tbl["B"] == LABEL["hist20"]),
        "p-value"].iloc[0]
    resid_vs_hist20_diff = dm_tbl.loc[
        (dm_tbl["A"] == LABEL["har_resid_xgb"]) & (dm_tbl["B"] == LABEL["hist20"]),
        "sharpe(A)-sharpe(B)"].iloc[0]

    md = f"""# Week 5 波動預測接部位管理

模型不變（見 `week04_xgb.md`、`week04_review_notes.md`）；本週只把預測轉成部位、回測、評估。
程式：`src/strategies.py`（部位規則）、`src/backtest.py`（執行、成本、評估、block bootstrap）。
只讀 `data/processed/predictions/`，不重新訓練任何模型。

## 結論

**重要更正（見 `week05_review_notes.md` 開頭）**：`src/strategies.py::historical_vol_weight`
原本有前視偏誤——20 日滾動視窗包含了「當天自己」的收盤到收盤報酬，但這個權重接著被拿去賺當天
的報酬，等於用還沒發生的資訊決定當天的部位。修好之後（改成嚴格用 t-20..t-1，排除 t 本身），
20 日歷史波動的夏普從原本的 1.4715 掉到 {scores.loc['hist20','sharpe']:.4f}，「20 日歷史波動
顯著贏 HAR」和第十三節「報酬擇時」的結論都不成立了——那個顯著性本身就是 bug 造成的。以下是修正
後的結論。

**波動預測的改進（Week 4 的 HAR+殘差XGB，日內 RV 的 QLIKE 比 HAR 低 7.5%）沒有讓部位管理
策略變好，但也沒有變差——修正後，HAR、HAR+殘差XGB、20 日歷史波動三者的夏普統計上沒有差異。**

* HAR+殘差XGB 對 HAR：夏普 {scores.loc['har_resid_xgb','sharpe']:.4f} 對
  {scores.loc['har','sharpe']:.4f}，幾乎打平，block bootstrap p = {resid_vs_har_p}（不顯著）。
  這個結論從一開始就沒變過，不受這次修正影響。
* **HAR+殘差XGB 對 20 日歷史波動：夏普 {scores.loc['har_resid_xgb','sharpe']:.4f}
  對 {scores.loc['hist20','sharpe']:.4f}，差 {resid_vs_hist20_diff:+.3f}，p = {resid_vs_hist20_p}
  ——沒有顯著差異**。修正前這裡是 p=0.001（顯著更差），是 bug 造成的假訊號。全部 9 組
  σ目標 × 槓桿上限（第九節）數字上大多是 20 日歷史波動略高，但差距都在雜訊範圍內。
* 誤差較低（QLIKE 更好）沒有轉成策略顯著更好，但也沒有更差——QLIKE 跟夏普本來就是不同的
  目標函數，原因見第三節「兩層評估總表」與討論。
* 波動目標確實把最大回撤從買進持有的 {scores.loc['buy_hold','max_drawdown']:.1%} 壓到約
  {scores.loc['har','max_drawdown']:.1%}（HAR 系列）、{scores.loc['hist20','max_drawdown']:.1%}
  （20 日歷史波動）——這個風險管理的價值不受這次修正影響，仍然穩健成立。
* 第十、十一、十二節（平滑／換目標／加特徵）的診斷數字本身沒有錯（都不牽涉 `historical_vol_
  weight`），但原本「能不能追上 20 日歷史波動」的框架已經不成立，因為現在沒有顯著的差距需要
  追——這些小節保留下來當作獨立的模型診斷結果，但不再是「解謎」的敘事。第十三節「報酬擇時」的
  結論是本次修正前最主要的誤判，已經整節改寫成如實記錄這次修正的過程。

完整解讀見 `week05_review_notes.md`。

## 一、部位規則與設定

* 波動目標部位：`w(t) = min(vol_target / σ̂(t), leverage_cap)`，預設 σ目標 = 15%、槓桿上限 = 1.5。
* 尺度校準：日內 RV 低估收盤到收盤風險（漏掉隔夜跳空），每折用**訓練段**估
  `E[rv_on]/E[rv]` 比例係數，套用到該折的測試段（`src/strategies.py::scale_ratios`）。
* 完美預知直接用實際 `rv_on`（已實現的收盤到收盤變異數代理），不需要校準。
* 20 日歷史波動不需要模型或校準：對收盤到收盤報酬取過去 20 個交易日（t-20..t-1，**不含當天**，
  否則會用到當天自己還沒發生的報酬去決定當天的部位）已實現變異數，本身就在收盤到收盤尺度上。
* 主設定：t 日收盤算預測、收盤調整部位、賺 t 到 t+1 收盤報酬（`timing="close"`）；換手成本
  單邊 1 bp，無 no-trade 門檻。
* 比較樣本：HAR 與 HAR+殘差XGB 都有預測的共同日期，{scores.loc['har','n']:.0f} 天。

## 二、五個預測來源

`不調整`（買進持有 QQQ）、`20 日歷史波動`（主要對手）、`HAR`、`HAR+殘差XGB`（主模型）、
`完美預知`（用實際隔日 `rv_on`，理論上限）。

## 三、兩層評估總表：QLIKE（預測品質）vs 策略結果

**注意這裡的 QLIKE 跟 `week04_xgb.md` 的 QLIKE 不是同一個數字**：Week 4 比較的是「日內 RV
的 log 預測」對「當天的日內 RV」（HAR {QLIKE['har']}、HAR+殘差XGB {QLIKE['har_resid_xgb']}）；
這裡比較的是「校準後的收盤到收盤變異數預測」對「實際收盤到收盤變異數（`rv_on`）」，是部位管理
真正在用的那個數字，也是唯一能把 20 日歷史波動、完美預知一起放進同一張表比較的方式。兩者原始
排序一致（HAR+殘差XGB 都比 HAR 好），但拿來跟 20 日歷史波動比就會出現下面這個反差。

{_md_table(two_layer.reset_index(names="source"))}

**這就是「誤差較低但策略沒改善」的具體樣子**：HAR+殘差XGB 的 QLIKE 比 HAR 低（不論用哪個定義），
但兩者的夏普幾乎相同；20 日歷史波動的 QLIKE 明顯比兩個模型都差，夏普數字上略高一點，但差距在
統計雜訊範圍內（第六節）。QLIKE 評的是「變異數這個數字猜得準不準」，策略夏普評的是「部位隨時間
怎麼變動、換手多少、跟報酬的時間對不對得上」——兩者不是同一件事，這張表本身就是最直接的證據：
QLIKE 排名跟夏普排名對不上。

![equity and drawdown]({figs['eq']})

## 四、追蹤誤差

![tracking]({figs['track']})

灰色虛線是 15% 目標。所有波動目標策略都貼著目標波動，買進持有（灰線）明顯偏離且波動的波動本身
也大很多。

## 五、逐年夏普與危機切片

逐年夏普：

{_md_table(yearly)}

covid_2020、tariff_2025 切片夏普：

{_md_table(crisis_sharpe.reset_index())}

同切片最大回撤：

{_md_table(crisis_dd.reset_index())}

兩個切片本身報酬都是負的（尤其 tariff_2025），所以切片夏普全面為負；波動目標策略的價值在切片
內的最大回撤明顯小於買進持有，不在切片夏普轉正。HAR 與 HAR+殘差XGB 在兩個切片內互有輸贏、差距
都很小，跟 Week 4 「HAR+殘差XGB 在 covid_2020、tariff_2025 的 QLIKE 都比 HAR 好」的結論方向
不一致——見 `week05_review_notes.md` 的討論。

## 六、夏普差異顯著性（block bootstrap，區塊 20 天，3000 次）

{_md_table(dm_tbl)}

## 七、策略 N：波動預測 vs 加上溢酬微調

溢酬 = VXN 隱含年化變異數 − 模型年化變異數預測；低於訓練段門檻（25 分位數）時部位打
{cfg.derate} 折。

{_md_table(stratn_tbl)}

微調後的夏普、報酬都略低，最大回撤略淺、追蹤誤差轉負（波動比目標低一些）；對照第六節最後一列，
p = {dm_tbl.iloc[-1]['p-value']}，沒有顯著幫助。
{rf_section}
## 九、穩健性

執行時點／成本／門檻：

{_md_table(rob_tbl)}

HAR+殘差XGB 對 HAR 在每一種變化下都是打平（差距都在 0.01–0.05 夏普點以內），這個結論穩健。

σ目標 × 槓桿上限（9 組）：

{_md_table(grid)}

HAR+殘差XGB 在 9 組裡有 {n_beats_har}/9 贏過 HAR、{n_beats_hist20}/9 贏過 20 日歷史波動。
`resid_beats_hist20` 幾乎全部是 False，但配合第六節的顯著性檢定看，這只代表「數字上沒有一組
贏」，不代表差距顯著——20 日歷史波動在這裡的小幅數字優勢多半在雜訊範圍內（見結論的更正說明）。

## 十、平滑診斷：換手率對夏普的影響（背景：這節當初是為了解釋一個後來發現是 bug 的差距）

這節當初的動機是「20 日歷史波動顯著贏，是不是因為它比較平滑」；後來發現那個顯著差距本身是
`historical_vol_weight` 的前視偏誤造成的（見結論的更正說明），所以「追上 hist20」這個框架已
經不成立。以下數字本身沒有錯，只是重新定位成一個獨立的問題：把 HAR+殘差XGB 的預測本身取過去
N 天平均（`src/strategies.py::smooth_variance_forecast`，只用該模型自己過去的預測、不引入新
資訊），換手率會不會影響夏普：

{_md_table(smooth_tbl)}

換手率從原始的 11.2% 壓到 10 天平均的 2.0%，夏普從 1.1375 微升到 1.1674（對未平滑版
p = {smooth_tbl.iloc[0]['p vs unsmoothed']}，不顯著）；窗口拉到 20、40 天，換手率降得更低，
但 QLIKE 變差、夏普不升反降（40 天版本連 HAR 都輸）。**結論（修正後）：換手率高低對 HAR+殘差
XGB 自己的夏普沒有顯著影響**——原本「靠平滑追上 20 日歷史波動」的框架不再適用，因為沒有顯著
差距需要追。

## 十一、隔夜資訊診斷：換成含隔夜的目標會更好嗎？

這節當初也是為了解釋「20 日歷史波動顯著贏」而做的（假說：日內 RV 目標漏掉隔夜跳空，靠一個
每折固定的尺度係數補回收盤到收盤，但隔夜變異數佔比逐月變動很大——
{on_share_monthly.min():.1%}–{on_share_monthly.max():.1%}，標準差
{on_share_monthly.std():.1%}，逐年均值從 2014–2018 年的約
{on_share_yearly.loc[2014:2018].mean():.1%} 緩升到 2025–2026 年的約
{on_share_yearly.loc[2025:2026].mean():.1%}，固定係數補不到這個變動）；那個「顯著差距」後來
確認是 bug，但這節的實驗本身（換成 `rv_on` 目標直接訓練，不用尺度係數）不牽涉
`historical_vol_weight`，結果依然有效、獨立成立：

{_md_table(on_tbl)}

**直接預測 `rv_on` 比原本「預測 `rv`、乘固定係數」的做法明顯更差**：`har_resid_xgb_rvon` 對
原本 rv 目標版本 p = {on_tbl.iloc[1]['p vs rv-target']}（顯著更差，夏普
{on_tbl.iloc[1]['sharpe']} 對 {ev_base['sharpe']:.4f}）。QLIKE 同樣變差：直接預測 `rv_on` 的
QLIKE（HAR {on_tbl.iloc[0]['QLIKE']}、HAR+殘差XGB {on_tbl.iloc[1]['QLIKE']}）都比原本「預測
`rv`、乘上固定係數」的版本（HAR {qlike_har_rv}、HAR+殘差XGB {qlike_resid_rv}）差。**這個結論
不受 hist20 的 bug 影響，站得住**：固定尺度係數雖然無法反映隔夜佔比的月度變動，但直接對雜訊
遠大於日內 RV 的隔夜變異數建模，引入的預測雜訊比省下的那點偏誤更傷——隔夜報酬平方是單一一根
「跳空」，比累積一整天的已實現變異數雜訊大得多，HAR 的三個落後項對它的解釋力明顯更差。
{ccfeat_section}{timing_section}"""
    out_path = REPORTS / "week05_position.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")
    print(two_layer)


if __name__ == "__main__":
    main()
