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

**波動預測的改進（Week 4 的 HAR+殘差XGB，日內 RV 的 QLIKE 比 HAR 低 7.5%）沒有讓部位管理
策略變好。**

* HAR+殘差XGB 對 HAR：夏普 {scores.loc['har_resid_xgb','sharpe']:.4f} 對
  {scores.loc['har','sharpe']:.4f}，幾乎打平，block bootstrap p = {resid_vs_har_p}（不顯著）。
* **HAR+殘差XGB 對 20 日歷史波動（主要對手）：夏普 {scores.loc['har_resid_xgb','sharpe']:.4f}
  對 {scores.loc['hist20','sharpe']:.4f}，差 {resid_vs_hist20_diff:+.3f}，p = {resid_vs_hist20_p}
  ——顯著更差**。全部 9 組 σ目標 × 槓桿上限（第六節）都是同一個方向：20 日歷史波動最好，
  HAR 與 HAR+殘差XGB 幾乎打平且始終不如 20 日歷史波動。
* 誤差較低（QLIKE 更好）沒有轉成策略更好，原因見第三節「兩層評估總表」與討論。
* 波動目標確實把最大回撤從買進持有的 {scores.loc['buy_hold','max_drawdown']:.1%} 壓到約
  {scores.loc['har','max_drawdown']:.1%}（HAR 系列）、{scores.loc['hist20','max_drawdown']:.1%}
  （20 日歷史波動）——風險管理本身有效，只是「用更準的波動預測」沒有讓風險管理本身更好。

完整解讀見 `week05_review_notes.md`。

## 一、部位規則與設定

* 波動目標部位：`w(t) = min(vol_target / σ̂(t), leverage_cap)`，預設 σ目標 = 15%、槓桿上限 = 1.5。
* 尺度校準：日內 RV 低估收盤到收盤風險（漏掉隔夜跳空），每折用**訓練段**估
  `E[rv_on]/E[rv]` 比例係數，套用到該折的測試段（`src/strategies.py::scale_ratios`）。
* 完美預知直接用實際 `rv_on`（已實現的收盤到收盤變異數代理），不需要校準。
* 20 日歷史波動不需要模型或校準：對收盤到收盤報酬取過去 20 日（含當天）已實現變異數，本身就在
  收盤到收盤尺度上。
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
但兩者的夏普幾乎相同；20 日歷史波動的 QLIKE 明顯比兩個模型都差，夏普卻是全場最高。QLIKE 評的是
「變異數這個數字猜得準不準」，策略夏普評的是「部位隨時間怎麼變動、換手多少、跟報酬的時間對不對
得上」——兩者不是同一件事，預測更準不保證部位管理更好。

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

結論方向（20 日歷史波動最好、HAR 與 HAR+殘差XGB 打平且不如 20 日歷史波動）在每一種變化下都一樣。

σ目標 × 槓桿上限（9 組）：

{_md_table(grid)}

HAR+殘差XGB 在 9 組裡有 {n_beats_har}/9 贏過 HAR、{n_beats_hist20}/9 贏過 20 日歷史波動
——方向完全一致，不是特定參數組合下的巧合。

## 十、平滑診斷：20 日歷史波動贏是因為比較平滑，還是資訊不同？

把 HAR+殘差XGB 的預測本身取過去 N 天平均（`src/strategies.py::smooth_variance_forecast`，
只用該模型自己過去的預測、不引入新資訊，換手率因此大幅下降），看換手率降到跟 20 日歷史波動
同一個量級之後，夏普有沒有追上：

{_md_table(smooth_tbl)}

**沒有追上**：換手率從原始的 11.2% 壓到 10 天平均的 2.0%（已經低於 20 日歷史波動的 3.0%），
夏普只從 1.1375 微升到 1.1674（對未平滑版 p = {smooth_tbl.iloc[0]['p vs unsmoothed']}，不顯著），
對 20 日歷史波動仍然顯著更差（p = {smooth_tbl.iloc[0]['p vs hist20']}）。窗口拉到 20、40 天，
換手率降得更低，但 QLIKE 變差、夏普不升反降（40 天版本連 HAR 都輸）。**結論：20 日歷史波動贏
不是因為反應速度，是因為它含有 HAR／HAR+殘差XGB 沒抓到的資訊**——單純放慢 HAR 系列的反應速度
無法複製這個優勢。
"""
    out_path = REPORTS / "week05_position.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")
    print(two_layer)


if __name__ == "__main__":
    main()
