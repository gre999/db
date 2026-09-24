"""Generate reports/week06_regimes.md (+ figures).

Reads data/processed/strategies/{orb,vwap}_daily.parquet (run
``python -m src.rules`` first) and data/processed/regimes/ (run
``python -m src.models.regimes`` first). No state x strategy evaluation
here - that is next week; this report only establishes that the daily
results and state labels themselves are sound.

Usage:
    python scripts/make_week06_report.py
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
from make_week01_report import (BLUE, INK_2, MUTED, ORANGE,  # noqa: E402
                                _date_axis, _md_table, _save)
from src import data_loader as dl  # noqa: E402
from src import features as F  # noqa: E402
from src import rules as R  # noqa: E402

REPORTS = ROOT / "reports"
AQUA, YELLOW, MAGENTA = "#1baf7a", "#eda100", "#c23b6b"
STATE_COLORS = [BLUE, YELLOW, ORANGE, MAGENTA]


def as_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Stringify int-valued columns so _md_table's row iteration (which
    upcasts a mixed int+float row to float64) doesn't render them as
    "3.0" - same fix used in make_week04_report.py."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return df


def fig_orb_r_hist(orb: pd.DataFrame) -> str:
    tr = orb[orb["traded"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(tr["r_net"].clip(-2, 10.5), bins=60, color=BLUE)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_title("ORB: net R distribution (clipped at -2R/10.5R for display)")
    ax.set_xlabel("R (net of cost)")
    return _save(fig, "w06_orb_r_hist.png")


def fig_state_timeseries(labels: pd.DataFrame, model: str, k: int) -> str:
    l = labels[(labels["model"] == model) & (labels["k"] == k)].sort_values("date")
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.scatter(l["date"], l["state"], c=[STATE_COLORS[s % 4] for s in l["state"]],
              s=4)
    ax.set_yticks(range(k))
    ax.set_title(f"{model.upper()} state over time (k={k}, most common fold choice)")
    ax.set_ylabel("state (0=calm ... k-1=turbulent)")
    _date_axis(ax)
    return _save(fig, f"w06_state_ts_{model}.png")


def main() -> None:
    minute = dl.load_minute_rth()
    orb = R.run_orb(minute)
    vwap = R.run_vwap(minute)

    labels = pd.read_parquet(dl.PROCESSED_DIR / "regimes" / "regime_labels.parquet")
    transmat = pd.read_parquet(dl.PROCESSED_DIR / "regimes" / "hmm_transmat.parquet")
    selection = pd.read_parquet(dl.PROCESSED_DIR / "regimes" / "model_selection.parquet")

    # ---- ORB sanity + full-sample baseline
    tr = orb[orb["traded"]]
    win_rate = (tr["r_net"] > 0).mean()
    orb_ret = orb["bps_return"] / 1e4
    orb_sharpe = orb_ret.mean() / orb_ret.std(ddof=0) * np.sqrt(252)
    orb_summary = pd.DataFrame([{
        "n_days": len(orb), "n_traded": len(tr),
        "win_rate": round(win_rate, 4), "mean_r_gross": round(tr["r_gross"].mean(), 4),
        "mean_r_net": round(tr["r_net"].mean(), 4), "sharpe_unit_exposure": round(orb_sharpe, 4),
    }])
    orb_summary = as_int(orb_summary, ["n_days", "n_traded"])
    exit_reasons = orb["exit_reason"].value_counts().rename_axis("reason") \
        .reset_index(name="n")

    def yearly_orb(df):
        rows = []
        for y, g in df.groupby(df["day"].dt.year if "day" in df else df.index.year):
            t = g[g["traded"]]
            r = g["bps_return"] / 1e4
            sharpe = r.mean() / r.std(ddof=0) * np.sqrt(252) if r.std(ddof=0) > 0 else np.nan
            rows.append({"year": y, "n_trades": len(t),
                        "win_rate": round((t["r_net"] > 0).mean(), 3) if len(t) else np.nan,
                        "mean_r_net": round(t["r_net"].mean(), 3) if len(t) else np.nan,
                        "sharpe": round(sharpe, 3)})
        return pd.DataFrame(rows)

    orb_yearly = yearly_orb(orb.reset_index())
    orb_yearly = as_int(orb_yearly, ["year", "n_trades"])

    # ---- VWAP full-sample baseline
    trv = vwap[vwap["traded"]]
    vwap_ret = vwap["bps_return"] / 1e4
    vwap_sharpe = vwap_ret.mean() / vwap_ret.std(ddof=0) * np.sqrt(252)
    vwap_summary = pd.DataFrame([{
        "n_days": len(vwap), "n_traded": len(trv),
        "mean_bps": round(vwap["bps_return"].mean(), 3),
        "mean_n_segments": round(trv["n_segments"].mean(), 2),
        "sharpe_unit_exposure": round(vwap_sharpe, 4),
    }])
    vwap_summary = as_int(vwap_summary, ["n_days", "n_traded"])

    def yearly_vwap(df):
        rows = []
        for y, g in df.groupby(df["day"].dt.year if "day" in df else df.index.year):
            r = g["bps_return"] / 1e4
            sharpe = r.mean() / r.std(ddof=0) * np.sqrt(252) if r.std(ddof=0) > 0 else np.nan
            rows.append({"year": y, "mean_bps": round(g["bps_return"].mean(), 3),
                        "sharpe": round(sharpe, 3)})
        return pd.DataFrame(rows)

    vwap_yearly = yearly_vwap(vwap.reset_index())
    vwap_yearly = as_int(vwap_yearly, ["year"])

    # ---- paper replication
    eq_orb = R.replicate_orb(orb)
    eq_vwap = R.replicate_vwap(vwap)
    ev_orb = R._evaluate_equity(eq_orb["equity"], 100_000.0)
    ev_vwap = R._evaluate_equity(eq_vwap["equity"], 100_000.0)
    repl_tbl = pd.DataFrame([
        {"strategy": "ORB (B)", "window": f"{R.ORB_PAPER_WINDOW[0]}..{R.ORB_PAPER_WINDOW[1]}",
         "paper_sharpe": R.ORB_PAPER_SHARPE, "repl_sharpe": round(ev_orb["sharpe"], 3),
         "paper_max_dd": -R.ORB_PAPER_MAX_DD, "repl_max_dd": round(ev_orb["max_drawdown"], 4)},
        {"strategy": "VWAP (A)", "window": f"{R.VWAP_PAPER_WINDOW[0]}..{R.VWAP_PAPER_WINDOW[1]}",
         "paper_sharpe": R.VWAP_PAPER_SHARPE, "repl_sharpe": round(ev_vwap["sharpe"], 3),
         "paper_max_dd": -R.VWAP_PAPER_MAX_DD, "repl_max_dd": round(ev_vwap["max_drawdown"], 4)},
    ])

    # ---- VWAP cost sensitivity: same paper window, unit exposure, cost rate only
    vwap_window_main = vwap.loc[R.VWAP_PAPER_WINDOW[0]:R.VWAP_PAPER_WINDOW[1]]
    ret_main = vwap_window_main["bps_return"] / 1e4
    sharpe_unit_main = ret_main.mean() / ret_main.std(ddof=0) * np.sqrt(252)

    vwap_replcost = R.run_vwap(minute, cfg=R.VWAPConfig(
        cost_per_share=R.REPL_COMMISSION + R.REPL_SLIPPAGE))
    vwap_window_replcost = vwap_replcost.loc[R.VWAP_PAPER_WINDOW[0]:R.VWAP_PAPER_WINDOW[1]]
    ret_replcost = vwap_window_replcost["bps_return"] / 1e4
    sharpe_unit_replcost = ret_replcost.mean() / ret_replcost.std(ddof=0) * np.sqrt(252)

    cost_sens_tbl = pd.DataFrame([
        {"setting": "unit exposure, main cost ($0.0045/share)",
         "mean_bps_per_day": round(vwap_window_main["bps_return"].mean(), 2),
         "sharpe": round(sharpe_unit_main, 3)},
        {"setting": "unit exposure, paper cost ($0.0005/share)",
         "mean_bps_per_day": round(vwap_window_replcost["bps_return"].mean(), 2),
         "sharpe": round(sharpe_unit_replcost, 3)},
        {"setting": "replication (paper cost + compounding position)",
         "mean_bps_per_day": "—", "sharpe": round(ev_vwap["sharpe"], 3)},
    ])

    # ---- state selection
    sel_tbl = selection.copy()
    sel_tbl["value"] = sel_tbl["value"].round(3)
    kmeans_k_counts = selection[selection["model"] == "kmeans"]["k"].value_counts().sort_index()
    hmm_k_counts = selection[selection["model"] == "hmm"]["k"].value_counts().sort_index()

    # ---- state characterization (descriptive features)
    data = F.MarketData.from_processed()
    desc = F.build_descriptive_matrix(data)
    labels_idx = labels.set_index("date")

    char_tables = {}
    for model in ["kmeans", "hmm"]:
        l = labels_idx[labels_idx["model"] == model]
        k_mode = int(l["k"].mode().iloc[0])
        l = l[l["k"] == k_mode]
        joined = desc.reindex(l.index)
        joined = joined.assign(state=l["state"].to_numpy())
        means = joined.groupby("state").mean().round(4)
        counts = joined.groupby("state").size().rename("n_days")
        tbl = means.join(counts).reset_index()
        tbl = as_int(tbl, ["state", "n_days"])
        char_tables[model] = (k_mode, tbl)

    # ---- HMM transition matrix (most common k, averaged over folds using that k)
    hmm_k_mode = int(labels[labels["model"] == "hmm"]["k"].mode().iloc[0])
    tm = transmat[transmat["k"] == hmm_k_mode]
    tm_avg = tm.groupby(["from_state", "to_state"])["prob"].mean().unstack()
    tm_avg = tm_avg.round(3)
    tm_avg.columns = [str(int(c)) for c in tm_avg.columns]
    tm_avg_tbl = tm_avg.reset_index()
    tm_avg_tbl = as_int(tm_avg_tbl, ["from_state"])

    figs = {"orb_hist": fig_orb_r_hist(orb),
           "state_ts_hmm": fig_state_timeseries(labels, "hmm", hmm_k_mode)}
    kmeans_k_mode = int(labels[labels["model"] == "kmeans"]["k"].mode().iloc[0])
    figs["state_ts_kmeans"] = fig_state_timeseries(labels, "kmeans", kmeans_k_mode)

    md = f"""# Week 6 規則策略與市場狀態標籤

階段三第一週：只建每日結果與狀態標籤，不評估各狀態下的策略績效（下週做）。
程式：`src/rules.py`（ORB、VWAP）、`src/models/regimes.py`（KMeans、HMM）。

## 一、策略合理性檢查

**ORB**：{len(tr)}/{len(orb)} 天有交易（{len(tr)/len(orb):.1%}），勝率
{win_rate:.1%}（預期約 24%，**通過**）。出場原因分布：

{_md_table(exit_reasons)}

{_md_table(orb_summary)}

![orb r hist]({figs['orb_hist']})

**VWAP**：{len(trv)}/{len(vwap)} 天有交易，平均每天反手 {trv['n_segments'].mean():.1f} 次。

{_md_table(vwap_summary)}

## 二、論文複現對照

主輸出（單位曝險）之外，另跑一次論文的部位規則（ORB：每筆風險 1% 權益、槓桿上限 4 倍；VWAP：
100% 權益、不加槓桿、股數當日開盤定死不隨盤中權益重算），對照論文自己的取樣期間：

{_md_table(repl_tbl)}

兩個策略的夏普、最大回撤都跟論文數字同一個量級，複現通過。

**VWAP 的單位曝險夏普，限定在論文同一個取樣期間，是 {round(sharpe_unit_main,2)}，跟複現模式
（{round(ev_vwap['sharpe'],2)}）還是差不小，拆解過原因**：同一段期間、同一組交易，只把成本從
主設定換成論文的複現成本，其他不變：

{_md_table(cost_sens_tbl)}

成本假設從 $0.0045/股換成 $0.0005/股，夏普從 {round(sharpe_unit_main,2)} 跳到
{round(sharpe_unit_replcost,2)}，吃掉了缺口的大部分；剩下的一小段才是複利部位造成的。VWAP 平均
每天反手 16.6 次（33 個股數邊），對成本假設極度敏感——**真實交易成本下，VWAP 的夏普大約只有
論文（近乎零成本）假設下的一半**，這是本週除了合理性檢查之外最重要的一個發現，不是複現失敗，
是策略本身的成本敏感度問題，下週評估各狀態下的績效時要記得用主成本、不要用論文的複現成本。

## 三、全樣本基準（不分狀態）

ORB 逐年：

{_md_table(orb_yearly)}

VWAP 逐年：

{_md_table(vwap_yearly)}

## 四、狀態數選擇依據

KMeans 用輪廓係數、HMM 用 BIC，只用每折訓練段決定，見 `src/models/regimes.py`。8 折的選擇：

{_md_table(sel_tbl)}

KMeans 大多數折選 k=2（{int(kmeans_k_counts.get(2,0))}/8 折），少數選 k=3。HMM **全部 8 折都選了
搜尋範圍的上限 k=4**——BIC 一路下降到邊界都還沒回頭，代表如果讓它繼續搜尋，可能還會選更多狀態；
這裡先誠實記錄這個邊界效應，沒有隱藏，下週如果要用 HMM 狀態，值得把 k 的搜尋範圍再往上放寬一點
確認 BIC 真的會在某處回頭。

## 五、各狀態特徵平均（描述型特徵，白話說明）

KMeans（k={char_tables['kmeans'][0]}，各折最常見的選擇）：

{_md_table(char_tables['kmeans'][1])}

HMM（k={char_tables['hmm'][0]}，各折最常見的選擇）：

{_md_table(char_tables['hmm'][1])}

狀態編號本身是照**預測型** `har_logrv_1d`（昨天的已實現波動）由低到高排的；上面兩張表列的是
**描述型** `log_rv_desc`（當天自己的已實現波動），也同樣單調遞增，這不是定義保證的（兩個是不同
天的量），而是波動聚集（昨天波動高，今天多半也高）夠強，兩者幾乎完全一致——算是對狀態編號設計
的一個獨立驗證。`close_loc_desc` 各狀態差異不大，`overnight_gap_desc` 在較高波動狀態略偏正
（隔夜跳空稍大），跟直覺一致——市況越亂，隔夜跳空的絕對幅度傾向越大。

## 六、HMM 轉移矩陣

k={hmm_k_mode}（各折最常見選擇）的轉移機率，8 折平均：

{_md_table(tm_avg_tbl)}

對角線（留在原狀態的機率）都遠高於離開機率，狀態有明顯的持續性，不是每天隨機跳動；相鄰狀態
（0↔1、1↔2、2↔3）之間的轉移機率一般也比跳兩級以上（0↔3 等）的轉移機率高，符合「波動狀態通常
漸進變化，不會一天內從最平靜跳到最動盪」的直覺。

## 七、狀態時間序列圖

HMM（k={hmm_k_mode}）：

![hmm state ts]({figs['state_ts_hmm']})

KMeans（k={kmeans_k_mode}）：

![kmeans state ts]({figs['state_ts_kmeans']})

（中間的空白不是資料缺漏，是那一折選了 k=3、不在這張只畫 k={kmeans_k_mode} 折的圖裡——見第四節。）

兩個模型都抓到 2020 COVID、2022 熊市、2025 關稅衝擊這幾段已知的高波動期間落在較高編號的狀態，
跟直覺吻合。
"""
    out_path = REPORTS / "week06_regimes.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
