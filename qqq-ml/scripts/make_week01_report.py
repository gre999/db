"""Generate reports/week01_data_quality.md (+ figures and anomaly list).

Reads only data/processed/ (run ``python -m src.data_loader`` first).

Usage:
    python scripts/make_week01_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402

REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"

# Reference palette (light mode): categorical slots 1-2, text, surface, grid.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "figure.dpi": 110, "savefig.bbox": "tight",
})


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    return f"figures/{name}"


def fig_fill_hist(daily: pd.DataFrame) -> str:
    pct = daily["fill_ratio"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 100]
    counts = pd.cut(pct, bins, include_lowest=True).value_counts(sort=False)
    labels = ["≤0.1", "0.1–0.25", "0.25–0.5", "0.5–1", "1–2", "2–5", "5–10",
              "10–25", ">25"]
    ax.bar(labels, counts.to_numpy(), color=BLUE, width=0.8)
    for x, v in enumerate(counts.to_numpy()):
        if v:
            ax.text(x, v, f"{v:,}", ha="center", va="bottom", color=INK_2,
                    fontsize=8)
    ax.set_title("Daily RTH fill ratio: most sessions need almost no filling")
    ax.set_xlabel("Share of RTH minutes filled (%)")
    ax.set_ylabel("Sessions")
    ax.grid(axis="x", visible=False)
    return _save(fig, "fill_ratio_hist.png")


def _date_axis(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def fig_fill_time(daily: pd.DataFrame) -> str:
    pct = daily["fill_ratio"] * 100
    monthly = pct.resample("MS").mean()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.scatter(daily.index, pct, s=8, color=BLUE, alpha=0.4, linewidths=0)
    ax.plot(monthly.index + pd.Timedelta(days=14), monthly, color=BLUE,
            linewidth=2)
    cap = max(float(pct.quantile(0.995)) * 1.2, 0.5)
    n_above = int((pct > cap).sum())
    ax.set_ylim(0, cap)
    if n_above:
        ax.text(0.99, 0.97, f"{n_above} day(s) above {cap:.2f}% not shown",
                transform=ax.transAxes, ha="right", va="top", color=INK_2,
                fontsize=8)
    ax.set_title("Fill ratio over time (dots: days, line: monthly mean)")
    ax.set_ylabel("RTH minutes filled (%)")
    _date_axis(ax)
    return _save(fig, "fill_ratio_time.png")


def fig_anomalies_year(an: pd.DataFrame) -> str:
    by_year = an["day"].dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(by_year.index.astype(str), by_year.to_numpy(), color=BLUE,
           width=0.8)
    for x, v in enumerate(by_year.to_numpy()):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", color=INK_2,
                fontsize=8)
    ax.set_title("Flagged RTH minutes per year")
    ax.set_ylabel("Flagged minutes")
    ax.grid(axis="x", visible=False)
    return _save(fig, "anomalies_per_year.png")


def fig_close_check(daily: pd.DataFrame) -> str:
    bps = daily["close_vs_ibkr"] * 1e4
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.scatter(daily.index, bps, s=8, color=BLUE, alpha=0.6, linewidths=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Last RTH minute close vs IBKR daily close")
    ax.set_ylabel("Difference (bps)")
    _date_axis(ax)
    return _save(fig, "close_vs_daily.png")


def fig_raw_vs_adj(daily: pd.DataFrame) -> str:
    """Cumulative dividend adjustment: how far raw prices sit above adjusted."""
    d = daily.dropna(subset=["close_adj"])
    gap = (d["close"] / d["close_adj"] - 1) * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.step(d.index, gap, where="post", color=BLUE, linewidth=1.5)
    ax.set_title("Raw vs dividend-adjusted price gap (steps = ex-dividend dates)")
    ax.set_ylabel("Raw / adjusted - 1 (%)")
    _date_axis(ax)
    return _save(fig, "raw_vs_adjusted.png")


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.to_numpy()) + " |")
    return "\n".join(lines)


def main() -> None:
    meta = json.loads((dl.PROCESSED_DIR / "build_meta.json").read_text())
    daily = dl.load_daily()
    an = dl.load_anomalies()
    di = pd.read_parquet(dl.PROCESSED_DIR / "daily_ibkr.parquet")

    REPORTS.mkdir(exist_ok=True)
    an_out = an.copy()
    an_out["ts"] = an_out["ts"].dt.strftime("%Y-%m-%d %H:%M")
    an_out["day"] = an_out["day"].dt.strftime("%Y-%m-%d")
    an_out.sort_values("ts").to_csv(REPORTS / "week01_anomalies.csv",
                                    index=False, float_format="%.6g")

    figs = {
        "hist": fig_fill_hist(daily),
        "time": fig_fill_time(daily),
        "anom": fig_anomalies_year(an) if len(an) else None,
        "close": fig_close_check(daily),
        "adj": fig_raw_vs_adj(daily),
    }

    fr = daily["fill_ratio"] * 100
    q = fr.quantile([0.5, 0.9, 0.95, 0.99])
    worst = daily.nlargest(10, "fill_ratio")[
        ["fill_ratio", "missing_ratio", "n_minutes", "is_half_day"]].copy()
    worst.insert(0, "date", worst.index.strftime("%Y-%m-%d"))
    worst["fill_ratio"] = (worst["fill_ratio"] * 100).round(2).astype(str) + "%"
    worst["missing_ratio"] = (worst["missing_ratio"] * 100).round(2) \
        .astype(str) + "%"

    flag_counts = an["flags"].str.split(";").explode().value_counts()
    top_an = an.reindex(an["ret"].abs().sort_values(ascending=False).index) \
        .head(15)
    top_an = pd.DataFrame({
        "time (ET)": top_an["ts"].dt.strftime("%Y-%m-%d %H:%M"),
        "flags": top_an["flags"],
        "1-min return": (top_an["ret"] * 100).round(3).astype(str) + "%",
        "close": top_an["close"].round(2),
        "volume": top_an["volume"].astype(int),
    })

    ex = di[di["is_ex_div"]].loc[daily.index.min():daily.index.max()]
    ex_tbl = pd.DataFrame({"ex-date": ex.index.strftime("%Y-%m-%d"),
                           "dividend (USD)": ex["dividend"].round(4)})

    cd = daily["close_vs_ibkr"].abs() * 1e4
    n_days = len(daily)
    years = daily.index.year.value_counts().sort_index()

    md = f"""# Week 1 資料品質報告

產生時間：{pd.Timestamp.now(tz=dl.ET):%Y-%m-%d %H:%M} ET　·　資料建置：{meta['built_at'][:16]}

資料來源：IBKR TWS API（`scripts/download_ibkr.py`）。清理程式：`src/data_loader.py`。
人工檢視結論見 [`week01_review_notes.md`](week01_review_notes.md)。

## 1. 總覽

| 項目 | 數值 |
|---|---|
| 分鐘資料範圍 (ET) | {meta['minute_range'][0][:16]} → {meta['minute_range'][1][:16]} |
| 交易日（有 RTH 分鐘資料） | {n_days:,}（{daily.index.min():%Y-%m-%d} → {daily.index.max():%Y-%m-%d}） |
| 其中半日交易日 | {int(daily['is_half_day'].sum())} |
| RTH 1 分鐘 bar | {meta['rth_minutes']:,} |
| 盤前盤後 1 分鐘 bar | {meta['ext_minutes']:,} |
| 原始檔 / 原始列數 | {meta['minute_files']} 檔 / {meta['minute_rows_raw']:,} 列 |
| 週檔重疊去重 | {meta['dedup_dropped']:,} 列（值不一致的衝突：{meta['dedup_conflicts']}） |
| 丟棄的未完成交易日 | {', '.join(meta['dropped_incomplete_days']) or '無'} |
| 與 IBKR 日線比對缺少的交易日 | {', '.join(meta['missing_days_vs_ibkr_daily']) or '無'} |
| 不在 IBKR 日線中的交易日 | {', '.join(meta['extra_days_not_in_ibkr_daily']) or '無'} |

每年交易日數：{', '.join(f'{y}: {n}' for y, n in years.items())}

## 2. 缺失分鐘與補值

規則：RTH 每個應有的分鐘都必須存在（一般日 390 根、半日 210 根）。
缺的分鐘以**該分鐘之前**最後一筆成交價補上（OHLC 皆為此價，成交量 0），絕不往回補。
IBKR 自己送來的零成交 bar（volume=0、barCount=0）也計入補值。

| 指標 | 數值 |
|---|---|
| 補值分鐘合計 | {meta['rth_minutes_filled']:,}（{meta['rth_minutes_filled'] / meta['rth_minutes']:.3%}） |
| 　其中 IBKR 缺漏、由我們補上 | {meta['rth_minutes_missing']:,} |
| 　其中 IBKR 零成交 bar | {meta['rth_minutes_ibkr_zero']:,} |
| 每日補值比例 中位數 / P90 / P95 / P99 | {q[0.5]:.2f}% / {q[0.9]:.2f}% / {q[0.95]:.2f}% / {q[0.99]:.2f}% |
| 每日補值比例 最大 | {fr.max():.2f}% |
| 補值比例 > 5% 的交易日 | {int((fr > 5).sum())} |

![fill ratio histogram]({figs['hist']})

![fill ratio over time]({figs['time']})

補值比例最高的 10 天：

{_md_table(worst)}

## 3. 異常值（只標記，不刪除）

門檻（`CleanConfig`）：單分鐘 |log 報酬| > {meta['config']['anomaly_abs_return']:.0%}；
或 > {meta['config']['anomaly_sigma']:g} 倍「前 {meta['config']['anomaly_vol_window']} 根」報酬標準差（只用過去資料）；
或 bar 高低價差 > {meta['config']['anomaly_bar_range']:.0%}；或 OHLC 不合理。

被標記的分鐘：**{meta['anomalies']:,}**。完整清單：[`week01_anomalies.csv`](week01_anomalies.csv)（請人工檢視）。

依類型（一分鐘可有多種）：{', '.join(f'`{k}` {v:,}' for k, v in flag_counts.items()) or '無'}

{f'![anomalies per year]({figs["anom"]})' if figs['anom'] else ''}

|報酬|最大的 15 筆：

{_md_table(top_an) if len(top_an) else '（無）'}

## 4. 半日交易日

依交易所規則（感恩節隔日；7/3 與 12/24 落在週一至週四時）判定，並與資料交叉驗證：
半日交易日的盤後資料都在 16:59 結束。範圍內共 {len(meta['half_days'])} 天，
沒有資料的：{', '.join(meta['half_days_without_data']) or '無'}。

{', '.join(meta['half_days'])}

`load_daily(exclude_half_days=True)` 等函式可直接排除半日交易日。

## 5. 盤後提早結束的交易日

一般日盤後應到 19:59、半日到 16:59。以下日子最後一根盤外 bar 較早：

{_md_table(pd.DataFrame(list(meta['short_extended_sessions'].items()), columns=['date', 'last bar (ET)'])) if meta['short_extended_sessions'] else '（無）'}

## 6. 拆股與股息

* IBKR 的 TRADES 日線已做拆股調整（1999 年價格約 51，而實際掛牌價約 100，2000 年 1 拆 2）。2015 年後 QQQ 無拆股。
* 分鐘資料與 IBKR 日線收盤比對：|差異| 中位數 {cd.median():.1f} bps、最大 {cd.max():.1f} bps；
  超過 {meta['config']['split_check_tolerance']:.0%}（疑似未調整拆股）的交易日：{', '.join(meta['split_suspect_days']) or '無'}。
* 分鐘資料**未做股息調整**。股息由兩份日線的調整因子反推，範圍內共 {len(ex_tbl)} 次除息。
  跨日報酬（`ret_overnight`、`ret_cc`）已加回股息；日內報酬不受影響。
* 資料有缺口的前一交易日（`prev_day_missing`）不計算跨日報酬。

![raw vs adjusted]({figs['adj']})

![close check]({figs['close']})

{_md_table(ex_tbl)}

## 7. 未來資訊洩漏的防護

* 每根 bar 有 `bar_end`：1 分鐘 bar 在開始時間 +1 分鐘後才可用，5 分鐘 bar 在 +5 分鐘後。
* 補值只用該分鐘之前的價格；異常值的 σ 只用過去的報酬。
* `ADJUSTED_LAST` 是回溯調整，價格水準含有未來股息資訊：**只用它的報酬，不用它的價格水準**。
* 股息在除息日使用（當時已公告）；半日交易日由事先公布的規則判定。
* 下載當下尚未收盤的交易日已丟棄。

## 8. 產出檔案（`data/processed/`）

| 檔案 | 內容 |
|---|---|
| `minute_all_sessions.parquet` | 去重後的全部分鐘資料 + session 標籤 |
| `minute_rth.parquet` | RTH 1 分鐘完整網格，含 `is_filled`、`fill_source`、`bar_end` |
| `minute_ext.parquet` | 盤前、盤後（不補值） |
| `bars_5min.parquet` | 5 分鐘 RTH bar |
| `daily.parquet` | 由分鐘資料聚合的日線 + 品質指標 + 股息 + 報酬 |
| `daily_ibkr.parquet` | IBKR 日線（原始 + 調整）與推得的股息 |
| `anomalies.parquet` | 異常值清單 |
| `build_meta.json` | 本報告用到的統計與參數 |
"""
    path = REPORTS / "week01_data_quality.md"
    path.write_text(md, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
