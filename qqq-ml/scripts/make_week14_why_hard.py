"""Week 14 Part 2 (config/week14_final.toml): quantify "why this is hard" -
for each phase, lay the prediction-layer improvement, the strategy-layer
observed Sharpe difference, and the minimum detectable difference (MDE)
side by side, plus one chart pairing observed |diff| against MDE per
phase. All numbers read from reports/results_summary.json - nothing is
recomputed here. Sharpe figures are the arithmetic-mean definition (see
results_summary.json's own _cross_check_notes); the figure caption states
this explicitly, per week14_final.toml's requirement that every chart's
caption state its date range and Sharpe definition.

Writes reports/figures/w14_diff_vs_mde.png and prints the comparison
table (also embedded as reports/week14_why_hard.md).

Usage:
    python scripts/make_week14_why_hard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"

# Same style block as scripts/make_week01_report.py, so this figure matches
# every other week's look.
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "figure.dpi": 110, "savefig.bbox": "tight",
})

# English in-image text only (matches every prior week's figures - DejaVu
# Sans, the default font, has no CJK glyphs); Chinese stays in the
# surrounding markdown caption/prose.
PHASE_LABEL = {"phase2": "Phase 2\n(vol forecast)", "phase3": "Phase 3\n(regime label)",
              "phase4": "Phase 4\n(filter)", "phase5": "Phase 5\n(CNN)"}


def main() -> None:
    d = json.loads((REPORTS / "results_summary.json").read_text(encoding="utf-8"))

    rows = []
    # phase2/3 have primary+secondary or orb+vwap rows; pick the row that
    # matches each phase's own headline comparison (primary / orb, per
    # final_outline.md's key-numbers section).
    p2 = d["phase2"]["rows"][0]
    rows.append({"phase": "phase2", "pred_metric": "QLIKE差(HAR+殘差XGB−HAR)",
                "pred_value": p2["prediction_layer"]["diff"],
                "pred_sig": f"DM p={p2['prediction_layer']['p_value']:.5f}(顯著改善)",
                "strat_diff": p2["strategy_layer"]["diff"],
                "mde": p2["strategy_layer"]["mde"],
                "strat_p": p2["strategy_layer"]["p_value"]})

    p3 = d["phase3"]["rows"][0]  # orb row
    rows.append({"phase": "phase3", "pred_metric": "HMM state1/2 vs state0已實現波動差(t值)",
                "pred_value": None,
                "pred_sig": "t=9.93/12.20,p≈0(狀態標籤本身顯著,不是雜訊)",
                "strat_diff": p3["strategy_layer"]["diff"],
                "mde": p3["strategy_layer"]["mde"],
                "strat_p": p3["strategy_layer"]["p_value"]})

    p4 = d["phase4"]["rows"][0]
    rows.append({"phase": "phase4", "pred_metric": "AUC(logistic篩選ORB)",
                "pred_value": p4["prediction_layer"]["value"],
                "pred_sig": "中等,無對照基準模型可算DM/AUC差",
                "strat_diff": p4["strategy_layer"]["diff"],
                "mde": p4["strategy_layer"]["mde"],
                "strat_p": p4["strategy_layer"]["p_value"]})

    p5 = d["phase5"]["rows"][0]
    rows.append({"phase": "phase5", "pred_metric": "AUC差(CNN−手工特徵logistic)",
                "pred_value": p5["prediction_layer"]["diff"],
                "pred_sig": f"block bootstrap p={p5['prediction_layer']['p_value']:.4f}(顯著更差)",
                "strat_diff": p5["strategy_layer"]["diff"],
                "mde": p5["strategy_layer"]["mde"],
                "strat_p": p5["strategy_layer"]["p_value"]})

    # ---------------------------------------------------------------
    # Chart: SIGNED observed diff vs MDE (magnitude, unsigned by
    # construction), strategy layer, per phase. Sign matters here - phase 3
    # and phase 5's filters make things WORSE (negative), not just "not
    # significantly better" - collapsing to abs() would hide that.
    # ---------------------------------------------------------------
    phases = [r["phase"] for r in rows]
    obs = [r["strat_diff"] for r in rows]
    mde = [r["mde"] for r in rows]
    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.6))
    obs_colors = [BLUE if o >= 0 else "#c23b6b" for o in obs]
    ax.bar(x - width / 2, obs, width, label="Observed Sharpe diff (signed)", color=obs_colors)
    ax.bar(x + width / 2, mde, width, label="Min. detectable diff (5%/80% power, magnitude)",
          color=ORANGE)
    ax.bar(x + width / 2, [-m for m in mde], width, color=ORANGE, alpha=0.5)
    ax.axhline(0, color=INK, linewidth=0.8)
    for i, (o, m) in enumerate(zip(obs, mde)):
        va = "bottom" if o >= 0 else "top"
        off = 0.02 if o >= 0 else -0.02
        ax.text(i - width / 2, o + off, f"{o:+.3f}", ha="center", va=va, fontsize=8, color=INK)
        ax.text(i + width / 2, m + 0.02, f"±{m:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABEL[p] for p in phases], fontsize=9)
    ax.set_ylabel("Annualized Sharpe (arithmetic-mean definition)")
    ax.set_title("Strategy layer: signed observed diff vs detectability band (±MDE)")
    ax.legend(frameon=False, fontsize=8, loc="upper center",
             bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "w14_diff_vs_mde.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")

    # ---------------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------------
    lines = ["# Week 14 Part 2:「為什麼難」的量化\n",
            "資料範圍:各階段自己的OOS walk-forward測試段(階段二2015-2026、階段三"
            "2020-2026共同範圍、階段四2018-2026、階段五2019-2026,見各階段章節)。"
            "策略層Sharpe一律用**算術平均年化定義**(mean/std×√252,跟block bootstrap/"
            "MDE同一定義,見results_summary.json的_cross_check_notes),不是各週報告"
            "顯示的複利版本。\n",
            "## 一、預測層改善 vs 策略層觀察差 vs 最小可偵測差\n",
            "**判讀分兩類,不是同一種「測不到」**:",
            "- **階段二**:MDE只有0.069,遠小於其他三階段——不同波動預測模型接同一套"
            "vol_target部位規則後,產生的部位序列高度相關(配對比較的變異數小),檢定力本身",
            "  夠高。觀察差0.003遠低於這個小的MDE門檻,這是**有檢定力支持的『沒有實質差異』**,",
            "  不是測不到,是真的量不出差別。",
            "- **階段三到五**:MDE落在0.7–0.8這個大得多的量級——樣本量(交易日數)沒有本質",
            "  改變,但效果本身(篩選/狀態/CNN的貢獻)混在整體策略報酬的雜訊裡,檢定力不足。",
            "  這三個階段是**偵測不到**,不是**偵測到沒有差異**——如果背後真有一個中等大小的",
            "  效果,現有樣本量本來就看不出來,不能倒過來說『證明沒有效果』。\n",
            "| 階段 | 預測層指標 | 預測層顯著性 | 策略層觀察差(算術Sharpe,保留正負號) | "
            "策略層p值 | 最小可偵測差(MDE) | 達MDE? | 判讀 |",
            "|---|---|---|---|---|---|---|---|"]
    interp = {
        "phase2": "有檢定力支持的沒有實質差異(MDE小,觀察差遠低於MDE)",
        "phase3": "偵測不到(MDE大,檢定力不足;不代表沒有效果)",
        "phase4": "偵測不到(MDE大,檢定力不足;不代表沒有效果)",
        "phase5": "偵測不到(MDE大,檢定力不足;不代表沒有效果)",
    }
    for r in rows:
        detectable = "是" if abs(r["strat_diff"]) >= r["mde"] else "否"
        lines.append(f"| {r['phase']} | {r['pred_metric']} | {r['pred_sig']} | "
                     f"{r['strat_diff']:+.4f} | {r['strat_p']:.4f} | {r['mde']:.4f} | "
                     f"{detectable} | {interp[r['phase']]} |")

    lines += ["",
             "**型態很一致**:四個階段裡,預測層(QLIKE/AUC/外部驗證t值)常常有統計上站得住腳"
             "的訊號或差異,但策略層的觀察差全部遠小於5%顯著/80%檢定力下的最小可偵測差"
             "(四階段都是「否」)——但如上表「判讀」欄所分:階段二是量過、量出來真的很小"
             "(有檢定力支持的沒有實質差異);階段三到五是根本量不到(檢定力不足),兩者結論"
             "不能混為一談。",
             "",
             f"**第10週的反推**(`reports/week10_robustness.md`第五節):要用現有的資料頻率"
             f"偵測到0.2的夏普差,logistic模型需要約104年的OOS資料——這是全篇「為什麼難」"
             f"最直接的量化:不是換模型或多等一兩年能解決的量級落差。",
             "",
             "![diff vs mde](figures/w14_diff_vs_mde.png)",
             "",
             "圖說:縱軸是策略層年化Sharpe,**算術平均定義**(mean/std×√252,跟block bootstrap/"
             "MDE同一定義,不是各週報告顯示的複利版本);橫軸四個階段用各自的OOS walk-forward"
             "測試段(階段二2015-2026、階段三2020-2026共同範圍、階段四2018-2026、階段五"
             "2019-2026)。圖片內文字為英文(全專案圖表慣例,預設字型無中文字符),中文說明"
             "見本節文字與表格。",
             ""]
    out_path = REPORTS / "week14_why_hard.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
