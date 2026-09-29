"""Week 14 Part 5 (defense prep): two figures the slide deck needs that
were never saved as images in their own week's report (weeks 7-12 are
table-only - see final_outline.md's figure inventory). Both reuse
EXISTING, already-tested analysis functions with the exact parameters
each week's own report used - no new analysis, just a chart of numbers
that were already computed and reported as tables.

Writes:
- reports/figures/w08_threshold_curve.png (src.filter_eval.
  threshold_curve_with_random_band, same call as scripts/
  make_week10_report.py:183, n_reps=1000 seed=0)
- reports/figures/w12_cnn_vs_hand_auc_by_fold.png (per-fold AUC from
  data/processed/sequences/week12_task_a_fold_diagnostics.parquet [CNN]
  and data/processed/predictions/week11_task_a_hand_feature_logistic
  .parquet [hand-feature logistic], the same two series compared in
  reports/stage5_chapter.md's per-fold table)

Usage:
    python scripts/make_defense_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_loader as dl  # noqa: E402
from src import filter_eval as FE  # noqa: E402
from src.models.base import load_predictions  # noqa: E402

REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"

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


def week08_threshold_curve() -> Path:
    preds = load_predictions("week08_filter")
    orb = pd.read_parquet(dl.PROCESSED_DIR / "strategies" / "orb_daily.parquet")
    tc = FE.threshold_curve_with_random_band(preds, orb, n_reps=1000, seed=0)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = tc["retention"] * 100
    ax.fill_between(x, tc["random_p05"], tc["random_p95"], color=ORANGE, alpha=0.25,
                    label="Random-filter null, 5th-95th pct (n_reps=1000)")
    ax.plot(x, tc["random_mean"], color=ORANGE, linewidth=1, linestyle="--",
           label="Random-filter null mean")
    ax.plot(x, tc["sharpe"], color=BLUE, linewidth=2, marker="o", label="Logistic filter (actual)")
    ax.set_xlabel("Retention level (% of signal days kept)")
    ax.set_ylabel("Test-fold Sharpe (compounding)")
    ax.set_title("Week 8 filter: threshold curve vs random-filter band")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1,
             bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "w08_threshold_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def week12_per_fold_auc() -> Path:
    diag = pd.read_parquet(dl.PROCESSED_DIR / "sequences" / "week12_task_a_fold_diagnostics.parquet")
    cnn_a = load_predictions("week12_task_a_cnn").set_index("date")
    hand_a = load_predictions("week11_task_a_hand_feature_logistic").set_index("date")

    rows = []
    for f in sorted(cnn_a["fold"].unique()):
        c = cnn_a[cnn_a["fold"] == f]
        h = hand_a[hand_a["fold"] == f]
        common = c.index.intersection(h.index)
        rows.append({"fold": f, "cnn_auc": roc_auc_score(c.loc[common, "y_true"], c.loc[common, "y_pred_proba"]),
                    "hand_auc": roc_auc_score(h.loc[common, "y_true"], h.loc[common, "y_pred_proba"])})
    per_fold = pd.DataFrame(rows)
    seed_min = diag.set_index("fold")["test_auc_seed_min"]
    seed_max = diag.set_index("fold")["test_auc_seed_max"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = per_fold["fold"]
    ax.errorbar(x, per_fold["cnn_auc"], yerr=[per_fold["cnn_auc"] - seed_min.loc[x].to_numpy(),
                                             seed_max.loc[x].to_numpy() - per_fold["cnn_auc"]],
               fmt="o-", color=BLUE, capsize=3, label="CNN (mean of 5 seeds, bar = seed range)")
    ax.plot(x, per_fold["hand_auc"], "s-", color=ORANGE, label="Hand-feature logistic")
    ax.axhline(0.5, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Fold (test year 2019-2026)")
    ax.set_ylabel("Test-fold AUC")
    ax.set_title("Week 12: CNN vs hand-feature logistic, per fold")
    ax.set_xticks(list(x))
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2,
             bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    p = FIG_DIR / "w12_cnn_vs_hand_auc_by_fold.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    p1 = week08_threshold_curve()
    print(f"wrote {p1}")
    p2 = week12_per_fold_auc()
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
