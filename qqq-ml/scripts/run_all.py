"""Week 14 Part 4 (reproducibility): run the full pipeline from raw data
through every week's report, ending with reports/final_report.pdf.

This is the entry point named in the final report's methodology section
and in README.md. Steps are grouped by week; each step is a subprocess
call to an existing, already-tested script - this file does not contain
any new analysis logic itself.

CNN steps (weeks 11-13) are slow (walk-forward training many models).
Pass --skip-cnn to reuse already-saved predictions under data/processed/
predictions/ instead of retraining (the final report's numbers come from
those saved predictions either way - retraining only re-verifies they are
reproducible, it does not change what the report says).

Weeks 1-4 only rebuild data/processed/ (data_loader, features, HAR
baselines, week 4 models) by default - their own reports (reports/
week01_data_quality.md etc.) are NOT regenerated. Those report scripts
read src/features.py and src/data_loader.py as they exist TODAY, not as
they existed in week 1-4 - both modules grew new feature columns and
tests in later weeks, so regenerating an early week's report today
would silently diverge from the historical version committed to git at
the time (confirmed during week 14's clean-venv reproducibility check -
see README.md's "可重現性" section). Pass --rebuild-early-reports to
regenerate them anyway (e.g. to see how they'd read against the current
codebase) - nothing downstream depends on these four report files, only
on the data/processed/ and data/processed/predictions/ files those weeks
produce.

Usage:
    python scripts/run_all.py                  # full run, retrains CNN
    python scripts/run_all.py --skip-cnn        # skip weeks 11-13 training
    python scripts/run_all.py --from-week 8     # resume from a given week
    python scripts/run_all.py --pdf-only        # only (re)build the PDF
    python scripts/run_all.py --rebuild-early-reports  # also redo weeks 1-4's own .md reports
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"FAILED: {label} (exit {result.returncode})")
    print(f"--- {label} done in {time.time() - t0:.1f}s ---", flush=True)


def build_steps(skip_cnn: bool, rebuild_early_reports: bool = False) -> list[tuple[int, str, list[str]]]:
    """(week, label, argv) - argv[0] is always PY."""
    steps = [
        (1, "data loader (clean raw -> data/processed/)", [PY, "-m", "src.data_loader"]),
        (2, "features", [PY, "-m", "src.features"]),
        (3, "HAR baselines", [PY, "-m", "src.models.har"]),
        (4, "week 4 models (HAR-X, XGBoost, RF)", [PY, "-m", "src.models.week4"]),
    ]
    if rebuild_early_reports:
        steps += [
            (1, "week 1 report (--rebuild-early-reports)", [PY, "scripts/make_week01_report.py"]),
            (2, "week 2 report (--rebuild-early-reports)", [PY, "scripts/make_week02_report.py"]),
            (3, "week 3 report (--rebuild-early-reports)", [PY, "scripts/make_week03_report.py"]),
            (4, "week 4 report (--rebuild-early-reports)", [PY, "scripts/make_week04_report.py"]),
        ]
    steps += [
        (5, "week 5 report (runs the main backtest, save=True)", [PY, "scripts/make_week05_report.py"]),
        (6, "rule strategies + regime labels", [PY, "-m", "src.rules"]),
        (6, "week 6 report", [PY, "scripts/make_week06_report.py"]),
        (7, "week 7 report", [PY, "scripts/make_week07_report.py"]),
        (8, "logistic filter model", [PY, "-m", "src.models.filter"]),
        (8, "week 8 report", [PY, "scripts/make_week08_report.py"]),
        (9, "week 9 report", [PY, "scripts/make_week09_report.py"]),
        (10, "week 10 report", [PY, "scripts/make_week10_report.py"]),
    ]
    if not skip_cnn:
        steps += [
            (11, "sequence dataset + baselines", [PY, "-m", "src.sequences"]),
            (11, "week 11 baselines", [PY, "scripts/run_week11_baselines.py"]),
            (11, "week 11 report", [PY, "scripts/make_week11_report.py"]),
            (12, "CNN walk-forward (slow)", [PY, "scripts/run_week12_cnn_walkforward.py"]),
            (12, "CNN stacking", [PY, "scripts/run_week12_stacking.py"]),
            (12, "CNN strategy layer", [PY, "scripts/run_week12_strategy_layer.py"]),
            (12, "week 12 report", [PY, "scripts/make_week12_report.py"]),
            (13, "shuffled-label full check (slow)", [PY, "scripts/run_week13_shuffled_label_full.py"]),
        ]
    else:
        print("--skip-cnn: reusing already-saved data/processed/predictions/week11-13 "
              "output instead of retraining (final_report.md's numbers come from "
              "those files either way).")
    steps += [
        (14, "week 14 robustness gap-fill (slow: HMM refits + 1000-rep bootstrap)",
         [PY, "scripts/run_week14_robustness.py"]),
        (14, "collect_results.py -> results_summary.json", [PY, "scripts/collect_results.py"]),
        (14, "why-hard chart + table", [PY, "scripts/make_week14_why_hard.py"]),
        (14, "final_report.md from template", [PY, "scripts/make_final_report.py"]),
    ]
    return steps


def build_pdf() -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise SystemExit(
            "pandoc not found on PATH. Install it (e.g. `winget install "
            "JohnMacFarlane.Pandoc`) and a XeLaTeX-capable TeX distribution "
            "(e.g. `winget install MiKTeX.MiKTeX`) before running --pdf-only "
            "or the full pipeline. A Traditional Chinese font must also be "
            "installed - this report is written for \"Noto Sans TC\" "
            "(bundled with Windows as of this project's setup; verify with "
            "`Get-ChildItem C:\\Windows\\Fonts | Select-String NotoSansTC`).")
    if shutil.which("xelatex") is None:
        raise SystemExit("xelatex not found on PATH - install a TeX distribution (MiKTeX/TeX Live).")

    src = ROOT / "reports" / "final_report.md"
    out = ROOT / "reports" / "final_report.pdf"
    cmd = [pandoc, str(src), "-o", str(out), "--pdf-engine=xelatex", "--toc",
          f"--resource-path={ROOT / 'reports'}"]
    print(f"\n=== pandoc -> {out} ===", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    # MiKTeX prints a harmless "you have not checked for updates" nag to
    # stderr on every run; only treat this as a failure by exit code, not
    # by stderr being non-empty.
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(f"FAILED: pandoc (exit {result.returncode})")
    print(f"wrote {out}")


def build_slides() -> None:
    """Week 14 Part 5: regenerate the two figures defense_slides.pptx needs
    that aren't produced by any weekly report script, then rebuild the
    deck itself. Both scripts read reports/results_summary.json and
    already-saved prediction/diagnostic files only - no new analysis."""
    run([PY, "scripts/make_defense_figures.py"], "defense figures (week 8 threshold curve, week 12 per-fold AUC)")
    run([PY, "scripts/make_defense_slides.py"], "defense_slides.pptx from results_summary.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-cnn", action="store_true",
                    help="Skip weeks 11-13's training steps; reuse saved predictions.")
    ap.add_argument("--from-week", type=int, default=1,
                    help="Resume from this week number (skips earlier steps).")
    ap.add_argument("--pdf-only", action="store_true",
                    help="Only run scripts/make_final_report.py + the pandoc PDF build.")
    ap.add_argument("--rebuild-early-reports", action="store_true",
                    help="Also regenerate weeks 1-4's own .md reports (overwrites the "
                        "git-committed historical versions with today's src/ state - see "
                        "the module docstring). Off by default.")
    ap.add_argument("--slides", action="store_true",
                    help="Only rebuild reports/defense_slides.pptx (and the two figures "
                        "it needs) from the current reports/results_summary.json - "
                        "nothing else runs. Use after results_summary.json changes.")
    args = ap.parse_args()

    t_start = time.time()
    if args.slides:
        build_slides()
    elif args.pdf_only:
        run([PY, "scripts/make_final_report.py"], "final_report.md from template")
        build_pdf()
    else:
        for week, label, cmd in build_steps(args.skip_cnn, args.rebuild_early_reports):
            if week < args.from_week:
                continue
            run(cmd, f"week {week}: {label}")
        build_pdf()
    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
