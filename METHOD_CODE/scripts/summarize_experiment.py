"""
Summarize experiment results: CSV main table + comparison plots.

Usage:
    python scripts/summarize_experiment.py outputs/full_experiment_20260615_195407

Produces (inside the experiment dir):
    summary_table.csv          — flat table (method × stage × seed)
    summary_table_mean.csv     — mean±std table (method × stage)
    figures/
        bar_avg_map.png        — Avg-mAP bar chart with error bars
        bar_macro_f1.png       — Macro-F1 bar chart
        bar_variant_recall.png — Variant Recall bar chart
        line_avg_map.png       — Avg-mAP line plot across stages
        line_macro_f1.png      — Macro-F1 line plot across stages
"""

import os
import sys
import re
import json
import glob
import argparse
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ──────────────────────── Config ────────────────────────

CORE_METRICS = [
    "avg_map",
    "macro_f1",
    "micro_f1",
    "variant_recall",
    "auroc",
    "fpr95",
    "tail_recall",
    "cka",
]

# Display names for paper tables
METRIC_DISPLAY = {
    "avg_map": "Avg-mAP",
    "macro_f1": "Macro-F1",
    "micro_f1": "Micro-F1",
    "variant_recall": "Var-Recall",
    "auroc": "AUROC",
    "fpr95": "FPR95",
    "tail_recall": "Tail-Recall",
    "cka": "CKA",
}

# Method display order (Ours first, baselines, ablations last)
METHOD_ORDER = [
    "ours",
    "seq_finetune",
    "task_lora",
    "task_lora_msp",
    "task_lora_adb",
    "task_lora_maha",
    "o_lora",
    "ewc_lora",
    "l2p",
    "ablation_no_evo",
    "ablation_no_dual",
    "ablation_no_anchor",
]

# Color palette (colorblind-friendly)
COLORS = {
    "ours": "#0072B2",
    "seq_finetune": "#D55E00",
    "task_lora": "#009E73",
    "task_lora_msp": "#F0E442",
    "task_lora_adb": "#56B4E9",
    "task_lora_maha": "#E69F00",
    "o_lora": "#CC79A7",
    "ewc_lora": "#000000",
    "l2p": "#8B4513",
    "ablation_no_evo": "#999999",
    "ablation_no_dual": "#666666",
    "ablation_no_anchor": "#333333",
}
DEFAULT_COLOR = "#AAAAAA"

# ──────────────────────── Parsing ────────────────────────

# Pattern: {method}_seed{S}_stage{N}  (e.g. ours_seed42_stage0, task_lora_msp_seed42_stage1)
_DIR_RE = re.compile(r"^(.+?)_seed(\d+)_stage(\d+)$")


def parse_dirname(dirname):
    """Extract (method, seed, stage) from directory name."""
    base = os.path.basename(dirname)
    m = _DIR_RE.match(base)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None


def collect_results(exp_dir):
    """
    Scan runs/ subdirectories and collect all metrics.

    Returns list of dicts:
      [{"method": str, "seed": int, "stage": int, **metrics}, ...]
    """
    runs_dir = os.path.join(exp_dir, "runs")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"No runs/ directory found in {exp_dir}")

    records = []
    for entry in sorted(os.listdir(runs_dir)):
        full = os.path.join(runs_dir, entry)
        if not os.path.isdir(full):
            continue

        parsed = parse_dirname(entry)
        if parsed is None:
            continue
        method, seed, stage = parsed

        # Find metrics file: metrics_stage{N}.json
        metrics_file = os.path.join(full, f"metrics_stage{stage}.json")
        if not os.path.exists(metrics_file):
            print(f"[WARN] Missing {metrics_file}, skipping.")
            continue

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        record = {"method": method, "seed": seed, "stage": stage}
        for mk in CORE_METRICS:
            record[mk] = metrics.get(mk)
        # Also collect any extra numeric metrics
        for k, v in metrics.items():
            if k not in record and isinstance(v, (int, float)):
                record[k] = v
        records.append(record)

    records.sort(key=lambda r: (
        METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 999,
        r["stage"], r["seed"]
    ))
    return records


# ──────────────────────── CSV Output ────────────────────────

def _fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def write_csv_flat(records, out_path):
    """Write flat CSV: one row per (method, seed, stage)."""
    cols = ["method", "seed", "stage"] + CORE_METRICS
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            vals = [_fmt(r.get(c)) for c in cols]
            f.write(",".join(vals) + "\n")
    print(f"[CSV] Flat table: {out_path}")


def write_csv_mean(records, out_path):
    """Write mean±std CSV: one row per (method, stage), aggregated across seeds."""
    # Group by (method, stage)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["method"], r["stage"])
        for mk in CORE_METRICS:
            v = r.get(mk)
            if v is not None:
                grouped[key][mk].append(v)

    cols = ["method", "stage"] + CORE_METRICS
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for method in METHOD_ORDER:
            stages = sorted(set(s for (m, s) in grouped if m == method))
            for stage in stages:
                key = (method, stage)
                row = [method, str(stage)]
                for mk in CORE_METRICS:
                    vals = grouped[key].get(mk, [])
                    if vals:
                        mean = np.mean(vals)
                        std = np.std(vals, ddof=0)
                        if std < 1e-8:
                            row.append(f"{mean:.4f}")
                        else:
                            row.append(f"{mean:.4f}±{std:.4f}")
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")
    print(f"[CSV] Mean±std table: {out_path}")


# ──────────────────────── Plotting ────────────────────────

def _get_ordered_methods(records):
    """Get methods in display order, only those present in records."""
    present = set(r["method"] for r in records)
    return [m for m in METHOD_ORDER if m in present]


def _aggregate_by_method_stage(records):
    """Return {(method, stage): {"mean": ..., "std": ..., "values": [...]}} per metric."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["method"], r["stage"])
        for mk in CORE_METRICS:
            v = r.get(mk)
            if v is not None:
                grouped[key][mk].append(v)

    agg = {}
    for key, metrics_dict in grouped.items():
        agg[key] = {}
        for mk, vals in metrics_dict.items():
            agg[key][mk] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "values": vals,
            }
    return agg


def plot_bar_chart(agg, methods, stages, metric, out_path, title=None):
    """
    Grouped bar chart: x = stage, groups = methods, bars with error bars.
    """
    if not _HAS_MPL:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    n_methods = len(methods)
    n_stages = len(stages)
    bar_width = 0.8 / n_methods
    x = np.arange(n_stages)

    fig, ax = plt.subplots(figsize=(max(8, n_stages * 2.5), 6))

    for i, method in enumerate(methods):
        means = []
        stds = []
        for stage in stages:
            entry = agg.get((method, stage), {}).get(metric, {})
            means.append(entry.get("mean", 0))
            stds.append(entry.get("std", 0))

        color = COLORS.get(method, DEFAULT_COLOR)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, means, bar_width,
            yerr=stds if any(s > 0 for s in stds) else None,
            label=method, color=color, alpha=0.85,
            capsize=3, error_kw={"linewidth": 1},
        )

    ax.set_xlabel("FSCIL Stage", fontsize=13)
    ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=13)
    ax.set_title(title or f"{METRIC_DISPLAY.get(metric, metric)} across Stages", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=11)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out_path}")


def plot_line_chart(agg, methods, stages, metric, out_path, title=None):
    """
    Line plot: x = stage, one line per method with error band.
    """
    if not _HAS_MPL:
        print("[WARN] matplotlib not installed, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        means = []
        stds = []
        valid_stages = []
        for stage in stages:
            entry = agg.get((method, stage), {}).get(metric, {})
            if entry:
                means.append(entry["mean"])
                stds.append(entry["std"])
                valid_stages.append(stage)

        if not valid_stages:
            continue

        color = COLORS.get(method, DEFAULT_COLOR)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(valid_stages, means, marker="o", linewidth=2, color=color, label=method)
        if np.any(stds > 0):
            ax.fill_between(valid_stages, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xlabel("FSCIL Stage", fontsize=13)
    ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=13)
    ax.set_title(title or f"{METRIC_DISPLAY.get(metric, metric)} across Stages", fontsize=14)
    ax.set_xticks(stages)
    ax.set_xticklabels([f"Stage {s}" for s in stages], fontsize=11)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out_path}")


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarize FSCIL experiment: CSV tables + comparison plots"
    )
    parser.add_argument("exp_dir", type=str,
                        help="Experiment directory (e.g. outputs/full_experiment_20260615_195407)")
    parser.add_argument("--metrics", nargs="*", default=["avg_map", "macro_f1"],
                        help="Metrics to plot (default: avg_map macro_f1)")
    parser.add_argument("--dpi", type=int, default=150, help="Plot DPI")
    args = parser.parse_args()

    exp_dir = args.exp_dir.rstrip("/\\")
    if not os.path.isdir(exp_dir):
        print(f"Error: {exp_dir} is not a directory")
        sys.exit(1)

    # Collect
    print(f"[Scan] {exp_dir}")
    records = collect_results(exp_dir)
    if not records:
        print("No results found.")
        sys.exit(1)

    methods = _get_ordered_methods(records)
    stages = sorted(set(r["stage"] for r in records))
    seeds = sorted(set(r["seed"] for r in records))
    print(f"[Info] {len(records)} files: {len(methods)} methods, {len(stages)} stages, {len(seeds)} seeds")

    # CSV
    csv_flat = os.path.join(exp_dir, "summary_table.csv")
    csv_mean = os.path.join(exp_dir, "summary_table_mean.csv")
    write_csv_flat(records, csv_flat)
    write_csv_mean(records, csv_mean)

    # Plots
    if _HAS_MPL:
        fig_dir = os.path.join(exp_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        agg = _aggregate_by_method_stage(records)

        for metric in args.metrics:
            if metric not in CORE_METRICS:
                print(f"[WARN] Unknown metric '{metric}', skipping.")
                continue

            display = METRIC_DISPLAY.get(metric, metric)

            # Bar chart
            plot_bar_chart(
                agg, methods, stages, metric,
                os.path.join(fig_dir, f"bar_{metric}.png"),
                title=f"{display} — Method Comparison",
            )

            # Line chart
            plot_line_chart(
                agg, methods, stages, metric,
                os.path.join(fig_dir, f"line_{metric}.png"),
                title=f"{display} — Incremental Performance",
            )

        print(f"\n[Done] All outputs saved to {exp_dir}/")
    else:
        print("\n[Done] CSV generated. Install matplotlib for plots: pip install matplotlib")


if __name__ == "__main__":
    main()
