"""
Aggregate experiment results from outputs/ into readable CSV and Markdown tables.

Scans the outputs directory for metrics_stage*.json files and produces:
  - results_table.csv: flat table of all runs
  - results_table.md: Markdown formatted table, grouped by method
  - results_summary.json: structured summary with mean±std across seeds

Usage:
    python scripts/aggregate_results.py --outputs_dir ./outputs --out_dir ./results

The script auto-detects method, stage, and seed from directory names.
Supported patterns:
  - stage_{N}_seed{S}/               -> method=ours
  - ours_seed{S}_stage{N}/           -> method=ours
  - {method}_stage_{N}_seed{S}/      -> method={method}
  - ablation_{variant}_stage_{N}_seed{S}/ -> method=ablation_{variant}
  - _smoke_{method}_seed{S}/         -> skipped (smoke tests)
"""

import os
import sys
import argparse
import json
import glob
import re
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Key metrics to extract and display
CORE_METRICS = [
    "macro_f1",
    "micro_f1",
    "avg_map",
    "variant_recall",
    "auroc",
    "fpr95",
    "tail_recall",
    "cka",
]


def parse_dirname(dirname):
    """
    Extract (method, stage, seed) from directory name.
    Returns None if not a result directory.
    """
    base = os.path.basename(dirname)

    # Skip smoke tests and temp files
    if base.startswith("_smoke") or base.startswith("_hparam_temp"):
        return None

    # Pattern: stage_{N}_seed{S}
    m = re.match(r"stage_(\d+)_seed(\d+)", base)
    if m:
        return {"method": "ours", "stage": int(m.group(1)), "seed": int(m.group(2))}

    # Pattern: ours_seed{S}_stage{N}
    m = re.match(r"ours_seed(\d+)_stage(\d+)", base)
    if m:
        return {"method": "ours", "stage": int(m.group(2)), "seed": int(m.group(1))}

    # Pattern: ablation_{variant}_stage_{N}_seed{S}
    m = re.match(r"ablation_(\w+)_stage_(\d+)_seed(\d+)", base)
    if m:
        return {"method": f"ablation_{m.group(1)}", "stage": int(m.group(2)), "seed": int(m.group(3))}

    # Pattern: {method}_stage_{N}_seed{S}
    m = re.match(r"(\w+)_stage_(\d+)_seed(\d+)", base)
    if m:
        return {"method": m.group(1), "stage": int(m.group(2)), "seed": int(m.group(3))}

    # Pattern: _smoke_{method}_seed{S} -> skip
    m = re.match(r"_smoke_(\w+)_seed(\d+)", base)
    if m:
        return None

    return None


def load_metrics(metrics_path):
    """Load metrics JSON, return dict."""
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_value(v, decimals=4):
    """Format a metric value for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def format_mean_std(values, decimals=4):
    """Compute mean ± std from a list of values."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return "N/A"
    mean = np.mean(clean)
    std = np.std(clean, ddof=0)
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def aggregate(outputs_dir, out_dir, run_id_filter=""):
    os.makedirs(out_dir, exist_ok=True)

    # Find all metrics_stage*.json files recursively.
    # This supports both old flat layout and new run_id-based layout.
    pattern = os.path.join(outputs_dir, "**", "metrics_stage*.json")
    all_files = glob.glob(pattern, recursive=True)

    records = []
    for filepath in all_files:
        dir_path = os.path.dirname(filepath)
        meta = parse_dirname(dir_path)
        if meta is None:
            continue

        metrics = load_metrics(filepath)
        if not metrics:
            continue

        rel_path = os.path.relpath(dir_path, outputs_dir)
        rel_parts = rel_path.split(os.sep)
        run_id = rel_parts[0] if len(rel_parts) >= 2 else "legacy"

        record = {
            "method": meta["method"],
            "stage": meta["stage"],
            "seed": meta["seed"],
            "run_id": run_id,
            "dir": dir_path,
        }
        # Extract core metrics (without eval_ prefix if present)
        for mk in CORE_METRICS:
            # Try direct key, then eval_ prefix
            val = metrics.get(mk)
            if val is None:
                val = metrics.get(f"eval_{mk}")
            record[mk] = val

        # Also include all other metrics for completeness
        for k, v in metrics.items():
            if k not in record and isinstance(v, (int, float)):
                record[k] = v

        records.append(record)

    if run_id_filter:
        records = [r for r in records if r["run_id"] == run_id_filter]

    if not records:
        print(f"[Aggregate] No metrics files found in {outputs_dir}")
        return

    print(f"[Aggregate] Found {len(records)} result files.")

    # Sort: method -> stage -> seed
    records.sort(key=lambda r: (r["run_id"], r["method"], r["stage"], r["seed"]))

    # ---------- Flat CSV ----------
    all_keys = ["run_id", "method", "stage", "seed"] + CORE_METRICS
    # Add any extra numeric keys found
    extra_keys = set()
    for r in records:
        for k, v in r.items():
            if k not in all_keys and isinstance(v, (int, float)):
                extra_keys.add(k)
    all_keys += sorted(extra_keys)

    csv_path = os.path.join(out_dir, "results_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ",".join(all_keys)
        f.write(header + "\n")
        for r in records:
            row = ",".join(format_value(r.get(k), decimals=6) for k in all_keys)
            f.write(row + "\n")
    print(f"[Aggregate] CSV saved to {csv_path}")

    # ---------- Markdown Table (grouped by method) ----------
    md_path = os.path.join(out_dir, "results_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Results Summary\n\n")
        f.write("Auto-generated from `outputs/*/metrics_stage*.json`.\n\n")

        # Group by method
        by_method = defaultdict(list)
        for r in records:
            by_method[r["method"]].append(r)

        for method in sorted(by_method.keys()):
            f.write(f"## Method: `{method}`\n\n")

            # Group by stage
            by_stage = defaultdict(list)
            for r in by_method[method]:
                by_stage[r["stage"]].append(r)

            for stage in sorted(by_stage.keys()):
                f.write(f"### Stage {stage}\n\n")

                # Header
                headers = ["seed"] + CORE_METRICS
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

                # Rows (one per seed)
                stage_recs = sorted(by_stage[stage], key=lambda r: r["seed"])
                for r in stage_recs:
                    vals = [str(r["seed"])] + [format_value(r.get(mk)) for mk in CORE_METRICS]
                    f.write("| " + " | ".join(vals) + " |\n")

                # Mean±std row
                means = ["mean±std"]
                for mk in CORE_METRICS:
                    vals = [r.get(mk) for r in stage_recs]
                    means.append(format_mean_std(vals))
                f.write("| " + " | ".join(means) + " |\n")
                f.write("\n")

        # Cross-method comparison table (last stage only, mean across seeds)
        f.write("## Cross-Method Comparison (Last Stage, Mean±Std)\n\n")
        f.write("| run_id | method | stage | " + " | ".join(CORE_METRICS) + " |\n")
        f.write("|" + "|".join(["---"] * (3 + len(CORE_METRICS))) + "|\n")

        run_ids = sorted(set(r["run_id"] for r in records))
        for run_id in run_ids:
            methods_in_run = sorted(set(r["method"] for r in records if r["run_id"] == run_id))
            for method in methods_in_run:
                subset = [r for r in records if r["run_id"] == run_id and r["method"] == method]
                max_stage = max(r["stage"] for r in subset)
                stage_recs = [r for r in subset if r["stage"] == max_stage]
                row = [run_id, method, str(max_stage)]
                for mk in CORE_METRICS:
                    vals = [r.get(mk) for r in stage_recs]
                    row.append(format_mean_std(vals))
                f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

    print(f"[Aggregate] Markdown saved to {md_path}")

    # ---------- JSON Summary ----------
    summary = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = f"{r['run_id']}__{r['method']}_stage{r['stage']}"
        for mk in CORE_METRICS:
            v = r.get(mk)
            if v is not None:
                summary[key][mk].append(v)

    summary_agg = {}
    for key, metrics_dict in summary.items():
        summary_agg[key] = {}
        for mk, vals in metrics_dict.items():
            if vals:
                summary_agg[key][mk] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "n": len(vals),
                    "values": vals,
                }

    json_path = os.path.join(out_dir, "results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_agg, f, indent=2)
    print(f"[Aggregate] JSON summary saved to {json_path}")

    print("\n[Aggregate] Done. Readable reports generated.")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--outputs_dir", type=str, default="./outputs",
                        help="Directory containing experiment output folders")
    parser.add_argument("--out_dir", type=str, default="./results",
                        help="Directory to save aggregated reports")
    parser.add_argument("--run_id", type=str, default="",
                        help="Optional run_id filter. Empty means aggregate all runs.")
    args = parser.parse_args()
    aggregate(args.outputs_dir, args.out_dir, args.run_id)


if __name__ == "__main__":
    main()
