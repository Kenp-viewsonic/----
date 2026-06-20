"""
Result Visualization for FSCIL Toxic Comment Classification.

Plan Step 4.4 — produces:
  1. Incremental performance curves (per-stage Avg-mAP comparison)
  2. Plastic branch activation decay plot (self-proving consolidation story)
  3. Variant Recall radar chart
  4. Attention map qualitative comparison (placeholder API)

Usage:
    python scripts/plot_results.py --results_json ./results/summary.json --out_dir ./results/figures
"""

import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy matplotlib import — only required when actually plotting
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Color palette (colorblind-friendly) ──
COLORS = {
    "ours": "#0072B2",           # blue
    "seq_finetune": "#D55E00",   # orange
    "task_lora": "#009E73",      # green
    "task_lora_msp": "#F0E442",  # yellow
    "task_lora_adb": "#56B4E9",  # sky blue
    "task_lora_maha": "#E69F00", # gold
    "o_lora": "#CC79A7",         # pink
    "ewc_lora": "#000000",       # black
    "l2p": "#8B4513",            # brown
    "ablation_no_evo": "#999999",
    "ablation_no_dual": "#666666",
    "ablation_no_anchor": "#333333",
}
DEFAULT_COLOR = "#AAAAAA"


def get_color(method: str) -> str:
    """Get color for method, falling back to grey."""
    return COLORS.get(method, DEFAULT_COLOR)


def load_metrics_json(json_path: str) -> dict:
    """Load a stage-level metrics JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_aggregated_json(summary_path: str) -> dict:
    """
    Parse an aggregated results JSON (from aggregate_across_seeds or aggregate_results.py).
    
    Expected structure:
      { "stage_0": {"macro_f1": {"mean": ..., "std": ...}, ...},
        "stage_1": {...}, 
        "overall": {...} }
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════
# Plot 1: Incremental Performance Curves
# ══════════════════════════════════════════════════════════

def plot_incremental_curves(
    results: dict,
    metric: str = "avg_map",
    out_path: str = "incremental_curves.png",
    title: str = "Incremental Performance (Avg-mAP)",
):
    """
    Plot per-stage metric curves for multiple methods with error bars.

    Args:
        results: {method_name: summary_dict} where summary_dict is from parse_aggregated_json
        metric: metric key to plot (e.g. 'avg_map', 'macro_f1')
        out_path: output image path
        title: plot title
    """
    if not _HAS_MPL:
        print("[WARN] matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, summary in sorted(results.items()):
        stages = []
        means = []
        stds = []
        for key, val in sorted(summary.items()):
            if not key.startswith("stage_"):
                continue
            stage = int(key.split("_")[1])
            if metric in val:
                stages.append(stage)
                means.append(val[metric]["mean"])
                stds.append(val[metric]["std"])

        if not stages:
            continue

        color = get_color(method)
        ax.errorbar(
            stages, means, yerr=stds,
            marker="o", linewidth=2, capsize=4,
            color=color, label=method, alpha=0.85,
        )

    ax.set_xlabel("FSCIL Stage", fontsize=13)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.legend(loc="lower left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(stages))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Incremental curves saved to {out_path}")


# ══════════════════════════════════════════════════════════
# Plot 2: Plastic Branch Activation Decay
# ══════════════════════════════════════════════════════════

def plot_plastic_decay(
    decay_log_path: str,
    out_path: str = "plastic_decay.png",
):
    """
    Plot plastic branch ||delta_W|| across epochs (self-proving consolidation story).

    Args:
        decay_log_path: Path to JSON file with structure:
            { "stage_0": {"epoch": [1,2,3,4,5], "plastic_norm": [0.5, 0.3, 0.2, 0.15, 0.1]},
              "stage_1": {...} }
        out_path: output image path
    """
    if not _HAS_MPL:
        print("[WARN] matplotlib not installed; skipping plot.")
        return

    with open(decay_log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_stages = len(data)
    fig, axes = plt.subplots(1, num_stages, figsize=(5 * num_stages, 4), squeeze=False)

    for idx, (stage_key, stage_data) in enumerate(sorted(data.items())):
        ax = axes[0, idx]
        epochs = stage_data.get("epoch", [])
        norms = stage_data.get("plastic_norm", [])

        ax.plot(epochs, norms, marker="s", color=COLORS["ours"], linewidth=2)
        ax.fill_between(epochs, 0, norms, alpha=0.15, color=COLORS["ours"])
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("||Plastic||", fontsize=11)
        ax.set_title(f"Stage {stage_key}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Plastic Branch Activation Decay (Proof of Consolidation)", fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Plastic decay saved to {out_path}")


# ══════════════════════════════════════════════════════════
# Plot 3: Variant Recall Radar Chart
# ══════════════════════════════════════════════════════════

def plot_variant_radar(
    variant_metrics: dict,
    out_path: str = "variant_radar.png",
):
    """
    Radar chart comparing variant recall across methods.

    Args:
        variant_metrics: {
            "method_name": {"leet": 0.85, "space_evasion": 0.72, "symbol_insert": 0.68, "uncovered": 0.45},
            ...
        }
        out_path: output image path
    """
    if not _HAS_MPL:
        print("[WARN] matplotlib not installed; skipping plot.")
        return

    if not variant_metrics:
        print("[WARN] No variant metrics provided; skipping radar chart.")
        return

    # Collect all variant types
    variant_types = set()
    for m_metrics in variant_metrics.values():
        variant_types.update(m_metrics.keys())
    variant_types = sorted(variant_types)

    N = len(variant_types)
    if N < 3:
        print("[WARN] Need at least 3 variant types for radar chart; skipping.")
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for method, m_metrics in sorted(variant_metrics.items()):
        values = [m_metrics.get(vt, 0.0) for vt in variant_types]
        values += values[:1]
        color = get_color(method)
        ax.fill(angles, values, alpha=0.1, color=color)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=method)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([vt.replace("_", " ").title() for vt in variant_types], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Variant Recall Radar", fontsize=14, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Variant radar saved to {out_path}")


# ══════════════════════════════════════════════════════════
# Plot 4: Attention Map Comparison (placeholder API)
# ══════════════════════════════════════════════════════════

def plot_attention_comparison(
    model_ours,
    model_no_pe,
    tokenizer,
    texts: list,
    out_path: str = "attention_comparison.png",
    layer_idx: int = 11,  # last layer
    head_idx: int = 0,
):
    """
    Qualitative attention map comparison: with vs without ToxicAwarePE on l33t text.

    Plan Step 4.4: 证明 Tokenizer 缺陷需要特征先验兜底。

    Args:
        model_ours: Full model with ToxicAwarePE enabled
        model_no_pe: Model with ToxicAwarePE disabled
        tokenizer: tokenizer
        texts: List of texts to compare (e.g. leet variants)
        out_path: output image path
        layer_idx: which transformer layer to extract attention from
        head_idx: which attention head to visualize
    """
    if not _HAS_MPL:
        print("[WARN] matplotlib not installed; skipping plot.")
        return

    import torch

    def get_attention_map(model, text):
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = encoding["input_ids"].to(next(model.parameters()).device)
        attention_mask = encoding["attention_mask"].to(next(model.parameters()).device)

        with torch.no_grad():
            # Run with output_attentions=True via internal hook
            # NOTE: current RobertaToxicClassifier.forward doesn't support output_attentions,
            # so this is a placeholder. Real implementation would need a hook.
            _ = model(input_ids=input_ids, attention_mask=attention_mask, texts=[text], return_rejection=False)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # Placeholder: return random attention for demo structure
        return tokens, np.random.rand(len(tokens), len(tokens))

    n_texts = min(len(texts), 3)
    fig, axes = plt.subplots(2, n_texts, figsize=(4 * n_texts, 8))

    for i, text in enumerate(texts[:n_texts]):
        # With PE
        tokens_ours, attn_ours = get_attention_map(model_ours, text)
        ax = axes[0, i] if n_texts > 1 else axes[0]
        im = ax.imshow(attn_ours, cmap="YlOrRd", aspect="auto")
        ax.set_title(f"With ToxicPE\n{text[:30]}...", fontsize=9)
        ax.set_xticks(range(len(tokens_ours)))
        ax.set_xticklabels(tokens_ours, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tokens_ours)))
        ax.set_yticklabels(tokens_ours, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Without PE
        tokens_no_pe, attn_no_pe = get_attention_map(model_no_pe, text)
        ax = axes[1, i] if n_texts > 1 else axes[1]
        im = ax.imshow(attn_no_pe, cmap="YlOrRd", aspect="auto")
        ax.set_title(f"Without ToxicPE\n{text[:30]}...", fontsize=9)
        ax.set_xticks(range(len(tokens_no_pe)))
        ax.set_xticklabels(tokens_no_pe, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tokens_no_pe)))
        ax.set_yticklabels(tokens_no_pe, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Attention Map: ToxicPE Impact on Leet Variants", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Attention comparison saved to {out_path}")


# ══════════════════════════════════════════════════════════
# Plot 5: Consolidated Results Table (LaTeX format)
# ══════════════════════════════════════════════════════════

def generate_latex_table(
    results: dict,
    metrics: list = None,
    out_path: str = "results_table.tex",
):
    """
    Generate LaTeX-formatted results table for direct paper inclusion.

    Args:
        results: {method_name: summary_dict}
        metrics: list of metric keys to include (default: core set)
        out_path: output .tex path
    """
    if metrics is None:
        metrics = ["avg_map", "macro_f1", "micro_f1", "variant_recall", "auroc"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{FSCIL Results (mean$\pm$std across seeds)}")
    lines.append(r"\label{tab:fscil_results}")
    col_fmt = "l" + "c" * len(metrics)
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")

    # Header
    header = "Method & " + " & ".join(m.replace("_", "\\_") for m in metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in sorted(results.keys()):
        summary = results[method]
        if "overall" not in summary:
            continue
        overall = summary["overall"]
        cells = [method.replace("_", "\\_")]
        for m in metrics:
            if m in overall:
                mean = overall[m].get("mean", "N/A")
                std = overall[m].get("std", 0)
                if isinstance(mean, (int, float)):
                    cells.append(f"${mean:.3f}_{{\\pm{std:.3f}}}$")
                else:
                    cells.append("N/A")
            else:
                cells.append("N/A")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"[Plot] LaTeX table saved to {out_path}")
    return tex_content


# ══════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate all FSCIL result plots")
    parser.add_argument("--results_json", type=str, required=True,
                        help="Path to aggregated results JSON (from aggregate_results.py)")
    parser.add_argument("--out_dir", type=str, default="./results/figures",
                        help="Output directory for figures")
    parser.add_argument("--metric", type=str, default="avg_map",
                        help="Primary metric for incremental curves")
    parser.add_argument("--decay_log", type=str, default=None,
                        help="Path to plastic decay JSON log for decay plot")
    parser.add_argument("--variant_json", type=str, default=None,
                        help="Path to variant recall JSON for radar chart")
    parser.add_argument("--skip_latex", action="store_true",
                        help="Skip LaTeX table generation")
    args = parser.parse_args()

    if not _HAS_MPL:
        print("[ERROR] matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    results = parse_aggregated_json(args.results_json)

    # Determine if results dict is keyed by method or flat
    sample_val = next(iter(results.values()))
    if isinstance(sample_val, dict) and "stage_0" in sample_val:
        # Keyed by method: {"ours": {stage_0: ...}, "task_lora": {...}}
        method_results = results
    else:
        # Single method flat: wrap
        method_results = {"method": results}

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Incremental curves
    plot_incremental_curves(
        method_results,
        metric=args.metric,
        out_path=os.path.join(args.out_dir, "incremental_curves.png"),
        title=f"Incremental Performance ({args.metric.replace('_', ' ').title()})",
    )

    # 2. Plastic decay (if available)
    if args.decay_log and os.path.exists(args.decay_log):
        plot_plastic_decay(
            args.decay_log,
            out_path=os.path.join(args.out_dir, "plastic_decay.png"),
        )

    # 3. Variant radar (if available)
    if args.variant_json and os.path.exists(args.variant_json):
        with open(args.variant_json, "r", encoding="utf-8") as f:
            variant_metrics = json.load(f)
        plot_variant_radar(
            variant_metrics,
            out_path=os.path.join(args.out_dir, "variant_radar.png"),
        )

    # 4. LaTeX table
    if not args.skip_latex:
        generate_latex_table(
            method_results,
            out_path=os.path.join(args.out_dir, "results_table.tex"),
        )

    print(f"\n[Plot] All figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
