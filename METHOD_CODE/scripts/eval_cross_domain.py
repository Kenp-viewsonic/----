"""
Cross-Domain Robustness Evaluation (Plan Step 4.4).

Evaluates FSCIL-trained model on implicit-hate corpus (EMNLP 2021) to verify
that learned "toxicity semantic kernel" transfers beyond surface-form memorization.

The model is frozen after full Jigsaw FSCIL training; implicit-hate samples are
treated as "novel semantic variants" of known toxicity categories.

Usage:
    # Requires implicit-hate corpus downloaded separately (not bundled)
    python scripts/eval_cross_domain.py \
        --checkpoint outputs/stage_2_seed42/checkpoint-best \
        --method ours \
        --config configs/base.yaml \
        --implicit_hate_path /path/to/implicit_hate.csv \
        --mapping configs/implicit_hate_mapping.yaml

Expected output:
    - Per-category zero-shot macro-F1, AUROC, variant_recall
    - Overall cross-domain transfer score
"""

import os
import sys
import argparse
import json
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.roberta_classifier import RobertaToxicClassifier
from models.dual_lora import apply_dual_lora_to_roberta
from baselines.lora_utils import apply_lora_to_roberta
from utils.metrics import compute_fscil_metrics, compute_variant_recall


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint_state(checkpoint_dir):
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu")
    return None


def detect_num_classes(checkpoint_state):
    for key, tensor in checkpoint_state.items():
        if "classifier" in key and "weight" in key and len(tensor.shape) == 2:
            return tensor.shape[0]
    return None


def load_model_from_checkpoint(checkpoint_dir, method, cfg, num_classes, device):
    """Load model from checkpoint based on method type."""
    from models.roberta_classifier import RobertaToxicClassifier
    from baselines.runner import L2PClassifier
    from models.baseline_classifier import BaselineToxicClassifier

    is_baseline = method in ("seq_finetune", "task_lora", "task_lora_msp",
                             "task_lora_adb", "task_lora_maha", "o_lora", "ewc_lora")

    if method == "l2p":
        base_model = BaselineToxicClassifier(
            num_classes=num_classes,
            model_name=cfg["model"]["name"],
        )
        model = L2PClassifier(
            base_model=base_model,
            pool_size=cfg.get("l2p", {}).get("pool_size", 20),
            prompt_length=cfg.get("l2p", {}).get("prompt_length", 5),
            top_k=cfg.get("l2p", {}).get("top_k", 5),
            num_stages=cfg.get("l2p", {}).get("num_stages", 3),
        )
    elif is_baseline:
        from models.baseline_classifier import BaselineToxicClassifier
        model = BaselineToxicClassifier(
            num_classes=num_classes,
            model_name=cfg["model"]["name"],
        )
        lora_cfg = cfg.get("lora", {})
        # Inject LoRA for task_lora/o_lora/ewc variants
        if method in ("task_lora", "task_lora_msp", "task_lora_adb", "task_lora_maha"):
            from models.baseline_lora import TaskLoRALayer
            apply_lora_to_roberta(
                model.roberta, TaskLoRALayer,
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
                r=lora_cfg.get("rs", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.05),
            )
        elif method == "o_lora":
            from models.baseline_lora import OrthLoRALayer
            apply_lora_to_roberta(
                model.roberta, OrthLoRALayer,
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
                r=lora_cfg.get("rs", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.05),
            )
        elif method == "ewc_lora":
            from models.baseline_lora import SingleBranchLoRALayer
            apply_lora_to_roberta(
                model.roberta, SingleBranchLoRALayer,
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
                r=lora_cfg.get("rs", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                lora_dropout=lora_cfg.get("dropout", 0.05),
            )
    else:
        # Ours
        pe_cfg = cfg.get("toxic_pe", {})
        prefix_cfg = cfg.get("prefix", {})
        gate_cfg = cfg.get("rejection_gate", {})
        lora_cfg = cfg.get("lora", {})

        model = RobertaToxicClassifier(
            num_classes=num_classes,
            model_name=cfg["model"]["name"],
            prefix_cfg=prefix_cfg,
            pe_cfg=pe_cfg,
            gate_cfg=gate_cfg,
            lora_cfg=lora_cfg,
        )
        apply_dual_lora_to_roberta(
            model.roberta,
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
            r_stable=lora_cfg.get("rs", 8),
            r_plastic=lora_cfg.get("rp", 4),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
        )

    checkpoint_state = load_checkpoint_state(checkpoint_dir)
    if checkpoint_state is not None:
        # Pre-resize rejection_gate.V_known buffer if needed
        for key, tensor in checkpoint_state.items():
            if "rejection_gate.V_known" in key and hasattr(model, 'rejection_gate'):
                model.rejection_gate.register_buffer('V_known', torch.zeros(tensor.shape[0], dtype=torch.long))
                break
        model.load_state_dict(checkpoint_state, strict=False)
        print(f"[CrossDomain] Loaded checkpoint from {checkpoint_dir}")

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation on implicit-hate corpus"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to final-stage checkpoint directory")
    parser.add_argument("--method", type=str, required=True,
                        choices=["ours", "seq_finetune", "task_lora", "task_lora_msp",
                                 "task_lora_adb", "task_lora_maha", "o_lora", "ewc_lora", "l2p"])
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--implicit_hate_path", type=str, required=True,
                        help="Path to implicit-hate CSV (text, label columns)")
    parser.add_argument("--mapping", type=str, default="configs/implicit_hate_mapping.yaml")
    parser.add_argument("--output_dir", type=str, default="./outputs/cross_domain")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max implicit samples (-1 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config)
    mapping_cfg = load_config(args.mapping)
    class_mapping = mapping_cfg.get("mapping", {})
    eval_cfg = mapping_cfg.get("evaluation", {})
    threshold = eval_cfg.get("threshold", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint_state = load_checkpoint_state(args.checkpoint)
    num_classes = detect_num_classes(checkpoint_state) if checkpoint_state else 5
    if num_classes is None:
        num_classes = 5
    model = load_model_from_checkpoint(args.checkpoint, args.method, cfg, num_classes, device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    # Load implicit-hate data
    if not os.path.exists(args.implicit_hate_path):
        print(f"[ERROR] Implicit hate corpus not found: {args.implicit_hate_path}")
        print("  Download from: https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset")
        sys.exit(1)

    df = pd.read_csv(args.implicit_hate_path)
    print(f"[CrossDomain] Loaded {len(df)} implicit hate samples")

    if args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)

    # Jigsaw label names (order: obscene, insult, threat, identity_hate, severe_toxic)
    JIGSAW_LABELS = ["obscene", "insult", "threat", "identity_hate", "severe_toxic"]

    # Evaluate per-category
    all_results = {}
    all_probs_list = []
    all_binary_labels = []

    for category, cat_info in class_mapping.items():
        target_labels = cat_info.get("target_labels", ["toxic"])
        cat_df = df[df.get("label", df.get("class", pd.Series([category] * len(df)))) == category]

        if len(cat_df) == 0:
            print(f"  [Skip] Category '{category}': no samples found")
            continue

        texts = cat_df["text"].astype(str).tolist()
        n = len(texts)
        print(f"\n[CrossDomain] Category: {category} ({n} samples)")

        # Tokenize
        encoding = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=cfg["data"].get("max_length", 128),
            return_tensors="pt",
        )

        dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"])
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_logits = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                texts=texts[:len(input_ids)],
                                return_rejection=False)
                all_logits.append(outputs["logits"].cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)  # [N, C]
        probs = 1.0 / (1.0 + np.exp(-all_logits))

        # Build multi-label targets based on mapping
        labels = np.zeros((n, num_classes))
        for tl in target_labels:
            if tl == "toxic":
                # Map "toxic" to any toxicity → set all non-severe_toxic classes
                labels[:, :4] = 1
            elif tl in JIGSAW_LABELS:
                idx = JIGSAW_LABELS.index(tl)
                if idx < num_classes:
                    labels[:, idx] = 1

        # Metrics
        cat_metrics = compute_fscil_metrics(probs, labels, threshold=threshold)

        # Variant recall: any positive detection
        detected = (probs > threshold).any(axis=1).mean()
        cat_metrics["variant_recall"] = float(detected)

        print(f"  Macro-F1: {cat_metrics['macro_f1']:.4f}, "
              f"Variant Recall: {detected:.4f}, "
              f"Avg-mAP: {cat_metrics['avg_map']:.4f}")

        all_results[category] = {
            "n_samples": n,
            **{k: float(v) if isinstance(v, (np.floating,)) else v
               for k, v in cat_metrics.items() if k != "per_class_recall"},
        }
        all_probs_list.append(probs)
        all_binary_labels.append(labels)

    # Overall cross-domain metrics
    if all_probs_list:
        all_probs = np.concatenate(all_probs_list, axis=0)
        all_labels = np.concatenate(all_binary_labels, axis=0)
        overall = compute_fscil_metrics(all_probs, all_labels, threshold=threshold)
        overall_detected = (all_probs > threshold).any(axis=1).mean()
        overall["variant_recall"] = float(overall_detected)
        all_results["overall"] = {k: float(v) if isinstance(v, (np.floating,)) else v
                                   for k, v in overall.items() if k != "per_class_recall"}

        print(f"\n[CrossDomain] OVERALL: "
              f"Macro-F1={overall['macro_f1']:.4f}, "
              f"Variant Recall={overall_detected:.4f}")

    # Save results
    output_path = os.path.join(args.output_dir, f"cross_domain_{args.method}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[CrossDomain] Results saved to {output_path}")


if __name__ == "__main__":
    main()
