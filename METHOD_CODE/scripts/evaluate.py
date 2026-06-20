"""
Unified evaluation script for all methods (Ours, baselines, ablations).

Loads a checkpoint and evaluates on:
  - Stage-specific test set
  - Cumulative test set (all seen classes)
  - OOD test set (next-stage unseen classes)
  - Variant robustness test

Usage:
    # Evaluate main method
    python scripts/evaluate.py --checkpoint outputs/stage_2_seed42/checkpoint-best \
                               --method ours --stage 2 --config configs/base.yaml

    # Evaluate baseline
    python scripts/evaluate.py --checkpoint outputs/task_lora_stage_2_seed42/checkpoint-best \
                               --method task_lora --stage 2 --config configs/base.yaml

    # Evaluate with previous stage checkpoint for CKA stability
    python scripts/evaluate.py --checkpoint outputs/stage_2_seed42/checkpoint-best \
                               --prev_checkpoint outputs/stage_0_seed42/checkpoint-best \
                               --method ours --stage 2 --config configs/base.yaml
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.toxic_dataset import ToxicCommentDataset
from data.fscil_split import FSCILSplitProtocol
from data.variant_generator import VariantGenerator
from utils.evaluator import evaluate_model


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base_cfg, stage_cfg):
    merged = base_cfg.copy()
    for key, val in stage_cfg.items():
        if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(val)
        else:
            merged[key] = val
    return merged


class ActiveLabelDataset:
    def __init__(self, subset, active_indices):
        self.subset = subset
        self.active_indices = active_indices

    def __getitem__(self, idx):
        item = dict(self.subset[idx])
        if "labels" in item:
            item["labels"] = item["labels"][self.active_indices]
        return item

    def __len__(self):
        return len(self.subset)


def load_model_from_checkpoint(checkpoint_dir, method, cfg, num_classes, device):
    """Load model from checkpoint based on method type."""
    from models.roberta_classifier import RobertaToxicClassifier
    from models.dual_lora import apply_dual_lora_to_roberta
    from baselines.lora_utils import apply_lora_to_roberta

    is_baseline = method in ("seq_finetune", "task_lora", "task_lora_msp",
                             "task_lora_adb", "o_lora", "ewc_lora")
    is_ablation = method.startswith("ablation_")

    # Determine prefix/PE/gate config
    if is_baseline:
        prefix_cfg = None
        pe_cfg = {"enable": False}
        gate_cfg = None
    else:
        prefix_cfg = cfg.get("prefix")
        pe_cfg = cfg.get("toxic_pe")
        gate_cfg = cfg.get("rejection_gate")

    model = RobertaToxicClassifier(
        num_classes=num_classes,
        model_name=cfg["model"]["name"],
        prefix_cfg=prefix_cfg,
        lora_cfg=cfg.get("lora") if not is_baseline else None,
        pe_cfg=pe_cfg,
        gate_cfg=gate_cfg,
    )

    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get("target_modules", ["query", "value"])
    rs = lora_cfg.get("rs", 8)
    rp = lora_cfg.get("rp", 4)
    alpha = lora_cfg.get("alpha", 16)
    dropout = lora_cfg.get("dropout", 0.05)

    if is_baseline:
        if method in ("task_lora", "task_lora_msp", "task_lora_adb", "o_lora"):
            apply_lora_to_roberta(
                model.roberta, target_modules, r=rs, lora_alpha=alpha,
                lora_dropout=dropout, multi_stage=True,
            )
        elif method == "ewc_lora":
            apply_lora_to_roberta(
                model.roberta, target_modules, r=rs, lora_alpha=alpha,
                lora_dropout=dropout, multi_stage=False,
            )
    elif is_ablation and "no_dual" in method:
        apply_lora_to_roberta(
            model.roberta, target_modules, r=rs, lora_alpha=alpha,
            lora_dropout=dropout, multi_stage=False,
        )
    else:
        apply_dual_lora_to_roberta(
            model.roberta, target_modules, r_stable=rs, r_plastic=rp,
            lora_alpha=alpha, lora_dropout=dropout,
        )

    # Load weights
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state = load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_path):
        state = torch.load(pytorch_path, map_location="cpu")
    else:
        state = None
        print(f"[Eval] Warning: no checkpoint found at {checkpoint_dir}")

    if state is not None:
        # Pre-resize rejection_gate.V_known buffer if needed
        for key, tensor in state.items():
            if "rejection_gate.V_known" in key and hasattr(model, 'rejection_gate'):
                model.rejection_gate.register_buffer('V_known', torch.zeros(tensor.shape[0], dtype=torch.long))
                break
        model.load_state_dict(state, strict=False)
        print(f"[Eval] Loaded checkpoint from {checkpoint_dir}")

    model.to(device)
    model.eval()
    return model


def extract_cls_representations(model, dataloader, device):
    """Extract CLS hidden states from a dataloader."""
    all_cls = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            texts = batch.get("texts", [])
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts if hasattr(model.forward, "__code__") and "texts" in model.forward.__code__.co_varnames else None,
                return_rejection=False,
            )
            all_cls.append(outputs["cls_hidden"].cpu().numpy())
    return np.concatenate(all_cls, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--prev_checkpoint", type=str, default=None,
                        help="Previous stage checkpoint for CKA stability computation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    cfg = base_cfg

    output_dir = args.output_dir or os.path.join(args.checkpoint, "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataset = ToxicCommentDataset(
        csv_path=cfg["data"]["dataset_path"],
        tokenizer_name=cfg["model"]["name"],
        max_length=cfg["data"]["max_length"],
        filter_toxic=cfg["data"].get("filter_toxic", True),
    )

    stage_defs = cfg.get("fscil", {}).get("stages")
    if stage_defs is not None:
        stage_definitions = {}
        for i, st in enumerate(stage_defs):
            stage_definitions[i] = {
                "classes": st["classes"],
                "shots": st["shots_per_class"],
            }
    else:
        stage_definitions = None
    
    split_protocol = FSCILSplitProtocol(
        dataset=dataset,
        output_dir=os.path.join(output_dir, "data_splits"),
        seed=args.seed,
        stage_definitions=stage_definitions,
    )
    split_protocol.create_splits()
    split_protocol.save_splits()

    active_label_indices = split_protocol.get_active_labels(args.stage)
    num_classes = len(active_label_indices)

    # Cumulative test set
    cumulative_indices = []
    for s in range(args.stage + 1):
        cumulative_indices.extend(split_protocol.get_stage_dataset(s, "test").indices)
    cumulative_dataset = ActiveLabelDataset(
        torch.utils.data.Subset(dataset, cumulative_indices),
        active_label_indices,
    )
    cumulative_loader = DataLoader(
        cumulative_dataset,
        batch_size=cfg["training"].get("per_device_eval_batch_size", 16),
        shuffle=False,
        collate_fn=ToxicCommentDataset.collate_fn,
    )

    # OOD loader (next stage unseen classes)
    ood_loader = None
    if args.stage < 2:
        try:
            next_stage_classes = split_protocol.get_active_labels(args.stage + 1)
            if len(next_stage_classes) > 0:
                ood_dataset = ActiveLabelDataset(
                    split_protocol.get_stage_dataset(args.stage + 1, "train"),
                    next_stage_classes,
                )
                ood_loader = DataLoader(
                    ood_dataset,
                    batch_size=cfg["training"].get("per_device_eval_batch_size", 16),
                    shuffle=False,
                    collate_fn=ToxicCommentDataset.collate_fn,
                )
        except Exception as e:
            print(f"[OOD] Could not build OOD loader: {e}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.method, cfg, num_classes, device)

    # Reference CLS representations for CKA (from prev checkpoint)
    ref_reps = None
    if args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
        print("[Eval] Loading previous stage model for CKA...")
        prev_model = load_model_from_checkpoint(
            args.prev_checkpoint, args.method, cfg, num_classes, device
        )
        ref_reps = extract_cls_representations(prev_model, cumulative_loader, device)
        del prev_model
        torch.cuda.empty_cache()

    # Variant generator
    variant_gen = VariantGenerator(seed=args.seed)

    # Tail class = last added class in this stage
    tail_class_indices = active_label_indices[-1:] if active_label_indices else None

    # Run unified evaluation
    print("\n[Eval] Running unified evaluation...")
    metrics = evaluate_model(
        model=model,
        dataloader=cumulative_loader,
        device=device,
        tokenizer=dataset.tokenizer,
        variant_generator=variant_gen,
        ood_dataloader=ood_loader,
        ref_representations=ref_reps,
        tail_class_indices=tail_class_indices,
        threshold=0.5,
    )

    print(f"\n[Eval] Results for {args.method} stage {args.stage}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save results
    result_path = os.path.join(output_dir, f"eval_{args.method}_stage{args.stage}.json")
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Eval] Results saved to {result_path}")


if __name__ == "__main__":
    main()