"""
Ablation study runner for the proposed method.

Variants:
  - no_evo:     disable semantic evolution loss (lambda_evo = 0)
  - no_dual:    disable plastic branch (rp = 0, only stable LoRA)
  - no_anchor:  use random prefix initialization instead of K-means anchor

Usage:
    python scripts/run_ablation.py --variant no_evo --stage 0 --config configs/base.yaml
    python scripts/run_ablation.py --variant no_dual --stage 0 --config configs/base.yaml
    python scripts/run_ablation.py --variant no_anchor --stage 0 --config configs/base.yaml
"""

import os
import sys
import argparse
import yaml
import json
import math
import torch
from transformers import TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.toxic_dataset import ToxicCommentDataset
from data.fscil_split import FSCILSplitProtocol
from data.variant_generator import VariantGenerator
from models.roberta_classifier import RobertaToxicClassifier
from models.dual_lora import apply_dual_lora_to_roberta
from trainers.incremental_trainer import IncrementalTrainer
from utils.evaluator import evaluate_model


class ActiveLabelDataset:
    """Wrapper to extract only active labels from a Subset."""
    def __init__(self, subset, active_indices):
        self.subset = subset
        self.active_indices = active_indices

    def __getitem__(self, idx):
        item = self.subset[idx]
        item = dict(item)
        if "labels" in item:
            item["labels"] = item["labels"][self.active_indices]
        return item

    def __len__(self):
        return len(self.subset)


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


def detect_num_classes_from_checkpoint(checkpoint_state):
    for key, tensor in checkpoint_state.items():
        if "classifier" in key and "weight" in key and len(tensor.shape) == 2:
            return tensor.shape[0]
    return None


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True,
                        choices=["no_evo", "no_dual", "no_anchor"])
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--prev_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--stages_config", type=str, default=None,
                        help="Override path to stages.yaml (used by smoke tests).")
    args = parser.parse_args()

    # Load configs
    base_cfg = load_config(args.config)
    stages_cfg_path = args.stages_config or "configs/stages.yaml"
    if os.path.exists(stages_cfg_path):
        stages_cfg = load_config(stages_cfg_path)
        stage_key = ["base", "stage1", "stage2"][args.stage]
        if stage_key in stages_cfg:
            cfg = merge_configs(base_cfg, stages_cfg[stage_key])
        else:
            cfg = base_cfg
    else:
        cfg = base_cfg

    # Apply ablation overrides
    if args.variant == "no_evo":
        cfg.setdefault("loss_weights", {})
        cfg["loss_weights"]["lambda_evo"] = 0.0
        print("[Ablation] Disabled evolution loss (lambda_evo = 0)")
    elif args.variant == "no_dual":
        cfg.setdefault("lora", {})
        cfg["lora"]["rp"] = 0
        print("[Ablation] Disabled plastic branch (rp = 0, single stable LoRA)")
    elif args.variant == "no_anchor":
        cfg.setdefault("prefix", {})
        cfg["prefix"]["init_random"] = True
        print("[Ablation] Using random prefix initialization (no K-means anchor)")

    output_dir = args.output_dir or f"./outputs/ablation_{args.variant}_stage_{args.stage}_seed{args.seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Save modified config for reproducibility
    with open(os.path.join(output_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = ToxicCommentDataset(
        csv_path=cfg["data"]["dataset_path"],
        tokenizer_name=cfg["model"]["name"],
        max_length=cfg["data"]["max_length"],
        filter_toxic=cfg["data"].get("filter_toxic", True),
    )
    print(f"[Data] Loaded {len(dataset)} toxic samples.")

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
    splits = split_protocol.create_splits()
    split_protocol.save_splits()

    active_label_indices = split_protocol.get_active_labels(args.stage)
    num_active_classes = len(active_label_indices)

    # Stage > 0: mix coreset replay samples into training
    if args.stage > 0:
        coreset_indices_for_replay = split_protocol.splits.get(args.stage, {}).get("coreset", [])
        if coreset_indices_for_replay:
            stage_train_indices = split_protocol.splits[args.stage]["train"]
            n_new = len(stage_train_indices)
            n_replay = len(coreset_indices_for_replay)
            # Oversample new-class data to prevent replay from overwhelming new-class signal
            if n_replay > n_new and n_new > 0:
                oversample_ratio = math.ceil(n_replay / n_new)
                oversampled_new = stage_train_indices * oversample_ratio
            else:
                oversampled_new = stage_train_indices
            combined_indices = oversampled_new + coreset_indices_for_replay
            combined_subset = torch.utils.data.Subset(dataset, combined_indices)
            train_dataset = ActiveLabelDataset(combined_subset, active_label_indices)
            print(f"[Replay] New-class: {n_new} x{len(oversampled_new)//n_new if n_new>0 else 1}={len(oversampled_new)}, "
                  f"replay: {n_replay}, total: {len(train_dataset)}")
        else:
            train_dataset = ActiveLabelDataset(
                split_protocol.get_stage_dataset(args.stage, "train"),
                active_label_indices,
            )
    else:
        train_dataset = ActiveLabelDataset(
            split_protocol.get_stage_dataset(args.stage, "train"),
            active_label_indices,
        )
    eval_dataset = ActiveLabelDataset(
        split_protocol.get_stage_dataset(args.stage, "test"),
        active_label_indices,
    )

    # Cumulative test set
    cumulative_indices = []
    for s in range(args.stage + 1):
        cumulative_indices.extend(split_protocol.get_stage_dataset(s, "test").indices)
    cumulative_dataset = ActiveLabelDataset(
        torch.utils.data.Subset(dataset, cumulative_indices),
        active_label_indices,
    )

    # Old validation for semantic consolidation
    old_val_dataset = None
    if args.stage > 0:
        old_val_dataset = ActiveLabelDataset(
            split_protocol.get_stage_dataset(0, "test"),
            split_protocol.get_active_labels(0),
        )

    # Model init
    init_num_classes = num_active_classes
    checkpoint_state = None
    if args.stage > 0 and args.prev_checkpoint:
        checkpoint_state = load_checkpoint_state(args.prev_checkpoint)
        detected = detect_num_classes_from_checkpoint(checkpoint_state) if checkpoint_state else None
        if detected:
            init_num_classes = detected

    model = RobertaToxicClassifier(
        num_classes=init_num_classes,
        model_name=cfg["model"]["name"],
        prefix_cfg=cfg.get("prefix"),
        lora_cfg=cfg.get("lora"),
        pe_cfg=cfg.get("toxic_pe"),
        gate_cfg=cfg.get("rejection_gate"),
    )

    lora_cfg = cfg.get("lora", {})
    apply_dual_lora_to_roberta(
        model.roberta,
        target_modules=lora_cfg.get("target_modules", ["query", "value"]),
        r_stable=lora_cfg.get("rs", 8),
        r_plastic=lora_cfg.get("rp", 4),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
    )

    if checkpoint_state is not None:
        # Pre-resize rejection_gate.V_known buffer to match checkpoint shape
        for key, tensor in checkpoint_state.items():
            if "rejection_gate.V_known" in key:
                model.rejection_gate.register_buffer('V_known', torch.zeros(tensor.shape[0], dtype=torch.long))
                print(f"[Checkpoint] Pre-resized V_known buffer to shape {tensor.shape}")
                break
        model.load_state_dict(checkpoint_state, strict=False)
        if num_active_classes > init_num_classes:
            model.expand_classifier(num_active_classes)

    model.set_stage(args.stage)
    model.to(device)

    # Stage > 0: freeze stable branch and base RoBERTa
    if args.stage > 0:
        print(f"[Stage {args.stage}] Freezing stable branch and base RoBERTa, training plastic + new head.")
        for param in model.roberta.parameters():
            param.requires_grad = False
        from models.dual_lora import DualBranchLoRALayer
        for module in model.modules():
            if isinstance(module, DualBranchLoRALayer):
                module.freeze_stable()
                module.unfreeze_plastic()
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.rejection_gate.parameters():
            param.requires_grad = True

    # Training args
    training_cfg = cfg.get("training", {})
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        num_train_epochs=training_cfg.get("num_train_epochs", 5),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        logging_steps=training_cfg.get("logging_steps", 50),
        eval_strategy=training_cfg.get("eval_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_avg_map"),
        greater_is_better=training_cfg.get("greater_is_better", True),
        fp16=training_cfg.get("fp16", False),
        seed=args.seed,
        report_to=[],
    )

    loss_weights = cfg.get("loss_weights", {})

    # --- Build coreset dataloader for semantic interference evaluation (#1 fix) ---
    coreset_dataloader = None
    if args.stage > 0:
        coreset_indices = split_protocol.splits.get(args.stage, {}).get("coreset", [])
        if coreset_indices:
            coreset_dataset = torch.utils.data.Subset(dataset, coreset_indices)
            coreset_dataloader = torch.utils.data.DataLoader(
                coreset_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=ToxicCommentDataset.collate_fn,
            )
            print(f"[Coreset] Built coreset dataloader with {len(coreset_indices)} samples.")

    # --- Pre-stage 0: initialize prefix from K-means BEFORE training (#3 fix) ---
    if args.stage == 0 and not model.prefix_module._proto_initialized and args.variant != "no_anchor":
        print("[Prefix] Running pre-training K-means initialization on base training data...")
        pre_train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=ToxicCommentDataset.collate_fn,
        )
        cls_embeddings = []
        max_samples = 500
        count = 0
        model.eval()
        with torch.no_grad():
            for batch in pre_train_loader:
                if count >= max_samples:
                    break
                batch_tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = model(input_ids=batch_tensors["input_ids"], attention_mask=batch_tensors["attention_mask"],
                            texts=batch_tensors.get("texts"), return_rejection=False)
                cls_embeddings.append(out["cls_hidden"].cpu())
                count += batch_tensors["input_ids"].size(0)
        if cls_embeddings:
            all_cls = torch.cat(cls_embeddings, dim=0)
            model.prefix_module.init_from_kmeans(all_cls)
            if hasattr(model.prefix_module, '_kmeans_centroids_'):
                model.rejection_gate.set_prototypes(model.prefix_module._kmeans_centroids_)
        model.train()
        print("[Prefix] K-means initialization complete.")

    # --- Initialize known toxic vocabulary for rejection gate (#7 fix) ---
    if not model.rejection_gate.vocab_initialized:
        toxic_subwords = [
            "hate", "kill", "die", "stupid", "idiot", "dumb", "ugly", "fat", "suck",
            "worst", "terrible", "awful", "horrible", "pathetic", "useless", "trash",
            "garbage", "crap", "damn", "hell", "fuck", "shit", "bastard", "ass",
            "moron", "loser", "racist", "nazi", "sexist",
            "h4te", "k1ll", "d1e", "st0pid", "1diot",
            "Ġhate", "Ġkill", "Ġdie", "Ġstupid", "Ġidiot",
        ]
        known_ids = []
        for word in toxic_subwords:
            token_id = dataset.tokenizer.convert_tokens_to_ids(word)
            if token_id != dataset.tokenizer.unk_token_id:
                known_ids.append(token_id)
        known_ids = list(set(known_ids))
        if known_ids:
            model.rejection_gate.set_known_vocab(known_ids, tokenizer=dataset.tokenizer)
            print(f"[Gate] Initialized known toxic vocabulary with {len(known_ids)} token IDs.")

    trainer = IncrementalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=dataset.tokenizer,
        stage_id=args.stage,
        loss_weights=loss_weights,
        old_val_dataset=old_val_dataset,
        coreset_dataloader=coreset_dataloader,
        data_collator=ToxicCommentDataset.collate_fn,
    )

    print(f"\n[Ablation {args.variant}] Stage {args.stage} training...")
    print(f"  Active classes: {active_label_indices}")
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Eval size: {len(eval_dataset)}")
    trainer.train()

    # Unified evaluation via evaluator.py
    print(f"\n[Ablation {args.variant}] Stage {args.stage} unified evaluation...")

    cumulative_loader = torch.utils.data.DataLoader(
        cumulative_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=ToxicCommentDataset.collate_fn,
    )

    variant_gen = VariantGenerator(seed=args.seed)

    ood_dataloader = None
    if args.stage < 2:
        try:
            next_stage_classes = split_protocol.get_active_labels(args.stage + 1)
            if len(next_stage_classes) > 0:
                ood_dataset = ActiveLabelDataset(
                    split_protocol.get_stage_dataset(args.stage + 1, "train"),
                    next_stage_classes,
                )
                ood_dataloader = torch.utils.data.DataLoader(
                    ood_dataset,
                    batch_size=training_args.per_device_eval_batch_size,
                    shuffle=False,
                    collate_fn=ToxicCommentDataset.collate_fn,
                )
        except Exception as e:
            print(f"[OOD] Could not build OOD loader: {e}")

    metrics = evaluate_model(
        model=model,
        dataloader=cumulative_loader,
        device=device,
        tokenizer=dataset.tokenizer,
        variant_generator=variant_gen,
        ood_dataloader=ood_dataloader,
        tail_class_indices=[active_label_indices[-1]] if active_label_indices else None,
        threshold=0.5,
    )

    print(f"\n[Ablation {args.variant}] Stage {args.stage} metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    metrics_path = os.path.join(output_dir, f"metrics_stage{args.stage}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_path = os.path.join(output_dir, "checkpoint-best")
    trainer.save_model(save_path)
    print(f"\n[Ablation {args.variant}] Model saved to {save_path}")


if __name__ == "__main__":
    main()
