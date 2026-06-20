"""
Unified baseline runner for FSCIL Toxic Comment Classification.

Methods:
  - seq_finetune: Sequential fine-tuning without anti-forgetting
  - task_lora: Per-stage independent LoRA, freeze old LoRAs
  - task_lora_msp: Task-LoRA + Max Softmax Probability for OOD
  - task_lora_adb: Task-LoRA + Adaptive Decision Boundary for OOD
  - task_lora_maha: Task-LoRA + Mahalanobis Distance for OOD
  - o_lora: Orthogonal LoRA (O-LoRA, 2601.02232)
  - ewc_lora: EWC + LoRA

Usage:
    python scripts/run_baseline.py --method task_lora --stage 0 --config configs/base.yaml
    python scripts/run_baseline.py --method o_lora --stage 1 --prev_checkpoint outputs/o_lora_stage_0_seed42/checkpoint-best
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from transformers import TrainingArguments, Trainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.toxic_dataset import ToxicCommentDataset
from data.fscil_split import FSCILSplitProtocol
from models.roberta_classifier import RobertaToxicClassifier
from baselines.lora_utils import (
    apply_lora_to_roberta,
    MultiStageLoRALayer,
    SingleBranchLoRALayer,
    EWCManager,
)
from utils.metrics import (
    compute_fscil_metrics,
    compute_auroc_fpr95,
    compute_tail_recall,
    compute_variant_recall,
    compute_cka,
    compute_forgetting,
)


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


class BaselineTrainer(Trainer):
    """
    Custom trainer for baselines.
    Supports O-LoRA orthogonality loss and EWC penalty.
    """

    def __init__(
        self,
        model,
        args,
        stage_id=0,
        baseline_method=None,
        ewc_manager=None,
        orth_lambda=0.0,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.stage_id = stage_id
        self.baseline_method = baseline_method
        self.ewc_manager = ewc_manager
        self.orth_lambda = orth_lambda
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            texts=inputs.get("texts"),
            labels=labels,
            return_rejection=False,
        )
        loss = outputs.get("loss", torch.tensor(0.0, device=inputs["input_ids"].device))

        # O-LoRA orthogonality loss
        if self.orth_lambda > 0 and self.baseline_method == "o_lora":
            orth_loss = 0.0
            for module in self.model.modules():
                if isinstance(module, MultiStageLoRALayer):
                    orth_loss = orth_loss + module.get_orthogonality_loss()
            if isinstance(orth_loss, torch.Tensor):
                loss = loss + self.orth_lambda * orth_loss

        # EWC penalty
        if self.ewc_manager is not None and self.baseline_method == "ewc_lora":
            loss = loss + self.ewc_manager.penalty(model)

        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return metrics

        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_logits = []
        all_labels = []
        all_cls = []
        all_texts = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    texts=batch.get("texts"),
                    labels=batch.get("labels"),
                    return_rejection=False,
                )
                all_logits.append(outputs["logits"].cpu())
                all_cls.append(outputs["cls_hidden"].cpu())
                all_labels.append(batch["labels"].cpu())
                if "texts" in batch:
                    all_texts.extend(batch["texts"])

        all_logits = torch.cat(all_logits).numpy()
        all_cls = torch.cat(all_cls).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid

        # FSCIL metrics
        fscil = compute_fscil_metrics(all_probs, all_labels)
        for k, v in fscil.items():
            if isinstance(v, dict):
                continue
            metrics[f"{metric_key_prefix}_{k}"] = v

        # Tail recall
        tail = compute_tail_recall(all_probs, all_labels)
        metrics[f"{metric_key_prefix}_tail_recall"] = tail["tail_recall"]

        # OOD metrics for MSP / ADB / Maha baselines
        if self.baseline_method in ("task_lora_msp", "task_lora_adb", "task_lora_maha"):
            known_mask = (all_labels >= 0).any(axis=1)
            if (~known_mask).sum() > 0 and known_mask.sum() > 0:
                max_prob = all_probs.max(axis=1)
                # MSP: lower max_prob -> more uncertain -> more OOD-like
                known_scores = -max_prob[known_mask]
                unknown_scores = -max_prob[~known_mask]
                ood = compute_auroc_fpr95(known_scores, unknown_scores)
                metrics[f"{metric_key_prefix}_auroc"] = ood["auroc"]
                metrics[f"{metric_key_prefix}_fpr95"] = ood["fpr95"]

        self.model.train()
        return metrics


def setup_model_for_stage(model, method, stage_id, cfg, device):
    """
    Configure model parameters for the current stage based on baseline method.
    """
    # Ensure classifier head is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    if method == "seq_finetune":
        # Full fine-tune: freeze base RoBERTa, tune classifier + last 2 encoder layers
        for param in model.roberta.parameters():
            param.requires_grad = False
        # Unfreeze last 2 layers
        for layer in model.roberta.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"[seq_finetune] Stage {stage_id}: classifier + last 2 encoder layers trainable.")

    elif method in ("task_lora", "task_lora_msp", "task_lora_adb", "task_lora_maha", "o_lora"):
        # Multi-stage LoRA: freeze base RoBERTa, add new LoRA stage
        for param in model.roberta.parameters():
            param.requires_grad = False
        # Add new LoRA stage if stage > 0
        if stage_id > 0:
            for module in model.modules():
                if isinstance(module, MultiStageLoRALayer):
                    module.add_stage()
        # Ensure all LoRA parameters are trainable (new stage only after add_stage)
        for module in model.modules():
            if isinstance(module, MultiStageLoRALayer):
                for p in module.parameters():
                    p.requires_grad = True
        print(f"[{method}] Stage {stage_id}: base frozen, new LoRA stage added.")

    elif method == "ewc_lora":
        # Single LoRA: freeze base, keep LoRA trainable
        for param in model.roberta.parameters():
            param.requires_grad = False
        for module in model.modules():
            if isinstance(module, SingleBranchLoRALayer):
                for p in module.parameters():
                    p.requires_grad = True
        print(f"[ewc_lora] Stage {stage_id}: base frozen, LoRA trainable.")

    model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["seq_finetune", "task_lora", "task_lora_msp",
                                 "task_lora_adb", "task_lora_maha", "o_lora", "ewc_lora"])
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

    output_dir = args.output_dir or f"./outputs/{args.method}_stage_{args.stage}_seed{args.seed}"
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Data
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

    train_dataset = ActiveLabelDataset(
        split_protocol.get_stage_dataset(args.stage, "train"),
        active_label_indices,
    )
    eval_dataset = ActiveLabelDataset(
        split_protocol.get_stage_dataset(args.stage, "test"),
        active_label_indices,
    )

    # Old validation for EWC Fisher computation (stage > 0)
    # Use active_label_indices (all seen classes) so label dimensions match model output
    old_val_dataset = None
    if args.stage > 0:
        old_val_dataset = ActiveLabelDataset(
            split_protocol.get_stage_dataset(0, "test"),
            active_label_indices,
        )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init model
    init_num_classes = num_active_classes
    checkpoint_state = None
    if args.stage > 0 and args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
        print(f"[Checkpoint] Loading from {args.prev_checkpoint}")
        safetensors_path = os.path.join(args.prev_checkpoint, "model.safetensors")
        pytorch_path = os.path.join(args.prev_checkpoint, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            checkpoint_state = load_file(safetensors_path, device="cpu")
        elif os.path.exists(pytorch_path):
            checkpoint_state = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.prev_checkpoint}")

        for key, tensor in checkpoint_state.items():
            if "classifier" in key and "weight" in key and len(tensor.shape) == 2:
                init_num_classes = tensor.shape[0]
                print(f"[Checkpoint] Detected {init_num_classes} classes.")
                break

    # Baselines do NOT use toxic semantic prefix or toxic PE by default
    model = RobertaToxicClassifier(
        num_classes=init_num_classes,
        model_name=cfg["model"]["name"],
        prefix_cfg=None,  # no prefix for baselines
        lora_cfg=None,
        pe_cfg={"enable": False},  # no toxic PE for baselines
        gate_cfg=None,  # no rejection gate for baselines
    )

    # Apply LoRA based on method
    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get("target_modules", ["query", "value"])
    r = lora_cfg.get("r", 8)
    alpha = lora_cfg.get("alpha", 16)
    dropout = lora_cfg.get("dropout", 0.05)

    if args.method in ("task_lora", "task_lora_msp", "task_lora_adb", "o_lora"):
        apply_lora_to_roberta(
            model.roberta,
            target_modules=target_modules,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            multi_stage=True,
        )
    elif args.method == "ewc_lora":
        apply_lora_to_roberta(
            model.roberta,
            target_modules=target_modules,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            multi_stage=False,
        )

    # Load checkpoint
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state, strict=False)
        if num_active_classes > init_num_classes:
            model.expand_classifier(num_active_classes)

    model.set_stage(args.stage)
    setup_model_for_stage(model, args.method, args.stage, cfg, device)

    # EWC setup
    ewc_manager = None
    if args.method == "ewc_lora" and args.stage > 0:
        ewc_manager = EWCManager(model, importance=cfg.get("ewc_importance", 1000.0), device=device)
        if old_val_dataset is not None:
            old_loader = torch.utils.data.DataLoader(
                old_val_dataset,
                batch_size=cfg["training"].get("per_device_eval_batch_size", 16),
                shuffle=False,
                collate_fn=ToxicCommentDataset.collate_fn,
            )
            print("[EWC] Computing Fisher Information...")
            ewc_manager.compute_fisher(old_loader, num_batches=50)
            ewc_manager.store_optimal_params()
            print("[EWC] Fisher computed and optimal params stored.")

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

    # Orth lambda for O-LoRA
    orth_lambda = cfg.get("o_lora", {}).get("orth_lambda", 1e-3) if args.method == "o_lora" else 0.0

    trainer = BaselineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=dataset.tokenizer,
        stage_id=args.stage,
        baseline_method=args.method,
        ewc_manager=ewc_manager,
        orth_lambda=orth_lambda,
        data_collator=ToxicCommentDataset.collate_fn,
    )

    print(f"\n[Stage {args.stage}] Starting baseline '{args.method}'...")
    print(f"  Active classes: {active_label_indices}")
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Eval size: {len(eval_dataset)}")

    trainer.train()

    # --- Comprehensive evaluation via evaluator.py ---
    print(f"\n[Stage {args.stage}] Running comprehensive evaluation...")

    # Cumulative test set
    cumulative_indices = []
    for s in range(args.stage + 1):
        cumulative_indices.extend(split_protocol.get_stage_dataset(s, "test").indices)
    cumulative_dataset = ActiveLabelDataset(
        torch.utils.data.Subset(dataset, cumulative_indices),
        active_label_indices,
    )
    cumulative_loader = torch.utils.data.DataLoader(
        cumulative_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=ToxicCommentDataset.collate_fn,
    )

    # OOD loader
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

    from data.variant_generator import VariantGenerator
    from utils.evaluator import evaluate_model
    variant_gen = VariantGenerator(seed=args.seed)

    comprehensive_metrics = evaluate_model(
        model=model,
        dataloader=cumulative_loader,
        device=device,
        tokenizer=dataset.tokenizer,
        variant_generator=variant_gen,
        ood_dataloader=ood_dataloader,
        tail_class_indices=[active_label_indices[-1]] if active_label_indices else None,
        threshold=0.5,
    )

    print(f"\n[Stage {args.stage}] Comprehensive metrics:")
    for k, v in comprehensive_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    metrics_path = os.path.join(output_dir, f"metrics_stage{args.stage}.json")
    with open(metrics_path, "w") as f:
        json.dump(comprehensive_metrics, f, indent=2)
    print(f"[Stage {args.stage}] Metrics saved to {metrics_path}")

    # Save model
    save_path = os.path.join(output_dir, "checkpoint-best")
    trainer.save_model(save_path)
    print(f"\n[Stage {args.stage}] Model saved to {save_path}")


if __name__ == "__main__":
    main()
