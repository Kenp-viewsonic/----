"""
Single-stage training script for FSCIL Toxic Comment Classification.

Usage:
    python run_stage.py --stage 0 --config configs/base.yaml
    python run_stage.py --stage 1 --prev_checkpoint outputs/stage_0/checkpoint-best
"""

import os
import sys
import random
import argparse
import json
import math
import yaml
import torch
import numpy as np
from transformers import TrainingArguments

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.toxic_dataset import ToxicCommentDataset
from data.fscil_split import FSCILSplitProtocol
from models.roberta_classifier import RobertaToxicClassifier
from models.dual_lora import apply_dual_lora_to_roberta
from trainers.incremental_trainer import IncrementalTrainer


class ActiveLabelDataset:
    """Wrapper to extract only active labels from a Subset."""
    def __init__(self, subset, active_indices, stage_first_new_idx=None, mask_future_labels=True):
        self.subset = subset
        self.active_indices = active_indices
        self.stage_first_new_idx = stage_first_new_idx
        self.mask_future_labels = mask_future_labels
    
    def __getitem__(self, idx):
        item = self.subset[idx]
        item = dict(item)  # shallow copy to avoid mutating original
        if "labels" in item:
            labels = item["labels"][self.active_indices]
            # For replay samples, optionally mark classes introduced in the current stage
            # as -1 (ignore). Leaving them unmasked provides explicit negatives/positives
            # for new-class calibration when those labels are known in the dataset.
            if self.stage_first_new_idx is not None and self.mask_future_labels:
                labels = labels.clone()
                labels[self.stage_first_new_idx:] = -1.0
            item["labels"] = labels
        return item
    
    def __len__(self):
        return len(self.subset)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base_cfg, stage_cfg):
    """Deep merge stage-specific overrides into base config."""
    merged = base_cfg.copy()
    for key, val in stage_cfg.items():
        if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(val)
        else:
            merged[key] = val
    return merged


def prepare_model_for_checkpoint_state(model, checkpoint_state):
    """Pre-create dynamic buffers/modules so strict=False does not drop them."""
    # Pre-resize rejection_gate.V_known buffer to match checkpoint shape
    # (stage 0 initializes it as [0], but after training it becomes [N])
    for key, tensor in checkpoint_state.items():
        if "rejection_gate.V_known" in key:
            model.rejection_gate.register_buffer('V_known', torch.zeros(tensor.shape[0], dtype=torch.long))
            print(f"[Checkpoint] Pre-resized V_known buffer to shape {tensor.shape}")
            break
    
    # Pre-populate frozen_plastics ModuleLists so load_state_dict can match keys.
    # Without this, frozen historical patches are silently dropped as unexpected keys.
    frozen_slots_needed = {}  # (module_path, idx) -> {'A': shape, 'B': shape}
    for key, tensor in checkpoint_state.items():
        if '.frozen_plastics.' not in key:
            continue
        parts = key.split('.frozen_plastics.')
        module_path = parts[0]
        rest = parts[1]  # e.g., "0.A" or "0.B"
        idx_str, buf_name = rest.split('.', 1)
        idx = int(idx_str)
        slot_key = (module_path, idx)
        if slot_key not in frozen_slots_needed:
            frozen_slots_needed[slot_key] = {}
        frozen_slots_needed[slot_key][buf_name] = tuple(tensor.shape)
    
    for (module_path, idx), shapes in frozen_slots_needed.items():
        try:
            target = model
            for p in module_path.split('.'):
                if p.isdigit():
                    target = target[int(p)]
                else:
                    target = getattr(target, p)
        except (AttributeError, IndexError, TypeError):
            continue
        
        from models.dual_lora import DualBranchLoRALayer
        if not isinstance(target, DualBranchLoRALayer):
            continue
        
        while len(target.frozen_plastics) <= idx:
            dummy = torch.nn.Module()
            shape_a = shapes.get('A', (1,))
            shape_b = shapes.get('B', (1,))
            dummy.register_buffer('A', torch.zeros(*shape_a))
            dummy.register_buffer('B', torch.zeros(*shape_b))
            target.frozen_plastics.append(dummy)
    
    if frozen_slots_needed:
        print(f"[Checkpoint] Pre-created {len(frozen_slots_needed)} frozen_plastics slots from checkpoint keys.")


def build_cumulative_eval_loaders(args, cfg, dataset, split_protocol, active_label_indices, training_args):
    """Build cumulative seen-class loader and next-stage OOD loader for comprehensive evaluation."""
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
    
    return cumulative_loader, ood_dataloader


def evaluate_comprehensive(model, cumulative_loader, ood_dataloader, ref_reps, dataset, active_label_indices, device, seed):
    from data.variant_generator import VariantGenerator
    from utils.evaluator import evaluate_model
    variant_gen = VariantGenerator(seed=seed)

    return evaluate_model(
        model=model,
        dataloader=cumulative_loader,
        device=device,
        tokenizer=dataset.tokenizer,
        variant_generator=variant_gen,
        ood_dataloader=ood_dataloader,
        ref_representations=ref_reps,
        tail_class_indices=[active_label_indices[-1]] if active_label_indices else None,
        threshold=0.5,
    )


def compute_class_prototypes(model, dataloader, device, target_class_indices):
    """Compute class prototypes from current support set CLS embeddings."""
    prototypes = {int(class_idx): [] for class_idx in target_class_indices}
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(
                input_ids=batch_tensors["input_ids"],
                attention_mask=batch_tensors["attention_mask"],
                texts=batch_tensors.get("texts"),
                return_rejection=False,
            )
            cls_hidden = outputs["cls_hidden"]
            labels = batch_tensors.get("labels")
            if labels is None:
                continue
            for class_idx in target_class_indices:
                positive_mask = labels[:, class_idx] > 0.5
                if positive_mask.any():
                    prototypes[int(class_idx)].append(cls_hidden[positive_mask].detach())

    reduced = {}
    for class_idx, chunks in prototypes.items():
        if chunks:
            reduced[class_idx] = torch.cat(chunks, dim=0).mean(dim=0)
        else:
            reduced[class_idx] = None
    model.train()
    return reduced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, help="Stage ID: 0, 1, 2, ...")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--prev_checkpoint", type=str, default=None, help="Path to previous stage checkpoint dir")
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
    
    # Override output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./outputs/stage_{args.stage}_seed{args.seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seeds for full reproducibility across machines
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load dataset
    dataset = ToxicCommentDataset(
        csv_path=cfg["data"]["dataset_path"],
        tokenizer_name=cfg["model"]["name"],
        max_length=cfg["data"]["max_length"],
        filter_toxic=cfg["data"].get("filter_toxic", True),
    )
    print(f"[Data] Loaded {len(dataset)} toxic samples.")
    print(f"[Data] Label distribution: {dataset.get_label_distribution()}")
    replay_cfg = cfg.get("replay", {})
    
    # FSCIL split (allow stage definitions from config)
    stage_defs = cfg.get("fscil", {}).get("stages")
    if stage_defs is not None:
        # Convert list format to dict format expected by FSCILSplitProtocol
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
        coreset_size_per_class=replay_cfg.get("coreset_size_per_class", 10),
        new_class_negative_ratio=replay_cfg.get("new_class_negative_ratio", 0.0),
        confusion_negatives_enabled=replay_cfg.get("confusion_negatives_enabled", False),
        confusion_negatives_extra_per_class=replay_cfg.get("confusion_negatives_extra_per_class", 0),
        confusion_negatives_extra_by_class=replay_cfg.get("confusion_negatives_extra_by_class", None),
        confusion_negatives_old_classes=replay_cfg.get("confusion_negatives_old_classes", None),
        stage_definitions=stage_definitions,
    )
    splits = split_protocol.create_splits()
    split_protocol.save_splits()
    
    # Get active label indices up to this stage
    active_label_indices = split_protocol.get_active_labels(args.stage)
    num_active_classes = len(active_label_indices)
    mask_replay_new_labels = replay_cfg.get("mask_new_class_labels", True)
    
    # Datasets for this stage (wrapped to only return active labels)
    # Stage > 0: mix coreset replay samples into training to prevent new-class-only bias
    if args.stage > 0:
        coreset_indices_for_replay = split_protocol.splits.get(args.stage, {}).get("coreset", [])
        if coreset_indices_for_replay:
            stage_train_indices = split_protocol.splits[args.stage]["train"]
            stage_negative_indices = split_protocol.splits[args.stage].get("negatives", [])
            stage_new_indices = sorted(list(set(stage_train_indices + stage_negative_indices)))
            n_new = len(stage_train_indices)
            n_negative = len(stage_negative_indices)
            n_replay = len(coreset_indices_for_replay)
            # Oversample new-class data to prevent replay from overwhelming new-class signal
            if n_replay > len(stage_new_indices) and len(stage_new_indices) > 0:
                oversample_ratio = math.ceil(n_replay / len(stage_new_indices))
                oversampled_new = stage_new_indices * oversample_ratio
            else:
                oversampled_new = stage_new_indices
                oversample_ratio = 1
            # Determine where new classes start in the active label list
            prev_classes = []
            for s in range(args.stage):
                prev_classes.extend(split_protocol.get_active_labels(s))
            first_new_idx = len(set(prev_classes))
            # New-class dataset: full labels (including 0 for negatives)
            new_subset = torch.utils.data.Subset(dataset, oversampled_new)
            new_ds = ActiveLabelDataset(new_subset, active_label_indices)
            # Replay dataset: mask classes not yet introduced with -1
            replay_subset = torch.utils.data.Subset(dataset, coreset_indices_for_replay)
            replay_ds = ActiveLabelDataset(
                replay_subset, active_label_indices,
                stage_first_new_idx=first_new_idx,
                mask_future_labels=mask_replay_new_labels,
            )
            train_dataset = torch.utils.data.ConcatDataset([new_ds, replay_ds])
            replay_label_mode = "masked" if mask_replay_new_labels else "full-label-calibrated"
            coreset_by_class = split_protocol.splits.get(args.stage, {}).get("coreset_by_class", {})
            coreset_summary = {cls: len(indices) for cls, indices in coreset_by_class.items()}
            negatives_by_class = split_protocol.splits.get(args.stage, {}).get("negatives_by_class", {})
            negatives_summary = {cls: len(indices) for cls, indices in negatives_by_class.items()}
            print(f"[Replay] New-class positives: {n_new}, explicit negatives: {n_negative}, "
                f"combined x{oversample_ratio}={len(oversampled_new)}, "
                  f"replay: {n_replay} ({replay_label_mode} cls>{first_new_idx-1}), total: {len(train_dataset)}")
            if negatives_summary:
                print(f"[Replay] New-class negatives by class: {negatives_summary}")
            if coreset_summary:
                print(f"[Replay] Coreset by old class: {coreset_summary}")
        else:
            train_dataset = ActiveLabelDataset(
                split_protocol.get_stage_dataset(args.stage, "train"),
                active_label_indices
            )
    else:
        train_dataset = ActiveLabelDataset(
            split_protocol.get_stage_dataset(args.stage, "train"),
            active_label_indices
        )
    eval_dataset = ActiveLabelDataset(
        split_protocol.get_stage_dataset(args.stage, "test"),
        active_label_indices
    )
    
    # Old validation dataset for semantic consolidation (stage > 0)
    old_val_dataset = None
    if args.stage > 0:
        # Use stage 0 test with base labels only
        old_val_dataset = ActiveLabelDataset(
            split_protocol.get_stage_dataset(0, "test"),
            split_protocol.get_active_labels(0)
        )
    
    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine initial num_classes: if stage > 0 with checkpoint, detect from checkpoint
    init_num_classes = num_active_classes
    checkpoint_state = None
    if args.stage > 0 and args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
        print(f"[Checkpoint] Loading from {args.prev_checkpoint}")
        # Try safetensors first, then pytorch_model.bin
        safetensors_path = os.path.join(args.prev_checkpoint, "model.safetensors")
        pytorch_path = os.path.join(args.prev_checkpoint, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            checkpoint_state = load_file(safetensors_path, device="cpu")
        elif os.path.exists(pytorch_path):
            checkpoint_state = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.prev_checkpoint}")
        
        # Detect number of classes from checkpoint classifier weight
        for key, tensor in checkpoint_state.items():
            if "classifier" in key and "weight" in key and len(tensor.shape) == 2:
                init_num_classes = tensor.shape[0]
                print(f"[Checkpoint] Detected {init_num_classes} classes from checkpoint.")
                break
    
    model = RobertaToxicClassifier(
        num_classes=init_num_classes,
        model_name=cfg["model"]["name"],
        prefix_cfg=cfg.get("prefix"),
        lora_cfg=cfg.get("lora"),
        pe_cfg=cfg.get("toxic_pe"),
        gate_cfg=cfg.get("rejection_gate"),
    )
    
    # Apply DualBranchLoRA to RoBERTa
    apply_dual_lora_to_roberta(
        model.roberta,
        target_modules=cfg["lora"].get("target_modules", ["query", "value"]),
        r_stable=cfg["lora"]["rs"],
        r_plastic=cfg["lora"]["rp"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
    )
    
    # Load checkpoint weights
    if checkpoint_state is not None:
        prepare_model_for_checkpoint_state(model, checkpoint_state)
        model.load_state_dict(checkpoint_state, strict=False)
        
        # Expand classifier for new classes
        if num_active_classes > init_num_classes:
            model.expand_classifier(num_active_classes)
    
    model.set_stage(args.stage)
    model.to(device)

    # Previous-stage teacher for old-class logit distillation during incremental training.
    teacher_model = None
    old_num_classes = init_num_classes if args.stage > 0 else 0
    if args.stage > 0 and checkpoint_state is not None:
        print(f"[Teacher] Building previous-stage teacher with {old_num_classes} old classes.")
        teacher_model = RobertaToxicClassifier(
            num_classes=old_num_classes,
            model_name=cfg["model"]["name"],
            prefix_cfg=cfg.get("prefix"),
            lora_cfg=cfg.get("lora"),
            pe_cfg=cfg.get("toxic_pe"),
            gate_cfg=cfg.get("rejection_gate"),
        )
        apply_dual_lora_to_roberta(
            teacher_model.roberta,
            target_modules=cfg["lora"].get("target_modules", ["query", "value"]),
            r_stable=cfg["lora"]["rs"],
            r_plastic=cfg["lora"]["rp"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
        )
        prepare_model_for_checkpoint_state(teacher_model, checkpoint_state)
        teacher_model.load_state_dict(checkpoint_state, strict=False)
        teacher_model.set_stage(max(args.stage - 1, 0))
        teacher_model.to(device)
        teacher_model.eval()
    
    # Stage > 0: freeze stable branch, only train plastic + new classifier head
    if args.stage > 0:
        print(f"[Stage {args.stage}] Freezing stable branch and base RoBERTa, training plastic + new head.")
        # Freeze base RoBERTa parameters
        for param in model.roberta.parameters():
            param.requires_grad = False
        
        # Freeze stable branch, unfreeze plastic branch in DualLoRA layers
        from models.dual_lora import DualBranchLoRALayer
        for module in model.modules():
            if isinstance(module, DualBranchLoRALayer):
                module.freeze_stable()
                module.unfreeze_plastic()
        
        # Optionally unfreeze top stable layers for controlled semantic reorganization.
        # Disabled by default (can conflict with weak prefix anchoring).
        stable_partial_cfg = cfg.get("stable_partial", {})
        if stable_partial_cfg.get("enable", False):
            unfreeze_top_layers = stable_partial_cfg.get("unfreeze_top_layers", 2)
            if unfreeze_top_layers > 0:
                total_layers = 12
                unfreeze_start = total_layers - unfreeze_top_layers
                for layer_idx, layer in enumerate(model.roberta.encoder.layer):
                    if layer_idx >= unfreeze_start:
                        attn = layer.attention.self
                        for attn_part_name in ["query", "value"]:
                            lo = getattr(attn, attn_part_name, None)
                            if isinstance(lo, DualBranchLoRALayer):
                                lo.unfreeze_stable()
                print(f"[PartialStable] Unfroze stable branch on top {unfreeze_top_layers} layers (indices {unfreeze_start}..{total_layers-1}).")
        
        # Ensure classifier head is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Ensure rejection gate is trainable
        for param in model.rejection_gate.parameters():
            param.requires_grad = True

        proto_cfg = cfg.get("prototype_init", {})
        if proto_cfg.get("enable", False) and num_active_classes > old_num_classes:
            new_class_indices = list(range(old_num_classes, num_active_classes))
            proto_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=training_args.per_device_eval_batch_size if 'training_args' in locals() else 32,
                shuffle=False,
                collate_fn=ToxicCommentDataset.collate_fn,
            )
            class_prototypes = compute_class_prototypes(model, proto_loader, device, new_class_indices)
            model.initialize_classifier_from_prototypes(
                class_prototypes=class_prototypes,
                normalize=proto_cfg.get("normalize", True),
                init_bias=proto_cfg.get("init_bias", -0.5),
            )
            initialized = [idx for idx, proto in class_prototypes.items() if proto is not None]
            print(f"[PrototypeInit] Initialized classifier rows from support prototypes for classes: {initialized}")
    
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
    
    # Loss weights
    loss_weights = cfg.get("loss_weights", {})
    
    # Data max_length
    data_max_length = cfg["data"].get("max_length", 128)
    
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
            print(f"[Coreset] Built coreset dataloader with {len(coreset_indices)} samples for delta_k evaluation.")
        else:
            # Fallback: use old_val_dataset as coreset
            print("[Coreset] No coreset indices found, falling back to old_val_dataset.")
    
    # --- Pre-stage 0: initialize prefix from K-means BEFORE training (#3 fix) ---
    if args.stage == 0 and not model.prefix_module._proto_initialized:
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
                batch_tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                out = model(
                    input_ids=batch_tensors["input_ids"],
                    attention_mask=batch_tensors["attention_mask"],
                    texts=batch_tensors.get("texts"),
                    return_rejection=False,
                )
                cls_embeddings.append(out["cls_hidden"].cpu())
                count += batch_tensors["input_ids"].size(0)
        
        if cls_embeddings:
            all_cls = torch.cat(cls_embeddings, dim=0)
            model.prefix_module.init_from_kmeans(all_cls)
            # Set rejection gate prototypes to K-means centroids
            if hasattr(model.prefix_module, '_kmeans_centroids_'):
                model.rejection_gate.set_prototypes(model.prefix_module._kmeans_centroids_)
        
        model.train()
        print("[Prefix] K-means initialization complete — prefix is now active for stage 0 training.")
    
    # --- Initialize known toxic vocabulary for rejection gate surface anomaly (#7 fix) ---
    if not model.rejection_gate.vocab_initialized:
        # Extract top toxic-indicative tokens from the tokenizer vocabulary
        # Strategy: collect token IDs for known toxic words and common insults
        toxic_subwords = [
            "hate", "kill", "die", "stupid", "idiot", "dumb", "ugly", "fat", "suck",
            "worst", "terrible", "awful", "horrible", "pathetic", "useless", "trash",
            "garbage", "crap", "damn", "hell", "fuck", "shit", "bastard", "ass",
            "moron", "loser", "racist", "nazi", "sexist",
            # Common leet/obfuscated patterns
            "h4te", "k1ll", "d1e", "st0pid", "1diot",
            # Toxic prefix/suffix fragments
            "Ġhate", "Ġkill", "Ġdie", "Ġstupid", "Ġidiot",
        ]
        known_ids = []
        for word in toxic_subwords:
            token_id = dataset.tokenizer.convert_tokens_to_ids(word)
            if token_id != dataset.tokenizer.unk_token_id:
                known_ids.append(token_id)
        # Also include tokenizer IDs for the special tokens
        known_ids = list(set(known_ids))
        if known_ids:
            model.rejection_gate.set_known_vocab(known_ids, tokenizer=dataset.tokenizer)
            print(f"[Gate] Initialized known toxic vocabulary with {len(known_ids)} token IDs.")
    
    # Trainer
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
        teacher_model=teacher_model,
        old_num_classes=old_num_classes,
        use_balanced_bce=loss_weights.get("balanced_bce", False),
        balanced_bce_max_pos_weight=loss_weights.get("balanced_bce_max_pos_weight", 10.0),
        data_collator=ToxicCommentDataset.collate_fn,
        max_length=data_max_length,
    )
    
    # Train
    print(f"\n[Stage {args.stage}] Starting training...")
    print(f"  Active classes: {active_label_indices}")
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Eval size: {len(eval_dataset)}")
    
    trainer.train()

    # If ablation: disable frozen plastics for all evaluation forward passes.
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("eval_disable_frozen_plastics", False):
        model.set_disable_frozen_plastics(True)
        print("[Ablation] Frozen plastics DISABLED for evaluation.")

    # Build cumulative evaluation loaders immediately after training. At this
    # point, stage > 0 models have already run semantic consolidation, because
    # the callback executes in on_train_end.
    cumulative_loader, ood_dataloader = build_cumulative_eval_loaders(
        args=args,
        cfg=cfg,
        dataset=dataset,
        split_protocol=split_protocol,
        active_label_indices=active_label_indices,
        training_args=training_args,
    )

    # Evaluate the saved pre-consolidation model as a diagnostic. This tells us
    # whether degradation happens during stage training or during consolidation
    # / frozen patch accumulation.
    pre_consolidation_metrics = None
    pre_consolidation_path = os.path.join(output_dir, "checkpoint-pre-consolidation", "pytorch_model.bin")
    if args.stage > 0 and os.path.exists(pre_consolidation_path):
        print(f"\n[Stage {args.stage}] Running pre-consolidation diagnostic evaluation...")
        pre_state = torch.load(pre_consolidation_path, map_location="cpu")
        pre_model = RobertaToxicClassifier(
            num_classes=num_active_classes,
            model_name=cfg["model"]["name"],
            prefix_cfg=cfg.get("prefix"),
            lora_cfg=cfg.get("lora"),
            pe_cfg=cfg.get("toxic_pe"),
            gate_cfg=cfg.get("rejection_gate"),
        )
        apply_dual_lora_to_roberta(
            pre_model.roberta,
            target_modules=cfg["lora"].get("target_modules", ["query", "value"]),
            r_stable=cfg["lora"]["rs"],
            r_plastic=cfg["lora"]["rp"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
        )
        prepare_model_for_checkpoint_state(pre_model, pre_state)
        missing, unexpected = pre_model.load_state_dict(pre_state, strict=False)
        if missing or unexpected:
            print(f"[PreConsolidation] load_state_dict missing={len(missing)}, unexpected={len(unexpected)}")
        pre_model.set_stage(args.stage)
        pre_model.to(device)
        pre_model.eval()
        if lora_cfg.get("eval_disable_frozen_plastics", False):
            pre_model.set_disable_frozen_plastics(True)
        pre_consolidation_metrics = evaluate_comprehensive(
            model=pre_model,
            cumulative_loader=cumulative_loader,
            ood_dataloader=ood_dataloader,
            ref_reps=None,
            dataset=dataset,
            active_label_indices=active_label_indices,
            device=device,
            seed=args.seed,
        )
        pre_metrics_path = os.path.join(output_dir, f"metrics_stage{args.stage}_pre_consolidation.json")
        with open(pre_metrics_path, "w") as f:
            json.dump(pre_consolidation_metrics, f, indent=2)
        print(f"[Stage {args.stage}] Pre-consolidation metrics saved to {pre_metrics_path}")
        del pre_model
        torch.cuda.empty_cache()

    # --- Comprehensive evaluation via evaluator.py ---
    print(f"\n[Stage {args.stage}] Running comprehensive evaluation...")

    # Reference CLS for CKA (load previous stage checkpoint if available)
    ref_reps = None
    if args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
        print("[Eval] Loading previous stage model for CKA...")
        prev_state = None
        safetensors_path = os.path.join(args.prev_checkpoint, "model.safetensors")
        pytorch_path = os.path.join(args.prev_checkpoint, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            prev_state = load_file(safetensors_path, device="cpu")
        elif os.path.exists(pytorch_path):
            prev_state = torch.load(pytorch_path, map_location="cpu")
        if prev_state is not None:
            # Use init_num_classes to match the checkpoint's classifier shape
            prev_num_classes = init_num_classes
            for key, tensor in prev_state.items():
                if "classifier" in key and "weight" in key and len(tensor.shape) == 2:
                    prev_num_classes = tensor.shape[0]
                    break
            prev_model = RobertaToxicClassifier(
                num_classes=prev_num_classes,
                model_name=cfg["model"]["name"],
                prefix_cfg=cfg.get("prefix"),
                lora_cfg=cfg.get("lora"),
                pe_cfg=cfg.get("toxic_pe"),
                gate_cfg=cfg.get("rejection_gate"),
            )
            apply_dual_lora_to_roberta(
                prev_model.roberta,
                target_modules=cfg["lora"].get("target_modules", ["query", "value"]),
                r_stable=cfg["lora"]["rs"],
                r_plastic=cfg["lora"]["rp"],
                lora_alpha=cfg["lora"]["alpha"],
                lora_dropout=cfg["lora"]["dropout"],
            )
            prepare_model_for_checkpoint_state(prev_model, prev_state)
            prev_model.load_state_dict(prev_state, strict=False)
            prev_model.to(device)
            prev_model.eval()
            if lora_cfg.get("eval_disable_frozen_plastics", False):
                prev_model.set_disable_frozen_plastics(True)
            ref_cls = []
            with torch.no_grad():
                for batch in cumulative_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    texts = batch.get("texts", [])
                    out = prev_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        texts=texts if hasattr(prev_model.forward, "__code__") and "texts" in prev_model.forward.__code__.co_varnames else None,
                        return_rejection=False,
                    )
                    ref_cls.append(out["cls_hidden"].cpu().numpy())
            ref_reps = np.concatenate(ref_cls, axis=0)
            del prev_model
            torch.cuda.empty_cache()

    comprehensive_metrics = evaluate_comprehensive(
        model=model,
        cumulative_loader=cumulative_loader,
        ood_dataloader=ood_dataloader,
        ref_reps=ref_reps,
        dataset=dataset,
        active_label_indices=active_label_indices,
        device=device,
        seed=args.seed,
    )

    print(f"\n[Stage {args.stage}] Comprehensive metrics:")
    for k, v in comprehensive_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save comprehensive metrics
    metrics_path = os.path.join(output_dir, f"metrics_stage{args.stage}.json")
    with open(metrics_path, "w") as f:
        json.dump(comprehensive_metrics, f, indent=2)
    print(f"[Stage {args.stage}] Metrics saved to {metrics_path}")

    # Save model
    trainer.save_model(os.path.join(output_dir, "checkpoint-best"))
    print(f"\n[Stage {args.stage}] Model saved to {output_dir}/checkpoint-best")


if __name__ == "__main__":
    main()
