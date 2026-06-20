"""
Shared baseline experiment runner.

Implements the full training + evaluation pipeline for all baseline methods.
Each baseline script in this directory calls run_experiment(method=...).
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.toxic_dataset import ToxicCommentDataset
from data.fscil_split import FSCILSplitProtocol
from data.variant_generator import VariantGenerator
from models.baseline_classifier import BaselineToxicClassifier
from models.baseline_lora import (
    TaskLoRALayer,
    OrthLoRALayer,
    SingleBranchLoRALayer,
    apply_lora_to_roberta,
    EWCLoRAWrapper,
)
from trainers.baseline_trainer import BaselineTrainer
from utils.evaluator import evaluate_model


class L2PPromptPool(nn.Module):
    """
    Learning to Prompt (L2P) pool for text classification.

    Maintains a pool of K learnable prompt-key pairs.
    At inference, selects top-M prompts via key-query similarity.
    New prompts are added at each FSCIL stage.

    Args:
        hidden_size: RoBERTa hidden dimension
        pool_size: Total number of prompts in the pool (K)
        prompt_length: Number of tokens per prompt
        top_k: Number of prompts to select per input (M)
        num_stages: Total number of FSCIL stages (for prompt allocation)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        pool_size: int = 20,
        prompt_length: int = 5,
        top_k: int = 5,
        num_stages: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.num_stages = num_stages

        # Keys: [pool_size, hidden_size] — used for prompt retrieval
        self.keys = nn.Parameter(torch.randn(pool_size, hidden_size) * 0.02)

        # Prompts: [pool_size, prompt_length, hidden_size]
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_length, hidden_size) * 0.02)

        # Query projection for computing key similarity
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Stage tracking: which prompt indices belong to which stage
        prompts_per_stage = pool_size // num_stages
        self.stage_prompt_indices = {}
        for s in range(num_stages):
            start = s * prompts_per_stage
            end = start + prompts_per_stage if s < num_stages - 1 else pool_size
            self.stage_prompt_indices[s] = list(range(start, end))

        self.current_stage = 0

    def set_stage(self, stage: int):
        """Set current stage and enable corresponding prompts for training."""
        self.current_stage = stage

    def get_active_prompt_indices(self) -> list:
        """Get indices of prompts available at the current stage."""
        indices = []
        for s in range(self.current_stage + 1):
            indices.extend(self.stage_prompt_indices.get(s, []))
        return indices

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """
        Select top-M prompts based on key-query similarity.

        Args:
            cls_hidden: [B, H] (used as query source; simplified from original L2P)

        Returns:
            selected_prompts: [B, top_k * prompt_length, H]
        """
        B = cls_hidden.shape[0]
        active_idx = self.get_active_prompt_indices()
        active_idx_t = torch.tensor(active_idx, device=cls_hidden.device, dtype=torch.long)

        # Active keys and prompts
        active_keys = self.keys[active_idx_t]        # [A, H]
        active_prompts = self.prompts[active_idx_t]   # [A, L, H]
        A = len(active_idx)

        # Query similarity: query @ keys^T
        query = self.query_proj(cls_hidden)  # [B, H]
        similarity = query @ active_keys.T   # [B, A]

        # Select top-k
        if A <= self.top_k:
            selected_idx = torch.arange(A, device=cls_hidden.device).unsqueeze(0).expand(B, -1)
        else:
            _, selected_idx = torch.topk(similarity, self.top_k, dim=-1)  # [B, K]

        # Gather selected prompts: [B, K, L, H] -> [B, K*L, H]
        selected = active_prompts[selected_idx]  # [B, K, L, H]
        selected = selected.reshape(B, self.top_k * self.prompt_length, self.hidden_size)

        return selected


class L2PClassifier(nn.Module):
    """
    Wrapper: BaselineToxicClassifier + L2PPromptPool.

    Freezes RoBERTa base and injects L2P prompts before the encoder.
    Only the prompt pool and classification head are trained.
    """

    def __init__(
        self,
        base_model: BaselineToxicClassifier,
        pool_size: int = 20,
        prompt_length: int = 5,
        top_k: int = 5,
        num_stages: int = 3,
    ):
        super().__init__()
        self.base_model = base_model
        self.prompt_pool = L2PPromptPool(
            hidden_size=base_model.hidden_size,
            pool_size=pool_size,
            prompt_length=prompt_length,
            top_k=top_k,
            num_stages=num_stages,
        )
        self.prompt_length = prompt_length

        # Freeze base RoBERTa
        for param in self.base_model.roberta.parameters():
            param.requires_grad = False

        self.current_stage = 0

    def set_stage(self, stage: int):
        self.current_stage = stage
        self.prompt_pool.set_stage(stage)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                return_rejection=False, **kwargs):
        B = input_ids.shape[0]
        device = input_ids.device

        # Get token embeddings
        token_embeddings = self.base_model.roberta.embeddings(input_ids=input_ids)

        # Use a preliminary pass to get [CLS] for prompt retrieval
        # (simplified: use mean pooling of embeddings as query source)
        with torch.no_grad():
            prelim_out = self.base_model.roberta(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
            )
        cls_query = prelim_out.last_hidden_state[:, 0, :]

        # Select prompts
        selected_prompts = self.prompt_pool(cls_query)  # [B, K*L, H]

        # Prepend prompts to token embeddings
        inputs_embeds = torch.cat([selected_prompts, token_embeddings], dim=1)

        # Extend attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(B, selected_prompts.shape[1], dtype=attention_mask.dtype, device=device)
            extended_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            extended_mask = None

        # Forward through encoder again (now with prompts)
        outputs = self.base_model.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.base_model.dropout(cls_hidden)
        logits = self.base_model.classifier(cls_hidden)

        result = {"logits": logits, "cls_hidden": cls_hidden}

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            mask = (labels >= 0).float()
            active_logits = logits * mask
            active_labels = labels * mask
            result["loss"] = loss_fct(active_logits, active_labels)

        if return_rejection:
            probs = torch.sigmoid(logits)
            max_prob = probs.max(dim=-1)[0]
            result["rejection"] = {
                "probs": probs,
                "u_t": 1.0 - max_prob,
                "decision": ["predicted"] * B,
                "max_prob": max_prob,
                "entropy": torch.zeros_like(max_prob),
                "d_proto": torch.zeros_like(max_prob),
                "s_surface": torch.zeros_like(max_prob),
            }

        return result


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


def freeze_base_roberta(model):
    for param in model.roberta.parameters():
        param.requires_grad = False


def freeze_old_loras(model):
    from models.baseline_lora import TaskLoRALayer, OrthLoRALayer
    for module in model.modules():
        if isinstance(module, (TaskLoRALayer, OrthLoRALayer)):
            module.freeze_active()


def run_experiment(method: str, stage: int, config_path: str = "configs/base.yaml",
                   prev_checkpoint: str = None, seed: int = 42, output_dir: str = None,
                   eval_only: bool = False):
    """
    Run a single baseline experiment.

    Args:
        method: baseline method name
        stage: FSCIL stage index
        config_path: path to base config YAML
        prev_checkpoint: path to previous stage checkpoint directory
        seed: random seed
        output_dir: override output directory
        eval_only: if True, skip training and only evaluate
    """
    base_cfg = load_config(config_path)
    stages_cfg_path = "configs/stages.yaml"
    if os.path.exists(stages_cfg_path):
        stages_cfg = load_config(stages_cfg_path)
        stage_key = ["base", "stage1", "stage2"][stage]
        if stage_key in stages_cfg:
            cfg = merge_configs(base_cfg, stages_cfg[stage_key])
        else:
            cfg = base_cfg
    else:
        cfg = base_cfg

    output_dir = output_dir or f"./outputs/{method}_stage_{stage}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(output_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = ToxicCommentDataset(
        csv_path=cfg["data"]["dataset_path"],
        tokenizer_name=cfg["model"]["name"],
        max_length=cfg["data"]["max_length"],
        filter_toxic=cfg["data"].get("filter_toxic", True),
    )
    print(f"[Data] Loaded {len(dataset)} toxic samples.")

    # FSCIL split (allow stage definitions from config)
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
        seed=seed,
        stage_definitions=stage_definitions,
    )
    split_protocol.create_splits()
    split_protocol.save_splits()

    active_label_indices = split_protocol.get_active_labels(stage)
    num_active_classes = len(active_label_indices)

    train_dataset = ActiveLabelDataset(
        split_protocol.get_stage_dataset(stage, "train"),
        active_label_indices,
    )
    eval_dataset = ActiveLabelDataset(
        split_protocol.get_stage_dataset(stage, "test"),
        active_label_indices,
    )

    # Cumulative test set
    cumulative_indices = []
    for s in range(stage + 1):
        cumulative_indices.extend(split_protocol.get_stage_dataset(s, "test").indices)
    cumulative_dataset = ActiveLabelDataset(
        torch.utils.data.Subset(dataset, cumulative_indices),
        active_label_indices,
    )

    # Old validation for EWC
    old_val_dataset = None
    if stage > 0:
        old_val_dataset = ActiveLabelDataset(
            split_protocol.get_stage_dataset(0, "test"),
            split_protocol.get_active_labels(0),
        )

    # Model init
    init_num_classes = num_active_classes
    checkpoint_state = None
    if stage > 0 and prev_checkpoint:
        checkpoint_state = load_checkpoint_state(prev_checkpoint)
        detected = detect_num_classes_from_checkpoint(checkpoint_state) if checkpoint_state else None
        if detected:
            init_num_classes = detected

    model = BaselineToxicClassifier(
        num_classes=init_num_classes,
        model_name=cfg["model"]["name"],
    )

    # Inject LoRA or L2P prompt pool if needed
    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get("target_modules", ["query", "value"])
    r = lora_cfg.get("rs", 8)
    alpha = lora_cfg.get("alpha", 16)
    dropout = lora_cfg.get("dropout", 0.05)

    if method == "l2p":
        # Wrap in L2P prompt pool (freeze base, only train prompts + classifier)
        model = L2PClassifier(
            base_model=model,
            pool_size=cfg.get("l2p", {}).get("pool_size", 20),
            prompt_length=cfg.get("l2p", {}).get("prompt_length", 5),
            top_k=cfg.get("l2p", {}).get("top_k", 5),
            num_stages=cfg.get("l2p", {}).get("num_stages", 3),
        )
    elif method in ("task_lora", "task_lora_msp", "task_lora_adb", "task_lora_maha"):
        apply_lora_to_roberta(
            model.roberta, TaskLoRALayer,
            target_modules=target_modules, r=r, lora_alpha=alpha, lora_dropout=dropout,
        )
    elif method == "o_lora":
        apply_lora_to_roberta(
            model.roberta, OrthLoRALayer,
            target_modules=target_modules, r=r, lora_alpha=alpha, lora_dropout=dropout,
        )
    elif method == "ewc_lora":
        apply_lora_to_roberta(
            model.roberta, SingleBranchLoRALayer,
            target_modules=target_modules, r=r, lora_alpha=alpha, lora_dropout=dropout,
        )
    # seq_finetune: no LoRA

    if checkpoint_state is not None:
        if method == "l2p":
            model.load_state_dict(checkpoint_state, strict=False)
            if num_active_classes > init_num_classes:
                model.base_model.expand_classifier(num_active_classes)
        else:
            model.load_state_dict(checkpoint_state, strict=False)
            if num_active_classes > init_num_classes:
                model.expand_classifier(num_active_classes)

    model.to(device)

    # Stage setup
    if method == "l2p":
        model.set_stage(stage)
    elif stage > 0:
        if method != "seq_finetune":
            if method == "l2p":
                # L2P base model is already frozen
                pass
            else:
                freeze_base_roberta(model)
        if method in ("task_lora", "task_lora_msp", "task_lora_adb", "task_lora_maha", "o_lora"):
            freeze_old_loras(model)
            for p in model.classifier.parameters():
                p.requires_grad = True

    # EWC wrapper
    ewc_wrapper = None
    if method == "ewc_lora":
        ewc_path = os.path.join(prev_checkpoint if prev_checkpoint else "", "ewc_fisher.pt")
        if stage > 0 and os.path.exists(ewc_path):
            ewc_wrapper = EWCLoRAWrapper(model, importance_weight=1e4)
            ewc_data = torch.load(ewc_path, map_location="cpu")
            ewc_wrapper.fisher = ewc_data["fisher"]
            ewc_wrapper.optimal_params = ewc_data["optimal_params"]
            print(f"[EWC] Loaded Fisher from {ewc_path}")
        elif stage == 0:
            ewc_wrapper = EWCLoRAWrapper(model, importance_weight=1e4)

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
        seed=seed,
        report_to=[],
    )

    trainer = BaselineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=dataset.tokenizer,
        method=method,
        ewc_wrapper=ewc_wrapper,
        data_collator=ToxicCommentDataset.collate_fn,
    )

    if not eval_only:
        print(f"\n[{method}] Stage {stage} training...")
        print(f"  Active classes: {active_label_indices}")
        print(f"  Train size: {len(train_dataset)}")
        print(f"  Eval size: {len(eval_dataset)}")
        trainer.train()

        # Save EWC Fisher after stage 0
        if method == "ewc_lora" and stage == 0 and ewc_wrapper is not None:
            old_val_loader = torch.utils.data.DataLoader(
                old_val_dataset if old_val_dataset else eval_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=ToxicCommentDataset.collate_fn,
            )
            ewc_wrapper.compute_fisher(old_val_loader, device, num_batches=100)
            ewc_save_path = os.path.join(output_dir, "checkpoint-best", "ewc_fisher.pt")
            os.makedirs(os.path.dirname(ewc_save_path), exist_ok=True)
            torch.save({
                "fisher": ewc_wrapper.fisher,
                "optimal_params": ewc_wrapper.optimal_params,
            }, ewc_save_path)
            print(f"[EWC] Saved Fisher to {ewc_save_path}")

    # Unified evaluation
    print(f"\n[{method}] Stage {stage} unified evaluation...")

    cumulative_loader = torch.utils.data.DataLoader(
        cumulative_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=ToxicCommentDataset.collate_fn,
    )

    variant_gen = VariantGenerator(seed=seed)

    ood_dataloader = None
    if stage < 2:
        try:
            next_stage_classes = split_protocol.get_active_labels(stage + 1)
            if len(next_stage_classes) > 0:
                ood_dataset = ActiveLabelDataset(
                    split_protocol.get_stage_dataset(stage + 1, "train"),
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

    print(f"\n[{method}] Stage {stage} metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ── Mahalanobis-specific OOD evaluation ──
    if method == "task_lora_maha" and ood_dataloader is not None:
        maha_metrics = _compute_mahalanobis_ood(
            model=model,
            known_loader=cumulative_loader,
            ood_loader=ood_dataloader,
            device=device,
        )
        metrics.update(maha_metrics)
        print(f"  [Mahalanobis] AUROC: {maha_metrics.get('maha_auroc', 0):.4f}, "
              f"FPR95: {maha_metrics.get('maha_fpr95', 0):.4f}")

    metrics_path = os.path.join(output_dir, f"metrics_stage{stage}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if not eval_only:
        save_path = os.path.join(output_dir, "checkpoint-best")
        trainer.save_model(save_path)
        print(f"\n[{method}] Model saved to {save_path}")

    return metrics


def _compute_mahalanobis_ood(model, known_loader, ood_loader, device):
    """
    Compute OOD detection metrics using Mahalanobis distance in [CLS] space.

    For each known class, compute the mean of [CLS] representations.
    Use a shared covariance matrix (regularized).
    At test time, score = min_d over all class means of Mahalanobis distance.

    Returns:
        dict with 'maha_auroc' and 'maha_fpr95' keys.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    model.eval()

    # ── Step 1: Collect known-class [CLS] representations and labels ──
    known_cls = []
    known_labels = []
    with torch.no_grad():
        for batch in known_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_rejection=False,
            )
            known_cls.append(outputs["cls_hidden"].cpu().numpy())
            known_labels.append(batch["labels"].cpu().numpy())

    known_cls = np.concatenate(known_cls, axis=0)      # [N, H]
    known_labels = np.concatenate(known_labels, axis=0) # [N, C]

    # ── Step 2: Compute per-class means and shared covariance ──
    C = known_labels.shape[1]  # number of classes
    H = known_cls.shape[1]

    class_means = np.zeros((C, H))
    for c in range(C):
        mask = known_labels[:, c] == 1
        if mask.sum() > 0:
            class_means[c] = known_cls[mask].mean(axis=0)
        else:
            class_means[c] = known_cls.mean(axis=0)  # fallback

    # Compute shared covariance with regularization
    centered = known_cls - known_cls.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    reg = 1e-4 * np.eye(H)
    cov_reg = cov + reg
    cov_inv = np.linalg.pinv(cov_reg)

    # ── Step 3: Compute Mahalanobis distances ──
    def maha_distance(X):
        """X: [N, H], returns [N] minimum Mahalanobis distance to any class mean."""
        dists = np.zeros((X.shape[0], C))
        for c in range(C):
            diff = X - class_means[c]
            dists[:, c] = np.sum(diff @ cov_inv * diff, axis=1)
        return dists.min(axis=1)

    known_dists = maha_distance(known_cls)

    # Collect OOD representations
    ood_cls_list = []
    with torch.no_grad():
        for batch in ood_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_rejection=False,
            )
            ood_cls_list.append(outputs["cls_hidden"].cpu().numpy())

    if not ood_cls_list:
        return {"maha_auroc": 0.5, "maha_fpr95": 1.0}

    ood_cls = np.concatenate(ood_cls_list, axis=0)
    ood_dists = maha_distance(ood_cls)

    # ── Step 4: Compute AUROC and FPR95 ──
    y_true = np.concatenate([np.zeros(len(known_dists)), np.ones(len(ood_dists))])
    y_score = np.concatenate([known_dists, ood_dists])

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = 0.5

    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

    return {"maha_auroc": auroc, "maha_fpr95": fpr95}
