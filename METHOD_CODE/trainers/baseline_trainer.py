"""
Generic baseline trainers for FSCIL ablation and comparison.

Supports:
  - seq_finetune: vanilla sequential fine-tuning (no LoRA, catastrophic forgetting)
  - task_lora: per-stage independent LoRA, freeze old LoRAs
  - task_lora_msp: Task-LoRA + Max Softmax Prob for OOD (training same as task_lora)
  - task_lora_adb: Task-LoRA + Adaptive Decision Boundary (ADB) rejection
  - o_lora: Orthogonal LoRA (subspace orthogonality constraint)
  - ewc_lora: EWC regularization over LoRA parameters
"""

import math
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, TrainerCallback
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from models.baseline_lora import TaskLoRALayer, OrthLoRALayer, EWCLoRAWrapper


class StageTransitionCallback(TrainerCallback):
    """
    Callback for task-based incremental LoRA methods.
    At the end of each stage, freezes the active LoRA branch so the next
    stage starts with a fresh branch.
    """

    def __init__(self, trainer_ref):
        self.trainer = trainer_ref

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            model = self.trainer.model
        method = getattr(self.trainer, "method", "")
        if method in ("task_lora", "task_lora_msp", "task_lora_adb", "o_lora"):
            print(f"[StageTransition] Freezing active LoRA branches for method={method}")
            for module in model.modules():
                if isinstance(module, (TaskLoRALayer, OrthLoRALayer)):
                    module.freeze_active()
        return control


class BaselineTrainer(Trainer):
    """
    Generic trainer for baseline methods.

    Args:
        method: one of ['seq_finetune', 'task_lora', 'task_lora_msp',
                        'task_lora_adb', 'o_lora', 'ewc_lora']
        ewc_wrapper: EWCLoRAWrapper instance (required for ewc_lora)
        adb_margin: margin for ADB (default 1.0)
    """

    def __init__(
        self,
        model=None,
        args=None,
        method: str = "seq_finetune",
        ewc_wrapper=None,
        adb_margin: float = 1.0,
        **kwargs
    ):
        # transformers >= 5.x renamed 'tokenizer' to 'processing_class'
        if "tokenizer" in kwargs and "processing_class" not in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(model=model, args=args, **kwargs)
        self.method = method
        self.ewc_wrapper = ewc_wrapper
        self.adb_margin = adb_margin

        # ADB: per-class learnable boundary scalars (initialized to 1.0)
        if method == "task_lora_adb":
            num_classes = self._detect_num_classes(model)
            self.adb_scales = nn.Parameter(torch.ones(num_classes))
            # Register as buffer so it's saved with model; but we want it learnable
            # Actually Parameter is automatically in model.parameters() if attached.
            # We'll attach to model manually in compute_loss if not present.

        self.add_callback(StageTransitionCallback(self))

    def _detect_num_classes(self, model):
        """Heuristic to detect number of classes from classifier head."""
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # Assume last Linear is classifier
                out_f = m.out_features
        return getattr(model, "num_classes", out_f if "out_f" in dir() else 5)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            return_rejection=False,
        )

        loss = outputs.get("loss", torch.tensor(0.0, device=inputs["input_ids"].device))
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=inputs["input_ids"].device)

        # --- O-LoRA orthogonality penalty ---
        if self.method == "o_lora":
            orth_loss = 0.0
            for module in model.modules():
                if isinstance(module, OrthLoRALayer):
                    orth_loss += module.orthogonality_loss()
            if isinstance(orth_loss, torch.Tensor) and orth_loss.item() > 0:
                loss = loss + orth_loss * 1e-3  # small weight

        # --- EWC penalty ---
        if self.method == "ewc_lora" and self.ewc_wrapper is not None:
            ewc_pen = self.ewc_wrapper.penalty(model)
            if isinstance(ewc_pen, torch.Tensor) and ewc_pen.item() > 0:
                loss = loss + ewc_pen

        # --- ADB rejection training ---
        if self.method == "task_lora_adb":
            loss = self._adb_loss(model, inputs, outputs, loss)

        if return_outputs:
            return loss, outputs
        return loss

    def _adb_loss(self, model, inputs, outputs, base_loss):
        """
        Adaptive Decision Boundary (ADB) for open-set rejection.
        Learns per-class scaling factors on logits to push known samples
        away from boundary and OOD samples below it.

        Simplified implementation: per-class logit scaling + margin loss.
        """
        logits = outputs["logits"]  # [B, C]
        labels = inputs.get("labels")
        if labels is None:
            return base_loss

        # Ensure adb_scales is a learnable parameter on the right device
        num_classes = logits.size(-1)
        if not hasattr(self, "_adb_scales"):
            self._adb_scales = nn.Parameter(torch.ones(num_classes, device=logits.device))
        else:
            if self._adb_scales.device != logits.device:
                self._adb_scales = nn.Parameter(self._adb_scales.to(logits.device))
            if self._adb_scales.size(0) != num_classes:
                # Expand if classifier grew
                new_scales = torch.ones(num_classes, device=logits.device)
                old_n = self._adb_scales.size(0)
                new_scales[:old_n] = self._adb_scales.data[:old_n]
                self._adb_scales = nn.Parameter(new_scales)

        # Scale logits per class
        scaled_logits = logits * self._adb_scales.unsqueeze(0)

        # Margin loss: for positive labels, encourage scaled_logits > margin
        # for negative labels, encourage scaled_logits < -margin
        mask_pos = (labels > 0).float()
        mask_neg = (labels == 0).float()

        margin = self.adb_margin
        pos_loss = (mask_pos * torch.relu(margin - scaled_logits)).sum() / (mask_pos.sum() + 1e-8)
        neg_loss = (mask_neg * torch.relu(margin + scaled_logits)).sum() / (mask_neg.sum() + 1e-8)

        adb_loss = pos_loss + neg_loss
        return base_loss + adb_loss * 0.1

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override to compute standard FSCIL metrics."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return metrics

        dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        all_probs = []
        all_labels = []
        all_u = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_rejection=False,
                )
                logits = outputs["logits"]
                probs = torch.sigmoid(logits)
                labels = batch["labels"]

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Uncertainty score for OOD
                max_prob = probs.max(dim=-1)[0]
                all_u.append((1.0 - max_prob).cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_u = np.concatenate(all_u, axis=0)

        preds = (all_probs >= 0.5).astype(int)

        macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)

        per_class_ap = []
        for c in range(all_labels.shape[1]):
            if all_labels[:, c].sum() > 0:
                ap = average_precision_score(all_labels[:, c], all_probs[:, c])
                per_class_ap.append(ap)
        avg_map = np.mean(per_class_ap) if per_class_ap else 0.0

        metrics[f"{metric_key_prefix}_macro_f1"] = float(macro_f1)
        metrics[f"{metric_key_prefix}_micro_f1"] = float(micro_f1)
        metrics[f"{metric_key_prefix}_avg_map"] = float(avg_map)

        self.model.train()
        return metrics
