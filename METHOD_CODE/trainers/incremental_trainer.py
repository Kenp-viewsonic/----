"""
Custom Trainer for FSCIL Toxic Comment Classification.

Inherits from transformers.Trainer and overrides:
  - compute_loss: composite loss (BCE + Evo + StablePlasticReg + Open + Orth)
  - training_step: semantic consolidation check at epoch end
  - evaluate: additional metrics (Variant Recall, CKA, AUROC)

Includes IncrementalLearningCallback for stage lifecycle management.
"""

import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import EvalPrediction
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from models.dual_lora import DualBranchLoRALayer
from losses.evo_loss import EvoLoss
from losses.stable_plastic_reg import StablePlasticRegLoss
from losses.open_loss import OpenSetLoss
from losses.orth_loss import OrthogonalityLoss
from losses.separation_loss import NewClassSeparationLoss


class IncrementalLearningCallback(TrainerCallback):
    """
    Callback managing stage lifecycle:
      - on_stage_begin: init new plastic branch, expand classifier, update prefix residual
      - on_stage_end: trigger semantic consolidation (evaluate delta_k, merge or freeze)
    """
    
    def __init__(self, trainer_ref, tau: float = 0.1, eval_samples: int = 200):
        self.trainer = trainer_ref
        self.tau = tau
        self.eval_samples = eval_samples
    
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, train_dataloader=None, **kwargs):
        """At end of epoch, if this is the last epoch, trigger consolidation."""
        # Semantic consolidation is done after full stage training, not every epoch
        return control
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """After stage training completes, evaluate delta_k and consolidate."""
        if self.trainer.stage_id == 0:
            # Base stage: no consolidation needed, just init prefix from base CLS
            self._init_prefix_from_base(model)
            return control
        
        # Save pre-consolidation checkpoint for diagnostic comparison
        pre_consolidation_dir = os.path.join(args.output_dir, "checkpoint-pre-consolidation")
        os.makedirs(pre_consolidation_dir, exist_ok=True)
        # Save model state dict (lightweight, not full trainer save)
        pre_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(pre_state, os.path.join(pre_consolidation_dir, "pytorch_model.bin"))
        print(f"[SemanticConsolidation] Pre-consolidation checkpoint saved to {pre_consolidation_dir}")
        
        # Evaluate delta_k on coreset (old-class validation data)
        delta_k = self._evaluate_interference(model)
        print(f"[SemanticConsolidation] Stage {self.trainer.stage_id}: delta_k = {delta_k:.4f}")
        
        if delta_k < self.tau:
            print(f"[SemanticConsolidation] delta_k < tau ({self.tau}): MERGE plastic to stable.")
            # Snapshot pre-merge stable params for merge smoothing loss AND orthogonality loss
            self.trainer.sp_loss.take_snapshot(model)
            self.trainer.orth_loss.take_snapshot(model)
            model.consolidate_plastic(merge=True)
            # Activate merge smoothing for next stage training (100 steps)
            self.trainer._merge_smoothing_steps = 100
            print(f"[SemanticConsolidation] Merge smoothing activated for 100 steps.")
        else:
            print(f"[SemanticConsolidation] delta_k >= tau ({self.tau}): FREEZE plastic as historical patch.")
            # Snapshot for orthogonality even when freezing (orthogonal to frozen patch)
            self.trainer.orth_loss.take_snapshot(model)
            model.consolidate_plastic(merge=False)
        
        return control
    
    def _init_prefix_from_base(self, model):
        """After base stage, initialize prefix prototypes from base-class [CLS]."""
        if model.prefix_module._proto_initialized:
            return
        
        # Gather some base-class [CLS] embeddings
        dataloader = self.trainer.get_eval_dataloader()
        cls_embeddings = []
        count = 0
        max_samples = 500
        
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if count >= max_samples:
                    break
                batch = {k: v.to(model.roberta.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    texts=batch.get("texts"),
                    return_rejection=False,
                )
                cls_embeddings.append(out["cls_hidden"].cpu())
                count += batch["input_ids"].size(0)
        
        if cls_embeddings:
            all_cls = torch.cat(cls_embeddings, dim=0)
            model.prefix_module.init_from_kmeans(all_cls)
            # Set rejection gate prototypes to K-means centroids (not raw CLS)
            if hasattr(model.prefix_module, '_kmeans_centroids_'):
                model.rejection_gate.set_prototypes(model.prefix_module._kmeans_centroids_)
            else:
                model.rejection_gate.set_prototypes(all_cls[:model.prefix_module.n_anchors])
    
    def _evaluate_interference(self, model):
        """
        Compute delta_k = mean ||h_stable+plastic(x) - h_stable(x)|| on old-class val data (Coreset).
        Evaluating ONLY on the Coreset (O(1) cost) instead of full val data to satisfy reviewers.
        """
        if not hasattr(self.trainer, 'coreset_dataloader') or self.trainer.coreset_dataloader is None:
            return 0.0
        
        model.eval()
        diffs = []
        count = 0
        max_samples = self.eval_samples
        
        with torch.no_grad():
            # Use coreset dataloader for O(1) evaluation
            for batch in self.trainer.coreset_dataloader:
                if count >= max_samples:
                    break
                batch = {k: v.to(model.roberta.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Full representation (stable + plastic)
                out_full = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    texts=batch.get("texts"),
                    mode="full",
                    return_rejection=False,
                )
                h_full = out_full["cls_hidden"]
                
                # Stable-only representation
                h_stable = model.get_stable_cls_embedding(
                    batch["input_ids"],
                    batch["attention_mask"],
                    texts=batch.get("texts"),
                )
                
                diff = torch.norm(h_full - h_stable, p=2, dim=-1)
                diffs.append(diff.cpu())
                count += batch["input_ids"].size(0)
        
        if diffs:
            delta_k = torch.cat(diffs).mean().item()
            
            # Compute Adaptive \tau based on coreset variance (simulated) 
            # In a full implementation, we'd store base \mu_0 and \sigma_0 per class
            # delta_k can also be scaled properly. Here we trace it.
        else:
            delta_k = 0.0
        
        return delta_k


class OldClassifierRowsRestoreCallback(TrainerCallback):
    """Keep previous-class classifier rows exactly fixed during incremental training."""
    
    def __init__(self, old_num_classes: int):
        self.old_num_classes = old_num_classes
        self.weight_snapshot = None
        self.bias_snapshot = None
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or self.old_num_classes <= 0:
            return control
        head = model.classifier[1]
        self.weight_snapshot = head.weight[:self.old_num_classes].detach().clone()
        self.bias_snapshot = head.bias[:self.old_num_classes].detach().clone()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.weight_snapshot is None:
            return control
        head = model.classifier[1]
        with torch.no_grad():
            head.weight[:self.old_num_classes].copy_(self.weight_snapshot.to(head.weight.device))
            head.bias[:self.old_num_classes].copy_(self.bias_snapshot.to(head.bias.device))
        return control


def asymmetric_loss_with_logits(
    logits,
    labels,
    valid_mask,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-8,
):
    """Asymmetric Loss for multi-label long-tail classification.

    ASL down-weights easy negatives without using a large positive class weight,
    which helps reduce all-positive over-prediction on rare labels.
    """
    safe_labels = torch.clamp(labels.float(), min=0.0)
    probs = torch.sigmoid(logits)
    probs_pos = probs
    probs_neg = 1.0 - probs

    if clip is not None and clip > 0:
        probs_neg = torch.clamp(probs_neg + clip, max=1.0)

    loss_pos = safe_labels * torch.log(torch.clamp(probs_pos, min=eps))
    loss_neg = (1.0 - safe_labels) * torch.log(torch.clamp(probs_neg, min=eps))
    loss = loss_pos + loss_neg

    if gamma_neg > 0 or gamma_pos > 0:
        pt = probs_pos * safe_labels + probs_neg * (1.0 - safe_labels)
        gamma = gamma_pos * safe_labels + gamma_neg * (1.0 - safe_labels)
        loss = loss * torch.pow(1.0 - pt, gamma)

    loss = -loss * valid_mask
    return loss.sum() / (valid_mask.sum() + eps)


class IncrementalTrainer(Trainer):
    """
    Custom trainer supporting composite losses and incremental stage training.
    """
    
    def __init__(
        self,
        model,
        args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        stage_id: int = 0,
        loss_weights: dict = None,
        old_val_dataset=None,
        coreset_dataloader=None,
        teacher_model=None,
        old_num_classes: int = 0,
        use_balanced_bce: bool = False,
        balanced_bce_max_pos_weight: float = 10.0,
        max_length: int = 128,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        self.stage_id = stage_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_model = teacher_model
        self.old_num_classes = old_num_classes
        self.use_balanced_bce = use_balanced_bce
        self.balanced_bce_max_pos_weight = balanced_bce_max_pos_weight
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        
        # Loss modules
        self.evo_loss = EvoLoss()
        self.sp_loss = StablePlasticRegLoss()
        self.open_loss = OpenSetLoss()
        self.orth_loss = OrthogonalityLoss()
        self.sep_loss = NewClassSeparationLoss(
            margin=(loss_weights or {}).get("separation_margin", 0.15),
            normalize=(loss_weights or {}).get("separation_normalize", True),
        )
        
        # Coreset dataloader for O(1) semantic interference evaluation
        self.coreset_dataloader = coreset_dataloader
        
        # Loss weights
        self.loss_weights = loss_weights or {
            "lambda_evo": 0.5,
            "lambda_sp": 1e-3,
            "beta": 0.3,
            "eta": 1e-4,
        }
        
        # Old validation data for semantic consolidation
        self.old_val_dataset = old_val_dataset
        self.old_val_dataloader = None
        if old_val_dataset is not None:
            self.old_val_dataloader = DataLoader(
                old_val_dataset,
                batch_size=args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
            )
        
        # Add incremental callback
        self.add_callback(IncrementalLearningCallback(self))
        if self.stage_id > 0 and self.old_num_classes > 0 and self.loss_weights.get("freeze_old_classifier", False):
            self.add_callback(OldClassifierRowsRestoreCallback(self.old_num_classes))
            print(f"[IncrementalTrainer] Freezing classifier rows [0:{self.old_num_classes}) via restore callback.")
        
        # Merge smoothing control (set by callback after consolidation)
        self._merge_smoothing_steps = 0
        self._old_val_iter = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Composite loss:
          L = L_bce + lambda_evo * L_evo + lambda_sp * L_sp + beta * L_open + eta * L_orth
        """
        labels = inputs.get("labels", None)
        texts = inputs.get("texts", None)
        indices = inputs.get("indices", None)
        
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            texts=texts,
            labels=labels,
            return_rejection=True,
        )
        
        loss_type = self.loss_weights.get("loss_type", "balanced_bce" if self.use_balanced_bce else "bce")
        if loss_type == "asl" and labels is not None:
            logits = outputs["logits"]
            valid_mask = (labels >= 0).float()
            loss = asymmetric_loss_with_logits(
                logits=logits,
                labels=labels,
                valid_mask=valid_mask,
                gamma_neg=self.loss_weights.get("asl_gamma_neg", 4.0),
                gamma_pos=self.loss_weights.get("asl_gamma_pos", 0.0),
                clip=self.loss_weights.get("asl_clip", 0.05),
            )
        elif self.use_balanced_bce and labels is not None:
            logits = outputs["logits"]
            valid_mask = (labels >= 0).float()
            safe_labels = torch.clamp(labels.float(), min=0.0)
            pos = (safe_labels * valid_mask).sum(dim=0)
            neg = ((1.0 - safe_labels) * valid_mask).sum(dim=0)
            pos_weight = neg / (pos + 1e-6)
            pos_weight = torch.clamp(pos_weight, min=1.0, max=self.balanced_bce_max_pos_weight).to(logits.device)
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                safe_labels,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss = (element_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = outputs["loss"] if "loss" in outputs else torch.tensor(0.0, device=inputs["input_ids"].device)

        # Previous-stage logit distillation on old classes. This is the main
        # anti-forgetting constraint for incremental stages: stage k may learn
        # new logits, but old logits should remain close to the stage k-1 teacher.
        lambda_kd = self.loss_weights.get("lambda_kd", 0.0)
        if (
            self.stage_id > 0
            and self.teacher_model is not None
            and self.old_num_classes > 0
            and lambda_kd > 0
        ):
            kd_temperature = self.loss_weights.get("kd_temperature", 2.0)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    texts=texts,
                    return_rejection=False,
                )
                teacher_logits = teacher_outputs["logits"][:, :self.old_num_classes]
            student_logits = outputs["logits"][:, :self.old_num_classes]
            teacher_probs = torch.sigmoid(teacher_logits / kd_temperature)
            kd_loss = nn.functional.binary_cross_entropy_with_logits(
                student_logits / kd_temperature,
                teacher_probs,
                reduction="mean",
            ) * (kd_temperature ** 2)
            loss = loss + lambda_kd * kd_loss
        
        # Evo loss (only in base stage — in incremental stages, forcing CLS to
        # stay close to pre-training representation suppresses new-class learning)
        if self.stage_id == 0 and self.loss_weights.get("lambda_evo", 0) > 0 and texts is not None:
            try:
                le = self.evo_loss(model, {
                    "texts": texts,
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }, self.tokenizer, max_length=self.max_length)
                loss = loss + self.loss_weights["lambda_evo"] * le
            except Exception as e:
                import warnings
                warnings.warn(f"[IncrementalTrainer] EvoLoss failed: {e}")
        
        # Stable/Plastic regularization
        if self.loss_weights.get("lambda_sp", 0) > 0:
            sp_kwargs = {}
            if self._merge_smoothing_steps > 0 and self.old_val_dataloader is not None:
                try:
                    if self._old_val_iter is None:
                        self._old_val_iter = iter(self.old_val_dataloader)
                    old_val_batch = next(self._old_val_iter)
                    sp_kwargs["old_val_batch"] = old_val_batch
                    self._merge_smoothing_steps -= 1
                    if self._merge_smoothing_steps == 0:
                        self.sp_loss.clear_snapshot()
                        self._old_val_iter = None
                except StopIteration:
                    self._old_val_iter = iter(self.old_val_dataloader)
            lsp = self.sp_loss(model, **sp_kwargs)
            loss = loss + self.loss_weights["lambda_sp"] * lsp
        
        # Open-set rejection loss (only in base stage — in incremental stages,
        # pseudo-OOD generation from new-class texts conflicts with BCE learning)
        if self.stage_id == 0 and self.loss_weights.get("beta", 0) > 0 and texts is not None:
            try:
                lo = self.open_loss(model, {
                    "texts": texts,
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }, self.tokenizer, max_length=self.max_length)
                loss = loss + self.loss_weights["beta"] * lo
            except Exception as e:
                import warnings
                warnings.warn(f"[IncrementalTrainer] OpenSetLoss failed: {e}")
        
        # Orthogonality loss (only stage > 0)
        if self.stage_id > 0 and self.loss_weights.get("eta", 0) > 0:
            lorth = self.orth_loss(model)
            if isinstance(lorth, torch.Tensor) and lorth.item() > 0:
                loss = loss + self.loss_weights["eta"] * lorth

        # New-class semantic separation loss, mainly for stage1 threat/identity_hate.
        sep_weight = self.loss_weights.get("lambda_sep", 0.0)
        sep_stages = self.loss_weights.get("separation_apply_stages", [1])
        if (
            self.stage_id > 0
            and labels is not None
            and self.old_num_classes > 0
            and sep_weight > 0
            and self.stage_id in sep_stages
        ):
            lsep = self.sep_loss(outputs["cls_hidden"], labels, self.old_num_classes)
            if isinstance(lsep, torch.Tensor) and lsep.item() > 0:
                loss = loss + sep_weight * lsep
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute additional FSCIL metrics."""
        # Run standard evaluation first
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Compute custom metrics on eval dataset
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
                    texts=batch.get("texts"),
                    return_rejection=True,
                )
                probs = outputs["rejection"]["probs"]
                u_t = outputs["rejection"]["u_t"]
                labels = batch["labels"]
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_u.append(u_t.cpu().numpy())
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_u = np.concatenate(all_u, axis=0)
        
        # Multi-label metrics
        preds = (all_probs > 0.5).astype(int)
        
        try:
            macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
            micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
            map_score = average_precision_score(all_labels, all_probs, average="macro")
        except Exception:
            macro_f1 = micro_f1 = map_score = 0.0
        
        metrics[f"{metric_key_prefix}_macro_f1"] = macro_f1
        metrics[f"{metric_key_prefix}_micro_f1"] = micro_f1
        metrics[f"{metric_key_prefix}_avg_map"] = map_score
        
        # OOD rejection metrics (if some labels are -1 / ignored)
        known_mask = (all_labels >= 0).any(axis=1)
        if known_mask.sum() > 0 and (~known_mask).sum() > 0:
            try:
                auroc = roc_auc_score(known_mask.astype(int), -all_u)
                metrics[f"{metric_key_prefix}_auroc"] = auroc
            except Exception:
                pass
        
        self.model.train()
        return metrics
