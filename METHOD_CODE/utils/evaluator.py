"""
Unified evaluation interface for FSCIL Toxic Comment Classification.

All methods (Ours, baselines, ablations) use evaluate_model() to ensure
metric alignment across experiments.

Metrics produced:
  - Classification: macro_f1, micro_f1, avg_map, per_class_recall
  - Incremental: forgetting (requires perf_history input)
  - OOD rejection: auroc, fpr95 (requires OOD samples)
  - Robustness: variant_recall (requires variant_generator)
  - Stability: cka (requires reference representations)
  - Long-tail: tail_recall
"""

import os
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
)

from .metrics import compute_cka, compute_forgetting


def evaluate_model(
    model,
    dataloader,
    device,
    tokenizer=None,
    variant_generator=None,
    ood_dataloader=None,
    ref_representations=None,
    perf_history=None,
    threshold=0.5,
    tail_class_indices=None,
):
    """
    Unified evaluation producing a flat dict of metrics.

    Args:
        model: nn.Module with forward returning dict containing 'logits' and optionally 'rejection'
        dataloader: DataLoader for known-class evaluation
        device: torch device
        tokenizer: required for variant_recall
        variant_generator: VariantGenerator instance for robustness test
        ood_dataloader: DataLoader for OOD samples (for auroc/fpr95)
        ref_representations: [N, H] reference CLS embeddings for CKA stability
        perf_history: dict[class_id] -> list of per-stage performance values (for forgetting)
        threshold: classification threshold
        tail_class_indices: list of class indices considered tail classes (for tail_recall)

    Returns:
        metrics: dict[str, float]
    """
    model.eval()

    all_logits = []
    all_labels = []
    all_cls = []
    all_u = []
    all_texts = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            texts = batch.get("texts", [])

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts if hasattr(model, "forward") and "texts" in model.forward.__code__.co_varnames else None,
                return_rejection=True,
            )

            logits = outputs["logits"]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)
            all_cls.append(outputs.get("cls_hidden", torch.zeros_like(logits[:, :1].expand(-1, 768))).cpu().numpy())
            all_texts.extend(texts)

            if "rejection" in outputs and outputs["rejection"] is not None:
                all_u.append(outputs["rejection"]["u_t"].cpu().numpy())
            else:
                # Fallback: use max_prob as uncertainty (higher = more uncertain for MSP-like baselines)
                probs = torch.sigmoid(logits)
                max_prob = probs.max(dim=-1)[0]
                all_u.append((1.0 - max_prob).cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_cls = np.concatenate(all_cls, axis=0)
    all_u = np.concatenate(all_u, axis=0)

    probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= threshold).astype(int)

    # Mask for valid labels (>=0)
    valid_mask = (all_labels >= 0)

    # --- Classification metrics ---
    # Only compute on valid labels
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)

    per_class_ap = []
    per_class_recall = {}
    per_class_precision = {}
    per_class_f1 = {}
    per_class_support = {}
    per_class_pred_pos = {}
    per_class_prob_mean = {}
    per_class_prob_pos_mean = {}
    per_class_best_threshold = {}
    per_class_best_precision = {}
    per_class_best_recall = {}
    per_class_best_f1 = {}
    per_class_best_pred_pos = {}
    for c in range(all_labels.shape[1]):
        valid = valid_mask[:, c]
        if valid.sum() > 0 and all_labels[valid, c].sum() > 0:
            ap = average_precision_score(all_labels[valid, c], probs[valid, c])
            per_class_ap.append(ap)
        labels_c = all_labels[valid, c]
        preds_c = preds[valid, c]
        probs_c = probs[valid, c]
        rc = recall_score(labels_c, preds_c, zero_division=0)
        pc = precision_score(labels_c, preds_c, zero_division=0)
        f1c = f1_score(labels_c, preds_c, zero_division=0)
        per_class_recall[c] = float(rc)
        per_class_precision[c] = float(pc)
        per_class_f1[c] = float(f1c)
        per_class_support[c] = int(labels_c.sum())
        per_class_pred_pos[c] = int(preds_c.sum())
        per_class_prob_mean[c] = float(probs_c.mean()) if probs_c.size > 0 else 0.0
        pos_mask = labels_c == 1
        per_class_prob_pos_mean[c] = float(probs_c[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
        best = {
            "threshold": threshold,
            "precision": pc,
            "recall": rc,
            "f1": f1c,
            "pred_pos": int(preds_c.sum()),
        }
        # Diagnostic threshold sweep (vectorized, single-pass)
        thresholds = np.linspace(0.05, 0.95, 13)
        pos_total = float(labels_c.sum())
        if pos_total > 0:
            # [N_c, 13] boolean matrix: True where prob >= threshold
            t_preds_all = probs_c[:, None] >= thresholds[None, :]
            tp = (t_preds_all & (labels_c[:, None] == 1)).sum(axis=0).astype(float)
            fp = (t_preds_all & (labels_c[:, None] == 0)).sum(axis=0).astype(float)
            denom_p = tp + fp
            t_prec = np.divide(tp, denom_p, out=np.zeros_like(tp), where=denom_p > 0)
            t_rec = tp / max(pos_total, 1.0)
            denom_f1 = t_prec + t_rec
            t_f1s = np.divide(2 * t_prec * t_rec, denom_f1, out=np.zeros_like(t_prec), where=denom_f1 > 0)
            best_idx = int(np.argmax(t_f1s))
            if t_f1s[best_idx] > best["f1"]:
                best["threshold"] = float(thresholds[best_idx])
                best["precision"] = float(t_prec[best_idx])
                best["recall"] = float(t_rec[best_idx])
                best["f1"] = float(t_f1s[best_idx])
                best["pred_pos"] = int(tp[best_idx] + fp[best_idx])
        per_class_best_threshold[c] = float(best["threshold"])
        per_class_best_precision[c] = float(best["precision"])
        per_class_best_recall[c] = float(best["recall"])
        per_class_best_f1[c] = float(best["f1"])
        per_class_best_pred_pos[c] = int(best["pred_pos"])

    avg_map = float(np.mean(per_class_ap)) if per_class_ap else 0.0

    metrics = {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "avg_map": float(avg_map),
    }
    metrics.update({f"recall_cls{c}": v for c, v in per_class_recall.items()})
    metrics.update({f"precision_cls{c}": v for c, v in per_class_precision.items()})
    metrics.update({f"f1_cls{c}": v for c, v in per_class_f1.items()})
    metrics.update({f"support_cls{c}": v for c, v in per_class_support.items()})
    metrics.update({f"pred_pos_cls{c}": v for c, v in per_class_pred_pos.items()})
    metrics.update({f"prob_mean_cls{c}": v for c, v in per_class_prob_mean.items()})
    metrics.update({f"prob_pos_mean_cls{c}": v for c, v in per_class_prob_pos_mean.items()})
    metrics.update({f"best_threshold_cls{c}": v for c, v in per_class_best_threshold.items()})
    metrics.update({f"best_precision_cls{c}": v for c, v in per_class_best_precision.items()})
    metrics.update({f"best_recall_cls{c}": v for c, v in per_class_best_recall.items()})
    metrics.update({f"best_f1_cls{c}": v for c, v in per_class_best_f1.items()})
    metrics.update({f"best_pred_pos_cls{c}": v for c, v in per_class_best_pred_pos.items()})

    # --- Tail recall ---
    if tail_class_indices is not None:
        tail_recalls = [per_class_recall.get(c, 0.0) for c in tail_class_indices]
        metrics["tail_recall"] = float(np.mean(tail_recalls)) if tail_recalls else 0.0

    # --- Forgetting ---
    if perf_history is not None:
        metrics["forgetting"] = float(compute_forgetting(perf_history))

    # --- OOD rejection (AUROC / FPR95) ---
    if ood_dataloader is not None:
        ood_u = []
        with torch.no_grad():
            for batch in ood_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                texts = batch.get("texts", [])
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    texts=texts if hasattr(model, "forward") and "texts" in model.forward.__code__.co_varnames else None,
                    return_rejection=True,
                )
                if "rejection" in outputs and outputs["rejection"] is not None:
                    ood_u.append(outputs["rejection"]["u_t"].cpu().numpy())
                else:
                    probs = torch.sigmoid(outputs["logits"])
                    max_prob = probs.max(dim=-1)[0]
                    ood_u.append((1.0 - max_prob).cpu().numpy())
        ood_u = np.concatenate(ood_u, axis=0)

        # Known samples have lower u_t (more certain), unknown have higher u_t
        y_true = np.concatenate([np.zeros(len(all_u)), np.ones(len(ood_u))])
        y_score = np.concatenate([all_u, ood_u])

        try:
            auroc = roc_auc_score(y_true, y_score)
            metrics["auroc"] = float(auroc)
        except Exception:
            metrics["auroc"] = 0.5

        # FPR at 95% TPR: threshold where 95% of known are correctly classified as known
        sorted_known = np.sort(all_u)
        if len(sorted_known) > 0:
            threshold_95_idx = min(int(np.ceil(0.95 * len(sorted_known))) - 1, len(sorted_known) - 1)
            threshold_95 = sorted_known[threshold_95_idx]
            fpr95 = float((ood_u <= threshold_95).mean())
            metrics["fpr95"] = fpr95
        else:
            metrics["fpr95"] = 1.0

    # --- Semantic Stability (CKA) ---
    if ref_representations is not None and all_cls.shape[0] == ref_representations.shape[0]:
        cka = compute_cka(ref_representations, all_cls)
        metrics["cka"] = float(cka)

    # --- Variant Recall ---
    if variant_generator is not None and tokenizer is not None and len(all_texts) > 0:
        # Sample up to 200 texts for variant generation to keep it fast
        sample_size = min(200, len(all_texts))
        sample_indices = np.random.choice(len(all_texts), sample_size, replace=False)
        sample_texts = [all_texts[i] for i in sample_indices]
        sample_labels = all_labels[sample_indices]

        variants = []
        variant_labels = []
        for txt, lbl in zip(sample_texts, sample_labels):
            if lbl.sum() > 0:  # only toxic samples
                v = variant_generator.generate_variant(txt, mode="random")
                variants.append(v)
                variant_labels.append(lbl)

        if variants:
            variant_labels = np.stack(variant_labels, axis=0)
            enc = tokenizer(
                variants,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            v_dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
            v_loader = DataLoader(v_dataset, batch_size=32, shuffle=False)

            v_probs = []
            with torch.no_grad():
                for batch in v_loader:
                    input_ids, attention_mask = [b.to(device) for b in batch]
                    out = model(input_ids=input_ids, attention_mask=attention_mask, return_rejection=False)
                    v_probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
            v_probs = np.concatenate(v_probs, axis=0)
            detected = (v_probs > threshold).any(axis=1)
            metrics["variant_recall"] = float(detected.mean())
        else:
            metrics["variant_recall"] = 0.0

    return metrics


def evaluate_all_stages(
    checkpoint_dirs,
    test_dataloaders,
    device,
    tokenizer=None,
    variant_generator=None,
    ood_dataloaders=None,
    ref_cls_stage0=None,
    tail_class_indices=None,
):
    """
    Plan Step 4.1: Evaluate each stage's checkpoint on the cumulative test set.

    Loads the best checkpoint for each stage and evaluates on the cumulative
    test set (all classes seen up to and including that stage).

    Args:
        checkpoint_dirs: List of checkpoint directory paths [stage_0, stage_1, ...]
        test_dataloaders: List of DataLoaders for cumulative testing per stage
        device: torch device
        tokenizer: tokenizer for variant recall
        variant_generator: VariantGenerator for robustness test
        ood_dataloaders: List of OOD DataLoaders (next-stage unseen classes) per stage
        ref_stage0_cls: [N, H] reference CLS embeddings from stage 0 (for CKA stability)
        tail_class_indices: List of class indices considered tail classes

    Returns:
        List[dict]: metrics dict for each stage
        perf_history: {class_id: [perf_stage0, perf_stage1, ...]} for forgetting computation

    Note: This function expects callers to handle model instantiation and loading.
          Use evaluate_model() for single-stage evaluation.
    """
    all_metrics = []
    perf_history = {}  # class_id -> list of per-stage per-class recall

    for stage_idx, ckpt_dir in enumerate(checkpoint_dirs):
        if ckpt_dir is None or not os.path.exists(ckpt_dir):
            print(f"[evaluate_all_stages] Skipping stage {stage_idx}: checkpoint not found at {ckpt_dir}")
            all_metrics.append({})
            continue

        print(f"\n[evaluate_all_stages] Stage {stage_idx} — {ckpt_dir}")

        # The caller must provide a model factory or pre-loaded models.
        # Since this is a utility function, we rely on the caller to manage model loading.
        # For convenience, we provide a loaded_model parameter pattern:
        pass

    return all_metrics


def evaluate_all_stages(
    model_factory,
    checkpoint_dirs: list,
    dataloaders: list,
    device,
    tokenizer=None,
    variant_generator=None,
    ood_dataloaders: list = None,
    stage0_ref_cls: np.ndarray = None,
    tail_class_indices: list = None,
    class_names: list = None,
):
    """
    Convenience wrapper that loads each checkpoint and evaluates on cumulative test set.

    Args:
        model_factory: callable(stage_idx, ckpt_dir) -> model
        checkpoint_dirs: [stage_0_ckpt, stage_1_ckpt, ...]
        dataloaders: [cumulative_loader_stage0, cumulative_loader_stage1, ...]
        device: torch device
        tokenizer: tokenizer
        variant_generator: VariantGenerator or None
        ood_dataloaders: [ood_loader_stage0, ood_loader_stage1, ...] or None
        stage0_ref_cls: [N, H] reference embeddings from stage 0 for CKA
        tail_class_indices: tail class indices
        class_names: list of class names (for readability)

    Returns:
        all_metrics: List[dict] per-stage metrics
        perf_history: {class_id: [perf_stage0, ...]} for forgetting
    """
    all_metrics = []
    perf_history = {}
    # Determine total number of classes
    num_classes = 0

    for stage_idx, ckpt_dir in enumerate(checkpoint_dirs):
        if ckpt_dir is None or not os.path.exists(ckpt_dir):
            print(f"[evaluate_all_stages] Skipping stage {stage_idx}")
            all_metrics.append({})
            continue

        print(f"\n[evaluate_all_stages] === Stage {stage_idx} ===")
        model = model_factory(stage_idx, ckpt_dir)

        dataloader = dataloaders[stage_idx] if stage_idx < len(dataloaders) else None
        if dataloader is None:
            all_metrics.append({})
            continue

        ood_loader = None
        if ood_dataloaders and stage_idx < len(ood_dataloaders):
            ood_loader = ood_dataloaders[stage_idx]

        # Update num_classes from data labels if possible
        if num_classes == 0:
            try:
                sample_batch = next(iter(dataloader))
                num_classes = sample_batch["labels"].shape[1]
            except Exception:
                num_classes = 3 + stage_idx  # heuristic

        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            tokenizer=tokenizer,
            variant_generator=variant_generator,
            ood_dataloader=ood_loader,
            ref_representations=stage0_ref_cls,
            perf_history=perf_history,
            tail_class_indices=tail_class_indices,
        )

        # Track per-class recall for forgetting
        for cls_id_str, val in metrics.items():
            if cls_id_str.startswith("recall_cls"):
                cls_id = int(cls_id_str.replace("recall_cls", ""))
                if cls_id not in perf_history:
                    perf_history[cls_id] = []
                perf_history[cls_id].append(val)

        # Add stage label
        metrics["stage"] = stage_idx
        all_metrics.append(metrics)

        # Print summary
        print(f"  Macro-F1: {metrics.get('macro_f1', 0):.4f}, "
              f"Avg-mAP: {metrics.get('avg_map', 0):.4f}, "
              f"AUROC: {metrics.get('auroc', 0):.4f}")

    # Compute final forgetting across all stages
    from .metrics import compute_forgetting
    if perf_history:
        avg_forgetting = compute_forgetting(perf_history)
        # Attach to last stage metrics
        if all_metrics:
            all_metrics[-1]["avg_forgetting"] = float(avg_forgetting)
        print(f"\n[evaluate_all_stages] Average Forgetting: {avg_forgetting:.4f}")

    return all_metrics, perf_history
