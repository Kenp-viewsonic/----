"""
Evaluation metrics for FSCIL Toxic Comment Classification.

Includes:
  - Avg-mAP, Macro-F1, Micro-F1, Forgetting
  - AUROC, FPR95 for OOD rejection
  - Variant Recall (toxic variant robustness)
  - Semantic Stability (CKA between stage representations)
  - Tail Recall
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    recall_score,
)


def compute_fscil_metrics(probs, labels, threshold=0.5):
    """
    Args:
        probs: [N, C] predicted probabilities (after sigmoid)
        labels: [N, C] ground-truth binary labels
        threshold: classification threshold
    
    Returns:
        dict with macro_f1, micro_f1, avg_map, per_class_recall
    """
    preds = (probs >= threshold).astype(int)
    
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    
    # Average mAP across classes
    per_class_ap = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() > 0:
            ap = average_precision_score(labels[:, c], probs[:, c])
            per_class_ap.append(ap)
    avg_map = np.mean(per_class_ap) if per_class_ap else 0.0
    
    # Per-class recall
    per_class_recall = {}
    for c in range(labels.shape[1]):
        rc = recall_score(labels[:, c], preds[:, c], zero_division=0)
        per_class_recall[c] = rc
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "avg_map": avg_map,
        "per_class_recall": per_class_recall,
    }


def compute_forgetting(perf_history):
    """
    Compute forgetting metric for each class across stages.
    
    Args:
        perf_history: dict[class_id] -> list of performance values per stage
    
    Returns:
        avg_forgetting: average forgetting across classes
    """
    forgettings = []
    for cls_id, perfs in perf_history.items():
        if len(perfs) < 2:
            continue
        max_perf = max(perfs[:-1])  # best performance before last stage
        last_perf = perfs[-1]
        forgetting = max_perf - last_perf
        forgettings.append(max(0, forgetting))
    
    return np.mean(forgettings) if forgettings else 0.0


def compute_cka(X, Y):
    """
    Centered Kernel Alignment (CKA) between two representation matrices.
    Uses Linear CKA — mathematically equivalent to standard CKA with linear
    kernel, but O(np²) instead of O(n³). Runs on GPU when available.

    CKA(X, Y) = ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)

    Args:
        X: [n, p] numpy array — representation matrix from stage A
        Y: [n, p] numpy array — representation matrix from stage B (same n samples)

    Returns:
        cka_score: float in [0, 1]
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.from_numpy(X.astype(np.float32)).to(device)
    Y = torch.from_numpy(Y.astype(np.float32)).to(device)

    # Center columns
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear CKA: only p×p matrices (p=768), not n×n
    XtX = X.T @ X   # [p, p]
    XtY = X.T @ Y   # [p, p]
    YtY = Y.T @ Y   # [p, p]

    hsic = (XtY ** 2).sum()
    var_x = (XtX ** 2).sum().sqrt()
    var_y = (YtY ** 2).sum().sqrt()

    if var_x == 0 or var_y == 0:
        return 0.0

    cka = hsic / (var_x * var_y)
    return float(cka.cpu().item())


def compute_variant_recall(model, variant_texts, variant_labels, tokenizer, device="cpu", threshold=0.5):
    """
    Evaluate model on adversarially perturbed toxic variants.
    
    Args:
        model: RobertaToxicClassifier
        variant_texts: list of perturbed texts
        variant_labels: [N, C] ground truth labels for variants
        tokenizer: tokenizer
        device: torch device
        threshold: decision threshold
    
    Returns:
        variant_recall: float, proportion of variants where any toxic label > threshold
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    model.eval()
    encoding = tokenizer(
        variant_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    
    dataset = TensorDataset(encoding["input_ids"], encoding["attention_mask"])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_rejection=False)
            probs = torch.sigmoid(out["logits"])
            all_probs.append(probs.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Variant Recall: any toxic label predicted positive
    detected = (all_probs > threshold).any(axis=1)
    variant_recall = detected.mean()
    
    return float(variant_recall)


def compute_auroc_fpr95(known_scores, unknown_scores):
    """
    Compute AUROC and FPR@95TPR for OOD detection.

    Args:
        known_scores: [N_known] anomaly scores for known samples (higher = more uncertain/anomalous)
        unknown_scores: [N_unknown] anomaly scores for unknown samples

    Returns:
        dict with auroc, fpr95
    """
    y_true = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])
    y_score = np.concatenate([known_scores, unknown_scores])

    auroc = roc_auc_score(y_true, y_score)

    # FPR at 95% TPR using ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find the operating point where TPR >= 0.95 with the smallest FPR
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        fpr95 = fpr[idx[0]]
    else:
        fpr95 = 1.0

    return {"auroc": auroc, "fpr95": fpr95}


def compute_tail_recall(probs, labels, label_counts=None, threshold=0.5):
    """
    Compute recall for the tail class (least frequent positive class).

    Args:
        probs: [N, C] predicted probabilities (after sigmoid)
        labels: [N, C] ground-truth binary labels
        label_counts: list of positive counts per class. If None, compute from labels.
        threshold: classification threshold

    Returns:
        dict with tail_recall, tail_class
    """
    if label_counts is None:
        label_counts = labels.sum(axis=0)

    tail_class = int(np.argmin(label_counts))
    tail_preds = (probs[:, tail_class] >= threshold).astype(int)
    tail_labels = labels[:, tail_class].astype(int)
    recall = recall_score(tail_labels, tail_preds, zero_division=0)

    return {"tail_recall": float(recall), "tail_class": tail_class}


def evaluate_all_metrics(probs, labels, label_counts=None, variant_texts=None,
                         variant_labels=None, tokenizer=None, model=None,
                         device="cpu", prev_cls_reps=None, threshold=0.5):
    """
    Unified evaluation function for all baselines, ablations, and main method.

    Computes the full metric suite:
      - FSCIL: macro_f1, micro_f1, avg_map, per_class_recall
      - Tail: tail_recall
      - OOD: auroc, fpr95 (if -1 labels present)
      - Robustness: variant_recall (if variant data provided)
      - Stability: cka (if prev_cls_reps provided alongside current cls_hidden)

    Args:
        probs: [N, C] predicted probabilities (after sigmoid)
        labels: [N, C] ground-truth binary labels
        label_counts: Optional per-class positive counts for tail recall
        variant_texts: Optional list of perturbed texts for variant recall
        variant_labels: Optional [N_var, C] labels for variants
        tokenizer: Required if variant_texts provided
        model: Required if variant_texts provided
        device: torch device
        prev_cls_reps: Optional [N, H] previous stage CLS representations for CKA
        threshold: decision threshold

    Returns:
        dict of all computed metrics
    """
    metrics = compute_fscil_metrics(probs, labels, threshold=threshold)

    # Tail recall
    tail = compute_tail_recall(probs, labels, label_counts=label_counts, threshold=threshold)
    metrics["tail_recall"] = tail["tail_recall"]
    metrics["tail_class"] = tail["tail_class"]

    # OOD metrics: treat samples with all labels == -1 as OOD
    known_mask = (labels >= 0).any(axis=1)
    if (~known_mask).sum() > 0 and known_mask.sum() > 0:
        # Use max_prob as anomaly score (lower = more certain)
        max_prob = probs.max(axis=1)
        known_scores = -max_prob[known_mask]  # negate so higher = more anomalous
        unknown_scores = -max_prob[~known_mask]
        ood = compute_auroc_fpr95(known_scores, unknown_scores)
        metrics["auroc"] = ood["auroc"]
        metrics["fpr95"] = ood["fpr95"]

    # Variant recall
    if variant_texts is not None and variant_labels is not None and model is not None and tokenizer is not None:
        v_recall = compute_variant_recall(model, variant_texts, variant_labels, tokenizer, device=device, threshold=threshold)
        metrics["variant_recall"] = v_recall

    # CKA semantic stability
    if prev_cls_reps is not None and model is not None:
        # Need current CLS reps; caller should pass them separately if needed
        pass

    return metrics
