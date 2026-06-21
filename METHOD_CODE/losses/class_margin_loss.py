"""
Class-specific margin loss for incremental stages.

For each newly introduced class, penalizes the model when the new-class logit
is NOT higher than the maximum old-class logit by at least `margin`.
Only activates for samples where the new class label is positive.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassMarginLoss(nn.Module):
    """Margin-based new-class boundary enforcement on logits."""

    def __init__(self, margin: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        old_num_classes: int,
    ) -> torch.Tensor:
        if labels is None or old_num_classes <= 0:
            return logits.new_zeros(())

        num_classes = logits.size(1)
        if old_num_classes >= num_classes:
            return logits.new_zeros(())

        valid_mask = labels >= 0
        positive_mask = labels > 0.5

        old_logits = logits[:, :old_num_classes]
        old_logits_masked = old_logits.clone()
        old_logits_masked[~(valid_mask[:, :old_num_classes])] = float("-inf")
        max_old = old_logits_masked.max(dim=1).values

        losses = []
        for class_offset, class_idx in enumerate(range(old_num_classes, num_classes)):
            class_pos = positive_mask[:, class_idx] & valid_mask[:, class_idx]
            if not class_pos.any():
                continue
            new_logit = logits[class_pos, class_idx]
            old_max = max_old[class_pos]
            losses.append(F.relu(self.margin - new_logit + old_max))

        if not losses:
            return logits.new_zeros(())

        stacked = torch.cat(losses, dim=0)
        if self.reduction == "mean":
            return stacked.mean()
        return stacked.sum()
