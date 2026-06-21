"""
Stage-wise new-class separation loss.

Designed for incremental FSCIL stages where newly introduced classes should be
pulled away from previously seen toxic clusters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewClassSeparationLoss(nn.Module):
    """
    Margin-based prototype separation on [CLS] representations.

    For each sample carrying a newly introduced class, encourage similarity to
    the corresponding new-class prototype to exceed similarity to old-class
    prototypes by a margin.
    """

    def __init__(self, margin: float = 0.15, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        cls_hidden: torch.Tensor,
        labels: torch.Tensor,
        old_num_classes: int,
    ) -> torch.Tensor:
        if labels is None or cls_hidden is None:
            return cls_hidden.new_zeros(())

        num_classes = labels.size(1)
        if old_num_classes <= 0 or old_num_classes >= num_classes:
            return cls_hidden.new_zeros(())

        valid_mask = labels >= 0
        positive_mask = labels > 0.5
        new_positive_mask = positive_mask[:, old_num_classes:]

        if not new_positive_mask.any():
            return cls_hidden.new_zeros(())

        reps = cls_hidden
        if self.normalize:
            reps = F.normalize(reps, dim=-1)

        prototypes = []
        for class_idx in range(num_classes):
            class_mask = positive_mask[:, class_idx] & valid_mask[:, class_idx]
            if class_mask.any():
                proto = reps[class_mask].mean(dim=0)
                if self.normalize:
                    proto = F.normalize(proto.unsqueeze(0), dim=-1).squeeze(0)
                prototypes.append(proto)
            else:
                prototypes.append(None)

        losses = []
        old_proto_indices = [idx for idx in range(old_num_classes) if prototypes[idx] is not None]
        if not old_proto_indices:
            return cls_hidden.new_zeros(())

        old_protos = torch.stack([prototypes[idx] for idx in old_proto_indices], dim=0)
        old_sims_all = reps @ old_protos.t()

        for class_idx in range(old_num_classes, num_classes):
            class_proto = prototypes[class_idx]
            if class_proto is None:
                continue

            sample_mask = positive_mask[:, class_idx] & valid_mask[:, class_idx]
            if not sample_mask.any():
                continue

            sample_reps = reps[sample_mask]
            pos_sim = (sample_reps * class_proto.unsqueeze(0)).sum(dim=-1)

            old_sim = old_sims_all[sample_mask].max(dim=-1).values

            same_sample_old = positive_mask[sample_mask, :old_num_classes] & valid_mask[sample_mask, :old_num_classes]
            if same_sample_old.any():
                masked_old_logits = old_sims_all[sample_mask].masked_fill(~same_sample_old, float("-inf"))
                hard_old = masked_old_logits.max(dim=-1).values
                hard_old = torch.where(torch.isfinite(hard_old), hard_old, old_sim)
                old_sim = torch.maximum(old_sim, hard_old)

            losses.append(F.relu(self.margin - pos_sim + old_sim))

        if not losses:
            return cls_hidden.new_zeros(())

        return torch.cat(losses, dim=0).mean()