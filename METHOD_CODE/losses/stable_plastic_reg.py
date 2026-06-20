"""
Stable/Plastic Regularization Loss (Section 3.3 of problemDef).

Comprises:
  1. Stable branch subspace decorrelation (ELLA-style)
  2. Plastic branch L1 sparsity
  3. Merge smoothing loss (if consolidation occurred)
"""

import torch
import torch.nn as nn

from models.dual_lora import DualBranchLoRALayer


class StablePlasticRegLoss(nn.Module):
    """
    Args:
        lambda_sp: Weight for plastic L1 sparsity
        lambda_stable: Weight for stable decorrelation
        lambda_merge: Weight for merge smoothing
    """
    
    def __init__(
        self,
        lambda_sp: float = 1e-3,
        lambda_stable: float = 1e-4,
        lambda_merge: float = 1e-3,
    ):
        super().__init__()
        self.lambda_sp = lambda_sp
        self.lambda_stable = lambda_stable
        self.lambda_merge = lambda_merge
        self.pre_merge_snapshots = {}
    
    def take_snapshot(self, model):
        """Snapshot current stable branch parameters before merge."""
        self.pre_merge_snapshots = {}
        for name, module in model.named_modules():
            if isinstance(module, DualBranchLoRALayer):
                self.pre_merge_snapshots[name] = {
                    'A': module.stable_A.detach().clone(),
                    'B': module.stable_B.detach().clone(),
                }
    
    def clear_snapshot(self):
        """Clear snapshot after merge smoothing period ends."""
        self.pre_merge_snapshots = {}
    
    def forward(self, model, old_val_batch=None):
        """
        Args:
            model: RobertaToxicClassifier
            old_val_batch: Validation batch from old classes for merge smoothing (optional)
        
        Returns:
            loss: scalar
        """
        total_loss = 0.0
        
        for name, module in model.named_modules():
            if not isinstance(module, DualBranchLoRALayer):
                continue
            
            # 1. Plastic L1 sparsity
            if module.r_plastic > 0:
                l1_plastic = (
                    module.plastic_A.abs().sum() + module.plastic_B.abs().sum()
                )
                total_loss += self.lambda_sp * l1_plastic
            
            # 2. Stable subspace decorrelation (ELLA-style)
            # Penalize overlap with past stable directions
            if module.r_stable > 0 and hasattr(module, '_stable_accumulator'):
                # Approximate: penalize stable parameters that align with accumulator
                acc = module._stable_accumulator
                stable_delta = module.stable_A @ module.stable_B
                overlap = torch.sum(stable_delta * acc)
                total_loss += self.lambda_stable * (overlap ** 2)
        
        # 3. Merge smoothing: enforce h_stable_old ~= h_stable_new on old-class samples
        if old_val_batch is not None and self.pre_merge_snapshots:
            device = old_val_batch["input_ids"].device
            
            # 3a. Compute h_stable_new (current stable branch)
            with torch.no_grad():
                out_new = model(
                    input_ids=old_val_batch["input_ids"],
                    attention_mask=old_val_batch["attention_mask"],
                    texts=old_val_batch.get("texts"),
                    mode="stable_only",
                    return_rejection=False,
                )
                h_new = out_new["cls_hidden"]
            
            # 3b. Temporarily restore pre-merge stable parameters
            restored = {}
            for name, module in model.named_modules():
                if isinstance(module, DualBranchLoRALayer) and name in self.pre_merge_snapshots:
                    snap = self.pre_merge_snapshots[name]
                    restored[name] = {
                        'A': module.stable_A.data.clone(),
                        'B': module.stable_B.data.clone(),
                    }
                    module.stable_A.data.copy_(snap['A'])
                    module.stable_B.data.copy_(snap['B'])
            
            # 3c. Compute h_stable_old
            with torch.no_grad():
                out_old = model(
                    input_ids=old_val_batch["input_ids"],
                    attention_mask=old_val_batch["attention_mask"],
                    texts=old_val_batch.get("texts"),
                    mode="stable_only",
                    return_rejection=False,
                )
                h_old = out_old["cls_hidden"]
            
            # 3d. Restore current stable parameters
            for name, module in model.named_modules():
                if isinstance(module, DualBranchLoRALayer) and name in restored:
                    module.stable_A.data.copy_(restored[name]['A'])
                    module.stable_B.data.copy_(restored[name]['B'])
            
            # 3e. L2 smoothing loss
            smooth_loss = torch.mean((h_new - h_old) ** 2)
            total_loss += self.lambda_merge * smooth_loss
        
        return total_loss
