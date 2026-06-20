"""
Cross-Stage Orthogonality Loss (Section 3.6 of problemDef).

Lightweight orthogonality on stable branch to prevent interference
between stable semantics from different stages.

L_orth = ||stable^T @ stable_old||_F^2  (only active when stage > 0)
"""

import torch
import torch.nn as nn

from models.dual_lora import DualBranchLoRALayer


class OrthogonalityLoss(nn.Module):
    """
    Penalizes overlap between current stable branch and a snapshot
    of the stable branch from the previous stage.
    """
    
    def __init__(self, eta: float = 1e-4):
        super().__init__()
        self.eta = eta
        self.prev_stable_snapshots = {}
    
    def take_snapshot(self, model):
        """Call at the end of each stage to snapshot stable branches."""
        snap = {}
        for name, module in model.named_modules():
            if isinstance(module, DualBranchLoRALayer):
                snap[name] = {
                    'A': module.stable_A.detach().clone(),
                    'B': module.stable_B.detach().clone(),
                }
        self.prev_stable_snapshots = snap
    
    def forward(self, model):
        """
        Returns orthogonality loss between current stable and previous snapshot.
        If no snapshot exists (stage 0), returns 0.
        """
        if not self.prev_stable_snapshots:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        count = 0
        
        for name, module in model.named_modules():
            if not isinstance(module, DualBranchLoRALayer):
                continue
            if name not in self.prev_stable_snapshots:
                continue
            
            prev = self.prev_stable_snapshots[name]
            # Compute current stable delta
            curr_delta = module.stable_A @ module.stable_B
            prev_delta = prev['A'] @ prev['B']
            
            # Frobenius inner product
            overlap = torch.sum(curr_delta * prev_delta)
            total_loss += overlap ** 2
            count += 1
        
        if count == 0:
            return torch.tensor(0.0)
        
        return self.eta * (total_loss / count)
