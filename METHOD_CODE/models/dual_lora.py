"""
Semantic-Stability-Aware Dual-Branch LoRA (Section 3.3 of problemDef).

Decomposes each adapter update into:
  - Stable branch (cross-stage accumulated, strong regularization)
  - Plastic branch (per-stage, sparse, independent)

At inference stage k:
  W_eff = W_base + stable_A@stable_B + sum(frozen_plastic) + active_plastic_A@active_plastic_B
"""

import torch
import torch.nn as nn
import copy


class DualBranchLoRALayer(nn.Module):
    """
    Single linear layer with dual-branch LoRA injection.
    
    Args:
        base_layer: The original nn.Linear to wrap
        r_stable: Rank of stable branch
        r_plastic: Rank of plastic branch
        lora_alpha: Scaling factor (usually 16)
        lora_dropout: Dropout on LoRA path
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r_stable: int = 8,
        r_plastic: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r_stable = r_stable
        self.r_plastic = r_plastic
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r_stable if r_stable > 0 else 1.0
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Stable branch (cross-stage)
        self.stable_A = nn.Parameter(torch.zeros(in_features, r_stable))
        self.stable_B = nn.Parameter(torch.zeros(r_stable, out_features))
        
        # Current active plastic branch
        self.plastic_A = nn.Parameter(torch.zeros(in_features, r_plastic))
        self.plastic_B = nn.Parameter(torch.zeros(r_plastic, out_features))
        
        # Frozen historical plastic branches (list of ParameterDict or raw tensors)
        self.frozen_plastics = nn.ModuleList()
        
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        self.reset_stable_parameters()
        self.reset_plastic_parameters()
    
    def reset_stable_parameters(self):
        """Xavier init for stable branch."""
        if self.r_stable > 0:
            nn.init.xavier_uniform_(self.stable_A)
            nn.init.zeros_(self.stable_B)
    
    def reset_plastic_parameters(self):
        """Xavier init for plastic branch."""
        if self.r_plastic > 0:
            nn.init.xavier_uniform_(self.plastic_A)
            nn.init.zeros_(self.plastic_B)
    
    def freeze_stable(self):
        self.stable_A.requires_grad = False
        self.stable_B.requires_grad = False
    
    def unfreeze_stable(self):
        self.stable_A.requires_grad = True
        self.stable_B.requires_grad = True
    
    def freeze_plastic(self):
        self.plastic_A.requires_grad = False
        self.plastic_B.requires_grad = False
    
    def unfreeze_plastic(self):
        self.plastic_A.requires_grad = True
        self.plastic_B.requires_grad = True
    
    def merge_plastic_to_stable(self):
        """
        Semantic consolidation: merge current plastic branch into stable branch
        while MAINTAINING low-rank structure via SVD compression.
        """
        if self.r_plastic == 0:
            return
        
        device = self.stable_A.device
        
        # Current stable component
        stable_delta = self.stable_A @ self.stable_B  # [in, out]
        
        # Plastic component
        plastic_delta = self.plastic_A @ self.plastic_B  # [in, out]
        
        # Combined matrix (sum of two low-rank components)
        combined = stable_delta + plastic_delta  # [in, out]
        
        # SVD to re-factorize into rank-r_stable
        if combined.abs().sum() > 0:
            U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
            r = min(self.r_stable, len(S))
            Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
            sqrt_S = torch.sqrt(Sr + 1e-10)
            new_A = (Ur * sqrt_S.unsqueeze(0)).contiguous().to(device)
            new_B = (sqrt_S.unsqueeze(1) * Vhr).contiguous().to(device)
            
            with torch.no_grad():
                # In-place update to preserve optimizer references
                self.stable_A.set_(new_A)
                self.stable_B.set_(new_B)
        
        # Clear old accumulator if it exists
        if hasattr(self, '_stable_accumulator'):
            acc = self._stable_accumulator.to(device)
            if acc.abs().sum() > 0:
                combined_full = self.stable_A @ self.stable_B + acc
                U, S, Vh = torch.linalg.svd(combined_full, full_matrices=False)
                r = min(self.r_stable, len(S))
                Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
                sqrt_S = torch.sqrt(Sr + 1e-10)
                with torch.no_grad():
                    self.stable_A.set_((Ur * sqrt_S.unsqueeze(0)).contiguous().to(device))
                    self.stable_B.set_((sqrt_S.unsqueeze(1) * Vhr).contiguous().to(device))
            del self._stable_accumulator
        
        self.reset_plastic_parameters()
        print("[DualLoRA] Plastic SVD-merged to stable (rank preserved).")
    
    def freeze_current_plastic(self):
        """
        Freeze current plastic branch as historical patch (not merged to stable).
        Create a new plastic branch for next stage.
        """
        if self.r_plastic == 0:
            return
        
        # Store frozen plastic as a non-trainable module
        frozen = nn.Module()
        frozen.register_buffer('A', self.plastic_A.detach().clone())
        frozen.register_buffer('B', self.plastic_B.detach().clone())
        self.frozen_plastics.append(frozen)
        
        # Re-initialize new plastic branch
        self.reset_plastic_parameters()
        print(f"[DualLoRA] Plastic frozen as historical patch #{len(self.frozen_plastics)}.")
    
    def forward(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        """
        Args:
            x: Input tensor
            mode: One of ['full', 'stable_only', 'base_only']
        """
        # Base output
        base_out = self.base_layer(x)
        
        if mode == "base_only":
            return base_out
        
        result = base_out
        
        # Stable branch
        if self.r_stable > 0:
            h = self.dropout(x @ self.stable_A) @ self.stable_B
            result = result + h * self.scaling
        
        if mode == "stable_only":
            return result
        
        # Accumulated stable (from past plastic merges)
        if hasattr(self, '_stable_accumulator'):
            result = result + x @ self._stable_accumulator
        
        # Active plastic branch
        if self.r_plastic > 0 and self.plastic_A.requires_grad or not self.training:
            h = self.dropout(x @ self.plastic_A) @ self.plastic_B
            result = result + h * (self.lora_alpha / self.r_plastic if self.r_plastic > 0 else 1.0)
        
        # Frozen historical plastics
        for frozen in self.frozen_plastics:
            h = x @ frozen.A @ frozen.B
            result = result + h * (self.lora_alpha / self.r_plastic if self.r_plastic > 0 else 1.0)
        
        return result


def apply_dual_lora_to_roberta(
    model,
    target_modules: list = ["query", "value"],
    r_stable: int = 8,
    r_plastic: int = 4,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """
    Replace specified linear layers in a RobertaModel with DualBranchLoRALayer.
    
    Args:
        model: transformers RobertaModel instance
        target_modules: List of attention sub-module names to adapt (e.g. ['query', 'value'])
    """
    adapted_count = 0
    
    encoder = model.encoder
    for layer_idx, layer in enumerate(encoder.layer):
        attn = layer.attention.self
        for attn_part_name in target_modules:
            attn_part = getattr(attn, attn_part_name, None)
            if isinstance(attn_part, nn.Linear):
                lora_layer = DualBranchLoRALayer(
                    base_layer=attn_part,
                    r_stable=r_stable,
                    r_plastic=r_plastic,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(attn, attn_part_name, lora_layer)
                adapted_count += 1
    
    print(f"[DualLoRA] Adapted {adapted_count} layers: {target_modules}")
    return model
