"""
Baseline LoRA variants for FSCIL ablation and comparison.

Includes:
  - SingleBranchLoRA: standard LoRA (one low-rank pair per layer)
  - TaskLoRA: per-stage independent LoRA with optional freezing of old stages
  - OrthLoRA: O-LoRA (orthogonal subspace constraint across stages)
  - EWCLoRAWrapper: wraps any LoRA model with EWC Fisher regularization
"""

import copy
import torch
import torch.nn as nn


class SingleBranchLoRALayer(nn.Module):
    """
    Standard single-branch LoRA (Hu et al. 2022).

    W_eff = W_base + alpha/r * (A @ B)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self.r == 0:
            return base
        h = self.dropout(x @ self.lora_A) @ self.lora_B
        return base + h * self.scaling


class TaskLoRALayer(nn.Module):
    """
    Task-specific LoRA for incremental learning.
    Each stage gets a new LoRA branch; old branches are frozen.

    Inference: W_eff = W_base + sum_{t<=k} frozen_t(A_t @ B_t) + active(A_k @ B_k)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        # Active branch for current stage
        self.active_A = nn.Parameter(torch.zeros(in_f, r))
        self.active_B = nn.Parameter(torch.zeros(r, out_f))
        self.reset_active()

        # Frozen historical branches
        self.frozen_branches = nn.ModuleList()

    def reset_active(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.active_A, a=5 ** 0.5)
            nn.init.zeros_(self.active_B)

    def freeze_active(self):
        """Freeze current active branch and start a new one."""
        if self.r == 0:
            return
        frozen = nn.Module()
        frozen.register_buffer("A", self.active_A.detach().clone())
        frozen.register_buffer("B", self.active_B.detach().clone())
        self.frozen_branches.append(frozen)
        self.reset_active()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self.r == 0:
            return base

        # Active branch
        h = self.dropout(x @ self.active_A) @ self.active_B
        out = base + h * self.scaling

        # Frozen branches
        for fb in self.frozen_branches:
            h = x @ fb.A @ fb.B
            out = out + h * self.scaling

        return out


class OrthLoRALayer(nn.Module):
    """
    Orthogonal LoRA (O-LoRA, from 2601.02232).

    Each stage adds a new LoRA branch constrained to be orthogonal
    to the subspace spanned by all previous branches.

    During training, an orthogonality loss penalizes:
        ||A_new^T @ A_past||_F^2 + ||B_new @ B_past^T||_F^2
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        self.active_A = nn.Parameter(torch.zeros(in_f, r))
        self.active_B = nn.Parameter(torch.zeros(r, out_f))
        self.reset_active()

        # Historical branches (frozen)
        self.frozen_branches = nn.ModuleList()

    def reset_active(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.active_A, a=5 ** 0.5)
            nn.init.zeros_(self.active_B)

    def freeze_active(self):
        """Finalize current branch and add to history."""
        if self.r == 0:
            return
        frozen = nn.Module()
        frozen.register_buffer("A", self.active_A.detach().clone())
        frozen.register_buffer("B", self.active_B.detach().clone())
        self.frozen_branches.append(frozen)
        self.reset_active()

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Compute orthogonality penalty between active and frozen branches.
        Returns 0 if no frozen branches exist.
        """
        if len(self.frozen_branches) == 0 or self.r == 0:
            return torch.tensor(0.0, device=self.active_A.device)

        loss = 0.0
        for fb in self.frozen_branches:
            # Penalize overlap in input subspace (A) and output subspace (B)
            overlap_A = torch.norm(self.active_A.t() @ fb.A, p="fro") ** 2
            overlap_B = torch.norm(self.active_B @ fb.B.t(), p="fro") ** 2
            loss += overlap_A + overlap_B

        return loss / len(self.frozen_branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        if self.r == 0:
            return base

        h = self.dropout(x @ self.active_A) @ self.active_B
        out = base + h * self.scaling

        for fb in self.frozen_branches:
            h = x @ fb.A @ fb.B
            out = out + h * self.scaling

        return out


class EWCLoRAWrapper:
    """
    EWC (Elastic Weight Consolidation) wrapper for any LoRA model.

    Usage:
        ewc = EWCLoRAWrapper(model, importance_weight=1e4)
        # After base stage training:
        ewc.compute_fisher(dataloader, device)
        # During incremental training:
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(self, model: nn.Module, importance_weight: float = 1e4):
        self.model = model
        self.importance_weight = importance_weight
        self.fisher = {}  # param_name -> Tensor
        self.optimal_params = {}  # param_name -> Tensor

    def compute_fisher(self, dataloader, device, num_batches: int = 100):
        """
        Approximate Fisher Information Matrix diagonal on base-stage data.
        """
        self.model.eval()
        self.fisher = {}
        self.optimal_params = {}

        # Register optimal params (current values)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.optimal_params[n] = p.detach().clone()
                self.fisher[n] = torch.zeros_like(p)

        count = 0
        for batch in dataloader:
            if count >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            self.model.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_rejection=False,
            )
            loss = outputs.get("loss", torch.tensor(0.0, device=device))
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.fisher[n] += (p.grad.detach() ** 2)

            count += 1

        # Average
        for n in self.fisher:
            self.fisher[n] /= max(count, 1)

        print(f"[EWC] Computed Fisher diagonal for {len(self.fisher)} parameters over {count} batches.")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        EWC penalty: sum_i F_i * (theta_i - theta*_i)^2
        """
        if not self.fisher:
            return torch.tensor(0.0)

        loss = 0.0
        device = next(model.parameters()).device
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.optimal_params:
                _opt = self.optimal_params[n].to(device)
                _fish = self.fisher[n].to(device)
                loss += (_fish * (p - _opt) ** 2).sum()

        return self.importance_weight * loss


def apply_lora_to_roberta(
    model,
    lora_class,
    target_modules: list = ["query", "value"],
    **lora_kwargs
):
    """
    Replace specified linear layers in RobertaModel with a LoRA layer.

    Args:
        model: transformers RobertaModel
        lora_class: e.g. SingleBranchLoRALayer, TaskLoRALayer, OrthLoRALayer
        target_modules: list of attention sub-module names
        **lora_kwargs: passed to lora_class constructor
    """
    adapted_count = 0
    for layer in model.encoder.layer:
        attn = layer.attention.self
        for name in target_modules:
            mod = getattr(attn, name, None)
            if isinstance(mod, nn.Linear):
                lora_layer = lora_class(base_layer=mod, **lora_kwargs)
                setattr(attn, name, lora_layer)
                adapted_count += 1
    print(f"[{lora_class.__name__}] Adapted {adapted_count} layers: {target_modules}")
    return model
