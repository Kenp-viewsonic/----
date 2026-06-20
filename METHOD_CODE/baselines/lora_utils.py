"""
Shared LoRA utilities for baselines.

Supports:
  - Single-branch LoRA (standard, used by seq_finetune/ewc_lora)
  - Multi-stage parallel LoRA (used by task_lora, task_lora_msp, task_lora_adb, o_lora)
  - EWC (Elastic Weight Consolidation) regularization
  - Orthogonality constraint for O-LoRA
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleBranchLoRALayer(nn.Module):
    """
    Standard single-branch LoRA adapter wrapping a linear layer.
    Used by seq_finetune (as a single adapter) and ewc_lora.
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
        self.dropout_p = lora_dropout

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.r == 0:
            return result
        h = x @ self.lora_A @ self.lora_B
        if self.dropout_p > 0 and self.training:
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        result = result + h * self.scaling
        return result

    def merge(self):
        """Destructively merge LoRA weights into the base linear layer."""
        if self.r == 0:
            return
        delta = (self.lora_A @ self.lora_B) * self.scaling
        with torch.no_grad():
            # base_layer.weight is [out_f, in_f]; delta is [in_f, out_f]
            self.base_layer.weight += delta.T
        self.r = 0
        self.lora_A = None
        self.lora_B = None


class MultiStageLoRALayer(nn.Module):
    """
    Multi-stage parallel LoRA adapters.
    Each incremental stage adds a new LoRA branch; previous branches are frozen.
    Used by task_lora, task_lora_msp, task_lora_adb, and o_lora.
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
        self.dropout_p = lora_dropout
        self.stages = nn.ModuleList()
        self._add_new_stage()

    def _add_new_stage(self):
        in_f = self.base_layer.in_features
        out_f = self.base_layer.out_features
        stage = nn.Module()
        stage.A = nn.Parameter(torch.zeros(in_f, self.r))
        stage.B = nn.Parameter(torch.zeros(self.r, out_f))
        nn.init.kaiming_uniform_(stage.A, a=math.sqrt(5))
        nn.init.zeros_(stage.B)
        self.stages.append(stage)

    def add_stage(self):
        """Freeze all previous stages and append a new trainable stage."""
        for stage in self.stages:
            for p in stage.parameters():
                p.requires_grad = False
        self._add_new_stage()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
        for stage in self.stages:
            h = x @ stage.A @ stage.B
            # Apply dropout only to the currently active (last) stage during training
            is_active = any(p.requires_grad for p in stage.parameters())
            if self.training and is_active and self.dropout_p > 0:
                h = F.dropout(h, p=self.dropout_p, training=self.training)
            result = result + h * scaling
        return result

    def get_orthogonality_loss(self):
        """
        O-LoRA orthogonality loss: penalize overlap between the current stage
        and all previous stages.
        Returns a scalar tensor.
        """
        if len(self.stages) < 2:
            return torch.tensor(0.0, device=self.base_layer.weight.device)
        current = self.stages[-1]
        loss = 0.0
        for prev in self.stages[:-1]:
            # Penalize inner product of B matrices (output directions)
            overlap = torch.sum((current.B.T @ prev.B) ** 2)
            loss = loss + overlap
        return loss


class EWCManager:
    """
    Elastic Weight Consolidation (EWC) for parameters.
    Computes diagonal Fisher Information and stores optimal parameters
    from the previous stage to regularize parameter drift.
    """

    def __init__(self, model, importance: float = 1000.0, device="cpu"):
        self.model = model
        self.importance = importance
        self.device = device
        self.fisher = {}
        self.optimal_params = {}

    def compute_fisher(self, dataloader, num_batches: int = 100):
        """
        Accumulate squared gradients over a subset of the dataloader.
        """
        self.model.eval()
        self.fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)

        count = 0
        for batch in dataloader:
            if count >= num_batches:
                break
            self.model.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Standard BCE loss for Fisher
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_rejection=False,
            )
            loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in self.fisher:
                        self.fisher[name] += param.grad.data.detach() ** 2
            count += 1

        for name in self.fisher:
            self.fisher[name] /= max(count, 1)

    def store_optimal_params(self):
        """Snapshot current parameters as the optimal old parameters."""
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.detach().clone()

    def penalty(self, model) -> torch.Tensor:
        """Compute EWC penalty: sum(F_i * (theta_i - theta_i^*)^2)."""
        if not self.optimal_params or not self.fisher:
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.optimal_params and name in self.fisher:
                diff = param - self.optimal_params[name].to(param.device)
                loss = loss + torch.sum(self.fisher[name].to(param.device) * (diff ** 2))
        return self.importance * loss


def apply_lora_to_roberta(
    model,
    target_modules: list = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    multi_stage: bool = False,
):
    """
    Replace specified linear layers in a RobertaModel with LoRA variants.

    Args:
        model: transformers RobertaModel instance
        target_modules: List of attention sub-module names, e.g. ['query', 'value']
        r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout probability on LoRA path
        multi_stage: If True, use MultiStageLoRALayer; else SingleBranchLoRALayer

    Returns:
        model (modified in-place)
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    LayerCls = MultiStageLoRALayer if multi_stage else SingleBranchLoRALayer
    adapted_count = 0

    encoder = model.encoder
    for layer_idx, layer in enumerate(encoder.layer):
        attn = layer.attention.self
        for part_name in target_modules:
            part = getattr(attn, part_name, None)
            if isinstance(part, nn.Linear):
                lora_layer = LayerCls(
                    base_layer=part,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(attn, part_name, lora_layer)
                adapted_count += 1

    print(f"[LoRA] Adapted {adapted_count} layers with {LayerCls.__name__}: {target_modules}")
    return model


def get_lora_state_dict(model, include_base: bool = False):
    """
    Extract LoRA-specific state dict for checkpointing.
    Useful for saving/loading per-stage LoRA weights without duplicating base RoBERTa.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (SingleBranchLoRALayer, MultiStageLoRALayer)):
            prefix = name + "."
            for k, v in module.state_dict().items():
                if not include_base and k.startswith("base_layer."):
                    continue
                state[prefix + k] = v
    return state