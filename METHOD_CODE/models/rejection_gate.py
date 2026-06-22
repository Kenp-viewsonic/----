"""
Variant-Aware Hierarchical Rejection Gate (Section 3.5 of problemDef).

Produces a two-level rejection decision:
  1. Coarse: toxic vs non-toxic (theta_coarse)
  2. Fine: if toxic but low confidence -> "known toxic framework, unknown variant"

u_t = sigmoid(a*(1-max_prob) + b*H + c*d_proto + d*s_surface)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class HierarchicalRejectionGate(nn.Module):
    """
    Args:
        hidden_size: Dimension of [CLS] hidden state
        num_classes: Number of fine-grained toxic classes
        theta_coarse: Coarse rejection threshold
        theta_fine: Fine rejection threshold
        a, b, c, d: Coefficients for uncertainty score (learnable or fixed)
        learnable_weights: If True, a/b/c/d are nn.Parameter
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 5,
        theta_coarse: float = 0.5,
        theta_fine: float = 0.3,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        learnable_weights: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.theta_coarse = theta_coarse
        self.theta_fine = theta_fine
        
        if learnable_weights:
            self.a = nn.Parameter(torch.tensor(a))
            self.b = nn.Parameter(torch.tensor(b))
            self.c = nn.Parameter(torch.tensor(c))
            self.d = nn.Parameter(torch.tensor(d))
        else:
            self.register_buffer('a', torch.tensor(a))
            self.register_buffer('b', torch.tensor(b))
            self.register_buffer('c', torch.tensor(c))
            self.register_buffer('d', torch.tensor(d))
        
        # Learnable bias and temperature for u_t calibration
        self.gate_bias = nn.Parameter(torch.tensor(0.0))
        self.gate_log_temperature = nn.Parameter(torch.tensor(0.0))  # log(1.0) = 0
        
        # Prototype distance module (simple linear projection then distance)
        self.proto_proj = nn.Linear(hidden_size, 128)
        
        # Known toxic vocabulary for surface anomaly (populated externally)
        self.register_buffer('V_known', torch.zeros(0, dtype=torch.long))
        self.vocab_initialized = False
    
    def set_prototypes(self, prototypes: torch.Tensor):
        """
        Set class prototypes (K-means centroids from base stage).
        prototypes: [num_prototypes, hidden_size]
        """
        self.register_buffer('prototypes', prototypes)
    
    def set_known_vocab(self, token_ids: List[int], tokenizer=None):
        """Set known toxic token ids for surface anomaly computation."""
        self.register_buffer('V_known', torch.tensor(token_ids, dtype=torch.long))
        self.tokenizer = tokenizer
        self.vocab_initialized = True
    
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Multi-label entropy: average binary entropy per class.
        probs: [B, C] (after sigmoid)
        Returns: [B]
        """
        eps = 1e-8
        entropy = -(
            probs * torch.log(probs + eps) +
            (1 - probs) * torch.log(1 - probs + eps)
        )
        return entropy.mean(dim=-1)
    
    def compute_proto_distance(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """
        Minimum distance from cls_hidden to any prototype.
        cls_hidden: [B, H]
        Returns: [B]
        """
        if not hasattr(self, 'prototypes') or self.prototypes is None or self.prototypes.numel() == 0:
            return torch.zeros(cls_hidden.size(0), device=cls_hidden.device)
        
        z = self.proto_proj(cls_hidden)  # [B, 128]
        # Ensure prototypes are on the same device as the model
        protos = self.proto_proj(self.prototypes.to(cls_hidden.device))  # [P, 128]
        
        # Compute pairwise distances
        dists = torch.cdist(z, protos)  # [B, P]
        min_dist = dists.min(dim=-1)[0]  # [B]
        return min_dist
    
    def compute_surface_anomaly(self, texts: List[str]) -> torch.Tensor:
        """
        Lightweight surface anomaly score (vectorized, batch-friendly).
        """
        if not texts:
            return torch.zeros(1, dtype=torch.float32)

        batch_size = len(texts)
        # Character entropy & non-alphanumeric ratio via vectorized string ops
        max_len = max(len(t) for t in texts) if texts else 1
        scores = torch.zeros(batch_size, dtype=torch.float32)

        for i, text in enumerate(texts):
            if not text:
                scores[i] = 0.0
                continue
            lower = text.lower()
            # Non-alphanumeric ratio (fast)
            non_alpha = sum(1 for c in lower if not c.isalnum() and not c.isspace())
            oov_ratio = min(non_alpha / max(len(lower) * 0.3, 1.0), 1.0)
            scores[i] = oov_ratio * 0.5  # Simplified: skip char-entropy per-sample

        return scores
    
    def forward(
        self,
        cls_hidden: torch.Tensor,
        logits: torch.Tensor,
        texts: Optional[List[str]] = None,
    ) -> dict:
        """
        Args:
            cls_hidden: [B, H] [CLS] representation
            logits: [B, C] raw logits (before sigmoid)
            texts: Optional raw texts for surface anomaly
        
        Returns:
            dict with:
              - 'probs': [B, C] sigmoid probabilities
              - 'u_t': [B] unknown probability
              - 'decision': List[str] per-sample decision
              - 'max_prob': [B]
              - 'entropy': [B]
        """
        probs = torch.sigmoid(logits)
        max_prob = probs.max(dim=-1)[0]  # [B]
        H = self.compute_entropy(probs)  # [B]
        d_proto = self.compute_proto_distance(cls_hidden)  # [B]
        
        if texts is not None:
            s_surface = self.compute_surface_anomaly(texts).to(cls_hidden.device)
        else:
            s_surface = torch.zeros_like(max_prob)
        
        # Normalize distances to [0, 1] roughly
        d_proto_norm = torch.tanh(d_proto)
        
        # Composite unknown score with learnable bias and temperature
        temperature = torch.exp(self.gate_log_temperature).clamp(min=0.1, max=10.0)
        raw_score = (
            self.a * (1.0 - max_prob) +
            self.b * H +
            self.c * d_proto_norm +
            self.d * s_surface +
            self.gate_bias
        )
        u_t = torch.sigmoid(raw_score / temperature)
        
        # Hierarchical decisions
        decisions = []
        for i in range(u_t.size(0)):
            if u_t[i] > self.theta_coarse:
                decisions.append("unknown")
            elif max_prob[i] < self.theta_fine:
                decisions.append("known_toxic_framework_unknown_variant")
            else:
                decisions.append("predicted")
        
        return {
            "probs": probs,
            "u_t": u_t,
            "decision": decisions,
            "max_prob": max_prob,
            "entropy": H,
            "d_proto": d_proto_norm,
            "s_surface": s_surface,
        }
