"""
Smoke test: verify all modules can be imported and basic shapes are correct.
Run: python tests/test_imports.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

print("Testing imports...")

from data.toxic_dataset import ToxicCommentDataset
from data.fscil_split import FSCILSplitProtocol
from data.variant_generator import VariantGenerator

from models.toxic_prefix import ToxicSemanticPrefix
from models.dual_lora import DualBranchLoRALayer, apply_dual_lora_to_roberta
from models.toxic_pe import ToxicAwarePE
from models.rejection_gate import HierarchicalRejectionGate
from models.roberta_classifier import RobertaToxicClassifier

from losses.evo_loss import EvoLoss
from losses.stable_plastic_reg import StablePlasticRegLoss
from losses.open_loss import OpenSetLoss
from losses.orth_loss import OrthogonalityLoss

from trainers.incremental_trainer import IncrementalTrainer, IncrementalLearningCallback
from utils.metrics import compute_fscil_metrics, compute_cka, compute_variant_recall

print("All imports OK.")

# Quick shape test for prefix
prefix = ToxicSemanticPrefix(hidden_size=768, num_layers=12, prefix_length=10)
P = prefix.get_prefix(stage_idx=0)
assert P.shape == (12, 10, 768), f"Prefix shape mismatch: {P.shape}"
print("Prefix shape OK.")

# Quick test for ToxicAwarePE
pe = ToxicAwarePE(hidden_size=768, max_length=128)
additive = pe(["Hello world!", "Test text."], base_embeddings=torch.randn(2, 128, 768))
assert additive.shape == (2, 128, 768), f"PE shape mismatch: {additive.shape}"
print("ToxicAwarePE shape OK.")

# Quick test for rejection gate
gate = HierarchicalRejectionGate(hidden_size=768, num_classes=5)
cls_h = torch.randn(2, 768)
logits = torch.randn(2, 5)
out = gate(cls_h, logits, texts=["hello", "world"])
assert "u_t" in out and out["u_t"].shape == (2,), f"Gate output mismatch"
print("Rejection gate OK.")

print("\nAll smoke tests passed.")
