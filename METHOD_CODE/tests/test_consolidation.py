"""
Tests for semantic consolidation logic in DualBranchLoRA and RobertaToxicClassifier.

Covers:
  - merge_plastic_to_stable: accumulator update + plastic reset
  - freeze_current_plastic: frozen branch creation + plastic reset
  - consolidate_plastic(merge=True/False): recursive application to model
  - get_stable_cls_embedding: forward mode propagation (stable_only vs full)
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dual_lora import DualBranchLoRALayer, apply_dual_lora_to_roberta
from models.roberta_classifier import RobertaToxicClassifier


def test_dual_lora_merge():
    """Merge plastic into stable accumulator."""
    base = nn.Linear(768, 768)
    layer = DualBranchLoRALayer(base, r_stable=8, r_plastic=4, lora_alpha=16)
    
    # Give plastic a deterministic non-zero value
    with torch.no_grad():
        layer.plastic_A.fill_(0.1)
        layer.plastic_B.fill_(0.2)
    
    assert not hasattr(layer, '_stable_accumulator')
    layer.merge_plastic_to_stable()
    
    # Accumulator should now exist and be non-zero
    assert hasattr(layer, '_stable_accumulator')
    assert layer._stable_accumulator.abs().sum().item() > 0
    
    # Plastic should be reset to zero (B initialized to zeros)
    assert layer.plastic_B.abs().sum().item() == 0.0
    print("[PASS] merge_plastic_to_stable")


def test_dual_lora_freeze():
    """Freeze plastic as historical patch."""
    base = nn.Linear(768, 768)
    layer = DualBranchLoRALayer(base, r_stable=8, r_plastic=4, lora_alpha=16)
    
    with torch.no_grad():
        layer.plastic_A.fill_(0.1)
        layer.plastic_B.fill_(0.2)
    
    assert len(layer.frozen_plastics) == 0
    layer.freeze_current_plastic()
    
    assert len(layer.frozen_plastics) == 1
    frozen = layer.frozen_plastics[0]
    assert frozen.A.abs().sum().item() > 0
    assert frozen.B.abs().sum().item() > 0
    
    # Plastic should be reset
    assert layer.plastic_B.abs().sum().item() == 0.0
    print("[PASS] freeze_current_plastic")


def test_classifier_consolidate_merge():
    """consolidate_plastic(merge=True) touches every DualBranchLoRALayer."""
    model = RobertaToxicClassifier(num_classes=2, model_name="roberta-base")
    apply_dual_lora_to_roberta(model.roberta, target_modules=["query", "value"],
                               r_stable=4, r_plastic=2)
    
    # Seed plastic with non-zero values
    for module in model.modules():
        if isinstance(module, DualBranchLoRALayer):
            with torch.no_grad():
                module.plastic_A.fill_(0.05)
                module.plastic_B.fill_(0.05)
    
    model.consolidate_plastic(merge=True)
    
    dual_count = 0
    for module in model.modules():
        if isinstance(module, DualBranchLoRALayer):
            dual_count += 1
            assert hasattr(module, '_stable_accumulator')
            assert module._stable_accumulator.abs().sum().item() > 0
    
    assert dual_count > 0, "No DualBranchLoRALayer found in model"
    print(f"[PASS] consolidate_plastic(merge=True) over {dual_count} layers")


def test_classifier_consolidate_freeze():
    """consolidate_plastic(merge=False) freezes plastic on every layer."""
    model = RobertaToxicClassifier(num_classes=2, model_name="roberta-base")
    apply_dual_lora_to_roberta(model.roberta, target_modules=["query", "value"],
                               r_stable=4, r_plastic=2)
    
    for module in model.modules():
        if isinstance(module, DualBranchLoRALayer):
            with torch.no_grad():
                module.plastic_A.fill_(0.05)
                module.plastic_B.fill_(0.05)
    
    model.consolidate_plastic(merge=False)
    
    dual_count = 0
    for module in model.modules():
        if isinstance(module, DualBranchLoRALayer):
            dual_count += 1
            assert len(module.frozen_plastics) == 1
    
    assert dual_count > 0
    print(f"[PASS] consolidate_plastic(merge=False) over {dual_count} layers")


def test_stable_cls_embedding_excludes_plastic():
    """stable_only mode must exclude active and frozen plastic contributions."""
    model = RobertaToxicClassifier(num_classes=2, model_name="roberta-base")
    apply_dual_lora_to_roberta(model.roberta, target_modules=["query", "value"],
                               r_stable=4, r_plastic=2)
    
    # Seed plastic with large values so difference is obvious
    for module in model.modules():
        if isinstance(module, DualBranchLoRALayer):
            with torch.no_grad():
                module.plastic_A.fill_(0.5)
                module.plastic_B.fill_(0.5)
    
    dummy_input = torch.randint(0, 100, (2, 16))
    dummy_mask = torch.ones(2, 16, dtype=torch.long)
    
    with torch.no_grad():
        h_full = model(dummy_input, dummy_mask, mode="full", return_rejection=False)["cls_hidden"]
        h_stable = model.get_stable_cls_embedding(dummy_input, dummy_mask)
    
    diff = (h_full - h_stable).abs().mean().item()
    assert diff > 1e-3, f"stable_only output too close to full output (diff={diff})"
    print(f"[PASS] stable_only mode excludes plastic (mean diff={diff:.4f})")


if __name__ == "__main__":
    test_dual_lora_merge()
    test_dual_lora_freeze()
    test_classifier_consolidate_merge()
    test_classifier_consolidate_freeze()
    test_stable_cls_embedding_excludes_plastic()
    print("\nAll consolidation tests passed.")
