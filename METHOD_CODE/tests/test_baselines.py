"""
Smoke tests for baseline components.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import RobertaModel
from baselines.lora_utils import (
    SingleBranchLoRALayer,
    MultiStageLoRALayer,
    EWCManager,
    apply_lora_to_roberta,
)


def test_single_branch_lora():
    base = torch.nn.Linear(768, 768)
    lora = SingleBranchLoRALayer(base, r=8, lora_alpha=16)
    x = torch.randn(2, 10, 768)
    out = lora(x)
    assert out.shape == (2, 10, 768)
    print("[PASS] SingleBranchLoRALayer forward shape OK")


def test_multi_stage_lora():
    base = torch.nn.Linear(768, 768)
    lora = MultiStageLoRALayer(base, r=8, lora_alpha=16)
    x = torch.randn(2, 10, 768)
    out0 = lora(x)
    assert out0.shape == (2, 10, 768)

    # Add stage
    lora.add_stage()
    out1 = lora(x)
    assert out1.shape == (2, 10, 768)

    # Orthogonality loss
    orth = lora.get_orthogonality_loss()
    assert isinstance(orth, torch.Tensor)
    print("[PASS] MultiStageLoRALayer forward + add_stage + orth_loss OK")


def test_apply_lora_to_roberta():
    model = RobertaModel.from_pretrained("roberta-base")
    apply_lora_to_roberta(model, target_modules=["query", "value"], r=4, multi_stage=False)
    x = torch.randint(0, 100, (1, 10))
    out = model(x)
    assert out.last_hidden_state.shape == (1, 10, 768)
    print("[PASS] apply_lora_to_roberta (single) OK")

    model2 = RobertaModel.from_pretrained("roberta-base")
    apply_lora_to_roberta(model2, target_modules=["query", "value"], r=4, multi_stage=True)
    out2 = model2(x)
    assert out2.last_hidden_state.shape == (1, 10, 768)
    print("[PASS] apply_lora_to_roberta (multi) OK")


def test_ewc_manager():
    model = RobertaModel.from_pretrained("roberta-base")
    apply_lora_to_roberta(model, target_modules=["query"], r=4, multi_stage=False)
    ewc = EWCManager(model, importance=100.0, device="cpu")

    # EWC needs a model that outputs loss; RobertaModel doesn't.
    # We'll just test the API by monkey-patching forward.
    original_forward = model.forward
    def mock_forward(input_ids, attention_mask=None, labels=None, return_rejection=False, **kwargs):
        # Ensure input_ids is 2D [B, L]
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(1)
        out = original_forward(input_ids, attention_mask=attention_mask)
        result = {"logits": out.last_hidden_state[:, 0, :], "cls_hidden": out.last_hidden_state[:, 0, :]}
        if labels is not None:
            result["loss"] = torch.tensor(1.0, requires_grad=True)
        return result

    model.forward = mock_forward

    # Mock dataloader with proper 2D tensors
    class MockDataset:
        def __init__(self, n=4):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 100, (10,)),
                "attention_mask": torch.ones(10),
                "labels": torch.ones(5),
            }

    loader = torch.utils.data.DataLoader(MockDataset(), batch_size=2)
    ewc.compute_fisher(loader, num_batches=2)
    ewc.store_optimal_params()
    penalty = ewc.penalty(model)
    assert isinstance(penalty, torch.Tensor)
    print("[PASS] EWCManager compute_fisher + penalty OK")


if __name__ == "__main__":
    test_single_branch_lora()
    test_multi_stage_lora()
    test_apply_lora_to_roberta()
    test_ewc_manager()
    print("\nAll baseline smoke tests passed!")