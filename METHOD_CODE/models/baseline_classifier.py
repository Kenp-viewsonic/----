"""
Simplified baseline classifier for FSCIL comparison.

RoBERTa-base + optional LoRA + linear classifier.
No prefix, no ToxicAwarePE, no rejection gate — to ensure fair comparison
with our full method.

Supports multi-label classification with BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig


class BaselineToxicClassifier(nn.Module):
    """
    Args:
        num_classes: number of fine-grained toxic classes
        model_name: HuggingFace model name
        use_lora: if True, expects LoRA layers to be injected externally
                  via apply_lora_to_roberta()
    """

    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        self.current_stage = 0

    def expand_classifier(self, new_num_classes: int):
        """Expand classification head for new classes."""
        old_head = self.classifier
        old_num_classes = old_head.out_features
        if new_num_classes <= old_num_classes:
            return

        new_head = nn.Linear(self.hidden_size, new_num_classes)
        with torch.no_grad():
            new_head.weight[:old_num_classes, :] = old_head.weight
            new_head.bias[:old_num_classes] = old_head.bias
        self.classifier = new_head
        self.num_classes = new_num_classes
        print(f"[BaselineClassifier] Expanded from {old_num_classes} to {new_num_classes} classes.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_rejection=False,
        **kwargs
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [B, H]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)  # [B, C]

        result = {
            "logits": logits,
            "cls_hidden": cls_hidden,
        }

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            mask = (labels >= 0).float()
            active_logits = logits * mask
            active_labels = labels * mask
            loss = loss_fct(active_logits, active_labels)
            result["loss"] = loss

        if return_rejection:
            # Minimal rejection info for metric compatibility
            probs = torch.sigmoid(logits)
            max_prob = probs.max(dim=-1)[0]
            result["rejection"] = {
                "probs": probs,
                "u_t": 1.0 - max_prob,  # uncertainty = 1 - max_prob (MSP-style)
                "decision": ["predicted"] * probs.size(0),
                "max_prob": max_prob,
                "entropy": torch.zeros_like(max_prob),
                "d_proto": torch.zeros_like(max_prob),
                "s_surface": torch.zeros_like(max_prob),
            }

        return result
