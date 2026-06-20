"""
Semantic Evolution Consistency Loss (Section 3.6 of problemDef).

Requires character-perturbed positive sample pairs to have close [CLS] representations.
L_evo = sum ||h_cls(x) - h_cls(x')||^2
"""

import torch
import torch.nn as nn
import random


class EvoLoss(nn.Module):
    """
    Computes consistency between original and perturbed toxic samples.
    
    Perturbations applied during training:
      - Leet substitution (probabilistic)
      - Random space insertion
      - Random char duplication
    """
    
    LEET_MAP = {
        'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7', 'g': '9'
    }
    
    def __init__(self, perturb_prob: float = 0.3):
        super().__init__()
        self.perturb_prob = perturb_prob
    
    def perturb_text(self, text: str) -> str:
        """Apply random character perturbations (3 strategies, randomly chosen)."""
        chars = list(text)
        r = random.random()
        if r < 0.33:
            # Strategy 1: Leet substitution on random chars
            for i, c in enumerate(chars):
                if c.lower() in self.LEET_MAP and random.random() < self.perturb_prob:
                    chars[i] = self.LEET_MAP[c.lower()]
        elif r < 0.66:
            # Strategy 2: Random space insertion
            if len(chars) > 3:
                pos = random.randint(1, len(chars) - 2)
                chars.insert(pos, ' ')
        else:
            # Strategy 3: Random char duplication
            if len(chars) > 1:
                pos = random.randint(0, len(chars) - 1)
                chars.insert(pos, chars[pos])  # duplicate a random char
        
        return "".join(chars)
    
    def forward(self, model, batch, tokenizer, max_length: int = 128):
        """
        Args:
            model: RobertaToxicClassifier
            batch: dict with 'texts', 'input_ids', 'attention_mask'
            tokenizer: tokenizer for encoding perturbed texts
            max_length: max sequence length
        
        Returns:
            loss: scalar
        """
        texts = batch["texts"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Get original [CLS]
        with torch.no_grad():
            orig_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts,
                return_rejection=False,
            )
        orig_cls = orig_out["cls_hidden"].detach()
        
        # Generate perturbed texts
        perturbed_texts = [self.perturb_text(t) for t in texts]
        
        # Tokenize perturbed
        encoding = tokenizer(
            perturbed_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        p_input_ids = encoding["input_ids"].to(input_ids.device)
        p_attention_mask = encoding["attention_mask"].to(attention_mask.device)
        
        # Get perturbed [CLS]
        pert_out = model(
            input_ids=p_input_ids,
            attention_mask=p_attention_mask,
            texts=perturbed_texts,
            return_rejection=False,
        )
        pert_cls = pert_out["cls_hidden"]
        
        # L2 distance (MSE)
        loss = torch.mean((orig_cls - pert_cls) ** 2)
        return loss
