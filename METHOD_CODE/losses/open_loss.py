"""
Open-Set Rejection Loss (Section 3.6 of problemDef).

Trains the rejection gate to distinguish known-class samples from pseudo-OOD.
Pseudo-OOD: 70% toxic-specialized (strong perturbations of known toxic words)
             30% general random/non-toxic text.
Mixed into batch at ratio 1:3 (OOD:Known).
"""

import torch
import torch.nn as nn
import random


class OpenSetLoss(nn.Module):
    """
    Binary loss on rejection score u_t:
      - Known samples: target u_t -> 0
      - Pseudo-OOD samples: target u_t -> 1
    """
    
    def __init__(self, ood_ratio_in_batch: float = 0.25):
        super().__init__()
        self.ood_ratio = ood_ratio_in_batch
        self.bce = nn.BCELoss()
    
    def generate_pseudo_ood(self, texts: list, tokenizer, max_length: int = 128, device="cpu"):
        """
        Generate pseudo-OOD samples from known texts.
        
        Strategy (Refined for "Hard Benign Negatives" Defense):
          - Must rely on syntactically valid but non-toxic structure where possible, 
            rather than just random characters, to prevent the gate from learning brittle text-chaos detectors.
          - We sample from a predefined set of benign syntactically-valid templates 
            mixed with surface variations to create "Hard Benign Negatives".
        """
        # Hard Benign Negatives Templates
        hard_benign_templates = [
            "I completely disagree with your point, it makes no sense.",
            "You are wrong about this topic, please read the article again.",
            "Stop spreading misinformation, this is highly inaccurate.",
            "Why would you say such a ridiculous thing? It's absurd.",
            "I demand you remove this post, it is factually incorrect."
        ]
        
        ood_texts = []
        for t in texts:
            if random.random() < 0.5:
                # Type A: Randomly select a hard benign template and optionally add emphasis
                base = random.choice(hard_benign_templates)
                if random.random() < 0.3:
                    base = base.upper()
                if random.random() < 0.3:
                    base += "!!!"
                ood_texts.append(base)
            else:
                # Type B: Mild perturbation to known text (preserve morphology, break toxicity)
                words = t.split()
                if len(words) > 3:
                     # Replace random words with benign stop-words
                     for _ in range(max(1, len(words) // 4)):
                          idx = random.randint(0, len(words)-1)
                          words[idx] = random.choice(["flower", "peace", "water", "apple", "happy"])
                ood_texts.append(" ".join(words))
        
        encoding = tokenizer(
            ood_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].to(device),
            "attention_mask": encoding["attention_mask"].to(device),
            "texts": ood_texts,
            "is_ood": torch.ones(len(ood_texts), device=device),
        }
    
    def forward(self, model, batch, tokenizer, max_length: int = 128):
        """
        Args:
            model: RobertaToxicClassifier
            batch: Known-class batch dict
            tokenizer: tokenizer
            max_length: max length
        
        Returns:
            loss: scalar
        """
        device = batch["input_ids"].device
        texts = batch["texts"]
        
        # Determine number of OOD samples to match target ratio (OOD : Known = ratio : 1-ratio)
        n_known = len(texts)
        n_ood = max(1, int(n_known * self.ood_ratio / (1.0 - self.ood_ratio)))
        
        # Sample subset of known texts for OOD generation
        if n_ood < n_known:
            ood_source_texts = random.sample(texts, n_ood)
        else:
            ood_source_texts = texts
        
        # Generate pseudo-OOD
        ood_batch = self.generate_pseudo_ood(
            ood_source_texts, tokenizer, max_length, device
        )
        
        # Known forward
        known_out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            texts=batch["texts"],
            return_rejection=True,
        )
        known_u = known_out["rejection"]["u_t"]  # [B]
        
        # OOD forward
        ood_out = model(
            input_ids=ood_batch["input_ids"],
            attention_mask=ood_batch["attention_mask"],
            texts=ood_batch["texts"],
            return_rejection=True,
        )
        ood_u = ood_out["rejection"]["u_t"]  # [B_ood]
        
        # Targets
        known_target = torch.zeros_like(known_u)
        ood_target = torch.ones_like(ood_u)
        
        all_u = torch.cat([known_u, ood_u], dim=0)
        all_target = torch.cat([known_target, ood_target], dim=0)
        
        loss = self.bce(all_u, all_target)
        return loss
