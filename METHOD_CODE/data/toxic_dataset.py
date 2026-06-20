"""
Toxic Comment Dataset for FSCIL.
Loads Jigsaw CSV and supports multi-label fine-grained classification.
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class ToxicCommentDataset(Dataset):
    """
    Multi-label toxic comment dataset.
    
    Labels correspond to fine-grained toxic sub-classes:
    [obscene, insult, threat, identity_hate, severe_toxic]
    Note: 'toxic' is used as a filter (parent label), not a model target.
    """
    LABEL_NAMES = ["obscene", "insult", "threat", "identity_hate", "severe_toxic"]
    
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "roberta-base",
        max_length: int = 128,
        filter_toxic: bool = True,
        label_indices: list = None,
        transform_text=None,
    ):
        """
        Args:
            csv_path: Path to train.csv
            tokenizer_name: HuggingFace tokenizer name
            max_length: Max token length
            filter_toxic: If True, only keep samples where toxic==1
            label_indices: Subset of labels to use (list of ints). If None, use all 5.
            transform_text: Optional text augmentation/transform function
        """
        super().__init__()
        self.csv_path = csv_path
        self.max_length = max_length
        self.filter_toxic = filter_toxic
        self.label_indices = label_indices or list(range(len(self.LABEL_NAMES)))
        self.transform_text = transform_text
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._load_data()
    
    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        
        # Ensure required columns exist
        required = ["comment_text", "toxic"] + self.LABEL_NAMES
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Filter to toxic samples only if specified
        if self.filter_toxic:
            df = df[df["toxic"] == 1].reset_index(drop=True)
        
        self.texts = df["comment_text"].fillna("").astype(str).tolist()
        
        # Extract labels (all 5 sub-classes)
        all_labels = df[self.LABEL_NAMES].values.astype(float)
        self.labels = all_labels[:, self.label_indices] if self.label_indices else all_labels
        
        # Also keep original indices for split tracking
        self.indices = df.index.tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.transform_text:
            text = self.transform_text(text)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Zero-leakage Masking: 
        # If the stage is restricted, any label NOT in the allowed_classes for this stage must be forced to 0
        raw_labels = self.labels[idx].copy()
        if hasattr(self, 'allowed_class_indices') and self.allowed_class_indices is not None:
             mask = np.zeros_like(raw_labels)
             mask[self.allowed_class_indices] = 1.0
             raw_labels = raw_labels * mask
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(raw_labels, dtype=torch.float32),
            "text": text,
            "index": self.indices[idx],
        }
        return item
    
    def set_allowed_classes(self, allowed_class_names):
        """Restrict labels visible during this stage to prevent multi-label future leakage."""
        self.allowed_class_indices = [self.label_indices[n] for n in allowed_class_names if n in self.label_indices]
    
    def get_label_distribution(self):
        """Return per-label positive counts."""
        import numpy as np
        pos_counts = self.labels.sum(axis=0)
        return {
            self.LABEL_NAMES[i]: int(pos_counts[j])
            for j, i in enumerate(self.label_indices)
        }
    
    @classmethod
    def collate_fn(cls, batch):
        """Custom collate for DataLoader."""
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        texts = [b.get("text", "") for b in batch]
        indices = [b.get("index", -1) for b in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "texts": texts,
            "indices": indices,
        }
