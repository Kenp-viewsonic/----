"""
FSCIL Split Protocol for Toxic Comment Classification.

Splits data into base stage + K incremental stages.
Each stage provides N-shot samples per new class.
"""

import os
import json
import pickle
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import Subset

from .toxic_dataset import ToxicCommentDataset


class FSCILSplitProtocol:
    """
    Protocol:
      - Stage 0 (base): {obscene, insult}
      - Stage 1: {threat, identity_hate}
      - Stage 2: {severe_toxic}
      
    All samples are drawn from toxic=1 subset.
    Per class, we sample N-shot for training and reserve the rest for testing.
    """
    
    STAGE_DEFINITIONS = {
        0: {"classes": ["obscene", "insult"], "shots": 32},
        1: {"classes": ["threat", "identity_hate"], "shots": 16},
        2: {"classes": ["severe_toxic"], "shots": 16},
    }
    
    ALL_CLASSES = ["obscene", "insult", "threat", "identity_hate", "severe_toxic"]
    
    def __init__(
        self,
        dataset: ToxicCommentDataset,
        output_dir: str = "./data_splits",
        seed: int = 42,
        min_positive_per_class: int = 10,
        coreset_size_per_class: int = 10,
        new_class_negative_ratio: float = 0.0,
        stage_definitions: dict = None,
    ):
        self.dataset = dataset
        self.output_dir = output_dir
        self.seed = seed
        self.min_positive = min_positive_per_class
        self.coreset_size_per_class = coreset_size_per_class
        self.new_class_negative_ratio = new_class_negative_ratio
        
        # Allow external override (e.g. from YAML config)
        if stage_definitions is not None:
            self.STAGE_DEFINITIONS = stage_definitions
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Build class -> sample indices mapping
        self.class_to_indices = self._build_class_indices()
        
        # Store per-class fixed train/test split to ensure consistency across stages
        self.class_train_test = {}
        
        # Stage splits storage
        self.splits = {}
    
    def _build_class_indices(self) -> Dict[str, List[int]]:
        """Map each fine-grained class to dataset indices where label==1."""
        mapping = {cls: [] for cls in self.ALL_CLASSES}
        label_names = self.dataset.LABEL_NAMES
        
        for idx in range(len(self.dataset)):
            labels = self.dataset.labels[idx]  # shape: (num_active_labels,)
            for j, cls in enumerate(label_names):
                if j < len(labels) and labels[j] == 1.0:
                    mapping[cls].append(idx)
        
        # Report stats
        for cls, indices in mapping.items():
            print(f"  Class '{cls}': {len(indices)} positive samples")
        
        return mapping
    
    def create_splits(self) -> Dict[int, Dict[str, List[int]]]:
        """
        Create FSCIL splits for all stages.
        
        Returns dict:
          splits[stage]['train'] = list of dataset indices
          splits[stage]['test']  = list of dataset indices (cumulative seen classes)
          splits[stage]['ood']   = list of dataset indices (unseen classes up to this stage)
        """
        rng = np.random.RandomState(self.seed)
        splits = {}
        seen_classes = []
        
        for stage_id, cfg in self.STAGE_DEFINITIONS.items():
            stage_classes = cfg["classes"]
            shots = cfg["shots"]
            
            stage_train = []
            stage_test = []
            stage_negatives = []
            stage_negatives_by_class = {}
            
            for cls in stage_classes:
                cls_indices = self.class_to_indices[cls]
                if len(cls_indices) < self.min_positive:
                    raise ValueError(
                        f"Class '{cls}' has only {len(cls_indices)} positives, "
                        f"need at least {self.min_positive}"
                    )
                
                # Fixed shuffle per class, only once
                if cls not in self.class_train_test:
                    shuffled = list(cls_indices)
                    rng.shuffle(shuffled)
                    train_part = shuffled[:shots]
                    test_part = shuffled[shots:]
                    self.class_train_test[cls] = {
                        "train": train_part,
                        "test": test_part,
                    }
                
                stage_train.extend(self.class_train_test[cls]["train"])
                stage_test.extend(self.class_train_test[cls]["test"])
                seen_classes.append(cls)

                # Explicit current-class negatives for multi-label calibration.
                # These are toxic samples where the current class label is 0.
                # They are added to the stage training set with full labels so the
                # current class receives direct negative evidence.
                n_neg = int(round(shots * self.new_class_negative_ratio))
                if n_neg > 0:
                    positive_set = set(self.class_to_indices[cls])
                    used_positive_train = set(self.class_train_test[cls]["train"])
                    candidates = [
                        idx for idx in range(len(self.dataset))
                        if idx not in positive_set and idx not in used_positive_train
                    ]
                    rng.shuffle(candidates)
                    selected_neg = candidates[:min(n_neg, len(candidates))]
                    stage_negatives.extend(selected_neg)
                    stage_negatives_by_class[cls] = [int(i) for i in selected_neg]
            
            # For stage > 0, also include test data from all previously seen classes
            # And generate the Coreset for evaluating interference (O(1) cost instead of running full val)
            coreset_indices = []
            coreset_by_class = {}
            if stage_id > 0:
                for prev_stage in range(stage_id):
                    prev_cfg = self.STAGE_DEFINITIONS[prev_stage]
                    for cls in prev_cfg["classes"]:
                        stage_test.extend(self.class_train_test[cls]["test"])
                        # Class-balanced coreset selection: sample a fixed number of
                        # positive examples per previously seen class. This is used
                        # both for replay and for semantic interference evaluation.
                        candidates = list(self.class_train_test[cls]["test"])
                        rng.shuffle(candidates)
                        selected = candidates[:min(self.coreset_size_per_class, len(candidates))]
                        coreset_indices.extend(selected)
                        coreset_by_class[cls] = [int(i) for i in selected]
            
            # OOD: samples from classes not yet seen (but still toxic=1)
            unseen_classes = [c for c in self.ALL_CLASSES if c not in seen_classes]
            ood_indices = []
            for cls in unseen_classes:
                ood_indices.extend(self.class_to_indices[cls])
            
            # For base stage, OOD is empty (or could be non-toxic samples)
            if stage_id == 0:
                ood_indices = []
            
            splits[stage_id] = {
                "train": sorted(list(set(stage_train))),
                "negatives": sorted(list(set(stage_negatives))),
                "test": sorted(list(set(stage_test))),
                "ood": sorted(ood_indices),
                "coreset": sorted(list(set(coreset_indices))),
                "coreset_by_class": coreset_by_class,
                "negatives_by_class": stage_negatives_by_class,
                "classes": stage_classes,
                "seen_classes": list(seen_classes),
            }
        
        self.splits = splits
        return splits
    
    def save_splits(self, filename: str = None):
        """Save splits to JSON."""
        if filename is None:
            filename = os.path.join(self.output_dir, f"split_seed{self.seed}.json")
        
        # Convert numpy ints to Python ints for JSON serialization
        serializable = {}
        for stage_id, split in self.splits.items():
            stage_dict = {}
            for k, v in split.items():
                if k in ("train", "negatives", "test", "ood", "coreset") and isinstance(v, list):
                    stage_dict[k] = [int(i) for i in v]
                else:
                    stage_dict[k] = v
            serializable[str(stage_id)] = stage_dict
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Saved splits to {filename}")
    
    def load_splits(self, filename: str):
        """Load splits from JSON."""
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.splits = {int(k): v for k, v in data.items()}
        return self.splits
    
    def get_stage_dataset(self, stage_id: int, split: str = "train") -> Subset:
        """Get a torch Subset for a given stage and split (train/test/ood)."""
        if stage_id not in self.splits:
            raise ValueError(f"Stage {stage_id} not found. Run create_splits() first.")
        
        indices = self.splits[stage_id][split]
        return Subset(self.dataset, indices)
    
    def get_active_labels(self, stage_id: int) -> List[int]:
        """
        Return label indices (0-4) that are active up to and including stage_id.
        """
        active = []
        for sid in range(stage_id + 1):
            for cls in self.STAGE_DEFINITIONS[sid]["classes"]:
                active.append(self.ALL_CLASSES.index(cls))
        return sorted(list(set(active)))
