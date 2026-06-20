"""
Baselines and ablations for FSCIL Toxic Comment Classification.

All baseline experiments are launched via individual scripts in this directory
or through the unified entry points:
  - scripts/run_baseline.py   (for baselines)
  - scripts/run_ablation.py   (for ablations on our full method)
"""

from .lora_utils import (
    SingleBranchLoRALayer,
    MultiStageLoRALayer,
    EWCManager,
    apply_lora_to_roberta,
    get_lora_state_dict,
)

__all__ = [
    "SingleBranchLoRALayer",
    "MultiStageLoRALayer",
    "EWCManager",
    "apply_lora_to_roberta",
    "get_lora_state_dict",
]
