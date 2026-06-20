"""
Baseline: Task-LoRA + Mahalanobis Distance (Classic OOD Detection).

Plan Step 4.2, 流派三: 经典的特征空间异常检测基线。

For each seen class, compute the mean and shared covariance of [CLS] representations
on training data. At test time, score each sample by its minimum Mahalanobis distance
across all known-class distributions.

The Mahalanobis score serves as an uncertainty signal:
  - Low distance → likely known class
  - High distance → possible OOD / unknown class

Training is identical to Task-LoRA; rejection is only applied at evaluation time.

Usage:
    python baselines/task_lora_maha.py --stage 0 --config configs/base.yaml
    python baselines/task_lora_maha.py --stage 1 --prev_checkpoint outputs/task_lora_maha_stage_0_seed42/checkpoint-best
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Task-LoRA + Mahalanobis Distance baseline"
    )
    parser.add_argument("--stage", type=int, required=True,
                        help="FSCIL stage index (0, 1, 2, ...)")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to base config YAML")
    parser.add_argument("--prev_checkpoint", type=str, default=None,
                        help="Path to previous stage checkpoint directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate")
    args = parser.parse_args()

    run_experiment(
        method="task_lora_maha",
        stage=args.stage,
        config_path=args.config,
        prev_checkpoint=args.prev_checkpoint,
        seed=args.seed,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
