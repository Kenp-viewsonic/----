"""
Baseline: Task-LoRA + MSP (Max Softmax Probability for OOD detection).

Training is identical to Task-LoRA; evaluation uses MSP score for rejection.

Usage:
    python baselines/task_lora_msp.py --stage 0 --config configs/base.yaml
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.runner import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--prev_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    run_experiment(
        method="task_lora_msp",
        stage=args.stage,
        config_path=args.config,
        prev_checkpoint=args.prev_checkpoint,
        seed=args.seed,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
