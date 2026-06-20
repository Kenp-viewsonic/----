"""
Baseline: Sequential Fine-tuning (no LoRA, catastrophic forgetting).

Usage:
    python baselines/seq_finetune.py --stage 0 --config configs/base.yaml
    python baselines/seq_finetune.py --stage 1 --prev_checkpoint outputs/seq_finetune_stage_0_seed42/checkpoint-best
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
        method="seq_finetune",
        stage=args.stage,
        config_path=args.config,
        prev_checkpoint=args.prev_checkpoint,
        seed=args.seed,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
