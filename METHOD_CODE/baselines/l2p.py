"""
Baseline: L2P (Learning to Prompt) for FSCIL Toxic Comment Classification.

Plan Step 4.2, 流派二 — 基于提示的连续学习。

Adapts the L2P/DualPrompt paradigm to text toxicity classification:
  - Maintains a pool of K learnable prompt embeddings
  - At each stage, selects top-M prompts via key-query similarity (prompt retrieval)
  - Prepends selected prompts to the input sequence
  - New prompts are added at each stage to prevent forgetting

Implementation note:
  This is a simplified text-adapted version. Unlike the original L2P (which uses
  ViT for vision), we adapt the prompt pool concept to RoBERTa's embedding space.

Usage:
    python baselines/l2p.py --stage 0 --config configs/base.yaml
    python baselines/l2p.py --stage 1 --prev_checkpoint outputs/l2p_stage_0_seed42/checkpoint-best
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="L2P (Learning to Prompt) FSCIL baseline"
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
        method="l2p",
        stage=args.stage,
        config_path=args.config,
        prev_checkpoint=args.prev_checkpoint,
        seed=args.seed,
        output_dir=args.output_dir,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
