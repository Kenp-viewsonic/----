"""
Unified Ours entry point — post-ablation clean baseline.

Usage:
    python scripts/run_unified.py --seed 42
    python scripts/run_unified.py --seed 42 --output_dir ./outputs_unified_seed42
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd):
    print(f"\n{'='*60}")
    print(f"[run_unified] {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[run_unified] Command failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Unified Ours pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs_unified")
    parser.add_argument("--max_stage", type=int, default=2)
    args = parser.parse_args()

    config = "configs/unified.yaml"
    stages_config = "configs/unified_stages.yaml"
    prev_checkpoint = None

    for stage in range(args.max_stage + 1):
        output_dir = os.path.join(args.output_dir, f"stage_{stage}")
        cmd = [
            sys.executable, "scripts/run_stage.py",
            "--stage", str(stage),
            "--config", config,
            "--stages_config", stages_config,
            "--seed", str(args.seed),
            "--output_dir", output_dir,
        ]
        if prev_checkpoint is not None:
            cmd.extend(["--prev_checkpoint", prev_checkpoint])

        run_command(cmd)
        prev_checkpoint = os.path.join(output_dir, "checkpoint-best")

    print("\n[run_unified] All stages completed!")


if __name__ == "__main__":
    main()
