"""
Full FSCIL pipeline: sequentially train all stages.

Usage:
    python run_full_pipeline.py --config configs/base.yaml --seeds 42,43,44
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--max_stage", type=int, default=2)
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    for seed in seeds:
        prev_checkpoint = None
        for stage in range(args.max_stage + 1):
            output_dir = f"./outputs/stage_{stage}_seed{seed}"
            
            cmd = [
                sys.executable, "scripts/run_stage.py",
                "--stage", str(stage),
                "--config", args.config,
                "--seed", str(seed),
                "--output_dir", output_dir,
            ]
            
            if prev_checkpoint:
                cmd.extend(["--prev_checkpoint", prev_checkpoint])
            
            run_command(cmd)
            
            # Next stage loads best checkpoint from current stage
            prev_checkpoint = os.path.join(output_dir, "checkpoint-best")
    
    print("\n[Pipeline] All stages completed!")


if __name__ == "__main__":
    main()
