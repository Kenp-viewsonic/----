"""
通用 FSCIL 实验流水线脚本。

支持主方法 (Ours)、基线 (Baselines)、消融 (Ablations) 的完整多阶段训练。

Usage:
  # 主方法
  python scripts/run_pipeline.py --method ours --config configs/base.yaml --seeds 42

  # 单个基线
  python scripts/run_pipeline.py --method task_lora --config configs/base.yaml --seeds 42

  # 单个消融
  python scripts/run_pipeline.py --method ablation_no_evo --config configs/base.yaml --seeds 42

  # 批量跑所有基线（3 seeds）
  python scripts/run_pipeline.py --method all_baselines --config configs/base.yaml --seeds 42,43,44

  # 批量跑所有消融
  python scripts/run_pipeline.py --method all_ablations --config configs/base.yaml --seeds 42

  # 一键跑全部（主方法 + 所有基线 + 所有消融）
  python scripts/run_pipeline.py --method all --config configs/base.yaml --seeds 42
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASELINE_METHODS = [
    "seq_finetune",
    "task_lora",
    "task_lora_msp",
    "task_lora_adb",
    "task_lora_maha",
    "o_lora",
    "ewc_lora",
    "l2p",
]

ABLATION_METHODS = [
    "ablation_no_evo",
    "ablation_no_dual",
    "ablation_no_anchor",
]


def run_command(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=True)
    return result


def run_single_method(method, seed, config, max_stage):
    """Run a single method through all stages for one seed."""
    prev_checkpoint = None

    for stage in range(max_stage + 1):
        if method == "ours":
            output_dir = f"./outputs/stage_{stage}_seed{seed}"
            script = "scripts/run_stage.py"
            cmd = [
                sys.executable, script,
                "--stage", str(stage),
                "--config", config,
                "--seed", str(seed),
                "--output_dir", output_dir,
            ]
        elif method.startswith("ablation_"):
            variant = method.replace("ablation_", "")
            output_dir = f"./outputs/ablation_{variant}_stage_{stage}_seed{seed}"
            script = "scripts/run_ablation.py"
            cmd = [
                sys.executable, script,
                "--variant", variant,
                "--stage", str(stage),
                "--config", config,
                "--seed", str(seed),
                "--output_dir", output_dir,
            ]
        else:
            # baseline
            output_dir = f"./outputs/{method}_stage_{stage}_seed{seed}"
            script = "scripts/run_baseline.py"
            cmd = [
                sys.executable, script,
                "--method", method,
                "--stage", str(stage),
                "--config", config,
                "--seed", str(seed),
                "--output_dir", output_dir,
            ]

        if prev_checkpoint and os.path.exists(prev_checkpoint):
            cmd.extend(["--prev_checkpoint", prev_checkpoint])

        run_command(cmd)

        # Next stage loads best checkpoint from current stage
        prev_checkpoint = os.path.join(output_dir, "checkpoint-best")

    print(f"\n[Pipeline] Method '{method}' seed {seed} completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        help=("Method to run: ours, seq_finetune, task_lora, task_lora_msp, "
                              "task_lora_adb, o_lora, ewc_lora, ablation_no_evo, "
                              "ablation_no_dual, ablation_no_anchor, "
                              "all_baselines, all_ablations, all"))
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--max_stage", type=int, default=2)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Determine methods to run
    if args.method == "all":
        methods = ["ours"] + BASELINE_METHODS + ABLATION_METHODS
    elif args.method == "all_baselines":
        methods = BASELINE_METHODS
    elif args.method == "all_ablations":
        methods = ABLATION_METHODS
    else:
        methods = [args.method]

    for method in methods:
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"# Starting: method={method}, seed={seed}")
            print(f"{'#'*60}")
            try:
                run_single_method(method, seed, args.config, args.max_stage)
            except subprocess.CalledProcessError as e:
                print(f"\n[ERROR] Method '{method}' seed {seed} failed with exit code {e.returncode}")
                # Continue with next experiment rather than stopping everything
                continue

    print("\n" + "="*60)
    print("[Pipeline] All requested experiments completed!")
    print("="*60)


if __name__ == "__main__":
    main()