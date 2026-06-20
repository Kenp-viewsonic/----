"""
快速冒烟测试脚本：验证主方法、所有基线、所有消融的 stage-0 能正常运行。

每种方法只跑 stage 0（base 阶段），约 5 分钟以内，用于验证：
  - 模块导入无报错
  - 数据切分正常
  - 模型构建和前向传播正常
  - 训练循环一个 epoch 能跑完不 OOM
  - 评测指标计算正常

Usage:
    python scripts/run_smoke_test.py --config configs/base.yaml --max_epochs 1

可选参数：
    --methods: 逗号分隔的方法列表，默认全部
    --max_epochs: 快速测试时训练的 epoch 数（默认 1）
    --skip_ours / skip_baselines / skip_ablations: 跳过某类测试
"""

import os
import sys
import argparse
import subprocess
import tempfile
import yaml

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


def override_epochs_in_config(orig_config_path, max_epochs, tmp_dir):
    """Create a temporary config with reduced epochs for fast smoke testing."""
    with open(orig_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override training epochs at top level
    if "training" not in cfg:
        cfg["training"] = {}
    cfg["training"]["num_train_epochs"] = max_epochs
    cfg["training"]["logging_steps"] = 5
    cfg["training"]["save_strategy"] = "no"
    cfg["training"]["load_best_model_at_end"] = False
    cfg["training"]["eval_strategy"] = "epoch"

    tmp_config_path = os.path.join(tmp_dir, "base_smoke.yaml")
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # Also override stage-specific configs if they exist
    stages_path = os.path.join(os.path.dirname(orig_config_path), "stages.yaml")
    tmp_stages_path = os.path.join(tmp_dir, "stages.yaml")
    if os.path.exists(stages_path):
        with open(stages_path, "r", encoding="utf-8") as f:
            stages_cfg = yaml.safe_load(f)
        for key in stages_cfg:
            if "training" not in stages_cfg[key]:
                stages_cfg[key]["training"] = {}
            stages_cfg[key]["training"]["num_train_epochs"] = max_epochs
            stages_cfg[key]["training"]["save_strategy"] = "no"
            stages_cfg[key]["training"]["load_best_model_at_end"] = False
            stages_cfg[key]["training"]["eval_strategy"] = "epoch"
        with open(tmp_stages_path, "w", encoding="utf-8") as f:
            yaml.dump(stages_cfg, f)
    else:
        # Write a minimal stages.yaml so run_stage/run_baseline don't fall back to original
        with open(tmp_stages_path, "w", encoding="utf-8") as f:
            yaml.dump({}, f)

    return tmp_config_path, tmp_stages_path


def run_smoke(method, config_path, stages_config_path, seed=42):
    """Run stage 0 for a single method and return success status."""
    output_dir = f"./outputs/_smoke_{method}_seed{seed}"

    common_args = [
        "--config", config_path,
        "--stages_config", stages_config_path,
        "--seed", str(seed),
        "--output_dir", output_dir,
    ]

    if method == "ours":
        cmd = [
            sys.executable, "scripts/run_stage.py",
            "--stage", "0",
        ] + common_args
    elif method.startswith("ablation_"):
        variant = method.replace("ablation_", "")
        cmd = [
            sys.executable, "scripts/run_ablation.py",
            "--variant", variant,
            "--stage", "0",
        ] + common_args
    else:
        cmd = [
            sys.executable, "scripts/run_baseline.py",
            "--method", method,
            "--stage", "0",
        ] + common_args

    print(f"\n{'='*60}")
    print(f"[Smoke Test] Running: {method}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"[Smoke Test] {method} -> PASS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Smoke Test] {method} -> FAIL (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--methods", type=str, default="",
                        help="Comma-separated methods, e.g. 'ours,task_lora,o_lora'. Empty = all.")
    parser.add_argument("--max_epochs", type=int, default=1,
                        help="Number of epochs for quick smoke test")
    parser.add_argument("--skip_ours", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Determine methods to test
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = []
        if not args.skip_ours:
            methods.append("ours")
        if not args.skip_baselines:
            methods.extend(BASELINE_METHODS)
        if not args.skip_ablations:
            methods.extend(ABLATION_METHODS)

    # Create temporary config with reduced epochs
    with tempfile.TemporaryDirectory() as tmp_dir:
        smoke_config, smoke_stages = override_epochs_in_config(args.config, args.max_epochs, tmp_dir)

        print(f"\n{'#'*60}")
        print(f"# FSCIL Smoke Test")
        print(f"# Methods: {', '.join(methods)}")
        print(f"# Epochs: {args.max_epochs}")
        print(f"# Config: {smoke_config}")
        print(f"# Stages: {smoke_stages}")
        print(f"{'#'*60}")

        results = {}
        for method in methods:
            results[method] = run_smoke(method, smoke_config, smoke_stages, seed=args.seed)

    # Summary
    print(f"\n{'='*60}")
    print("[Smoke Test] Summary")
    print(f"{'='*60}")
    passed = sum(results.values())
    total = len(results)
    for method, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {method}")
    print(f"\nTotal: {passed}/{total} passed")

    if passed < total:
        print("\n[Smoke Test] SOME TESTS FAILED. Please check logs above.")
        sys.exit(1)
    else:
        print("\n[Smoke Test] ALL TESTS PASSED. Ready for full experiments.")


if __name__ == "__main__":
    main()