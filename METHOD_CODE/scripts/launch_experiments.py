"""
Unified experiment launcher for FSCIL Toxic Comment Classification.

Supports three preset scenarios:
  1. quick_dev      : Fast smoke test / code verification (2-5 min per stage)
  2. subset_hparam  : Subset hyperparameter search (1-3 min per stage)
  3. full_experiment: Full paper experiments with multiple seeds (30-60 min per seed)

Usage:
    # Quick dev: run all stages for "ours" with 1 seed
    python scripts/launch_experiments.py --scenario quick_dev --methods ours

    # Subset hparam: grid search over a few configs
    python scripts/launch_experiments.py --scenario subset_hparam --methods ours --hparam_grid grid_example

    # Full experiment: all methods, all seeds
    python scripts/launch_experiments.py --scenario full_experiment --methods all --seeds 42,43,44,45,46

    # Run only specific stages
    python scripts/launch_experiments.py --scenario quick_dev --methods ours --stages 0

    # Dry-run: print commands without executing
    python scripts/launch_experiments.py --scenario quick_dev --methods ours --dry_run
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Preset configurations
SCENARIOS = {
    "quick_dev": {
        "config": "configs/quick_dev.yaml",
        "stages_config": "configs/quick_dev_stages.yaml",
        "default_seeds": [42],
        "default_methods": ["ours"],
        "description": "Fast development / smoke test (2-5 min per stage)",
    },
    "subset_hparam": {
        "config": "configs/subset_hparam.yaml",
        "stages_config": "configs/subset_stages.yaml",
        "default_seeds": [42],
        "default_methods": ["ours"],
        "description": "Subset hyperparameter search (1-3 min per stage)",
    },
    "full_experiment": {
        "config": "configs/full_experiment.yaml",
        "stages_config": "configs/full_stages.yaml",
        "default_seeds": [42, 43, 44, 45, 46],
        "default_methods": ["ours", "seq_finetune", "task_lora", "task_lora_msp",
                           "task_lora_adb", "task_lora_maha", "o_lora", "ewc_lora", "l2p"],
        "description": "Full paper experiments with multiple seeds",
    },
    "full_experiment_kd_tune": {
        "config": "configs/full_experiment_kd_tune.yaml",
        "stages_config": "configs/full_stages_kd_tune.yaml",
        "default_seeds": [42],
        "default_methods": ["ours"],
        "description": "Full experiment variant with stronger stage2 KD and lower LR",
    },
}

ALL_METHODS = [
    "ours",
    "seq_finetune",
    "task_lora",
    "task_lora_msp",
    "task_lora_adb",
    "task_lora_maha",
    "o_lora",
    "ewc_lora",
    "l2p",
    "ablation_no_evo",
    "ablation_no_dual",
    "ablation_no_anchor",
]


def build_command(method, stage, config, stages_config, seed, output_dir, prev_checkpoint=None):
    """Build the command list for a single stage run."""
    if method == "ours":
        script = "scripts/run_stage.py"
        cmd = [
            sys.executable, script,
            "--stage", str(stage),
            "--config", config,
            "--stages_config", stages_config,
            "--seed", str(seed),
            "--output_dir", output_dir,
        ]
    elif method.startswith("ablation_"):
        variant = method.replace("ablation_", "")
        script = "scripts/run_ablation.py"
        cmd = [
            sys.executable, script,
            "--variant", variant,
            "--stage", str(stage),
            "--config", config,
            "--stages_config", stages_config,
            "--seed", str(seed),
            "--output_dir", output_dir,
        ]
    elif method == "l2p":
        script = "baselines/l2p.py"
        cmd = [
            sys.executable, script,
            "--stage", str(stage),
            "--config", config,
            "--seed", str(seed),
            "--output_dir", output_dir,
        ]
    else:
        script = "scripts/run_baseline.py"
        cmd = [
            sys.executable, script,
            "--method", method,
            "--stage", str(stage),
            "--config", config,
            "--stages_config", stages_config,
            "--seed", str(seed),
            "--output_dir", output_dir,
        ]
    
    if prev_checkpoint is not None:
        cmd.extend(["--prev_checkpoint", prev_checkpoint])
    
    return cmd


def run_command(cmd, dry_run=False):
    """Execute or print a command."""
    print(f"\n{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    if dry_run:
        print("[DRY RUN] Skipped execution.")
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def run_pipeline(method, scenario_cfg, seeds, stages, run_root, dry_run=False, skip_failed_seed=False):
    """Run a full pipeline for a single method across seeds and stages."""
    config = scenario_cfg["config"]
    stages_config = scenario_cfg["stages_config"]
    
    summary = {}
    
    for seed in seeds:
        prev_checkpoint = None
        seed_summary = []
        
        for stage in stages:
            output_dir = os.path.join(
                run_root,
                "runs",
                f"{method}_seed{seed}_stage{stage}"
            )
            
            cmd = build_command(
                method=method,
                stage=stage,
                config=config,
                stages_config=stages_config,
                seed=seed,
                output_dir=output_dir,
                prev_checkpoint=prev_checkpoint,
            )
            
            exit_code = run_command(cmd, dry_run=dry_run)
            success = (exit_code == 0)
            seed_summary.append({
                "stage": stage,
                "output_dir": output_dir,
                "exit_code": exit_code,
                "success": success,
            })
            
            if not success:
                print(f"[ERROR] Method={method}, Seed={seed}, Stage={stage} failed with code {exit_code}")
                if skip_failed_seed:
                    break
                # Continue to next stage anyway; user can decide whether to use the checkpoint
            
            # Next stage loads best checkpoint from current stage
            prev_checkpoint = os.path.join(output_dir, "checkpoint-best")
        
        summary[f"seed{seed}"] = seed_summary
    
    return summary


def run_hparam_grid(scenario_cfg, grid_name, seeds, stages, run_root, dry_run=False):
    """Run a predefined hyperparameter grid search."""
    # Example grids (can be extended)
    GRIDS = {
        "grid_example": [
            {"lora.rs": 4, "lora.rp": 2, "loss_weights.lambda_evo": 0.3},
            {"lora.rs": 8, "lora.rp": 4, "loss_weights.lambda_evo": 0.5},
            {"lora.rs": 8, "lora.rp": 4, "loss_weights.lambda_evo": 0.8},
            {"lora.rs": 12, "lora.rp": 6, "loss_weights.lambda_evo": 0.5},
        ],
        "tau_search": [
            {"semantic_consolidation.tau": 0.05},
            {"semantic_consolidation.tau": 0.1},
            {"semantic_consolidation.tau": 0.2},
        ],
        "prefix_search": [
            {"prefix.n_anchors": 3, "prefix.m": 5},
            {"prefix.n_anchors": 5, "prefix.m": 10},
            {"prefix.n_anchors": 8, "prefix.m": 10},
        ],
    }
    
    if grid_name not in GRIDS:
        print(f"[ERROR] Unknown grid name: {grid_name}. Available: {list(GRIDS.keys())}")
        return {}
    
    import yaml
    from copy import deepcopy
    
    base_config_path = scenario_cfg["config"]
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    
    all_summaries = {}
    
    for idx, overrides in enumerate(GRIDS[grid_name]):
        cfg = deepcopy(base_cfg)
        
        # Apply overrides
        for key_path, value in overrides.items():
            keys = key_path.split(".")
            node = cfg
            for k in keys[:-1]:
                if k not in node:
                    node[k] = {}
                node = node[k]
            node[keys[-1]] = value
        
        # Save temp config
        temp_config = os.path.join(run_root, "hparam_configs", f"_hparam_temp_{grid_name}_cfg{idx}.yaml")
        os.makedirs(os.path.dirname(temp_config) if os.path.dirname(temp_config) else ".", exist_ok=True)
        with open(temp_config, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        
        print(f"\n[HPARAM SEARCH] Grid point {idx+1}/{len(GRIDS[grid_name])}: {overrides}")
        
        summary = run_pipeline(
            method="ours",
            scenario_cfg={
                "config": temp_config,
                "stages_config": scenario_cfg["stages_config"],
            },
            seeds=seeds,
            stages=stages,
            run_root=run_root,
            dry_run=dry_run,
        )
        
        all_summaries[f"cfg{idx}_{json.dumps(overrides, sort_keys=True)}"] = summary
    
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Launch FSCIL experiments")
    parser.add_argument("--scenario", type=str, required=True,
                       choices=list(SCENARIOS.keys()),
                       help="Experiment scenario preset")
    parser.add_argument("--methods", type=str, default="",
                       help="Comma-separated methods, or 'all'. Empty=scenario default.")
    parser.add_argument("--seeds", type=str, default="",
                       help="Comma-separated seeds. Empty=scenario default.")
    parser.add_argument("--stages", type=str, default="0,1,2",
                       help="Comma-separated stage IDs to run")
    parser.add_argument("--hparam_grid", type=str, default="",
                       help="Hyperparameter grid name (for subset_hparam scenario)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--skip_failed_seed", action="store_true",
                       help="If a seed fails, skip remaining stages for that seed")
    parser.add_argument("--outputs_root", type=str, default="./outputs",
                       help="Root directory for all launch outputs")
    parser.add_argument("--run_id", type=str, default="",
                       help="Optional run id. Empty means auto-generated.")
    args = parser.parse_args()
    
    scenario = SCENARIOS[args.scenario]
    print(f"\n{'#'*60}")
    print(f"# Scenario: {args.scenario}")
    print(f"# {scenario['description']}")
    print(f"# Config: {scenario['config']}")
    print(f"# Stages Config: {scenario['stages_config']}")
    print(f"{'#'*60}")
    
    # Determine methods
    if args.methods.lower() == "all":
        methods = ALL_METHODS
    elif args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = scenario["default_methods"]
    
    # Determine seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = scenario["default_seeds"]
    
    # Determine stages
    stages = [int(s.strip()) for s in args.stages.split(",")]

    run_id = args.run_id.strip() if args.run_id else f"{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = os.path.join(args.outputs_root, run_id)
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(os.path.join(run_root, "runs"), exist_ok=True)
    
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Stages: {stages}")
    print(f"Run ID: {run_id}")
    print(f"Run Root: {run_root}")
    print(f"Dry run: {args.dry_run}")
    
    # Run
    all_summaries = {}
    
    if args.hparam_grid:
        # Hyperparameter search mode
        summary = run_hparam_grid(
            scenario_cfg=scenario,
            grid_name=args.hparam_grid,
            seeds=seeds,
            stages=stages,
            run_root=run_root,
            dry_run=args.dry_run,
        )
        all_summaries["hparam_search"] = summary
    else:
        # Standard pipeline mode
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Running method: {method}")
            print(f"{'='*60}")
            summary = run_pipeline(
                method=method,
                scenario_cfg=scenario,
                seeds=seeds,
                stages=stages,
                run_root=run_root,
                dry_run=args.dry_run,
                skip_failed_seed=args.skip_failed_seed,
            )
            all_summaries[method] = summary
    
    # Save summary and manifest under run_root
    summary_path = os.path.join(run_root, "launch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    manifest = {
        "run_id": run_id,
        "scenario": args.scenario,
        "description": scenario["description"],
        "config": scenario["config"],
        "stages_config": scenario["stages_config"],
        "methods": methods,
        "seeds": seeds,
        "stages": stages,
        "hparam_grid": args.hparam_grid if args.hparam_grid else None,
        "dry_run": args.dry_run,
        "skip_failed_seed": args.skip_failed_seed,
        "run_root": run_root,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary_file": summary_path,
    }
    manifest_path = os.path.join(run_root, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Launch complete. Summary saved to: {summary_path}")
    print(f"Run manifest saved to: {manifest_path}")
    print(f"{'='*60}")
    
    # Print quick summary
    for method, method_summary in all_summaries.items():
        print(f"\nMethod: {method}")
        if isinstance(method_summary, dict):
            # Check if this is a hparam search result (nested dicts) or standard pipeline
            first_val = next(iter(method_summary.values())) if method_summary else None
            if first_val is not None and isinstance(first_val, dict):
                # hparam search: keys are cfg names, values are standard summaries
                for cfg_key, cfg_summary in method_summary.items():
                    print(f"  Config: {cfg_key}")
                    for seed_key, seed_runs in cfg_summary.items():
                        if isinstance(seed_runs, list):
                            successes = sum(1 for r in seed_runs if r["success"])
                            total = len(seed_runs)
                            print(f"    {seed_key}: {successes}/{total} stages succeeded")
            elif first_val is not None and isinstance(first_val, list):
                # standard pipeline: keys are seed names, values are run lists
                for seed_key, seed_runs in method_summary.items():
                    successes = sum(1 for r in seed_runs if r["success"])
                    total = len(seed_runs)
                    print(f"  {seed_key}: {successes}/{total} stages succeeded")


if __name__ == "__main__":
    main()
