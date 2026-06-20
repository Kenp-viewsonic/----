"""
Experiment Logger — CSV/JSON 记录每阶段指标，支持多 seed 汇总。

Plan Step 4.4: 结果记录与可视化基础设施。

Usage:
    from utils.logger import ExperimentLogger

    logger = ExperimentLogger(output_dir="./outputs/ours_seed42")
    logger.log_stage_metrics(stage=0, metrics={"macro_f1": 0.85, "avg_map": 0.72})
    logger.log_stage_metrics(stage=1, metrics={"macro_f1": 0.78, "avg_map": 0.68})
    logger.save()  # writes metrics.csv and metrics.json
"""

import os
import json
import csv
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


class ExperimentLogger:
    """
    Collects per-stage metrics and serializes to CSV + JSON.

    Args:
        output_dir: Directory for output files (metrics.csv, metrics.json)
        method: Optional method label (e.g. 'ours', 'task_lora')
        seed: Optional random seed identifier
    """

    def __init__(
        self,
        output_dir: str,
        method: str = "unknown",
        seed: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.method = method
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)

        # stage -> metrics dict
        self.stage_metrics: Dict[int, dict] = {}
        # extra metadata
        self._config_snapshot = None

    def set_config(self, config: dict):
        """Record the config used for this run."""
        self._config_snapshot = config

    def log_stage_metrics(self, stage: int, metrics: dict):
        """
        Record metrics for a single stage.

        Args:
            stage: FSCIL stage index (0, 1, 2, ...)
            metrics: flat dict of metric_name -> float value
        """
        record = {"stage": stage}
        record.update({k: self._safe_float(v) for k, v in metrics.items()})
        self.stage_metrics[stage] = record

    def get_stage_metrics(self, stage: int) -> dict:
        """Retrieve recorded metrics for a stage."""
        return self.stage_metrics.get(stage, {})

    def save(self):
        """Persist all collected metrics to disk."""
        csv_path = os.path.join(self.output_dir, "metrics.csv")
        json_path = os.path.join(self.output_dir, "metrics.json")

        # Determine all metric keys across stages (for consistent CSV columns)
        all_keys = set()
        for rec in self.stage_metrics.values():
            all_keys.update(rec.keys())
        all_keys.discard("stage")

        sorted_keys = sorted(all_keys)
        fieldnames = ["stage"] + sorted_keys

        # Write CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for stage in sorted(self.stage_metrics.keys()):
                writer.writerow(self.stage_metrics[stage])

        # Write JSON
        payload = {
            "method": self.method,
            "seed": self.seed,
            "stages": {str(k): v for k, v in sorted(self.stage_metrics.items())},
        }
        if self._config_snapshot is not None:
            payload["config"] = self._config_snapshot

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        print(f"[Logger] Saved metrics to {csv_path} and {json_path}")

    @staticmethod
    def _safe_float(v):
        """Convert numpy types to native Python float for JSON serialization."""
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (dict,)):
            return {kk: ExperimentLogger._safe_float(vv) for kk, vv in v.items()}
        if isinstance(v, (list,)):
            return [ExperimentLogger._safe_float(vv) for vv in v]
        return v


def aggregate_across_seeds(
    loggers: List[ExperimentLogger],
    output_path: str,
) -> dict:
    """
    Compute mean ± std across multiple seed loggers and save aggregated JSON.

    Args:
        loggers: List of ExperimentLogger instances (one per seed)
        output_path: Path to save aggregated JSON (e.g. results/summary.json)

    Returns:
        summary: dict mapping metric_name -> {"mean": float, "std": float}
    """
    # Collect per-stage metrics across seeds
    # stage -> {metric -> [values]}
    stage_metric_values: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for logger in loggers:
        for stage, rec in logger.stage_metrics.items():
            for metric, val in rec.items():
                if metric == "stage":
                    continue
                if isinstance(val, (int, float)):
                    stage_metric_values[stage][metric].append(val)

    # Compute mean/std
    summary = {}
    for stage, metrics in sorted(stage_metric_values.items()):
        summary[f"stage_{stage}"] = {}
        for metric, vals in sorted(metrics.items()):
            arr = np.array(vals)
            summary[f"stage_{stage}"][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
            }

    # Overall average across stages
    all_metric_vals: Dict[str, List[float]] = defaultdict(list)
    for stage_data in stage_metric_values.values():
        for metric, vals in stage_data.items():
            all_metric_vals[metric].extend(vals)

    summary["overall"] = {}
    for metric, vals in sorted(all_metric_vals.items()):
        arr = np.array(vals)
        summary["overall"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(arr),
        }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Logger] Aggregated {len(loggers)} seeds -> {output_path}")
    return summary
