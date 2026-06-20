# FSCIL Toxic Comment Classification

Few-Shot Class-Incremental Learning (FSCIL) framework for toxic comment identification and fine-grained classification, with robustness to evolving surface forms (leet speak, evasion, black slang).

## Architecture

- **Backbone**: RoBERTa-base (125M params)
- **ToxicSemanticPrefix**: K-means anchored prefix-tuning (Section 3.2)
- **DualBranchLoRA**: Semantic-stability-aware stable + plastic branches (Section 3.3)
- **ToxicAwarePE**: Surface-structure-aware positional encoding (Section 3.4)
- **HierarchicalRejectionGate**: Two-level rejection (unknown / known-framework-unknown-variant) (Section 3.5)

## Project Structure

```
METHOD_CODE/
├── configs/           # YAML configurations
├── data/              # Dataset, FSCIL split protocol, variant generator
├── models/            # Core modules (prefix, LoRA, PE, gate, classifier)
├── losses/            # Composite losses (evo, stable/plastic reg, open, orth)
├── trainers/          # Custom Trainer with semantic consolidation
├── utils/             # Metrics (CKA, Variant Recall, AUROC, Forgetting)
└── scripts/           # Training scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place Jigsaw Toxic Comment CSV at:
```
jigsaw-toxic-comment-classification-challenge/train.csv/train.csv
```

## Quick Start

### Recommended: Use Preset Scenarios

We provide three preset experiment scenarios for different purposes:

```bash
# 1. Quick development / smoke test (2-5 min per stage)
python scripts/launch_experiments.py --scenario quick_dev --methods ours --stages 0

# 2. Hyperparameter search on subset data (1-3 min per stage)
python scripts/launch_experiments.py --scenario subset_hparam --hparam_grid tau_search --stages 0

# 3. Full paper experiments (30-60 min per seed)
python scripts/launch_experiments.py --scenario full_experiment --methods ours --seeds 42,43,44,45,46
```

See `EXPERIMENTS.md` for detailed usage, advanced options, and FAQ.

### Manual: Single Stage
```bash
python scripts/run_stage.py --stage 0 --config configs/base.yaml --seed 42
python scripts/run_stage.py --stage 1 --config configs/base.yaml --prev_checkpoint outputs/stage_0_seed42/checkpoint-best --seed 42
```

### Manual: Full Pipeline
```bash
python scripts/run_full_pipeline.py --config configs/base.yaml --seeds 42,43,44
```

## Stage Definitions

| Stage | New Classes | Shots/Class |
|-------|-------------|-------------|
| 0     | obscene, insult | 32 |
| 1     | threat, identity_hate | 16 |
| 2     | severe_toxic | 16 |

Note: `toxic` is used as a parent filter, not an FSCIL target class.

## Key Configurations

### Base Config
See `configs/base.yaml`:
- `lora.rs` / `lora.rp`: stable / plastic ranks (default 8 / 4)
- `prefix.m`: prefix length (default 10)
- `semantic_consolidation.tau`: merge threshold (default 0.1)
- `loss_weights`: weights for evo, stable/plastic, open, orth losses

### Experiment Presets
| Preset | Config | Stages | Purpose |
|--------|--------|--------|---------|
| Quick Dev | `configs/quick_dev.yaml` | `configs/quick_dev_stages.yaml` | Smoke test / debugging |
| Subset HParam | `configs/subset_hparam.yaml` | `configs/subset_stages.yaml` | Fast hyperparameter search |
| Full Experiment | `configs/full_experiment.yaml` | `configs/full_stages.yaml` | Paper experiments / reproduction |

See `CONFIG_GUIDE.md` for 11GB VRAM tuning recommendations and `EXPERIMENTS.md` for running instructions.

## Evaluation Metrics

- **Avg-mAP** / **Macro-F1** / **Micro-F1**
- **Forgetting** (per-class performance drop)
- **AUROC** / **FPR95** (OOD rejection)
- **Variant Recall** (toxic variant robustness)
- **Semantic Stability CKA** (cross-stage representation consistency)
- **Tail Recall** (rare class recall)

## Citation

This code implements the method described in the problem definition document.
