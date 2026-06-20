# 实验运行指南

本文档基于实际代码梳理所有实验入口，覆盖训练、评测、消融、基线、汇总全流程。

---

## 目录

1. [环境准备](#环境准备)
2. [入口脚本一览](#入口脚本一览)
3. [场景一：快速开发验证（quick_dev）](#场景一快速开发验证)
4. [场景二：超参搜索（subset_hparam）](#场景二超参搜索)
5. [场景三：完整论文实验（full_experiment）](#场景三完整论文实验)
6. [手动单阶段训练](#手动单阶段训练)
7. [基线实验](#基线实验)
8. [消融实验](#消融实验)
9. [评测与指标](#评测与指标)
10. [结果汇总与可视化](#结果汇总与可视化)
11. [跨领域鲁棒性验证](#跨领域鲁棒性验证)
12. [配置体系说明](#配置体系说明)
13. [输出目录结构](#输出目录结构)
14. [可运行方法清单](#可运行方法清单)

---

## 环境准备

```bash
# 在 METHOD_CODE 根目录下激活环境
conda activate .conda

# 验证依赖
python -c "import torch; import transformers; print(f'torch={torch.__version__}, transformers={transformers.__version__}')"

# 验证所有模块可导入
python scripts/run_smoke_test.py --config configs/quick_dev.yaml --max_epochs 1 --skip_baselines --skip_ablations
```

---

## 入口脚本一览

| 脚本 | 用途 | 典型调用者 |
|------|------|-----------|
| `scripts/launch_experiments.py` | **统一入口**：按场景批量跑所有方法/seed/stage | 直接使用 |
| `scripts/run_pipeline.py` | 轻量多阶段流水线（单方法/批量基线/消融） | 直接使用 |
| `scripts/run_stage.py` | 单阶段训练（ours 主方法） | `launch_experiments.py` 内部调用 |
| `scripts/run_baseline.py` | 单阶段训练（基线方法） | `launch_experiments.py` 内部调用 |
| `scripts/run_ablation.py` | 单阶段训练（消融变体） | `launch_experiments.py` 内部调用 |
| `scripts/evaluate.py` | 单 checkpoint 评测（全指标） | 手动调用 |
| `scripts/aggregate_results.py` | 扫描 outputs/ 生成 CSV + Markdown 表格 | 手动调用 |
| `scripts/plot_results.py` | 生成论文级可视化图表 | 手动调用 |
| `scripts/eval_cross_domain.py` | 跨领域零样本鲁棒性评测 | 手动调用 |
| `scripts/run_smoke_test.py` | 冒烟测试（所有方法 stage 0） | 手动调用 |
| `scripts/run_full_pipeline.py` | 简单全流程（仅 ours，旧版兼容） | 手动调用 |

---

## 场景一：快速开发验证

> 用途：验证代码、调试数据流、确认模块无报错。单 stage 约 2-5 分钟。

```bash
# 1) 仅跑 stage 0（最快反馈）
python scripts/launch_experiments.py --scenario quick_dev --methods ours --stages 0

# 2) 跑全阶段 0→1→2（约 10-15 分钟）
python scripts/launch_experiments.py --scenario quick_dev --methods ours --stages 0,1,2

# 3) 干运行：只打印命令不执行
python scripts/launch_experiments.py --scenario quick_dev --methods ours --dry_run
```

快速开发配置特点：`base 2 epoch / 增量 1 epoch`，不保存 checkpoint，`eval_samples=50`，1 seed。

---

## 场景二：超参搜索

> 用途：用子集数据快速筛选超参组合。单 stage 约 1-3 分钟。

```bash
# tau 搜索（语义沉淀阈值）
python scripts/launch_experiments.py --scenario subset_hparam --hparam_grid tau_search --stages 0,1

# 前缀配置搜索
python scripts/launch_experiments.py --scenario subset_hparam --hparam_grid prefix_search --stages 0

# LoRA 秩 + lambda_evo 综合搜索
python scripts/launch_experiments.py --scenario subset_hparam --hparam_grid grid_example --stages 0,1,2
```

超参搜索配置特点：`base 8-shot / 增量 4-shot`，`prefix.n_anchors` 同步下调，3 epoch，`eval_samples=30`。

预定义网格见 `launch_experiments.py` 中 `GRIDS` 字典：
- `grid_example`：LoRA 秩 + lambda_evo 网格
- `tau_search`：`{0.05, 0.1, 0.2}`
- `prefix_search`：`{n_anchors: 3/5/8, m: 5/10}`

---

## 场景三：完整论文实验

> 用途：论文最终实验与复现。单 seed 全阶段约 30-60 分钟。

```bash
# 1) 仅跑 ours（5 seeds）
python scripts/launch_experiments.py --scenario full_experiment --methods ours

# 2) 跑指定 seed
python scripts/launch_experiments.py --scenario full_experiment --methods ours --seeds 42,43

# 3) 跑全部方法（ours + 9 个基线 + 3 个消融），5 seeds
python scripts/launch_experiments.py --scenario full_experiment --methods all

# 4) 指定方法子集
python scripts/launch_experiments.py --scenario full_experiment --methods ours,o_lora,ewc_lora --seeds 42,43,44

# 5) 失败跳过（某 stage 失败则跳过该 seed 剩余 stage）
python scripts/launch_experiments.py --scenario full_experiment --methods ours --skip_failed_seed
```

完整实验配置特点：`base 5 epoch / 增量 3 epoch`，5 seeds，保存 best checkpoint，`metric_for_best_model=eval_avg_map`。

---

## 手动单阶段训练

> 当你需要精确控制某个阶段时，直接调用底层脚本。

### 主方法（Ours）

```bash
# Stage 0（base 阶段）
python scripts/run_stage.py --stage 0 --config configs/base.yaml --seed 42

# Stage 1（增量，加载 stage 0 的 checkpoint）
python scripts/run_stage.py --stage 1 --config configs/base.yaml --seed 42 \
    --prev_checkpoint ./outputs/stage_0_seed42/checkpoint-best

# Stage 2
python scripts/run_stage.py --stage 2 --config configs/base.yaml --seed 42 \
    --prev_checkpoint ./outputs/stage_1_seed42/checkpoint-best
```

### 通过 run_pipeline.py 跑全流程

```bash
# Ours 全阶段
python scripts/run_pipeline.py --method ours --config configs/base.yaml --seeds 42

# 多 seed
python scripts/run_pipeline.py --method ours --config configs/base.yaml --seeds 42,43,44

# 所有基线
python scripts/run_pipeline.py --method all_baselines --config configs/base.yaml --seeds 42

# 所有消融
python scripts/run_pipeline.py --method all_ablations --config configs/base.yaml --seeds 42

# 全部（ours + 基线 + 消融）
python scripts/run_pipeline.py --method all --config configs/base.yaml --seeds 42
```

---

## 基线实验

### 可用基线（9 个）

| 方法 | 标识 | 说明 | 论文流派 |
|------|------|------|----------|
| Sequential Fine-tune | `seq_finetune` | 无防遗忘的朴素连续微调 | 正则化/架构 |
| Task-LoRA | `task_lora` | 每阶段独立 LoRA，冻结旧阶段 | 正则化/架构 |
| Task-LoRA + MSP | `task_lora_msp` | Task-LoRA + Max Softmax Probability 拒识 | 开集拒识 |
| Task-LoRA + ADB | `task_lora_adb` | Task-LoRA + Adaptive Decision Boundary 拒识 | 开集拒识 |
| Task-LoRA + Mahalanobis | `task_lora_maha` | Task-LoRA + Mahalanobis 距离特征空间异常检测 | 开集拒识 |
| O-LoRA | `o_lora` | 正交 LoRA（2601.02232），正交参数隔离 | 正则化/架构 |
| EWC + LoRA | `ewc_lora` | EWC 正则化 + LoRA 参数重要性惩罚 | 正则化/架构 |
| L2P | `l2p` | Learning to Prompt，动态提示池适配版 | 提示连续学习 |
| DualPrompt | `l2p` | （等价 L2P，同一实现） | 提示连续学习 |

### 运行基线

```bash
# 单个基线，单阶段
python scripts/run_baseline.py --method o_lora --stage 0 --config configs/base.yaml --seed 42

# 单个基线，多阶段（手动衔接 checkpoint）
python scripts/run_baseline.py --method o_lora --stage 0 --config configs/base.yaml --seed 42
python scripts/run_baseline.py --method o_lora --stage 1 --config configs/base.yaml --seed 42 \
    --prev_checkpoint ./outputs/o_lora_stage_0_seed42/checkpoint-best
```

### 通过 launch_experiments.py 跑基线

```bash
# 指定基线
python scripts/launch_experiments.py --scenario full_experiment --methods o_lora,ewc_lora --seeds 42

# 全部基线（不含 ours 和消融）
python scripts/launch_experiments.py --scenario full_experiment \
    --methods seq_finetune,task_lora,task_lora_msp,task_lora_adb,task_lora_maha,o_lora,ewc_lora,l2p

# 或用 all（含 ours + 基线 + 消融）
python scripts/launch_experiments.py --scenario full_experiment --methods all
```

### 通过 run_pipeline.py 跑基线

```bash
# 所有基线
python scripts/run_pipeline.py --method all_baselines --config configs/base.yaml --seeds 42
```

---

## 消融实验

### 可用消融变体

| 变体标识 | 说明 | 控制开关 |
|----------|------|----------|
| `ablation_no_evo` | 移除语义演化一致性损失 L_evo | `loss_weights.lambda_evo=0` |
| `ablation_no_dual` | 退化为单分支 LoRA（rs=12, rp=0） | `lora.rs=12, lora.rp=0, eta=0` |
| `ablation_no_anchor` | K-means 锚定前缀降级为随机初始化 | `prefix.init_random=true` |

消融配置在 `configs/ablation.yaml` 中统一定义（还包含 `no_toxic_pe`, `no_plastic`, `always_add`, `always_freeze`, `no_rejection`, `no_open_loss`, `no_orth` 等扩展变体）。

### 运行消融

```bash
# 单个消融，单阶段
python scripts/run_ablation.py --variant no_evo --stage 0 --config configs/base.yaml --seed 42

# 通过 launch_experiments.py
python scripts/launch_experiments.py --scenario full_experiment \
    --methods ablation_no_evo,ablation_no_dual,ablation_no_anchor

# 通过 run_pipeline.py
python scripts/run_pipeline.py --method all_ablations --config configs/base.yaml --seeds 42
```

---

## 评测与指标

### 单 checkpoint 评测

```bash
# 评测 ours
python scripts/evaluate.py \
    --checkpoint ./outputs/stage_2_seed42/checkpoint-best \
    --method ours --stage 2 --config configs/base.yaml

# 评测基线
python scripts/evaluate.py \
    --checkpoint ./outputs/o_lora_stage_2_seed42/checkpoint-best \
    --method o_lora --stage 2 --config configs/base.yaml

# 带 CKA 稳定性评测（传入 stage 0 参考 checkpoint）
python scripts/evaluate.py \
    --checkpoint ./outputs/stage_2_seed42/checkpoint-best \
    --prev_checkpoint ./outputs/stage_0_seed42/checkpoint-best \
    --method ours --stage 2 --config configs/base.yaml
```

### 指标体系

| 指标 | 类型 | 说明 |
|------|------|------|
| `avg_map` | 持续学习 | 平均 mAP（论文主指标） |
| `macro_f1` | 持续学习 | 宏平均 F1 |
| `micro_f1` | 持续学习 | 微平均 F1 |
| `forgetting` | 持续学习 | 遗忘率（标准 CIL 定义） |
| `auroc` | 拒识 | 已知/未知 AUROC |
| `fpr95` | 拒识 | 95% TPR 时的误报率 |
| `variant_recall` | 演化鲁棒性 | 变体测试集上的召回率 |
| `cka` | 语义稳定性 | 阶段 k 与阶段 0 的 [CLS] CKA 相似度 |
| `tail_recall` | 长尾 | 最少样本毒性标签的单独 Recall |

---

## 结果汇总与可视化

### 汇总结果

```bash
# 扫描 outputs/ 生成 CSV + Markdown + JSON 汇总
python scripts/aggregate_results.py --outputs_dir ./outputs --out_dir ./results

# 指定 run_id 过滤（避免混合不同批次）
python scripts/aggregate_results.py --outputs_dir ./outputs/full_experiment_20260609_113000 --out_dir ./results
```

产物：
- `results/results_table.csv`：扁平表格
- `results/results_table.md`：Markdown 格式表格
- `results/results_summary.json`：按 method 分组的 mean±std 汇总

### 生成论文图表

```bash
# 生成增量性能曲线 + LaTeX 表格
python scripts/plot_results.py --results_json ./results/results_summary.json --out_dir ./results/figures

# 含 plastic 衰减图（需先记录衰减日志）
python scripts/plot_results.py --results_json ./results/results_summary.json \
    --decay_log ./results/plastic_decay.json --out_dir ./results/figures

# 含变异雷达图
python scripts/plot_results.py --results_json ./results/results_summary.json \
    --variant_json ./results/variant_recall.json --out_dir ./results/figures
```

产物（`results/figures/`）：
- `incremental_curves.png`：多方法逐阶段 Avg-mAP 对比曲线（含误差棒）
- `plastic_decay.png`：可塑分支激活度随 Epoch 衰减图（沉淀自证）
- `variant_radar.png`：变异鲁棒性雷达图（leet/空格/符号/未覆盖变异）
- `results_table.tex`：LaTeX 格式结果表格（可直接贴论文）

### 日志记录

```python
# 在训练脚本中使用 ExperimentLogger
from utils.logger import ExperimentLogger, aggregate_across_seeds

logger = ExperimentLogger(output_dir="./outputs/ours_seed42", method="ours", seed=42)
logger.log_stage_metrics(stage=0, metrics={"macro_f1": 0.85, "avg_map": 0.72})
logger.log_stage_metrics(stage=1, metrics={"macro_f1": 0.78, "avg_map": 0.68})
logger.save()  # → metrics.csv + metrics.json
```

---

## 跨领域鲁棒性验证

> Plan Step 4.4：验证学到的"毒性语义核"能迁移到隐式仇恨表达。

```bash
python scripts/eval_cross_domain.py \
    --checkpoint ./outputs/stage_2_seed42/checkpoint-best \
    --method ours \
    --config configs/base.yaml \
    --implicit_hate_path /path/to/implicit_hate.csv \
    --mapping configs/implicit_hate_mapping.yaml \
    --output_dir ./outputs/cross_domain
```

需要单独下载 implicit-hate 语料库（EMNLP 2021, ~22K tweets）。

映射配置在 `configs/implicit_hate_mapping.yaml`，将隐式仇恨 7 类映射到 Jigsaw 多标签空间。

---

## 配置体系说明

### 配置文件清单

| 文件 | 用途 |
|------|------|
| `configs/base.yaml` | 默认基础配置（模型、LoRA、前缀、拒识、训练、损失权重） |
| `configs/stages.yaml` | 各阶段覆盖（学习率、epoch、损失权重微调） |
| `configs/quick_dev.yaml` + `quick_dev_stages.yaml` | 快速开发场景 |
| `configs/subset_hparam.yaml` + `subset_stages.yaml` | 超参搜索场景 |
| `configs/full_experiment.yaml` + `full_stages.yaml` | 完整论文实验场景 |
| `configs/ablation.yaml` | 消融实验控制开关（11 个变体） |
| `configs/implicit_hate_mapping.yaml` | 跨领域评测映射 |

### 合并逻辑

`launch_experiments.py` / `run_stage.py` / `run_baseline.py` 内部的 `merge_configs()`：
- 先加载 `--config`（如 `base.yaml`）
- 再加载 `--stages_config`（如 `stages.yaml`）中对应 stage key（`base` / `stage1` / `stage2`）
- stage 级配置覆盖 base 配置

### 关键超参一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora.rs` | 8 | 稳定分支秩 |
| `lora.rp` | 4 | 可塑分支秩 |
| `prefix.m` | 10 | 前缀长度 |
| `prefix.n_anchors` | 5 | K-means 聚类数 |
| `prefix.alpha_anchor` | 0.7 | 锚定强度 |
| `rejection_gate.theta_coarse` | 0.5 | 粗拒识阈值 |
| `rejection_gate.theta_fine` | 0.3 | 细拒识阈值 |
| `semantic_consolidation.tau` | 0.1 | 语义沉淀阈值 |
| `loss_weights.lambda_evo` | 0.5 | 语义演化损失权重 |
| `loss_weights.lambda_sp` | 1e-3 | 可塑分支 L1 稀疏权重 |
| `loss_weights.beta` | 0.3 | 开集拒识损失权重 |
| `loss_weights.eta` | 1e-4 | 跨阶段正交性损失权重 |

---

## 输出目录结构

### launch_experiments.py 输出（推荐）

```
outputs/
├── quick_dev_20260609_101500/           # run_id（自动生成）
│   ├── run_manifest.json               # 参数与配置快照
│   ├── launch_summary.json             # 执行汇总
│   ├── hparam_configs/                 # 超参搜索临时配置（仅 hparam 场景）
│   └── runs/
│       ├── ours_seed42_stage0/
│       │   ├── checkpoint-best/
│       │   ├── metrics_stage0.json
│       │   └── data_splits/
│       ├── ours_seed42_stage1/
│       └── ours_seed42_stage2/
└── full_experiment_20260609_113000/
    └── runs/
        ├── o_lora_seed42_stage0/
        └── ewc_lora_seed42_stage0/
```

### run_pipeline.py / 手动运行输出

```
outputs/
├── stage_0_seed42/
├── stage_1_seed42/
├── stage_2_seed42/
├── task_lora_stage_0_seed42/
└── ablation_no_evo_stage_0_seed42/
```

---

## 可运行方法清单

### ALL_METHODS（launch_experiments.py 注册）

| # | 方法 | 脚本 | 说明 |
|---|------|------|------|
| 1 | `ours` | `run_stage.py` | 主方法：双分支 LoRA + ToxicPE + 前缀 + 拒识 |
| 2 | `seq_finetune` | `run_baseline.py` | 朴素连续微调 |
| 3 | `task_lora` | `run_baseline.py` | 每阶段独立 LoRA |
| 4 | `task_lora_msp` | `run_baseline.py` | Task-LoRA + MSP 拒识 |
| 5 | `task_lora_adb` | `run_baseline.py` | Task-LoRA + ADB 拒识 |
| 6 | `task_lora_maha` | `run_baseline.py` | Task-LoRA + Mahalanobis 距离 |
| 7 | `o_lora` | `run_baseline.py` | O-LoRA 正交隔离 |
| 8 | `ewc_lora` | `run_baseline.py` | EWC 正则化 + LoRA |
| 9 | `l2p` | `run_baseline.py` | Learning to Prompt（提示池） |
| 10 | `ablation_no_evo` | `run_ablation.py` | 移除 L_evo |
| 11 | `ablation_no_dual` | `run_ablation.py` | 退化为单分支 LoRA |
| 12 | `ablation_no_anchor` | `run_ablation.py` | 随机初始化前缀 |

### run_pipeline.py 特有快捷键

```bash
--method all_baselines   # 跑全部 9 个基线
--method all_ablations   # 跑全部 3 个消融
--method all             # 全部（ours + 基线 + 消融）
```
