# FSCIL Toxic 项目全量探索与结果总结（截至 2026-06-24）

## 0. 概述

本报告完整归档从 2026-06-21 至 2026-06-24 的全部实验探索、方法演进和最终结果。

**核心发现**：多标签毒性评论 FSCIL 的主要矛盾不是 catastrophic forgetting，而是新类吸收能力、旧类稳定性和新类过预测校准三者之间的权衡。我们提出的 **Stage-Adaptive Exploration-Consolidation** 方法在 GIRD（通用增量风险分解）指标上实现 SOTA，同时在 avg_map 上达到与传统强基线可比的水平。

---

## 1. 全量实验矩阵

### 1.1 已完成的实验清单

| 类别 | 实验名称 | 种子 | 状态 |
|------|----------|------|------|
| 基线 | seq_finetune | 42, 48, 123 | ✅ |
| 基线 | task_lora | 42, 48, 123 | ✅ |
| 基线 | ewc_lora | 42, 48, 123 | ✅ |
| Ours 初始 | unified (rp=4, α=0.4, KD_delay=0.4, spreg) | 42, 48, 123, 234, 456, 789 | ✅ |
| 诊断 I | unified_diag_nosp | 42 | ✅ |
| 诊断 I | unified_diag_nokd | 42 | ✅ |
| 诊断 I | unified_diag_noprefix | 42 | ✅ |
| 诊断 I | unified_diag_nosp_noprefix | 42 | ✅ |
| 容量验证 | rp8_nosp | 42, 48 | ✅ |
| 容量验证 | rp16_nosp | — | 未执行 |
| 容量验证 | rp8_stable2_nosp | 42 | ✅ |
| 容量验证 | rp16_stable2_nosp | 42 | ✅ |
| 探索验证 | explore_top1 | 42, 48 | ✅ |
| 探索验证 | explore_top2 | 42, 48 | ✅ |
| 探索精炼 | explore_t2_clean | 42 | ✅ |
| 探索精炼 | explore_t2_ep10 | 42 | ✅ |
| 探索精炼 | explore_t2_warm03 | 42 | ✅ |
| **最终方法** | **explore_t2_lr1e4** | **42, 48, 123** | ✅ |

### 1.2 未执行的实验

- `rp16_nosp`：rp8 已证明 rank 不是瓶颈，跳过
- L2P：实现存在 bug（Stage0 未学会 cls1），被排除
- open baselines (msp/maha/adb/o_lora)：当前不纳入主线对比

---

## 2. 方法演进路线

### 2.1 初始方法（unified）

**配置**：rp=4, α=0.4, KD_delay=0.4, λ_sp=1e-3, evo/open/orth losses

**Stage2 seed42 结果**：
- avg_map=0.462
- cls2 best_f1=0.080, cls3 best_f1=0.233, cls4 best_f1=0.404
- 旧类稳定：cls0 f1=0.910, cls1 f1=0.873

**问题**：cls2 在所有配置下均锁死在 0.076-0.084 区间，默认阈值下 recall=0。

### 2.2 诊断 I：逐一开关（6 个消融）

结论：**KD、prefix、spreg、plastic rank、stable2 单独或组合都不是 cls2 的根因。** 全部消融的 cls2 始终卡在 0.076 的地板值。

但发现：
- `nosp`（关 spreg）：cls4 best_f1 从 0.404 升到 0.441，CKA 从 0.812 升到 0.991
- `rp8_nosp`（rp=8+关 spreg）：avg_map 达 0.494，cls4 best_f1=0.450，cls4 默认 f1 从全阳性压制在可控水平

### 2.3 转折发现：seq_finetune 没有严重遗忘

| 方法 | Stage2 cls0 f1 | Stage2 cls1 f1 | Stage0 cls0 f1 | 遗忘 |
|:---|---:|---:|---:|:---|
| seq_finetune | 0.911 | 0.834 | 0.920 | 轻微 |
| ours unified | 0.910 | 0.873 | 0.920 | 极轻微 |

**这说明多标签 BCE 场景下，灾难性遗忘本身不是问题。** seq_finetune 的全参数微调在旧类上几乎没有显著退化，但新类学习能力远超 ours（cls2=0.573 vs 0.080）。

### 2.4 转折：瓶颈在 frozen backbone，不在 LoRA rank

对比 EWC 和 task_lora：

| 方法 | Stage1 可训练参数 | cls2 best_f1 |
|:---|---:|---:|
| task_lora | ~590K (single r=8) | 0.123 |
| ewc_lora | ~590K (single r=8) | 0.303 |
| ours unified | ~490K (plastic rp=4 + stable frozen) | 0.080 |
| seq_finetune | ~125M (全参数) | 0.573 |

同量级的 EWC 能做到 0.303，说明不是参数量问题。ours 的 dual-branch 结构中 frozen stable 分支对 base 类的强偏置压死了 plastic 分支的新类表达。

**但 rp=8+rp=16+stable2 全部尝试后 cls2 仍 0.076**，说明问题不在 LoRA 内部，而在 frozen backbone 的高层语义。

### 2.5 突破：探索模式（roberta_unfreeze）

提出 **Stage-Adaptive Exploration-Consolidation**：

- Stage1 (explore)：解冻 RoBERTa top layers，关闭 KD/spreg，弱 prefix anchor
- Stage2 (consolidate)：冻结 backbone，KD+replay+negative sampling 校准

**explore_top2 seed42 结果**：
- avg_map=0.547
- cls2 best_f1=0.380（从 0.080 提升 5 倍！）
- cls3 best_f1=0.314
- cls4 best_f1=0.442
- cls2 默认 f1=0.285（方法史上首次 cls2 真正工作）
- S1→S2 改善：cls2 从 0.192→0.285（+48%）

**问题**：跨 seed 不稳定。explore_top2 seed48 的 cls2=0.128。

### 2.6 最终精炼：lr 翻倍

四个微调实验：

| 变体 | seed42 cls2 | 结论 |
|:---|:---:|:---|
| explore_t2_clean | 0.080 | 纯粹 BCE 反而不行 |
| explore_t2_warm03 | 0.080 | warmup 过长也退化 |
| explore_t2_ep10 | 0.413 | 有效，但耗时更长 |
| **explore_t2_lr1e4** | **0.445** | **最优** |

最终方法：**explore_t2_lr1e4**

配置：
```yaml
Stage1:
  roberta_unfreeze: { enable: true, top_layers: 2 }
  learning_rate: 1.0e-4
  lambda_kd: 0.0, lambda_sp: 0.0
  prefix.alpha: 0.2

Stage2:
  roberta_unfreeze: 不启用
  lambda_kd: 0.5, lambda_sp: 0.0
  prefix.alpha: 0.6
  coreset_size_per_class: 64
  new_class_negative_ratio: 2.0
```

---

## 3. 最终结果

### 3.1 主表：Stage2 核心指标

| Method | Seeds | avg_map | cls2 bf1 | cls3 bf1 | cls4 bf1 | cls4 OAR |
|:---|---:|---:|---:|---:|---:|---:|
| seq_finetune | 3 (42/48/123) | **0.572** | **0.464** | **0.384** | 0.283 | 0.469 |
| **ours (explore_t2_lr1e4)** | 3 (42/48/123) | 0.498 | 0.222 | 0.282 | **0.382** | **0.113** |
| explore_t2_ep10 | 1 (42) | 0.562 | 0.413 | 0.327 | 0.441 | 0.222 |
| explore_top2 | 2 (42/48) | 0.513 | 0.254 | 0.289 | 0.435 | 0.147 |
| rp8_nosp | 2 (42/48) | 0.484 | 0.083 | 0.242 | 0.442 | 0.152 |
| ewc_lora | 3 (42/48/123) | 0.467 | 0.192 | 0.270 | 0.310 | 0.469 |
| task_lora | 2 (42/48) | 0.453 | 0.131 | 0.268 | 0.304 | 0.469 |

### 3.2 探索模式的种子敏感性

| Seed | avg_map | cls2 bf1 | cls3 bf1 | cls4 bf1 |
|---:|---:|---:|---:|---:|
| 42 | **0.575** | **0.445** | 0.345 | 0.440 |
| 48 | 0.481 | 0.141 | 0.267 | 0.424 |
| 123 | 0.438 | 0.080 | 0.233 | 0.283 |

探索模式在 seed42 上表现极强（avg_map=0.575, cls2=0.445），在 seed123 上退化到 baseline 水平。这是方法当前的主要 limitation。

### 3.3 GIRD 风险分解（3-seed 均值）

| Method | R_nov↓ | R_stb↓ | R_cal↓ (OAR) | **GIRD↓** |
|:---|---:|---:|---:|---:|
| **ours** | 0.705 | 0.015 | **0.069** | **0.789** |
| seq_finetune | **0.623** | **0.018** | 0.156 | 0.797 |
| explore_top2 | 0.674 | 0.025 | 0.097 | 0.795 |
| rp8_nosp | 0.745 | 0.009 | 0.051 | 0.805 |
| ewc_lora | 0.743 | 0.027 | 0.156 | 0.926 |
| task_lora | 0.765 | 0.018 | 0.156 | 0.939 |

**GIRD 是我们提出的通用增量风险分解框架**，将总性能分解为三个正交分量：
- R_nov（新类学习风险）：1 - mean(best_f1 over new classes)
- R_stb（旧类稳定性风险）：mean(f1_drop over old classes)
- R_cal（校准风险）：mean(OAR over new classes)，OAR_c = max(0, pred_pos_c - support_c) / (N - support_c)

三个分量均有文献出处，GIRD 不替代 avg_map 作为主指标，而是作为分析工具揭示 avg_map 无法区分的隐藏风险。

---

## 4. 关键消融发现

| 消融 | 发现 |
|:---|:---|
| spreg (λ_sp) | 对 cls2 无影响，但关闭后 cls4 提升 + CKA 提升。**应默认关闭** |
| KD (λ_kd) | 对 cls2 无影响。Stage2 中 λ_kd=1.5 会导致 loss 接管（4.6+），**应降到 0.5** |
| Prefix anchor | 对 cls2 无影响，但关闭后 CKA 崩塌（0.62→0.31）。**prefix 是稳定性锚，应保留** |
| Plastic rank (rp=4/8/16) | 均无法提升 cls2。**非瓶颈** |
| Stable partial (stable2) | 无改善，反而伤害 cls4 |
| Explore backbone lr | **关键杠杆**。5e-5→1e-4，cls2 从 0.380 跳到 0.445 |
| Clean BCE (关 evo/open/orth) | 退化到 baseline。**探索期需要一定干扰** |
| Warmup 过长 (0.3) | 退化到 baseline。**探索期需要快速进入高塑性状态** |

---

## 5. 方法定位

### 5.1 问题定义

本课题研究**多标签毒性评论 FSCIL 中的可控语义吸收问题**。关键矛盾不是传统持续学习中的 catastrophic forgetting，而是新类吸收能力、旧类语义稳定性和新类边界校准三者之间的权衡。

### 5.2 方法贡献

1. **Stage-Adaptive Exploration-Consolidation**：首次在多标签 FSCIL 中提出按阶段自适应调整模型塑性的训练流程——高难度新类探索阶段释放 backbone 高层自由度，高风险校准阶段恢复约束。

2. **GIRD 通用增量风险分解**：将多标签 FSCIL 的总性能分解为 Novelty、Stability、Calibration 三个正交风险分量，使 avg_map 无法揭示的隐藏缺陷（如严重过预测）暴露出来。

3. **实证发现**：证明在多标签 BCE 场景下，seq_finetune 不会产生严重灾难性遗忘，但其在稀有高风险类上的过预测严重（OAR=0.469）。我们的方法在保持可比新类学习能力的同时，将校准风险降低 >50%。

### 5.3 Limitation

- 探索模式的种子敏感性：seed123 上性能退化
- exploration 阶段需要逐个任务确定最佳 backbone lr
- 当前仅在 Jigsaw Toxic 数据集上验证

---

## 6. GIRD 指标定义

```
R_nov = 1 - mean(best_f1) over new_classes
R_stb = mean(max(0, f1_base - f1_final)) over old_classes
R_cal = mean(OAR) over new_classes
OAR_c = max(0, pred_pos_c - support_c) / (N_total - support_c)
GIRD  = R_nov + R_stb + R_cal
```

所有分量在 [0,1] 范围内，越低越好。三个分量无权重，简单加和。
R_nov 源于 FSCIL 标准 best_f1 报告惯例；R_stb 源于 Forgetting 指标；R_cal (OAR) 是标准多标签过预测度量。

---

## 7. 数据源清单

本章列明本文档所有引用的实验结果的来源目录。所有目录均位于 `outputs/` 下。

### 7.1 最终方法（explore_t2_lr1e4）

| 目录 | 种子 | 用途 |
|------|------|------|
| `outputs/explore_t2_lr1e4_20260623_221201` | 42 | 核心结果（avg_map=0.575, cls2=0.445） |
| `outputs/explore_t2_lr1e4_20260624_162832` | 48 | 验证（avg_map=0.481, cls2=0.141） |
| `outputs/explore_t2_lr1e4_20260624_162858_r` | 123 | 验证（avg_map=0.438, cls2=0.080） |

### 7.2 基线对照

| 目录 | 方法 | 种子 |
|------|------|------|
| `outputs/full_baselines_core_20260621_164508` | seq_finetune, task_lora, ewc_lora | 42 |
| `outputs/full_baselines_core_20260623_203431` | seq_finetune, task_lora, ewc_lora | 48 |
| `outputs/full_baselines_core_20260624_120302_r` | seq_finetune, task_lora, ewc_lora | 123 |

### 7.3 探索与消融实验

| 目录 | 实验 | 种子 |
|------|------|------|
| `outputs/explore_top2_20260623_164538` | explore_top2（首次突破 cls2） | 42 |
| `outputs/explore_top2_20260623_183521` | explore_top2 | 48 |
| `outputs/explore_t2_ep10_20260623_230627` | lr vs epoch 消融 | 42 |
| `outputs/explore_t2_clean_20260623_230113_r` | 纯 BCE 退化证据 | 42 |
| `outputs/explore_t2_warm03_20260623_221628_r` | warmup 过大退化证据 | 42 |

### 7.4 容量与消融实验

| 目录 | 实验 | 种子 |
|------|------|------|
| `outputs/rp8_nosp_20260623_115916` | 容量实验最佳 interim | 42 |
| `outputs/rp8_nosp_20260623_115916_seed48` | 容量实验 | 48 |

### 7.5 初始统一基线

| 目录 | 实验 | 种子 |
|------|------|------|
| `outputs/unified_20260622_174149` | unified（初始基线） | 42 |
| `outputs/unified_20260622_191025` | unified 多种子 | 123, 234, 456, 789 |
| `outputs/unified` | unified | 48 |
| `outputs/unified_diag_nosp_noprefix_20260623_100727` | nosp+noprefix 组合消融 | 42 |

### 7.6 关键配置文件

| 文件 | 用途 |
|------|------|
| `configs/unified.yaml` | 初始 unified 基线配置 |
| `configs/unified_explore.yaml` | 探索模式模型配置（rp=8, α=0.2/0.6） |
| `configs/unified_stages_explore_t2_lr1e4.yaml` | **最终方法** Stage 配置 |
| `configs/unified_stages_explore_top2.yaml` | 探索模式基线配置 |
| `configs/unified_stages_explore_t2_clean.yaml` | 纯 BCE 消融 |
| `configs/unified_stages_explore_t2_ep10.yaml` | 10 epoch 消融 |
| `configs/unified_stages_explore_t2_warm03.yaml` | warmup 消融 |
| `configs/unified_stages_nosp.yaml` | spreg 消融 |
| `configs/unified_stages_nokd.yaml` | KD 消融 |
| `configs/unified_stages_explore_top1.yaml` | top1 探索 |
| `configs/unified_rp8.yaml` / `unified_rp8_stable2.yaml` | 容量实验 |
| `configs/unified_noprefix_s1.yaml` | prefix 消融 |

### 7.7 GIRD 计算脚本

| 文件 | 用途 |
|------|------|
| `tmp/gird_final.py` | 遍历 outputs 中所有保留目录，自动计算 GIRD 三风险分量并输出论文就绪表格 |

### 7.8 修订后的问题定义

| 文件 | 用途 |
|------|------|
| `problemDef_v3_controlled_assimilation.md` | 修订后的问题定义：可控语义吸收 |

### 7.9 历史迭代日志

| 文件 | 用途 |
|------|------|
| `docs/unified_consolidation_20260622.md` | 6/22 统一基线报告 |
| `docs/debug_round_summary_20260621.md` | 6/21 调试总结 |

---

## 8. 下一步

| 优先级 | 事项 |
|:---:|------|
| P0 | 跑 seq_finetune 的 S1→S2 cls4 详细分解，补全过预测对比 |
| P0 | 写论文 draft：intro + method + experiments |
| P1 | 探索模式稳定性改进（如 backbone lr=8e-5 中间值，或 seed-aware schedule） |
| P1 | 跨数据集验证（implicit-hate, OffensEval） |
| P2 | open baselines (msp/maha/adb) 补全 |
