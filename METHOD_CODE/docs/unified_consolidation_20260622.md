# 架构收束与统一基线报告（截至 2026-06-22）

## 1. 背景

本报告衔接 `debug_round_summary_20260621.md` 的结论（"当前瓶颈已从 memory preservation 转向 new-class semantic separation"），记录后续完整的架构审视、大规模 ablation 实验、以及最终的模型收束过程。

**TL;DR**：经过 20+ 次独立 experiment launch，我们证实了多个方向的"单点有效性"和"组合失效"模式，最终将所有保留机制收敛为一份精简的 unified 配置。

---

## 2. 实验全景

### 2.1 实验路线图

```
debug_round (06-21)
  ├── separation_loss (表征级分离) → 失败，CKA 崩塌
  ├── proto_gated (原型初始化+弱锚定) → 失败，全阳性崩溃
  │
12-launch batch (06-21 夜)
  ├── G1 边界学习：hardneg_pc, hardneg_refined
  │   └── 结论：对 cls3 有微弱帮助，整体 map 拖累
  ├── G2 损失/训练：margin, delaykd, lrboost
  │   ├── margin → 失败
  │   ├── delaykd → ✅ 多轮最佳单一改动 (avg_map=0.488)
  │   └── lrboost → 稳定但无益
  ├── G3 自由度：stable2, noprefix_diag
  │   ├── stable2 → 小幅正收益
  │   └── noprefix_diag → 证实 prefix 压制 cls3
  └── G4 组合/容量：combo_hm/hd/hs/triple/quad, rp8
      ├── 所有 combo → 互相抵消或退化
      └── rp8 → 无效
  │
3-launch refinement (06-22 早)
  ├── delaykd_alpha03 → 0.487 (alpha=0.3 有效但不稳定)
  ├── delaykd_stable2 → 0.487 (stable2 有效)
  └── delaykd_hardneg_refined → 失败 (0.455)
  │
unified 收束 (06-22 午)
  ├── round 1: proto_init 默认开 → 全阳性崩溃 (0.385)
  ├── round 2: stable_partial+alpha=0.3 恶性协同 → CKA=0.18
  ├── round 3: seed 非确定性暴露 → 两台主机结果分歧
  └── round 4: 最终修复 → 待验证
```

### 2.2 关键指标对比（重要里程碑）

| 实验 | stage2 avg_map | cls2 best_f1 | cls3 best_f1 | cls4 best_f1 | 备注 |
|---|---:|---:|---:|---:|---|
| low_rank (历史基线) | 0.509 | 0.142 | 0.287 | 0.487 | rp=4 |
| delaykd | 0.488 | 0.089 | 0.254 | 0.457 | 最佳单一改动 |
| delaykd_alpha03 | 0.487 | 0.090 | 0.260 | 0.455 | alpha=0.3 |
| delaykd_stable2 | 0.487 | 0.090 | 0.259 | 0.454 | top-2 stable |
| sep_aggressive | 0.472 | 0.113 | 0.241 | 0.419 | 表征级分离 |
| proto_gated | 0.401 | 0.083 | 0.235 | 0.309 | 全阳性崩溃 |

---

## 3. 关键发现与判断

### 3.1 已被证实的有效机制

1. **延迟 KD（`kd_delay_ratio=0.4`）**
   - 多轮实验中最强的单一提升。
   - 旧类蒸馏在 stage1 前半程压制新类学习，延迟 40% 步数给新类起步空间。
   - **保留位置**：`trainers/incremental_trainer.py` 默认值设为 0.4。

2. **stage1 prefix 弱化锚定（`alpha=0.3~0.4`）**
   - 完全关闭 prefix（`noprefix_diag`）证明了 prefix 确实压制 cls3。
   - 但 alpha=0.3 处于稳定性悬崖边缘，不同 seed 下 CKA 从 0.97 漂到 0.33。
   - 最终采用 alpha=0.4 作为稳定分界点。
   - **保留位置**：`configs/unified.yaml` 的 `prefix.stage_alpha`。

3. **top-2 stable 分支解冻（`stable_partial.unfreeze_top_layers=2`）**
   - 单独使用时有效，CKA=0.90。
   - 但与 alpha=0.3 同时启用时产生恶性协同（CKA→0.18），因此**默认关闭**，通过显式 config 开关启用。
   - **保留方式**：代码已内化，通过 `stable_partial.enable=true` 显式启用。

4. **冻结旧类分类器行**
   - 防止新类梯度覆盖旧类 logit。
   - **保留位置**：`freeze_old_classifier: true`，默认开启。

5. **温和 balanced BCE（`max_pos_weight=3.0`）**
   - 过大的 pos_weight 造成高 recall+低 precision 的尾类过预测。
   - 收敛值 3.0 在各阶段通用。
   - **保留位置**：`unified_stages.yaml` 各阶段 loss_weights。

6. **类均衡 coreset replay**
   - 多轮确认有效，CKA 更稳定。
   - **保留位置**：`coreset_size_per_class: 64`。

7. **stage2 较高负样本比例**
   - stage2（单类 severe_toxic）需要更多负样本平衡。
   - `new_class_negative_ratio: 2.0`。

### 3.2 已被淘汰的机制

| 机制 | 淘汰原因 |
|------|---------|
| `NewClassSeparationLoss` | 表征级分离破坏稳定层，CKA 崩塌，对 cls2 无帮助 |
| `ClassMarginLoss` | 推高新类 logit 但不建立边界，伤害 cls4 |
| `ASL (Asymmetric Loss)` | 默认参数不适配当前数据分布 |
| `proto_init`（原型分类头初始化） | 导致全阳性崩溃，分离使用无益 |
| `hard negative replay`（全局/类特异/refined） | 对 cls3 有极微弱帮助，整体 map 拖累 |
| `layerwise prefix alpha` | 不如简单的 stage-level 弱化有效 |
| `plastic rank=8` | 增加参数但无益，甚至更差 |
| `newclass_lr_boost` | 稳定但无益，且实现 buggy |

### 3.3 组合失效模式

实验中多次观察到：**两个单独有效的改动，组合后反而退化**。已知的恶性协同包括：

- `stable_partial` + `alpha=0.3`：弱锚定 + 高层稳定分支解冻 → 小样本新类表征过度漂移（CKA→0.18）
- `proto_init` + `delaykd`：原型初值推高所有新类 logit + 延迟 KD 延缓旧类校准 → 全阳性崩溃

### 3.4 训练可复现性修复

发现 `run_stage.py:243` 仅设置了 `torch.manual_seed()`，缺少 GPU 和 cuDNN 确定性配置，导致两台不同 GPU 架构的主机产出不同结果。已修复为：

```python
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 4. 统一基线配置 (Unified)

### 4.1 架构组成

```
输入文本
  ├── ToxicAwarePE（保留，领域特化）
  ├── Prefix Anchor（base alpha=0.7, stage1 alpha=0.4, stage2 alpha=0.6）
  ├── RoBERTa + DualBranchLoRA (rs=8, rp=4)
  │     ├── stable（base 训练后冻结）
  │     └── plastic（每阶段独立）
  ├── Classifier Head（标准 Linear + BCE，旧行冻结）
  ├── Rejection Gate（保留，防御层）
  │
  └── 损失函数
        ├── BCE（balanced, max_pos_weight=3.0）
        ├── KD（lambda=0.5/1.5, 前 40% 步延迟）
        ├── stable/plastic reg（lambda_sp=1e-3）
        ├── orthogonality（eta=1e-4, 仅 stage>0）
        └── evo/open（仅 base stage）
```

### 4.2 配置文件

| 文件 | 用途 |
|------|------|
| `configs/unified.yaml` | 模型结构、prefix、训练参数 |
| `configs/unified_stages.yaml` | 阶段重载（学习率、epoch、loss weights） |
| `scripts/run_unified.py` | 三阶段流水线入口 |

### 4.3 运行方式

```bash
python scripts/launch_experiments.py --scenario unified
# 或
python scripts/run_unified.py --seed 42
```

### 4.4 与旧配置的关系

`unified` 等价于历史 `delaykd_alpha03` + seed 确定性 + alpha 收敛到 0.4 的安全区。代码层面收敛了：

- `kd_delay_ratio=0.4`：写在 trainer 默认值里，不再依赖 config 开关
- `stable_partial`：代码已支持，默认关闭（需显式 `enable: true`）
- `proto_init`：代码已支持，默认关闭（需显式 `enable: true`）

---

## 5. 清理内容

### 5.1 删除的 loss 文件

- `losses/separation_loss.py`
- `losses/class_margin_loss.py`

### 5.2 从 launch_experiments 移出的场景

| 移除的场景 | 原因 |
|------|------|
| `full_experiment_low_rank_sep*` | loss 文件已删除 |
| `full_experiment_low_rank_proto_gated*` | 实验路线已废弃 |
| `full_experiment_low_rank_hardneg_pc` | 实验路线已废弃 |
| `full_experiment_low_rank_margin` | loss 文件已删除 |
| `full_experiment_low_rank_delaykd` | 内化到统一基线 |
| `full_experiment_low_rank_lrboost` | buggy，已删除 |
| `full_experiment_low_rank_stable2` | 通过 stable_partial 开关内化 |
| `full_experiment_low_rank_noprefix_diag` | 实验路线已废弃 |
| `full_experiment_low_rank_rp8` | 实验路线已废弃 |
| `full_experiment_low_rank_combo_*` | 实验路线已废弃 |
| `full_experiment_low_rank_delaykd_*` | 内化到统一基线 |
| `full_experiment_kd_tune/stage1_free/frozen_ablate` | 早期实验已废弃 |

### 5.3 保留的场景

| 场景 | 用途 |
|------|------|
| `quick_dev` | 快速冒烟测试 |
| `subset_hparam` | 子集超参搜索 |
| `full_experiment` | 论文完整实验（含 baselines） |
| `full_experiment_low_rank` | 历史参考基线 |
| `unified` | **新统一基线** |
| `full_baselines_core` | 核心 baselines |
| `full_baselines_open` | 开集 baselines |

---

## 6. 未解决的问题

尽管经过多轮 ablation 和收敛，以下问题仍然是开放挑战：

1. **cls2 (threat) 在所有配置下均未能有效分离**
   - `best_f1` 始终徘徊在 0.08~0.11，且仅出现在 threshold sweep 中，默认阈值下恒为 0。
   - threat 与 obscene/insult 的三元共现结构可能是根本性限制。

2. **延迟 KD 的边际效应见顶**
   - `delaykd` 之后的所有组合改进（+alpha、+stable2）都没能在 `delaykd` 基础上再次突破 0.488 的天花板。

3. **表征稳定性和新类分化的 trade-off 尚未根本解决**
   - alpha=0.3 骑在稳定性悬崖上，alpha=0.4 收敛更安全但可能放弃部分 cls3 收益。
   - 这本质上是因为 prefix 机制默认是"吸附器"而非"分化器"。

4. **分类器仍是标准 Linear Head**
   - 前面所有模块（LoRA、prefix、ToxicPE）在精心塑造表示空间，但决策边界仍由普通线性层完成。
   - 更结构化的分类头设计（如 prototype-metric 混合、logit calibration）是潜在的下一方向。

---

## 7. 下一步建议

| 优先级 | 方向 | 说明 |
|---|---|---|
| P0 | 用 unified 稳定跑 multi-seed | 验证新基线的跨 seed 稳定性 |
| P0 | 补全 baseline 对比表 | 以 unified 为 ours，对比 seq_finetune/task_lora/ewc_lora |
| P1 | 论文写作 | 整理 methodology → experiment → ablation 的主线叙事 |
| P2 | 分类头 upgrade | 探索 prototype-calibrated head 或 logit calibration |
| P3 | threat 专向改进 | 考虑三元共现结构是否需要数据层面的专项处理 |

---

## 8. 结语

本轮工作从 6 月 21 日的"找出真正瓶颈"出发，经过了表征 separation、原型初始化、边界学习、训练策略四大方向的高密度实验探索，最终把模型从"外挂堆叠"收敛为一份精简的 unified 基线。核心收获是：

- **delay KD + 弱化 stage1 prefix anchor** 是最有效的两条改动
- **绝大多数 ablation（separation/margin/hardneg/proto_init/rp8/layerwise）被证实无效**
- **训练可复现性**是一个容易被忽视但影响判断精确度的基础问题
- 当前 unified 基线预期 `avg_map ≈ 0.48`，已从原始 `low_rank` 的 0.509 略有下降，但代码量减少了约 30%，可维护性和可解释性显著提升
