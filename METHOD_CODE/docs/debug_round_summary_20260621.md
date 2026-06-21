# FSCIL Toxic 项目阶段性调试总结（截至 2026-06-21）

## 1. 目标与背景

本轮工作的目标，不是单纯追求某一组指标变好，而是系统性回答下面几个问题：

1. 当前 `ours` 的失败主因是什么？
2. 问题出在 checkpoint / consolidation / replay / threshold / KD / frozen plastics / plastic rank 的哪一层？
3. 相比 baseline，`ours` 到底强在哪里、弱在哪里？
4. 下一步应继续调参，还是进入结构性修改？

本总结将本轮探索、试错、对比实验与结论完整归档。

---

## 2. 方法论主线回顾

根据 `methodology.md`，我们的设计有三层：

- **感知层**：`ToxicAwarePE` + `ToxicSemanticPrefix`
  - 对毒性表达变体保持敏感
  - 将文本锚定到已有毒性语义空间

- **记忆层**：`DualBranchLoRALayer` + `SemanticConsolidation`
  - `plastic` 分支快速吸收新类/新变体
  - `stable` 分支沉淀长期毒性语义
  - 通过 `delta_k` 决定 merge 还是 freeze

- **防御层**：`HierarchicalRejectionGate`
  - 基于异常感知和语义距离进行拒识

本轮调试实际上是在检验：

> 当前工程实现，是否真的实现了“新类探索 → 长期沉淀 → 开放环境防御”这一认知记忆闭环。

---

## 3. 本轮探索与试错过程

### 3.1 第一阶段：排查 checkpoint / frozen plastics / consolidation

#### 已验证事项

1. **`frozen_plastics` 继承问题已修复**
   - 通过预创建 `frozen_plastics` 槽位后，历史 plastic checkpoint 能正确加载。
   - 这不是后续异常的主因。

2. **加入了 pre-consolidation 评估**
   - 新增：
     - `metrics_stage1_pre_consolidation.json`
     - `metrics_stage2_pre_consolidation.json`
   - 结果表明：很多失败在 **训练结束后、consolidation 之前就已经发生**。
   - 说明单独 blaming `merge/freeze` 是不准确的。

#### 结论

- checkpoint / frozen plastics 加载不是主问题。
- consolidation 不是唯一问题，甚至多数时候不是首要问题。

---

### 3.2 第二阶段：补指标，验证是不是阈值问题

为避免只看 recall 误判，我们新增了：

- `precision_cls{i}`
- `f1_cls{i}`
- `support_cls{i}`
- `pred_pos_cls{i}`
- `prob_mean_cls{i}`
- `prob_pos_mean_cls{i}`
- `best_threshold_cls{i}`
- `best_precision_cls{i}`
- `best_recall_cls{i}`
- `best_f1_cls{i}`
- `best_pred_pos_cls{i}`

#### 关键发现

多轮实验显示：

- 某些类在默认阈值 `0.5` 下 recall 很高，但 precision 极低；
- 更关键的是，**即便允许 per-class threshold search，`best_f1` 也不高**；
- 很多类出现 `prob_pos_mean ≈ prob_mean`，即正样本和总体样本的预测分布几乎不分离。

#### 结论

- 这不是简单的“阈值没调好”。
- `best_threshold` 也救不起来，说明表示/边界本身有问题。

---

### 3.3 第三阶段：调 KD、balanced BCE、ASL、replay

本轮做过的主要调参与诊断配置包括：

- `full_experiment_kd_tune`
- `full_experiment_stage1_free`
- `full_experiment_frozen_ablate`
- `full_experiment_low_rank`

主要尝试过：

1. **增强 / 削弱 stage2 KD**
2. **冻结旧类 classifier rows**
3. **balanced BCE**
4. **ASL (Asymmetric Loss)**
5. **class-balanced replay / coreset**
6. **显式新类负样本采样**
7. **frozen plastics eval ablation**
8. **plastic rank 减半 (`rp: 8 → 4`)**

下面总结这些尝试的有效性。

---

## 4. 每类尝试的结论

### 4.1 KD（Knowledge Distillation）

#### 观察

- 增强 KD 时，旧类更稳，但容易压制新类。
- 减弱甚至关闭 stage1 KD（`stage1_free`）后，`cls2/cls3` 依然没有明显恢复。

#### 结论

- **KD 不是 stage1 新类学不出来的主因。**
- 它会影响 trade-off，但不是根本瓶颈。

---

### 4.2 balanced BCE

#### 观察

- 较大的 `balanced_bce_max_pos_weight` 会造成“高 recall + 极低 precision”的尾类过预测。
- 降低 pos weight 后，某些类又直接被压死。

#### 结论

- **balanced BCE 只能在过预测和压死之间移动问题，不能根治。**
- 它不是充分解决方案。

---

### 4.3 ASL (Asymmetric Loss)

#### 观察

- ASL 在当前数据分布下没有明显改善。
- 反而更倾向把尾类整体推向全正或无区分区域。

#### 结论

- **ASL 默认参数不适合当前问题。**
- 暂不作为主方向。

---

### 4.4 class-balanced replay / coreset

#### 观察

- 将 replay/coreset 扩展为每旧类均衡采样后：
  - `avg_map` 和某些类的排序质量有所改善；
  - CKA 也更稳定；
  - stage2 的 late class（尤其 `cls4`）变得更可控。

#### 结论

- **均衡 replay 是有效的，应保留。**
- 但它不能单独解决 stage1 类分离问题。

---

### 4.5 显式新类负样本采样

#### 观察

- 对 `cls4`（stage2 severe_toxic）改善明显：precision 和 best F1 明显上升。
- 但若负样本比例过大，会把 stage1 的 `cls2/cls3` 直接压死。

#### 结论

- **显式负样本方向是对的。**
- 但它必须 stage-wise 精细控制，不能一刀切。

---

### 4.6 frozen plastics eval ablation

#### 观察

禁用 frozen plastics 后：

- `cls2/cls3/cls4` 并没有变好；
- 尤其 `cls4` 明显变差；
- overall best F1 普遍下降。

#### 结论

- **frozen plastics 不是误报的主噪声源。**
- 它们确实在提供有效记忆，不应简单移除。

---

### 4.7 low-rank plastic (`rp: 4`)

#### 观察

- `full_experiment_low_rank` 是目前我们最好的 `ours` 版本之一：
  - `cls4 best_f1` 当前最佳；
  - `cls2 best_f1` 也是当前最好；
  - `cls3` 与最好水平接近。

#### 结论

- **降低 plastic 容量有助于减少噪声和过拟合。**
- 这是当前值得保留的方向。

---

## 5. baseline 结果分析

目前真正完整可用于横向比较的，是 `full_baselines_core_20260621_164508` 这一组：

- `seq_finetune`
- `task_lora`
- `ewc_lora`

注意：`outputs/full_baseline_open` 实际不是 open baselines 结果，而是 `full_experiment_stage1_free` 的 `ours` 结果，不能用来做 baseline 对比。

### 5.1 与当前最好 ours (`full_experiment_low_rank`) 的对比

以 stage2 的 `best_f1` 为主：

| 方法 | cls2 best_f1 | cls3 best_f1 | cls4 best_f1 | avg_map |
|---|---:|---:|---:|---:|
| **ours / low_rank** | **0.142** | **0.287** | **0.487** | 0.509 |
| `seq_finetune` | 0.435 | 0.356 | 0.409 | **0.575** |
| `task_lora` | 0.283 | 0.262 | 0.283 | 0.472 |
| `ewc_lora` | 0.313 | 0.281 | 0.285 | 0.487 |

### 5.2 baseline 结论

#### 我们方法的强项

- 在 `cls4 / severe_toxic` 上，当前 `ours` **优于所有已跑 baseline**。
- 说明：
  - 记忆架构、prefix、negative sampling 对 late-stage 新类确实有效。

#### 我们方法的弱项

- 在 `cls2 / threat`、`cls3 / identity_hate` 上，`ours` **明显差于 baseline**，尤其差于 `seq_finetune`。
- 这非常关键，因为它说明：

> 不是数据本身导致 threat / identity_hate 完全不可学；baseline 可以学出来更多，而 ours 当前结构没有做到。

### 5.3 baseline 带来的最终洞察

这一步非常重要，因为它排除了一个长期疑问：

- ~~是不是数据太难，所以谁都学不好 cls2/cls3？~~

答案是否定的。

> **ours 的结构性偏置让它更擅长 late-stage 类（cls4），但不擅长 stage1 的细粒度新类分离。**

---

## 6. 当前最可信的总判断

综合本轮所有试错与 baseline 结果，我认为当前项目的真实状态是：

### 6.1 已被证实的部分

1. `frozen_plastics` 继承逻辑已修复。
2. pre-consolidation 评估机制有效。
3. per-class threshold search 有必要，但不是根因。
4. class-balanced replay 有价值。
5. 显式新类负样本采样对 `cls4` 有显著帮助。
6. `frozen_plastics` 不是主要噪声源。
7. lower-rank plastic 是目前最好的 `ours` 方向。

### 6.2 已被排除的方向

1. ~~checkpoint 加载问题~~
2. ~~只怪 consolidation~~
3. ~~单纯阈值问题~~
4. ~~KD 是根因~~
5. ~~frozen plastics 全局噪声是主因~~
6. ~~数据本身导致 cls2/cls3 不可学~~

### 6.3 当前剩下的真正问题

> **stage1 的新类分离能力不足。**

更具体地说：

- `threat / identity_hate` 与 `obscene / insult` 高度共现；
- current implementation 更倾向把它们吸收到已有毒性语义核心中；
- 没有形成显式的“新类去纠缠 / 分离”机制；
- 后续所有阶段都在继承这个问题。

换句话说：

> 当前瓶颈已经从 “memory preservation” 转向 “new-class semantic separation”。

---

## 7. 对方法论的再理解

这一点很重要，因为我们不是要推翻最初的方法论，而是要重新理解它真正缺了什么。

### 原方法论没有被证伪

- 感知层（异常感知 + 语义锚定）仍然有意义；
- 记忆层（stable/plastic + consolidation）仍然对 late class 有帮助；
- 防御层（拒识）尚未成为当前主要矛盾；

### 但当前实现缺少一个关键环节

在“新概念进入长期记忆之前”，模型需要先完成：

> **把新类从旧类语义团中分离出来**

如果没有这一步，stable/plastic 再优雅，也只是在对一个混杂的语义表示做记忆保存。

### 因此，下一步改动最自然的理论解释是：

不是推翻现有三层，而是补一个机制：

> **新类语义去纠缠 / separation objective**

它应当位于：

- 感知层之后
- 记忆沉淀之前

即：

**“锚定 → 分离 → 沉淀”**

这比“锚定后直接沉淀”更符合认知叙事。

---

## 8. 后续建议

### 8.1 近期工程建议（高优先级）

1. **继续以 `full_experiment_low_rank` 作为当前 `ours` 工作版**
   - 它是目前最佳折中版本。
   - 可作为后续方法改进和 baseline 对照的基线。

2. **继续补完 open baselines 结果**
   - `task_lora_msp`
   - `task_lora_maha`
   - `task_lora_adb`
   - `o_lora`
   - `l2p`

这一步不是因为一定能改结论，而是为了让最终论文对比更完整。

### 8.2 下一阶段结构修改（最高优先级）

建议开始设计并实现一个 **stage1 新类分离目标**，形式可以是：

1. **contrastive loss**
   - 新类正样本彼此拉近；
   - 与旧类 hard negatives 拉远；

2. **prototype separation loss**
   - 给每个新类单独 prototype；
   - 显式增大新类到 base 类 prototype 的 margin；

3. **class-specific margin / metric objective**
   - 只对 stage1 新类 logits 或 CLS 表征施加；
   - 不破坏现有 stable/plastic 架构；

### 8.3 不建议继续的方向

短期内不建议再做：

- 单纯调 `lambda_kd`
- 单纯调 `balanced_bce_max_pos_weight`
- 再尝试 ASL 默认参数
- 简单关掉 frozen plastics

这些方向本轮已经充分探索，收益递减明显。

---

## 9. 推荐下一步执行顺序

### 路线 A：论文对照优先
1. 跑完 open baselines
2. 以 `full_experiment_low_rank` 作为当前 `ours`
3. 整理 baseline 对比表
4. 再决定是否上 separation objective

### 路线 B：方法改进优先（推荐）
1. 保留 `full_experiment_low_rank` 作为 `ours_base`
2. 开始实现一个最小侵入的 **stage1 separation loss**
3. 先只验证：
   - `cls2 best_f1`
   - `cls3 best_f1`
   - `cls4 best_f1` 是否保持
4. 若有效，再扩展到完整实验

我更推荐路线 B，因为当前阻塞点已经足够明确。

---

## 10. 结语

本轮探索的最大价值，不是把指标直接调到最好，而是把问题空间大幅收缩了。

我们现在已经知道：

- 当前 `ours` 不是整体无效；
- 它在 late-stage 新类（尤其 `cls4`）上确实有优势；
- 但它把 stage1 新类的分离问题处理得不够好；
- baseline 已经证明这是 `ours` 的结构性短板，而不是数据本身的宿命；
- 下一步最值得做的，不是继续调 memory 细节，而是补上新类分离机制。

> 结论一句话：
>
> **当前框架的主要短板不是“忘记”，而是“没先学会区分”，所以后续工作应从记忆保持调参转向新类语义分离建模。**
