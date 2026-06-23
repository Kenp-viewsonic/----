# 可控语义吸收：少样本增量毒性评论分类问题定义（v3）

## 0. 核心定位

本课题不再将主要矛盾定义为传统持续学习中的 **catastrophic forgetting**，而聚焦于多标签毒性评论 FSCIL 场景中更突出的现实问题：

> **少样本新毒性类别到来时，模型既要快速吸收新类语义，又要避免将大量旧类样本、边界样本或未知变体误吸收到新类中。**

我们将该问题定义为：

> **Few-shot Class-Incremental Toxic Comment Classification with Controlled Semantic Assimilation**  
> **面向少样本类别增量毒性评论分类的可控语义吸收问题。**

其中“可控语义吸收”强调两点：

1. **Assimilation / 吸收**：模型必须能从极少样本中学习新毒性类别，例如 `threat`、`identity_hate`、`severe_toxic`；
2. **Controlled / 可控**：模型不能因为新类样本少、类别共现强、语义边界模糊，而把大量非目标样本错误扩张到新类决策区域中。

---

## 1. 背景与动机

毒性评论分类通常是多标签任务。一个评论可以同时包含 `toxic`、`obscene`、`insult`、`threat`、`identity_hate`、`severe_toxic` 等多个标签。与通用单标签 FSCIL 不同，多标签毒性评论具有三个特殊性质：

### 1.1 新类和旧类高度共现

例如 `threat` 往往与 `toxic`、`insult`、`obscene` 共现，`identity_hate` 也常与一般攻击性表达混合出现。这导致新类不是远离旧类的新语义簇，而是嵌在旧毒性语义团中的细粒度子区域。

因此，难点不只是“记住旧类”，而是：

> 如何从已有毒性语义团中分离并吸收一个新的细粒度风险类别。

### 1.2 简单微调不一定严重遗忘，但容易边界失控

在当前 Jigsaw 多标签 FSCIL 实验中，Sequential Fine-tuning 在旧类 `obscene/insult` 上并未出现严重灾难性遗忘，说明传统 CL 叙事中的“遗忘”并不是该任务的唯一主要矛盾。

但简单微调存在另一类风险：

- 对新类学习较快；
- 但容易将新类决策边界扩得过大；
- 在少样本、高共现、长尾类别上产生大量过预测。

这在内容审核系统中尤其危险：高风险标签的全阳性倾向会导致误伤、审核资源浪费和策略报警泛滥。

### 1.3 过度记忆保护会抑制新类吸收

相反，过强的稳定性约束、冻结机制或旧类语义锚定虽然能保持旧类稳定，却可能使模型无法学习 `threat` 这类细粒度新类。

因此，本课题关注的不是单纯的：

> 新类学习 vs 旧类遗忘

而是更符合毒性多标签 FSCIL 的三元权衡：

> **新类吸收能力 vs 旧类语义稳定性 vs 新类边界可控性**

---

## 2. 研究目标

给定一个按阶段到来的毒性评论类别流：

- Stage 0：基础毒性类，例如 `obscene`、`insult`；
- Stage 1：新增细粒度风险类，例如 `threat`、`identity_hate`；
- Stage 2：新增高风险稀有类，例如 `severe_toxic`；

每个增量阶段仅提供少量新类标注样本。模型需要在每个阶段完成：

1. **旧类保持**：已见类别的判别能力不显著退化；
2. **新类吸收**：新类别在排序指标和 F1 指标上形成有效判别边界；
3. **边界控制**：避免新类预测区域过度扩张，减少全阳性式过预测；
4. **未知拒识**：对不确定样本、未知毒性变体或边界样本输出拒识/复核信号；
5. **表面形式鲁棒**：对毒性词变体、规避拼写、黑话表达保持一定识别能力。

本课题的核心目标是：

> 在少样本增量毒性分类中，实现对新毒性类别的可控吸收，使模型既能学习新类，又能避免将旧类和未知样本大规模误吸收到新类中。

---

## 3. 核心科学问题

### Q1. 新类吸收：如何在低样本、高共现条件下学习细粒度毒性类别？

`threat`、`identity_hate` 等类别常被已有攻击性语义覆盖。模型需要从旧类语义团中挖掘新类的细粒度判别特征。

关键问题：

- 少样本新类是否需要更高的阶段可塑性？
- 固定低秩参数高效更新是否足以支持新类分离？
- 何时应该释放更多模型自由度，例如提高 LoRA rank、解冻高层表示或使用阶段自适应塑性？

### Q2. 边界控制：如何防止少样本新类被过度泛化？

简单微调可能快速提高新类 recall，但也可能将大量旧类样本错误预测为新类。

关键问题：

- 如何度量新类的过预测和误吸收？
- 如何通过负样本回放、拒识门控、语义锚定和校准机制控制新类边界扩张？
- 高风险标签是否应采用更保守的吸收策略？

### Q3. 稳定-可塑调度：不同增量阶段是否需要不同学习策略？

不同新类的学习难度并不一致。`threat` 可能需要强可塑性，`severe_toxic` 则可能更需要过预测控制。

关键问题：

- 能否根据阶段难度动态调整 plastic capacity、KD 强度、prefix anchoring 和 replay ratio？
- 是否应采用“先探索、后固化”的流程，而不是每个阶段都强行保护旧知识？
- 如何判断一个阶段知识应被沉淀到稳定分支，还是作为风险补丁隔离保留？

### Q4. 未知与变体：如何区分新类、旧类变体和未知风险？

毒性表达会通过字符替换、空格规避、谐音、隐喻等方式持续演化。未知样本不应被强行硬归入已知类。

关键问题：

- 如何结合语义不确定性和表面形式异常检测未知毒性变体？
- 如何输出“已知毒性框架下的未知变体”而非单一 unknown？
- 拒识机制能否降低新类误吸收？

---

## 4. 方法论主线

本课题方法从原来的“记忆保持”转向“可控吸收”。整体框架可概括为：

> **阶段自适应塑性 + 语义锚定约束 + 负样本边界校准 + 变体感知拒识**

### 4.1 阶段自适应塑性（Stage-Adaptive Plasticity）

不同阶段采用不同可塑性预算：

- 对细粒度、高共现、难分离类别，释放更强 plastic capacity；
- 对高风险、易过预测类别，采用更强边界校准和拒识控制；
- 避免在新类尚未学会之前过早施加强记忆保护。

可选实现：

- 动态调整 plastic LoRA rank；
- 解冻高层 stable LoRA 或 RoBERTa top layers；
- stage-wise 调整 KD 强度；
- stage-wise 调整 prefix anchor 强度；
- stage-wise 调整新类负样本比例。

### 4.2 稳定-可塑双分支 LoRA（Stable-Plastic Dual-Branch LoRA）

保留双分支结构，但重新解释其功能：

- **stable branch**：不是单纯防遗忘，而是保存已知毒性语义锚点，防止新类边界无限扩张；
- **plastic branch**：不是微小补丁，而是阶段性新类吸收通道，其容量应根据阶段难度动态调整；
- **semantic consolidation**：阶段结束后判断 plastic 知识是否能安全沉淀，若对旧类干扰可控则合并，否则隔离为历史补丁。

### 4.3 毒性语义锚定前缀（Toxic Semantic Anchor Prefix）

使用 base 类毒性样本的表示质心初始化语义前缀，为模型提供已知毒性语义锚点。

新的解释是：

> prefix anchor 不是为了压制新类，而是为了限制新类吸收方向，避免新类边界无约束扩张。

在高可塑阶段可适当降低 anchor 强度，在校准阶段可提高 anchor 强度。

### 4.4 负样本边界校准（Negative Replay Boundary Calibration）

对新类构造显式负样本，尤其是与新类高度共现但标签为 0 的旧类样本，用于训练模型学习“什么不是该新类”。

该机制直接服务于可控吸收：

- 减少全阳性预测；
- 控制新类决策区域；
- 提升 precision 和 calibration。

### 4.5 变体感知层级拒识门控（Variant-Aware Hierarchical Rejection）

拒识机制不只用于 OOD 检测，也用于防止边界样本被错误吸收到已知新类中。

输出分为：

1. 已知毒性类别；
2. 已知毒性框架下的未知变体；
3. 非目标/低置信风险样本。

---

## 5. 训练目标

基础分类目标仍采用多标签 BCE：

$$
\mathcal{L}_{cls}=\mathrm{BCEWithLogits}(\hat{\mathbf y}, \mathbf y)
$$

在此基础上，引入围绕“可控吸收”的辅助目标：

$$
\mathcal{L}=\mathcal{L}_{cls}
+ \lambda_{cal}\mathcal{L}_{cal}
+ \lambda_{open}\mathcal{L}_{open}
+ \lambda_{evo}\mathcal{L}_{evo}
+ \lambda_{reg}\mathcal{L}_{reg}
$$

其中：

- $\mathcal{L}_{cal}$：新类边界校准损失，强调新类负样本与高共现负例；
- $\mathcal{L}_{open}$：拒识/未知变体损失；
- $\mathcal{L}_{evo}$：表面形式演化一致性损失，约束字符扰动前后表示一致；
- $\mathcal{L}_{reg}$：稳定-可塑结构正则，但不应在新类尚未学会时过强。

训练策略遵循：

> **先吸收，再校准，再沉淀。**

即：

1. 新类探索期：提高 plasticity，弱化旧类约束；
2. 边界校准期：增加负样本与拒识约束；
3. 语义沉淀期：评估是否合并 plastic 知识。

---

## 6. 评测指标

除常规分类指标外，本问题必须显式评估“可控性”。

### 6.1 分类性能

- Avg-mAP；
- Macro-F1 / Micro-F1；
- Per-class best F1；
- Per-class default-threshold F1；
- Tail-class recall。

### 6.2 旧类稳定性

- Forgetting；
- 旧类 f1 drop；
- Semantic Stability / CKA。

### 6.3 新类吸收能力

- 新类 best F1；
- 新类 positive mean probability vs global mean probability；
- 新类 ranking separation；
- 新类 threshold sensitivity。

### 6.4 过预测与误吸收控制

建议新增核心指标：

#### New-class Prediction Expansion Ratio

$$
\mathrm{PER}_c=\frac{\mathrm{pred\_pos}_c}{\mathrm{support}_c}
$$

若 $\mathrm{PER}_c \gg 1$，说明该类存在明显过预测。

#### Over-Assimilation Rate

$$
\mathrm{OAR}_c=\frac{\max(0,\mathrm{pred\_pos}_c-\mathrm{support}_c)}{N-\mathrm{support}_c}
$$

用于衡量非该类样本被错误吸收到新类中的比例。

#### New-class Precision at Default Threshold

默认阈值下的新类 precision 是衡量部署可控性的关键指标，不能只报告 best F1。

### 6.5 未知与变体鲁棒性

- AUROC / FPR95；
- Variant Recall；
- Unknown toxic variant rejection rate；
- Cross-domain implicit hate detection / rejection score distribution。

---

## 7. 基线与对照

### 7.1 必跑基线

- Sequential Fine-tuning：验证强可塑性上界与过预测风险；
- Task-LoRA：参数高效持续学习对照；
- EWC-LoRA：正则化记忆对照；
- Task-LoRA + MSP / ADB：开放集拒识对照。

### 7.2 关键消融

- 固定低秩 plastic vs 高秩 plastic；
- 无 stable branch / single-branch LoRA；
- 无 prefix anchor；
- 无负样本边界校准；
- 无 semantic consolidation；
- 无 rejection gate。

### 7.3 关键比较维度

不只比较 Avg-mAP，还必须比较：

1. 新类 best F1；
2. 新类默认 precision；
3. 新类 PER/OAR；
4. 旧类 f1 drop；
5. 未知拒识 AUROC/FPR95。

---

## 8. 数据协议

主数据集：Jigsaw Toxic Comment Classification Challenge。

推荐阶段划分：

- Stage 0：`obscene`, `insult`；
- Stage 1：`threat`, `identity_hate`；
- Stage 2：`severe_toxic`。

每个阶段采用 few-shot 新类训练样本，并在累计已见类上评估。

额外构造：

1. 新类负样本池：对每个新类采样标签为 0 但毒性相关的样本；
2. 变体测试集：leet speak、空格规避、符号插入、谐音替换；
3. 隐式仇恨迁移集：如 implicit-hate，用于验证语义层面泛化。

---

## 9. 预期贡献

本课题预期贡献从“防遗忘”调整为“可控吸收”：

1. **提出可控语义吸收问题定义**：指出多标签毒性 FSCIL 的关键矛盾是新类吸收、旧类稳定与过预测控制之间的权衡；
2. **提出阶段自适应稳定-可塑机制**：根据新类分离难度动态调整 plastic capacity 与稳定约束；
3. **提出面向毒性新类的边界校准策略**：通过显式负样本和拒识门控抑制新类误吸收；
4. **提出过预测/误吸收评测指标**：PER、OAR、默认阈值 precision 等，用于补足传统 Avg-mAP 和 best F1 的不足；
5. **验证简单微调的局限**：其新类学习能力强，但容易产生高风险新类过预测；本方法目标是在可接受新类性能下显著提升部署可控性。

---

## 10. 一句话摘要

本研究面向少样本增量毒性评论分类，重新定义“可控语义吸收”问题：模型不仅要从少量样本中学习新毒性类别，还要防止新类决策边界过度扩张。为此，本文拟提出阶段自适应稳定-可塑机制、毒性语义锚定、负样本边界校准与变体感知拒识门控，在新类吸收、旧类稳定和过预测控制之间取得平衡。
