# 语料中毒性评论识别与分类研究定义（FSCIL 聚焦，v2）

## 1. 目标 + 关键问题

### 1.1 研究目标

本课题聚焦单一设定：**有样本类增量学习（FSCIL）** 下的毒性评论识别与细粒度分类。

目标是在开放域语料流中实现以下闭环：

- 风险识别：判断样本是否存在毒性/有害倾向；
- 类别判别：在已知毒性标签集合中进行多标签判别；
- 未知拒识：对不确定样本输出未知风险而非硬分类；
- 增量更新：新类到来时快速学习，同时尽量保持旧类性能。

本版本明确边界：

- 聚焦文本单模态；
- 聚焦 FSCIL，不并行展开 ZSCIL；
- **创新重心放在"毒性语义演化感知的持续学习机制"**，围绕毒性表达的表面形式变异（黑话、避讳、字符替换）与语义标签持续演化的双重挑战展开。

### 1.2 关键科学问题

1. **毒性表达持续演化下的参数高效学习**：毒性评论的表面形式（黑话、变体拼写、避讳改写）持续快速迭代，不同毒性子类（identity_hate vs. threat vs. insult）的演化速度差异显著，参数高效模块如何在低秩约束下同时捕获**稳定语义核**与**演化表面形式**？
2. **在离散短评论场景下，如何提升对隐式毒性、反讽表达和标签噪声/群体偏置的鲁棒建模能力？**
3. **开放集门控能否降低未知毒性表达、变体辱骂和隐晦攻击被误归入已知类的风险？**
4. **在长尾分布下，如何提升尾部毒性类别召回而不过度牺牲总体精度？**

### 1.3 更收敛的 key issue

- 毒性评论语料流中，黑话/避讳/变体表达持续演化，如何用语义锚定 + 表面形式鲁棒编码实现持续识别并控制遗忘？
- 离散短评论条件下，如何增强对隐式毒性、反讽和群体偏置的鲁棒建模？
- 如何防止/减少未知毒性变体或分布外评论被误归入已知类？

---

## 2. 相关工作（按三大 key issue 精选）

### 2.1 筛选原则与证据来源

- 先按三大 key issue 做相关性筛选，再按录用信息排序（AAAI/ACL/EMNLP/NAACL/EACL 优先）。
- 保留文献均已做摘要核验，并优先检查 arXiv TeX source 中的数据集与基线段落。
- 对源码不完整或与当前任务边界偏离较大的条目，标记为"降权/不纳入主线"。

### 2.2 关键问题一：持续识别（FSCIL / CIL）

- **2104.11882**, Incremental Few-shot Text Classification with Multi-round New Classes（NAACL 2021）
	- 贡献：给出 NLP 场景的增量少样本文本分类设定与评测流程。
	- 资产：提供 IFS-Intent（基于 BANKING77）与 IFS-Relation（基于 FewRel）基准，并给出 ProtoNet、DyFewShot 和 entailment 式方法的对比。
	- **局限**：通用意图分类任务的标签语义稳定，未考虑毒性场景中表面形式快速演化的挑战。
- **2408.09053**, Learning to Route for Dynamic Adapter Composition in Continual Learning with Language Models（EMNLP 2024）
	- 贡献：参数高效持续学习中的"适配器组合与路由"范式。
	- 资产：MTL5、WOS 和 AfriSenti 基准，报告 DAM、EPI、MoCL 等对照。
- **2601.02232**, ELLA: Efficient Lifelong Learning for Adapters in Large Language Models（EACL 2026 Main）
	- 贡献：在 LoRA 持续学习中通过**子空间去相关正则**提升稳定-可塑平衡，不依赖回放。
	- **对本课题的启发**：ELLA 的"选择性子空间去相关"思想可直接迁移到毒性场景，但需升级为**先验引导的异构子空间分解**——因为毒性不同子类的演化速度差异巨大，不能对所有子空间一视同仁。
	- 资产：Standard CL、Long Sequence 和 TRACE 评测，以及 SeqFT、EWC、LwF、O-LoRA、LB-CL、DATA 等基线。
- **2012.15504**, Continual Learning in Task-Oriented Dialogue Systems
	- 贡献：对话场景连续学习基准（37 域）与 AdapterCL 架构。

> **为什么通用 CL 方法在毒性场景下不足**：毒性表达的对抗性演化导致特征漂移速度远超通用文本分类。例如 identity_hate 的污名化词汇可在数周内完成一轮迭代，而通用意图分类的语义表达相对稳定。这要求适配器不仅要"防遗忘"，还要**主动编码对表面形式变异的鲁棒性**。

### 2.3 关键问题二：短文本隐式语义与弱上下文鲁棒建模

- **2403.16504**, LARA: Linguistic-Adaptive Retrieval-Augmentation for Multi-Turn Intent Classification（EMNLP 2024）
	- 贡献：多轮会话意图分类中，检索增强与上下文利用策略。
	- 用途：可作为"弱上下文条件下语义增强"对照。
- **1909.08705**, CASA-NLU: Context-Aware Self-Attentive NLU（EMNLP 2019）
	- 贡献：显式引入前序意图、槽位、对话行为和历史话语进行上下文建模。
	- 用途：为短文本中隐式语义消歧与上下文敏感编码提供经典强基线参考。
- **毒性领域补充（待纳入）**：
	- HateXplain（可解释仇恨言论检测，提供标签层级与群体指向信息）
	- 对抗性仇恨言论/毒性变体检测相关研究（字符替换、leet speak、空格规避等对抗性改写策略的检测）

### 2.4 关键问题三：未知毒性变体误归类（开放集/OOD）

- **2412.13542**, Multi-Granularity Open Intent Classification via Adaptive Granular-Ball Decision Boundary（AAAI 2025）
	- 贡献：多粒度边界建模，直接针对已知/未知边界错分。
	- 资产：StackOverflow、SNIPS 和 BANKING77 基准。
- **2211.05561**, Estimating Soft Labels for Out-of-Domain Intent Detection（EMNLP 2022 Main Oral）
	- 贡献：ASoul 软标签伪 OOD 学习，缓解未知样本被硬归类。
- **2210.10722**, UniNL: Aligning Representation Learning with Scoring Function for OOD Detection（EMNLP 2022 Main）
	- 贡献：将表征学习目标与 OOD 打分函数对齐（KNCL + KNN score）。
- **2106.08616**, Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training（ACL 2021 Long Oral）
	- 贡献：通过伪异常样本构造进行端到端 OOS 训练。
- **2010.13009**, DNNC（EMNLP 2020 Main）与 **1909.02027**, CLINC150 数据集论文（EMNLP-IJCNLP 2019）
	- 贡献：前者是 few-shot + OOS 的经典 NLI/近邻范式，后者提供高覆盖 intent+OOS 评测基准。

> **毒性场景的特殊性**：通用 OOD 检测假设未知类与已知类语义远离，但毒性变体往往是**已知毒性词的轻微字符扰动**（如 "idiot" → "1d1ot"），在嵌入空间中与已知类高度接近。这要求拒识机制不能仅依赖语义距离，还需引入**表面形式异常通道**。

### 2.5 降权与不纳入主线条目

- **2411.14252**（CIKM 2025）已下载到源码包，但当前包内正文 TeX 证据不完整，暂不作为主线证据文献。
- **2505.11998**（PEARL）与本课题既定"固定秩 + 阶段冻结"路线存在方法族重叠风险，仅用于差异化声明，不作为主创新依赖。

### 2.6 当前结论（对本课题最有用的相关工作组合）

- **持续识别主线**：2104.11882 + 2408.09053 + 2601.02232（ELLA 的子空间去相关思想需场景化升级）。
- **弱上下文语义主线**：2403.16504 + 1909.08705 + 毒性领域变体检测文献（待补充）。
- **未知拒识主线**：2412.13542 + 2211.05561 + 2210.10722 + 2106.08616。

---

## 3. 方法论

### 3.1 总体框架

毒性评论的持续学习面临独特的**双重演化**挑战：

1. **语义标签演化**：新毒性子类不断出现（如从 general toxic 细分到针对特定群体的仇恨言论）；
2. **表面形式演化**：同一毒性语义通过黑话、避讳、变体拼写反复出现（如 "idiot" → "id1ot" → "1d10t" → "i d i o t"）。

因此，框架设计遵循 **"语义锚定 + 表面形式鲁棒编码 + 演化感知增量适配"** 三条主线。

采用 Transformer 主干 + **毒性语义锚定 KV 前缀** + **语义稳定性感知的双分支 LoRA** + 阶段冻结 + **毒性表达结构感知位置编码** + **变体感知层级拒识门控**。

输入为离散评论样本 $x_t$（默认不依赖会话历史，可选拼接最小上下文），输出为：

- 多标签毒性概率向量 $\hat{\mathbf{y}}_t\in[0,1]^{C_k}$；
- 综合风险分数 $s_t$；
- 未知风险概率 $u_t$（含"已知毒性框架的未知变体"细分提示）。

### 3.2 毒性语义锚定前缀（Toxic Semantic Anchor Prefix）

本方案的前缀不再完全随机初始化，而是引入**语义锚定**机制，使前缀在增量初期即具备毒性语义的先验结构。

**初始化**：对基类（阶段 0）的毒性样本，提取 [CLS] 表示并通过 K-means 聚类得到 $N_{anchor}$ 个语义质心，构成初始前缀分量 $P_{proto} \in \mathbb{R}^{m \times d}$：

$$
P_{proto} = \text{KMeans}\big(\{\text{[CLS]}_i \mid y_i \in \text{Base Toxic Classes}\}\big)
$$

**阶段级更新**：第 $k$ 个增量阶段的前缀为初始锚点与阶段可学习残差之组合：

$$
P^{(k)} = \alpha \cdot P_{proto} + (1-\alpha) \cdot \Theta_P^{(k)},\quad \Theta_P^{(k)}\in\mathbb{R}^{m\times d}
$$

其中 $\alpha \in (0,1)$ 为锚定强度超参。注意力计算保持：

$$
Q = HW_Q,\quad K = [P^{(k)};H]W_K,\quad V=[P^{(k)};H]W_V
$$

**演化感知更新**（可选增强）：当检测到当前批次中 OOV 比率或字符变异密度高于阈值时，临时提升 $(1-\alpha)$ 以增强前缀对表面形式变化的敏感度。

### 3.3 语义稳定性感知的双分支 LoRA（Semantic-Stability-Aware Dual-Branch LoRA）

**核心观察**：毒性类别在持续学习中面临两类知识：
1. **跨阶段稳定的语义核**：如"攻击性意图"、"负面情感指向"等抽象框架，一旦学会应长期保留；
2. **阶段特有的表面形式**：如新出现的污名化词汇、变体拼写、隐式威胁句式，可能只在一两个阶段内活跃后沉淀为常识，也可能随时间被淘汰。

如果将所有知识混在一个 LoRA 中更新，阶段特有的表面形式会冲散已稳定的语义核，导致遗忘。因此，我们将每个阶段的增量更新显式分解为**两个分支**，并在阶段结束时引入**"语义沉淀"机制**——类似于软件项目中 dev 分支在里程碑时审查并合并到 main 分支。

**分支一：语义稳定分支（Stable Branch / Main）**

$$
\Delta W_{stable}^{(k)} = A_{stable}^{(k)} B_{stable}^{(k)},\quad A_{stable}^{(k)}\in\mathbb{R}^{d\times r_s},\; B_{stable}^{(k)}\in\mathbb{R}^{r_s\times d_h}
$$

该分支是跨阶段累积的"主干"，负责学习所有已见毒性类别的**共享抽象语义**。其更新受强能量惩罚正则约束，防止被新阶段的表面形式干扰：

$$
\mathcal{L}_{stable} = \big\| \Delta W_{stable}^{(k)} * \mathcal{W}_{past}^{stable} \big\|_F^2
$$

其中 $\mathcal{W}_{past}^{stable} = \sum_{j<k} \Delta W_{stable}^{(j)}$ 为历史稳定分支累积更新，$*$ 表示逐元素乘积（借鉴 ELLA 的子空间去相关思想，但限定在稳定分支内）。

**分支二：语义可塑分支（Plastic Branch / Dev）**

$$
\Delta W_{plastic}^{(k)} = A_{plastic}^{(k)} B_{plastic}^{(k)},\quad A_{plastic}^{(k)}\in\mathbb{R}^{d\times r_p},\; B_{plastic}^{(k)}\in\mathbb{R}^{r_p\times d_h}
$$

该分支是每个阶段独立初始化的"补丁"，专门学习当前阶段新引入类别的**特有表面形式与细粒度语义**。允许较大更新自由度，但施加稀疏性约束防止过拟合：

$$
\mathcal{L}_{plastic} = \lambda_{sp} \big\| \Delta W_{plastic}^{(k)} \big\|_1
$$

**阶段结束时的"语义沉淀"机制（Semantic Consolidation）**

在每个增量阶段训练结束后，对当前 plastic 分支进行"沉淀审查"，评估其对旧类知识的干扰程度：

$$
\delta^{(k)} = \frac{1}{|D_{val}^{old}|} \sum_{x \in D_{val}^{old}} \Big\| h_{stable+plastic}(x) - h_{stable}(x) \Big\|_2
$$

其中 $D_{val}^{old}$ 为旧类验证样本（少量保留或通过生成式回放构造），$h_{stable}$ 为仅使用稳定分支的 [CLS] 表示，$h_{stable+plastic}$ 为加入当前 plastic 分支后的表示。如果 $\delta^{(k)} < \tau$（即 plastic 分支对旧类表示的干扰低于阈值），则将该 plastic 分支**累加合并**到 stable 分支：

$$
\mathcal{W}_{past}^{stable} \leftarrow \mathcal{W}_{past}^{stable} + \Delta W_{plastic}^{(k)}
$$

然后 plastic 分支重新初始化，进入下一阶段。如果 $\delta^{(k)} \ge \tau$，则 plastic 分支被**隔离冻结**（不合并到 stable，但保留为独立的历史补丁），stable 分支保持不变。

**推理时的参数组合（关键实现细节）**：

在第 $k$ 阶段推理时，有效投影矩阵为：

$$
W_{eff}^{(k)} = W_{init} + \underbrace{\mathcal{W}_{past}^{stable}}_{\text{已沉淀核心}} + \underbrace{\sum_{j \in \mathcal{F}_k} \Delta W_{plastic}^{(j)}}_{\text{冻结历史补丁}} + \underbrace{\Delta W_{plastic}^{(k)}}_{\text{当前活跃补丁}}
$$

其中 $\mathcal{F}_k = \{j \mid j < k, \; \delta^{(j)} \ge \tau\}$ 为截至阶段 $k$ 所有被冻结的 plastic 分支索引集合。

**为什么冻结分支仍需参与推理**：
- 每个被冻结的 plastic 分支承载了对应阶段新引入类别的**特有表面形式知识**。若推理时丢弃，则这些类别的样本将仅依赖 stable 分支的抽象语义核，丢失关键的细粒度判别特征。
- 但冻结分支对旧类存在干扰——这正是它们未能通过沉淀审查的原因。为此，我们依赖两个机制控制总干扰：
  1. **稀疏性约束**：每个 plastic 分支受 L1 稀疏正则，非零元素极少，多个分支叠加的有效秩仍远低于单一稠密矩阵；
  2. **stable 分支的鲁棒性**：stable 分支通过强子空间去相关正则，已建立对旧类足够鲁棒的语义表示，能容忍冻结分支带来的边际干扰。

**参数预算控制（可选增强）**：
若冻结分支数量随阶段增长导致参数量累积，可引入轻量"冻结分支剪枝"策略：对冻结时间超过 $T_{max}$ 个阶段且在该期间未被激活（即对应类别样本的推理置信持续低于阈值）的分支，予以丢弃。这确保了长期运行时的参数增量可控。

**为什么比单一 LoRA 更贴合场景**：
- 稳定分支长期维护毒性语义的"核心资产"（抽象攻击框架），不受每阶段新表面形式的噪音干扰；
- 可塑分支提供安全的"试验场"，学习新类特有表达，即使失败也不会破坏主分支；
- **语义沉淀机制实现了知识的自然分层**：经过验证的阶段知识晋升为核心知识（合并到 stable），未经验证的知识被隔离冻结但仍参与推理——这是通用持续学习方法不具备的"知识生命周期管理"能力；
- 无需人工预设类别级门控，沉淀阈值 $\tau$ 是统一超参数，在验证集上一次性调参即可。

### 3.4 毒性表达结构感知位置编码

使用复合位置编码，显式建模毒性评论的表层结构信号：

$$
\mathrm{PE}_i = \mathrm{PE}_{abs}(i) + W_q q_i + W_m m_i + W_l l_i + W_v v_i
$$

其中：

- **$q_i$（强调强度标记）**：基于标点密度与重复标点模式（如 "!!!"、"???"）。高强调密度与毒性情绪爆发高度相关。
- **$m_i$（大写/强调标记）**：连续大写序列（"STUPID IDIOT"）及社交媒体强调符（*星号*、_下划线_）的检测。
- **$l_i$（句法片段位置标记）**：反问句、条件威胁句（"if you... then I will..."）的句法模板位置检测。
- **$v_i$（字符级变异标记，新增）**：显式编码对抗性表面形式——leet speak 替换（a→4, e→3）、空格规避（"i d i o t"）、谐音替换等。$v_i$ 通过一个轻量的字符级规则或小型 CNN 检测器计算，无需外部词典。

### 3.5 变体感知层级拒识门控（Variant-Aware Hierarchical Rejection Gate）

基于多标签不确定性、原型距离与**表面形式异常分数**估计未知概率：

**表面形式异常分数**（新增）：

$$
s_{surface}(x_t) = \text{CharEntropy}(x_t) \cdot \text{OOV-Ratio}(x_t) \cdot \max_{w \in \mathcal{V}_{known}} \text{EditSim}(x_t, w)
$$

其中 $\mathcal{V}_{known}$ 为训练阶段见过的毒性关键词集合，$\text{EditSim}$ 为归一化编辑距离。该分数捕获"看起来像已知毒性词的变体但又不完全匹配"的分布外信号。

**综合未知概率**：

$$
\begin{aligned}
H(\hat{\mathbf{y}}_t) &= -\frac{1}{C_k}\sum_{j=1}^{C_k}\big[\hat y_{t,j}\log \hat y_{t,j} + (1-\hat y_{t,j})\log(1-\hat y_{t,j})\big] \\
u_t &= \sigma\!\left(a\big(1-\max_j \hat y_{t,j}\big) + b\,H(\hat{\mathbf{y}}_t) + c\,d_{proto}(x_t) + d\,s_{surface}(x_t)\right)
\end{aligned}
$$

**层级拒识**（新增）：不是单一阈值输出，而是**粗粒度-细粒度两级**：

1. **粗粒度层**：区分 toxic vs non-toxic（低阈值，高召回）；
2. **细粒度层**：若粗粒度判定为 toxic 但细粒度各类别置信均低，输出 **"已知毒性框架的未知变体"** 而非简单"未知"。

这对毒性场景极具价值：系统可提示"检测到疑似新型仇恨表达，建议人工复核"，而不是笼统拒识。

### 3.6 训练目标

精简为 **5 项核心损失**，避免工程堆砌：

$$
\begin{aligned}
\mathcal{L} ={}& \mathcal{L}_{bce} + \lambda_{evo}\mathcal{L}_{evo} + \lambda_{sp}\mathcal{L}_{stable/plastic} \\
&+ \beta\mathcal{L}_{open} + \eta\mathcal{L}_{orth}
\end{aligned}
$$

各项说明：

- **$\mathcal{L}_{bce}$**：多标签二元交叉熵。基础分类损失，对所有类别一视同仁。
- **$\mathcal{L}_{evo}$（语义演化一致性损失，核心新增）**：要求同一毒性语义的不同表面形式（通过对训练样本施加字符扰动生成的正样本对）在 [CLS] 表示空间中接近：
  $$
  \mathcal{L}_{evo} = \sum_{(x, x') \in \mathcal{P}} \big\| h_{CLS}(x) - h_{CLS}(x') \big\|_2^2
  $$
  其中 $\mathcal{P}$ 为字符扰动正样本对集合（leet 替换、空格插入等），$h_{CLS}$ 为 [CLS] 表示。这是直接针对"黑话变体"场景设计的核心损失。
- **$\mathcal{L}_{stable/plastic}$**：对应 3.3 中双分支 LoRA 的正则约束——稳定分支的 ELLA 式子空间去相关 + 可塑分支的 L1 稀疏约束；以及语义沉淀的合并损失（若当前阶段 plastic 分支通过沉淀审查并合并，需最小化新旧稳定分支在旧类样本上的输出差异，确保合并过程平滑）。
- **$\mathcal{L}_{open}$**：拒识损失，用于训练阶段区分已知类与构造的伪 OOD 样本。
- **$\mathcal{L}_{orth}$**：跨阶段适配器子空间正交性约束（轻量版，仅在稳定分支上施加，防止不同阶段的稳定语义核相互干扰）。

**删减说明**：原 ASL 并入演化一致性损失（$\mathcal{L}_{evo}$ 本身具有难样本聚焦效应）；长尾校正 $\mathcal{L}_{imb}$ 改用采样策略（类别重加权/过采样）替代；前缀条件一致性并入 $\mathcal{L}_{evo}$ 的表示约束中；蒸馏损失 $\mathcal{L}_{distill}$ 由稳定分支的强正则天然承担。

---

### short version

**PPT 第1页：方法框架**
- 任务：面向 FSCIL 的毒性评论识别与细粒度分类，核心挑战是**毒性表达的双重演化**（语义标签演化 + 表面形式演化）。
- 核心思路：**语义锚定 + 表面形式鲁棒编码 + 演化感知增量适配**。
  - 毒性语义锚定前缀：用基类毒性语义质心初始化前缀，而非随机初始化。
  - 语义稳定性感知的双分支 LoRA：**稳定分支（Main）**长期维护跨阶段毒性语义核，**可塑分支（Dev）**学习当前阶段特有表面形式；阶段结束时通过"语义沉淀"机制审查 plastic 分支对旧类的干扰程度，低干扰则合并晋升到 stable，高干扰则隔离冻结——类似于 git 分支的合并/保留策略。
  - 毒性表达结构感知位置编码：显式建模标点、强调、句法模板与字符级变异信号。
  - 变体感知层级拒识门控：两级拒识（粗粒度 toxic vs 细粒度"已知框架的未知变体"），引入表面形式异常分数。

**PPT 第2页：训练与输出**
- 核心新增损失：语义演化一致性损失 $\mathcal{L}_{evo}$，强制同一毒性语义的不同字符变体在表示空间中接近。
- 推理输出：
  - 多标签概率向量、综合风险分数、未知风险概率（含"已知毒性框架的未知变体"细分）。
  - 用 max-sigmoid 置信度 + 多标签熵 + 原型距离 + **表面形式异常分数** 做层级拒识。
- 评测指标：
  - 持续学习：Avg-mAP、Macro-F1、Forgetting
  - 拒识能力：AUROC、FPR95
  - **演化鲁棒性：Variant Recall（毒性词变体召回）、Semantic Stability（跨阶段语义核一致性）**
  - 长尾效果：Tail Recall
  - 工程代价：参数增量、推理时延
- 对照基线：
  - 精简主表：Sequential Fine-tune、Task-LoRA、Task-LoRA + MSP、Task-LoRA + ADB、MoCL（可选）
  - **新增消融**：单一分支 LoRA vs 双分支 LoRA（含/不含语义沉淀）、无语义锚定前缀 vs 有锚定、无演化一致性损失 vs 有演化一致性

---

## 4. 评测方案 + 基线 + 数据集

### 4.1 评测方案

增量流程建议：

- 阶段 0：基础类训练（如 {toxic, obscene, insult}）；
- 阶段 1..K：每阶段引入若干新类（如 {threat, identity_hate}、{severe_toxic}），每类少量标注样本；
- 每阶段结束后评测累计已见类别 + 未知类拒识能力。

核心指标：

- 平均增量性能（Avg-mAP / Avg-Macro-F1）
- 微/宏 F1（Micro-F1 / Macro-F1）
- 遗忘度（Forgetting）
- 未知类 AUROC / FPR95
- **Variant Recall（新增）**：已知毒性词的字符级变体（leet 替换、空格插入、谐音替换）的召回率
- **Semantic Stability（新增）**：跨阶段同一语义核的表示一致性（用中心核对齐 CKA 或余弦相似度衡量）
- 尾部标签召回率（Tail Recall）
- 参数增量与推理时延

### 4.2 基线（可直接落地）

为避免"只和弱基线比较"，建议按四组设置对照：

1. 增量学习基础对照（持续识别）
	- Sequential Fine-tune（无防遗忘）
	- Replay / A-GEM / LAMOL（有记忆回放）
	- EWC / LwF / L2（正则蒸馏系）
2. 参数高效持续学习对照（与本文最接近）
	- Task-LoRA（每阶段独立 LoRA，简单冻结）
	- AdapterCL（来自 2012.15504）
	- DAM / EPI / MoCL（来自 2408.09053）
	- O-LoRA / LB-CL / DATA / Recurrent-KIF（来自 2601.02232）
3. FSCIL 文本任务对照
	- ProtoNet（2104.11882 中文本化对照）
	- DyFewShot（2104.11882 中文本化对照）
	- Entailment / DNNC 系（2010.13009, 2104.11882）
4. 未知拒识与开放集对照
	- MSP / LOF / GDA / Energy
	- ADB / DA-ADB
	- KNN-CL / UniNL / ASoul / DCLOOS

**建议最小可执行基线子集（论文主表，精简版）**：

- 必跑 4 项：Sequential Fine-tune、Task-LoRA、Task-LoRA + MSP、Task-LoRA + ADB
- 可选加 1 项（算力允许）：MoCL（作为较强参数高效持续学习对照）

**新增消融基线**：
- Ours - $\mathcal{L}_{evo}$（去掉演化一致性损失）
- Ours - Dual Branch（退化为单一 LoRA）
- Ours - Anchor Prefix（退化为随机初始化前缀）

### 4.3 数据集（可参考开源）

优先选"可同时支持增量 + 未知拒识"的数据：

1. 主评测数据（建议必选）
	- Toxic Comment Classification Challenge / Jigsaw Toxic Comment（主场景数据；评论级毒性分类与多标签切分）
	- 标签形态说明：Jigsaw 为多标签体系（toxic、severe_toxic、obscene、threat、insult、identity_hate），训练端采用 sigmoid + 多标签损失，不使用单标签 softmax。
	- Jigsaw Unintended Bias in Toxicity Classification（偏置与亚群体鲁棒性补充）
	- HateXplain / OLID / OffensEval / HatEval（可作为毒性、仇恨与攻击性表达补充池）
2. 增量专用构造基准（建议至少选一套）
	- 由 Toxic Comment 分类标签构造 FSCIL 切分：base 类 + 细粒度增量毒性子类 + OOD 评论池
	- 可执行切分示例：阶段0 {toxic, obscene, insult}；阶段1 {threat, identity_hate}；阶段2 {severe_toxic} + OOD 评论池（可按样本量重排）
	- 由 Jigsaw / OffensEval 的攻击类型或语气标签构造多阶段新类
3. **毒性变体构造协议（新增，核心）**
	- 在每个增量阶段，对已知毒性关键词进行**字符级扰动**生成变体测试集：
		- Leet speak 替换：a→4, e→3, i→1, o→0, s→5 等
		- 空格规避："idiot" → "i d i o t"、"f u c k"
		- 符号插入："i.d.i.o.t"、"i*d*i*o*t"
		- 谐音/近似替换："phuck"、"azz"
	- 变体样本**不加入训练**，仅用于测试 Variant Recall，专门评测模型对"表面形式演化"的鲁棒性。
4. 上下文/语义演化补充集（用于 issue2）
	- 多轮评论线程或回复链数据，用于建模上下文反讽与指代攻击
	- 带上下文的社媒仇恨/攻击性数据，可用于验证语境依赖建模
5. 外部迁移与鲁棒性补充（可选）
	- SNIPS、ATIS（仅作通用文本分类迁移参照）
	- AfriSenti / WOS / MTL5（跨域持续学习鲁棒性验证）

建议的数据协议：

- 增量划分：固定 base + K 个增量阶段（每阶段新增类）。
- 未知划分：每阶段保留未见类作为 unknown pool，统一计算 AUROC/FPR95。
- **变体划分：每阶段对已知毒性词构造字符级变体，统一计算 Variant Recall。**
- 复现实验：至少报告 3 个随机划分种子并公开 split 脚本 + 变体构造脚本。

### 4.4 可直接执行的检索清单

1. 先定任务词：toxic comment / hateful comment / abusive language / harmful content
2. 再定学习词：few-shot class incremental learning / continual learning / adapter
3. 再定评测词：open set / OOD / unknown class detection
4. **新增：毒性变体词：adversarial text / leet speak / character-level perturbation / hate speech variant**
5. 最后定数据词：dataset / benchmark / split protocol

---

## 一句话版本（用于项目书摘要）

本研究面向 FSCIL 毒性评论识别，提出**"语义稳定性感知的双分支 LoRA + 毒性语义锚定前缀 + 表面形式鲁棒编码"**框架，通过语义演化一致性约束与变体感知层级拒识，实现对毒性表达持续演化的增量学习，在控制遗忘的同时保持对新黑话/变体辱骂的识别与拒识能力。
