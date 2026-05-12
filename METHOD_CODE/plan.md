## Plan: FSCIL毒性评论识别方法实现（METHOD_CODE）

**TL;DR**：在11G显存约束下，基于RoBERTa-base自研双分支LoRA与毒性语义锚定前缀，实现FSCIL毒性评论持续学习与拒识框架。代码模块化，兼容transformers Trainer，支持多阶段增量训练、语义沉淀、变体评测。

---

## Phase 1: 项目骨架与基础设施（1-2天）

**目标**：搭建可运行的项目结构，实现数据加载与基线模型推理。

**Step 1.1 — 目录结构初始化** *独立*
- `METHOD_CODE/configs/` — YAML/JSON配置（模型路径、LoRA秩、阶段定义、损失权重）
- `METHOD_CODE/data/` — 数据集加载器、FSCIL切分协议、变体构造脚本
- `METHOD_CODE/models/` — 主干封装、双分支LoRA、前缀模块、位置编码、拒识门控
- `METHOD_CODE/losses/` — 五项核心损失实现
- `METHOD_CODE/trainers/` — 继承transformers Trainer的自定义训练器（支持阶段冻结、语义沉淀钩子）
- `METHOD_CODE/utils/` — 日志、检查点、CKA相似度、Variant Recall评测
- `METHOD_CODE/scripts/` — 分阶段运行脚本（stage_0_base.sh, stage_1_incremental.sh等）
- `METHOD_CODE/requirements.txt` — 依赖清单

**Step 1.2 — 数据加载与FSCIL切分** *独立，可并行1.1*
- 实现 `ToxicCommentDataset(Dataset)`：读取Jigsaw CSV，支持多标签（toxic, severe_toxic, obscene, threat, insult, identity_hate）
- 实现 `FSCILSplitProtocol`：
  - 阶段0: {toxic, obscene, insult}
  - 阶段1: {threat, identity_hate}
  - 阶段2: {severe_toxic}（可调）
  - 每阶段每类N-shot（如16-shot或32-shot）
- 实现 `VariantGenerator`：leet替换、空格规避、符号插入（仅测试用，不加入训练）
- **关键输出**：`data_splits/` 下每个seed的pickle/JSON切分文件，确保可复现

**Step 1.3 — 基座模型封装（无LoRA）** *依赖1.1*
- 实现 `ToxicCommentClassifier(nn.Module)`：
  - 加载 `RobertaModel.from_pretrained('roberta-base')`
  - 分类头：Linear(hidden_size, num_classes) + Sigmoid（多标签）
- 支持标准Trainer训练一个非增量基线，验证数据流正确
- **验证**：单阶段在base类上训练，计算Macro-F1与mAP，确认与公开基线持平

**Step 1.4 — 依赖与环境** *独立*
- `requirements.txt` 锁定 `torch>=2.0`, `transformers>=4.35`, `scikit-learn`, `numpy`, `tqdm`, `pyyaml`
- 显存预算：RoBERTa-base(125M) + 双分支LoRA + 前缀，11G单卡batch_size估计=8~16（需梯度累积）

---

## Phase 2: 核心方法模块实现（3-4天）

**目标**：完整实现problemDef中3.2-3.5的所有创新模块。

**Step 2.1 — 毒性语义锚定前缀（3.2）** *依赖1.3*
- 实现 `ToxicSemanticPrefix(nn.Module)`：
  - `init_from_kmeans(base_cls_embeddings, n_anchors)`：对阶段0的[CLS]做K-means，得到 `P_proto`
  - `forward(stage_idx)`：返回 `P_k = alpha * P_proto + (1-alpha) * Theta_P[k]`
  - 在Transformer层中拼接前缀到K/V：`K = cat([P_k, H]) @ W_K`
- **参考模式**：仿照Prefix-Tuning论文，但替换随机初始化为K-means锚定
- **实现细节**：前缀长度建议 `m=10~20`，alpha默认 `0.7`

**Step 2.2 — 自研双分支LoRA（3.3）** *依赖1.3*
- 实现 `DualBranchLoRALayer(nn.Module)`：
  - `stable_A`, `stable_B`：跨阶段累积，初始化Xavier，秩 `r_s`
  - `plastic_A`, `plastic_B`：每阶段独立初始化，秩 `r_p`（建议 `r_s=8, r_p=4`）
  - `forward(x)`：`x @ (W_base + stable_A@stable_B + plastic_A@plastic_B)`
- 实现 `SemanticConsolidation` 类：
  - `evaluate_interference(model, old_val_loader)`：计算 `delta_k = mean ||h_stable+plastic - h_stable||`
  - 若 `delta_k < tau`：执行 `stable += plastic`（矩阵累加），然后 `plastic.reset_parameters()`
  - 若 `delta_k >= tau`：`plastic.freeze()`（保留为历史补丁），新 `plastic` 初始化
- **关键实现**：需要暴露 `h_stable`（仅stable分支）与 `h_full`（双分支）两种前向模式供沉淀审查使用
- **显存注意**：历史冻结的plastic分支需保留在模型中参与推理，但梯度关闭；若阶段多，需预留参数预算

**Step 2.3 — 毒性表达结构感知位置编码（3.4）** *依赖1.3*
- 实现 `ToxicAwarePE(nn.Module)`：
  - 绝对位置编码复用RoBERTa内置
  - `q_i`（强调强度）：标点密度与重复模式（正则 `[!?]{2,}`）
  - `m_i`（大写/强调）：连续大写序列、*星号*、_下划线_
  - `l_i`（句法片段）：反问句、条件威胁模板（规则匹配，如 `if you.*then I will`）
  - `v_i`（字符变异）：轻量规则检测leet替换、空格规避，可扩展为char-CNN
  - 输出：`PE_i = RoBERTa_PE + W_q*q_i + W_m*m_i + W_l*l_i + W_v*v_i`
- **实现方式**：在embedding层后、Transformer前注入，作为附加embedding加到token embedding上

**Step 2.4 — 变体感知层级拒识门控（3.5）** *依赖1.3*
- 实现 `HierarchicalRejectionGate(nn.Module)`：
  - `surface_anomaly_score(x)`：CharEntropy * OOV-Ratio * max_edit_sim(known_toxic_vocab)
  - 已知毒性词表 `V_known`：从训练集阶段0提取高频毒性token/词，持续更新
  - `forward(cls_hidden, probs)`：
    - `H(y)`：多标签熵（逐标签二元熵均值）
    - `max_prob`：各类sigmoid最大概率
    - `d_proto`：到最近原型（阶段0 K-means质心）的距离
    - `u = sigmoid(a*(1-max_prob) + b*H + c*d_proto + d*s_surface)`
  - **层级输出**：
    - 若 `u > theta_coarse` → "unknown"
    - 若 `u <= theta_coarse` 且 `max_prob < theta_fine` → "已知毒性框架的未知变体"
    - 否则 → 多标签预测
- **超参**：a,b,c,d可学习或固定；theta_coarse=0.5, theta_fine=0.3（验证集调参）

---

## Phase 3: 损失函数与训练流程（2-3天）

**Step 3.1 — 核心损失实现** *依赖Phase 2*
- `losses/bce_loss.py`：多标签二元交叉熵（已有，直接用BCEWithLogitsLoss）
- `losses/evo_loss.py`：语义演化一致性损失 `L_evo`
  - 需要字符扰动生成器（leet替换、空格插入）在训练时实时生成正样本对
  - 正样本对共享相同多标签，计算[CLS]的L2距离
- `losses/stable_plastic_reg.py`：
  - 稳定分支：ELLA式子空间去相关 `||stable * W_past_stable||_F^2`
  - 可塑分支：L1稀疏 `lambda_sp * ||plastic||_1`
  - 沉淀合并平滑损失（若本阶段通过审查）：`||h_old_stable(x) - h_new_stable(x)||` 在旧类样本上
- `losses/open_loss.py`：拒识损失
  - 构造伪OOD样本：从非毒性评论中采样，或随机文本，或已知毒性词的强扰动变体
  - 目标：已知类 `u_t -> 0`，伪OOD `u_t -> 1`
- `losses/orth_loss.py`：跨阶段正交性 `||stable^T @ stable_old||`（轻量，仅在阶段>0时激活）
- **组合**：`L = L_bce + lambda_evo*L_evo + lambda_sp*L_sp + beta*L_open + eta*L_orth`

**Step 3.2 — 自定义Trainer** *依赖3.1*
- 继承 `transformers.Trainer`，重写：
  - `compute_loss()`：调用上述复合损失
  - `training_step()`：在步骤结束时检查是否需要执行语义沉淀（建议每个epoch结束后执行，而非每step）
  - `evaluate()`：额外计算 Variant Recall、Semantic Stability(CKA)、AUROC/FPR95
- 实现 ` IncrementalLearningCallback(TrainerCallback)`：
  - 阶段开始时：初始化新plastic分支、更新前缀残差、扩充分类头（新类）
  - 阶段结束时：触发 `SemanticConsolidation.evaluate_and_merge()`，记录delta_k
  - 保存检查点：分阶段保存 `checkpoint-stage{k}/`，包含stable、所有plastic、前缀、词表

**Step 3.3 — 训练脚本与阶段管理** *依赖3.2*
- `scripts/run_stage.py`：单阶段入口，接收 `--stage 0/1/2` 和 `--prev_checkpoint` 路径
- `scripts/run_full_pipeline.py`：串行执行所有阶段，自动传递检查点路径
- 配置系统（推荐YAML）：
  - `configs/base.yaml`：模型、LoRA秩、前缀长度、损失权重
  - `configs/stages.yaml`：每阶段新增类、shot数、epoch数、学习率

---

## Phase 4: 评测与基线（2-3天）

**Step 4.1 — 评测指标实现** *依赖Phase 3*
- `utils/metrics.py`：
  - 持续学习：Avg-mAP、Macro-F1、Micro-F1、Forgetting（按标准CIL定义）
  - 拒识：AUROC、FPR95（对已知/未知样本的u_t打分排序）
  - 演化鲁棒性：**Variant Recall**（变体测试集上，若任毒性标签>0.5即算召回）
  - 语义稳定性：**Semantic Stability CKA**（中心核对齐，比较阶段k与阶段0的[CLS]表示矩阵相似度）
  - 长尾：Tail Recall（对最少样本的毒性标签单独计算Recall）
- 实现 `evaluate_all_stages()`：加载每个阶段的最终检查点，在累积测试集上评测

**Step 4.2 — 消融基线实现** *依赖4.1*
- `baselines/seq_finetune.py`：顺序微调（无防遗忘）
- `baselines/task_lora.py`：每阶段独立LoRA + 冻结旧LoRA（单分支对照）
- `baselines/task_lora_msp.py`：Task-LoRA + MSP（Max Softmax Prob作为OOD分数）
- `baselines/task_lora_adb.py`：Task-LoRA + ADB（Adaptive Decision Boundary）
- 消融变体（在Ours上开关模块）：
  - `Ours - L_evo`：evo_loss权重置0
  - `Ours - Dual Branch`：退化为单LoRA（plastic分支移除，稳定分支保留）
  - `Ours - Anchor Prefix`：前缀随机初始化

**Step 4.3 — 结果记录与可视化** *依赖4.2*
- `utils/logger.py`：CSV/JSON记录每阶段指标
- `scripts/plot_results.py`：绘制增量性能曲线（每阶段Avg-mAP）、遗忘柱状图、Variant Recall对比图
- 输出 `results/` 目录，包含模型预测、指标汇总、对比表格（LaTeX格式，可直接贴论文）

---

## Phase 5: 验证与文档（1-2天）

**Step 5.1 — 单元测试与冒烟测试**
- 测试双分支LoRA矩阵乘法正确性（输出shape、梯度流）
- 测试语义沉淀逻辑（delta_k < tau时stable是否累加，delta_k >= tau时plastic是否冻结并保留）
- 测试变体生成器（leet替换结果是否符合预期）
- 单epoch小规模训练（仅100条数据）确认训练循环无OOM、损失下降

**Step 5.2 — 代码文档与使用说明**
- `README.md`：环境安装、数据准备（需配置Jigsaw路径）、训练命令、评测命令
- 每个模块头部Docstring说明与problemDef对应章节
- `CONFIG_GUIDE.md`：解释各超参对11G显存的适配建议

---

## 关键文件清单

| 文件 | 说明 |
|------|------|
| `models/roberta_dual_lora.py` | 主干+双分支LoRA核心，含SemanticConsolidation钩子 |
| `models/toxic_prefix.py` | 毒性语义锚定前缀（K-means初始化+阶段残差） |
| `models/toxic_pe.py` | 结构感知位置编码（强调/大写/句法/变异） |
| `models/rejection_gate.py` | 层级拒识门控（表面异常分数+两级阈值） |
| `losses/evo_loss.py` | 字符扰动正样本对生成 + L2一致性损失 |
| `losses/stable_plastic_reg.py` | 稳定分支去相关 + 可塑分支L1 + 合并平滑 |
| `trainers/incremental_trainer.py` | 继承transformers Trainer，集成阶段回调与复合损失 |
| `data/fscil_split.py` | Jigsaw多标签切分协议（base+K阶段） |
| `data/variant_generator.py` | leet/空格/符号变体构造（评测专用） |
| `utils/metrics.py` | Variant Recall, CKA, AUROC, Forgetting |

---

## 技术决策与约束（已确认 + 待讨论）

### 已确认决策

- **基座**：RoBERTa-base（11G显存下安全选择）。RoBERTa-large暂列 backlog，仅在 base 结果稳定且显存允许（梯度检查点）时考虑。
- **LoRA 层范围**：**首版仅注入 `q_proj` 与 `v_proj`**（`attention.self.query / attention.self.value`）。
  - 理由：q/v 已覆盖毒性语义锚定与表面形式匹配的核心通路，参数量增幅约 **0.15%**；k_proj / o_proj 对表征收益有限但显著增加激活显存；fc1/fc2（FFN）与 ToxicAwarePE 易冲突，首版排除。
  - **开放**：若 Variant Recall 瓶颈明显，可逐步开放 `k_proj` 或 `o_proj`，通过配置 `layers_to_adapt` 列表切换，无需改代码结构。
- **LoRA 实现**：自研轻量版，不依赖 peft 库，直接对选定线性层注入双分支低秩矩阵。
- **伪 OOD 构造**：**70% 毒性特化 + 30% 通用随机**，训练时按 **1:3（OOD:Known）** 混入同一 batch。
  - Toxic-specific：对已知毒性词进行强字符扰动（leet、空格规避、符号插入）、句式反转、跨语言混排。
  - Generic：Wikipedia/News 短文本采样或随机高斯无意义串。
  - 两者通过配置开关切换，也可纯用一种做消融。
- **训练器**：基于 transformers Trainer 以减少样板代码，通过 Callback 和重写 `compute_loss` 注入自定义逻辑。
- **显存优化**：梯度累积（accumulation_steps=2 或 4）、混合精度（fp16=True）、必要时开启 `gradient_checkpointing=True`。

### 待讨论 / 可调整

- **双分支秩选择**：`r_s=8, r_p=4` 为经验初值，需根据阶段0验证集性能调参。若稳定分支容量不足，可上调 `r_s`；若可塑分支过拟合新类表面形式，可下调 `r_p` 或加大 L1 系数。
- **冻结 plastic 分支的长期参数预算**：当阶段数 K>5 时，冻结分支累积可能导致参数量上升。是否引入“冻结分支剪枝”（丢弃超过 `T_max` 阶段且低激活的分支）首版暂不实现，列 backlog。
- **K-means 前缀初始化 vs 随机初始化**：语义锚定前缀是核心创新点，但若聚类数 `n_anchors` 选择不当（过少则覆盖不足，过多则前缀冗长），可能反而引入噪声。需保留“随机初始化”对照作为 ablation。
- **变体生成器规则集 vs 小型 CNN**：当前计划用轻量规则（字符替换、正则）计算 `v_i` 与构造评测变体。若后续需要更复杂的形变（如 Unicode 同形字、emoji 穿插），是否引入小型 char-CNN 检测器，待评估工程收益。

---

## 验证步骤（可执行检查项）

1. [ ] `python -c "from models.roberta_dual_lora import DualBranchLoRALayer; print('OK')"` — 模块导入无报错
2. [ ] `python data/fscil_split.py --seed 42` — 生成切分文件，类别分布符合阶段定义
3. [ ] `python scripts/run_stage.py --stage 0 --config configs/base.yaml` — 阶段0训练1个epoch，损失下降，无OOM
4. [ ] `python -m pytest tests/test_consolidation.py` — 语义沉淀逻辑通过单元测试
5. [ ] `python scripts/evaluate.py --checkpoint checkpoint-stage2/ --variant_test data/variants_seed42.json` — 输出Variant Recall > 随机基线
6. [ ] `python scripts/run_baseline.py --method seq_finetune` — 基线可跑通，指标可复现

---

## 待确认与开放讨论（迁移前/后均可调整）

> 以下问题部分已有初步共识，但未在代码层面锁定；迁移后可根据目标机器环境、实验进展随时调整。建议在 `README.md` 中维护一张「决策日志」记录每次变更。

1. **Jigsaw 数据路径（已确认）**
   - 数据位于 `jigsaw-toxic-comment-classification-challenge/train.csv/train.csv`（解压后单文件）。
   - **开放**：若迁移后路径不同，仅修改 `configs/base.yaml` 中的 `data_path` 即可，无需改代码。

2. **FSCIL 阶段定义与 shot 数（建议初值，未锁定）**
   - 当前建议：阶段 0 `{toxic, obscene, insult}` → 阶段 1 `{threat, identity_hate}` → 阶段 2 `{severe_toxic}`。
   - shot 数建议 **16-shot**（11G 显存下稳定），若显存充裕或开启 gradient checkpointing 可尝试 32-shot。
   - **开放**：阶段数、每阶段类别数、类别组合均可调；Jigsaw 为多标签，FSCIL 通常按单标签任务切分，这里存在**多标签增量 vs 单标签增量**的范式选择，需后续验证哪种更符合毒性场景评测惯例。

3. **K-means 锚定前缀聚类数（建议初值）**
   - 建议 `n_anchors = 10~20`，与 hidden_size=768 匹配。
   - **开放**：若阶段 0 base 类样本聚类后质心分散，可适当增加；前缀长度 `m` 与聚类数直接相关，需与 `alpha`（锚定强度，建议 0.7）联合调参。

4. **双分支 LoRA 层选择（已确认首版范围，扩展开放）**
   - 首版仅 `q_proj/v_proj`，预留 `layers_to_adapt` 配置项。
   - **开放**：若消融实验显示表征瓶颈在 k_proj，可扩展；若显存吃紧，可回退到仅 v_proj。

5. **伪 OOD 构造（已确认混合策略，细节开放）**
   - 70% 毒性特化 + 30% 通用随机，1:3 混入 batch。
   - **开放**：毒性特化中的“跨语言混排”“句式反转”实现成本较高，首版可先用字符扰动替代；通用随机中“高斯无意义串”对语言模型过于简单，可能学偏，可替换为 Wikipedia 采样。

6. **损失权重与超参（全部未锁定）**
   - `lambda_evo, lambda_sp, beta, eta` 以及 `tau`（沉淀阈值）、`theta_coarse/fine`（拒识阈值）均需阶段 0 后在验证集上搜索。
   - **开放**：建议首版先用网格搜索/随机搜索确定一组基线值，写入 `configs/base.yaml`，后续实验再细化。

7. **评测指标权重（待讨论）**
   - Avg-mAP、Macro-F1、Forgetting 是 CIL 标准；Variant Recall 和 Semantic Stability(CKA) 是本方法特色指标。
   - **开放**：若审稿人质疑 CKA 计算开销大，可替换为更轻量的余弦相似度；若 Variant Recall 定义不清晰（多标签下“召回”如何判定），需统一口径。

8. **长期运行时的工程问题（backlog）**
   - 冻结 plastic 分支累积 → 参数量增长 → 推理时延上升。
   - **开放**：首版不实现分支剪枝，但在 `SemanticConsolidation` 类中预留剪枝接口；若 K>5 后时延显著，再实现基于激活频率的轻量剪枝。

9. **基线选择（建议最小集，可扩展）**
   - 必跑：Sequential Fine-tune、Task-LoRA、Task-LoRA+MSP、Task-LoRA+ADB。
   - **开放**：若算力允许，加入 MoCL、O-LoRA、ELLA 等强对照；若算力紧张，可仅跑 2~3 个基线。

10. **代码可移植性**
    - 计划使用 `transformers>=4.35`、`torch>=2.0`，迁移后需确认目标机器 CUDA 版本与 PyTorch 兼容性。
    - **开放**：若目标机器无网络或 huggingface 下载受限，需提前缓存 `roberta-base` 权重到 `MODEL_DIR`。
