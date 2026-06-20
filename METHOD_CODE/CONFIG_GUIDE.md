# CONFIG_GUIDE.md — FSCIL Toxic Comment Classification 配置指南

> 目标环境：单卡 **11 GB VRAM**（如 RTX 2080 Ti / RTX 3060）。
> 所有建议均基于 `roberta-base`（125 M 参数）。

---

## 1. 显存预算总览

| 组件 | 估算显存 |
|------|---------|
| RoBERTa-base 权重（fp32） | ~500 MB |
| 激活值（seq_len=128, batch=8） | ~3.5 GB |
| 双分支 LoRA（q+v, r_s=8, r_p=4） | ~30 MB |
| 毒性语义前缀（m=10, 12 层） | ~1.5 MB |
| ToxicAwarePE + 分类头 | ~5 MB |
| 优化器状态（AdamW, fp32） | ~1.5 GB |
| **总计（训练峰值）** | **~6–8 GB** |

**结论**：11 GB 显存下安全，预留约 3 GB 余量给动态分配和 evaluation。

---

## 2. 关键超参与显存/性能权衡

### 2.1 LoRA 配置 (`lora`)

| 参数 | 默认值 | 可调范围 | 影响 |
|------|--------|---------|------|
| `target_modules` | `["query", "value"]` | 可扩展 `key`, `dense`（即 o_proj） | 每增加一个模块，参数量 +~15 MB，激活显存略增 |
| `rs` (stable rank) | 8 | 4–16 | **稳定分支容量**。若 Variant Recall 低或遗忘严重，优先上调 `rs` |
| `rp` (plastic rank) | 4 | 2–8 | **可塑分支容量**。若新类过拟合表面形式，下调 `rp` 或加大 `lambda_sp` |
| `alpha` | 16 | 固定或 2×rs | 缩放因子，通常保持 `alpha = 2 * rs` |
| `dropout` | 0.05 | 0.0–0.1 | 正则化，显存无影响 |

**首版锁定 `q_proj + v_proj`** 的理由：
- q/v 已覆盖毒性语义锚定与表面形式匹配的核心通路。
- 扩展 `k_proj` 对表征收益有限，但显著增加激活显存（KV cache 翻倍）。
- 若后续消融发现表征瓶颈在 k_proj，可通过 `target_modules: ["query", "key", "value"]` 一键扩展。

### 2.2 前缀配置 (`prefix`)

| 参数 | 默认值 | 可调范围 | 影响 |
|------|--------|---------|------|
| `m` (prefix_length) | 10 | 5–20 | 每增加 1，显存 +~0.15 MB。过长可能稀释注意力 |
| `n_anchors` | 5 | 5–20 | K-means 聚类数。若 base 类语义分散（如 obscene/insult 差异大），上调至 10–15 |
| `alpha_anchor` | 0.7 | 0.5–0.9 | 锚定强度。越高越依赖 K-means 质心，越低越自由学习 |
| `apply_to_all_layers` | `true` | 可改为仅前 6 层 | 全层注入显存开销极小，但可尝试仅前 N 层以加速推理 |

**注意**：`n_anchors` 必须 ≤ base 阶段有效样本数（否则 K-means 报错）。

### 2.3 ToxicAwarePE (`toxic_pe`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable` | `true` | 可关闭以做消融 |
| `q_dim`, `m_dim`, `l_dim`, `v_dim` | 1 | 投影到 hidden_size 前的特征维度。保持 1 即可，增大对显存几乎无影响 |

### 2.4 拒识门控 (`rejection_gate`)

| 参数 | 默认值 | 调参建议 |
|------|--------|---------|
| `theta_coarse` | 0.5 | 粗拒识阈值。若伪阳性高（正常文本被判 unknown），上调至 0.6–0.7 |
| `theta_fine` | 0.3 | 细拒识阈值。若已知框架变体漏判，下调至 0.2 |
| `learnable_weights` | `false` | 固定权重已够用。若数据充足，可设为 `true` 让 a/b/c/d 自适应 |

---

## 3. 训练配置 (`training`)

### 3.1 Batch Size 与梯度累积

| 显存 | per_device_bs | gradient_accumulation_steps | 等效 batch |
|------|--------------|----------------------------|-----------|
| 11 GB | 8 | 2 | 16 |
| 11 GB | 16 | 2 | 32（需关闭 Prefix 或调短 seq_len） |
| 8 GB | 4 | 4 | 16 |

**建议**：11 GB 下保持 `per_device_train_batch_size=8, gradient_accumulation_steps=2`。
若开启 `fp16=true`，可尝试 `batch_size=16`，但注意 fp16 在 BCELoss 中偶有数值不稳定。

### 3.2 阶段与 Shot 数 (`fscil`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| shots_per_class (base) | 32 | 阶段 0 样本充足，可支撑较大前缀和 LoRA 初始化 |
| shots_per_class (inc) | 16 | 增量阶段样本少，epochs 不宜过多，防止过拟合 |
| num_seeds | 3 | 论文报告标准差所需 |

**若显存紧张**：
- 将 base shots 降至 24，增量 shots 降至 8。
- 同步缩短 `prefix.m` 至 5，`rs` 降至 4。

---

## 4. 损失权重 (`loss_weights`)

| 权重 | 默认值 | 作用 | 调参建议 |
|------|--------|------|---------|
| `lambda_evo` | 0.5 | 语义演化一致性 | 若 Variant Recall 低，上调至 0.8–1.0 |
| `lambda_sp` | 1e-3 | Plastic L1 稀疏 + Stable 去相关 | 若 plastic 过拟合，上调至 1e-2 |
| `beta` | 0.3 | 拒识损失 | 若 AUROC 低，上调至 0.5；若已知类误判多，下调至 0.1 |
| `eta` | 1e-4 | 跨阶段正交性 | 仅 stage>0 激活。若遗忘严重，上调至 1e-3 |

**注意**：`lambda_merge`（merge smoothing）在 `stable_plastic_reg.py` 内部默认 1e-3，一般无需调整。

---

## 5. 语义沉淀 (`semantic_consolidation`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau` | 0.1 | 沉淀阈值。delta_k < tau 时 merge，否则 freeze |
| `eval_samples` | 200 | 计算 delta_k 的样本数。增大可提高估计稳定性，但略微增加阶段切换时间 |

**tau 调参指南**：
- `tau` 过低（如 0.05）：倾向于 freeze，导致 plastic 分支累积，推理时延上升。
- `tau` 过高（如 0.3）：倾向于 merge，可能把新类的表面噪声混入 stable，损害旧类性能。
- **建议**：先在验证集上观察 delta_k 的分布，取中位数作为 tau 初值。

---

## 6. 快速显存检查清单

运行训练前，执行以下检查：

```bash
# 1. 模块导入
python -c "from models.roberta_classifier import RobertaToxicClassifier; print('OK')"

# 2. 单 batch 前向峰值显存
python -c "
import torch
from models.roberta_classifier import RobertaToxicClassifier
model = RobertaToxicClassifier(num_classes=2).cuda()
x = torch.randint(0, 100, (8, 128)).cuda()
m = torch.ones(8, 128).cuda()
torch.cuda.reset_peak_memory_stats()
_ = model(x, m)
print(f'Peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
"
```

若 Peak > 9 GB，请按以下顺序降显存：
1. 减小 `per_device_train_batch_size` 到 4
2. 增大 `gradient_accumulation_steps` 到 4
3. 缩短 `max_length` 到 96
4. 关闭 `toxic_pe.enable`
5. 减小 `prefix.m` 到 5

---

## 7. 配置修改决策日志（示例）

| 日期 | 修改项 | 原因 | 结果 |
|------|--------|------|------|
| 2026-05-18 | `n_anchors` 5 → 10 | obscene/insult 聚类分散 | Stage 0 mAP +2.1% |
| 2026-05-18 | `beta` 0.3 → 0.5 | AUROC 仅 0.71 | AUROC 0.71 → 0.84 |

建议在 `configs/` 下保留 `base.yaml` 不动，通过 `stages.yaml` 和命令行 `--output_dir` 做实验级覆盖。
