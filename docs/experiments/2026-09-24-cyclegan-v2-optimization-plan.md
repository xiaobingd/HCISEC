# CycleGAN v2 优化实验方案

- 日期：2026-09-24
- 目标：在不改变研究任务定义的前提下，系统优化当前 ACC ↔ Audio 个体化 CycleGAN 训练流程，并判断性能瓶颈究竟来自训练逻辑、数据归一化、损失设计还是模型容量。
- 原则：一次只改变一类因素；先在代表性用户上筛选，再扩展到 9 用户；所有正式结论必须基于严格训练/测试隔离协议。

## 1. 当前问题背景

现有实验显示 bo 的认证效果显著好于多数用户，推测一部分原因来自 bo 数据本身质量较高。但当前 CycleGAN 训练代码也存在若干可能放大用户差异的问题：

1. 动态 loss 权重调度存在阶段公式不一致；
2. validation 没有真正用于 best-checkpoint 选择；
3. 当前归一化逻辑存在实现和实验定义不清晰的问题；
4. generator 同时优化多个损失，可能在小样本条件下互相干扰；
5. 当前生成器容量相对单用户约 300 条数据可能偏大；
6. 当前数据是 ACC/Audio 成对采集，必须补充 paired supervised baseline，验证 CycleGAN 本身是否必要。

因此本轮优化不直接增加更复杂模型，而是优先清理训练逻辑并建立可解释对照。

---

## 2. 第一阶段：修复训练逻辑

### 2.1 修复 dynamic lambda schedule

当前阶段定义：

- Stage 1：epoch 0–49
- Stage 2：epoch 50–149
- Stage 3：epoch 150+

但 Stage 2 仍使用旧公式：

```python
p = (epoch - 30) / 70.0
```

这会导致 Stage 2 后半段 p > 1，从而让部分权重超出设计范围，甚至出现负权重。

应修改为：

```python
p = (epoch - 50) / 100.0
p = min(max(p, 0.0), 1.0)
```

建议 Stage 2 权重保持如下线性插值：

```text
adv        : 0.5 → 1.0
rec        : 10.0 → 5.0
cycle      : 1.0 → 8.0
identity   : 1.0 → 0.1
freq       : 3.0 → 2.0
smooth     : 0.5 → 0.2
feat_match : 2.0 → 4.0
```

并增加断言：

```python
assert lambda_identity >= 0
assert lambda_smooth >= 0
```

实验编号：

```text
C2-BASE-FIX
```

该实验只修正 schedule，不改网络、不改数据、不改 loss 组成。

---

## 3. 第二阶段：引入 validation best-checkpoint

当前训练主要按固定 epoch 保存模型，不同用户可能在不同训练阶段达到最优状态。

因此每 5 个 epoch 在 CycleGAN validation 上评估：

- ACC → Audio reconstruction L1；
- Audio → ACC reconstruction L1；
- cycle reconstruction L1；
- frequency-aware loss；
- 可选：paired correlation / cosine similarity。

推荐 validation score：

\[
L_{val}
=
L_{rec}
+
\lambda_c L_{cycle}
+
\lambda_f L_{freq}
\]

禁止使用最终 authentication test 选择 generator checkpoint。

保存：

```text
best_generator.pth
last_generator.pth
```

并记录：

```text
best_epoch
best_val_score
last_val_score
```

实验编号：

```text
C2-BESTCKPT
```

目标是回答：

> 各用户是否存在明显不同的最优训练 epoch，固定 epoch=200 是否导致部分用户过拟合？

---

## 4. 第三阶段：Normalization Ablation

当前数据加载器中的 normalize 参数实现需要统一，并明确所有统计量只能由训练集估计。

必须避免利用 validation/test 统计量进行预处理。

比较四种方案：

| 编号 | 方案 |
|---|---|
| N1 | 当前 per-sample normalization |
| N2 | train-set global min-max |
| N3 | train-set z-score |
| N4 | log-magnitude + train-set z-score |

### N1：当前方案

保留现有 normalize_mel_spectrogram / normalize_magnitude 逻辑，作为 baseline。

### N2：训练集全局 min-max

对每个模态只用 train set 计算：

\[
x' = \frac{x-x_{min}^{train}}{x_{max}^{train}-x_{min}^{train}}
\]

val/test 使用完全相同的训练统计量。

### N3：训练集 z-score

\[
x' = \frac{x-\mu_{train}}{\sigma_{train}}
\]

### N4：log magnitude + z-score

先：

\[
x' = \log(1+|x|)
\]

再使用 train-set mean/std 做标准化。

需要特别观察：

- 是否保留用户间能量差异；
- 低信噪比用户是否改善；
- bo 的优势是否显著缩小或保持。

实验编号：

```text
C2-NORM-N1
C2-NORM-N2
C2-NORM-N3
C2-NORM-N4
```

---

## 5. 第四阶段：Loss Ablation

当前 generator 同时使用：

\[
L_{adv},
L_{rec},
L_{cycle},
L_{identity},
L_{feature},
L_{freq},
L_{smooth}
\]

单用户约 300 条样本下，过多约束可能导致优化目标相互冲突。

建议先定义基础版本：

\[
L_{base}
=
L_{adv}
+
10L_{rec}
+
5L_{cycle}
\]

然后逐项增加。

| 编号 | 损失组成 |
|---|---|
| L0 | Base |
| L1 | Base + identity |
| L2 | Base + frequency |
| L3 | Base + feature matching |
| L4 | Base + smooth |
| L5 | Full current loss |

实验编号：

```text
C2-LOSS-L0
C2-LOSS-L1
C2-LOSS-L2
C2-LOSS-L3
C2-LOSS-L4
C2-LOSS-L5
```

注意：最终选择不能只看 generator validation loss。

必须同时看下游 authentication：

- EER；
- FAR；
- FRR；
- HTER；
- Balanced Accuracy；
- genuine / impostor separation。

核心问题：

> 哪些 CycleGAN loss 真正提升身份认证，而不仅仅让生成结果更平滑或更好看？

---

## 6. 第五阶段：Paired Supervised Baseline

由于 ACC 与 Audio 是同一次单词发声的成对数据，必须验证是否真的需要 CycleGAN。

建立 paired supervised mapping：

```text
ACC → Audio generator
Audio → ACC generator
```

不使用 discriminator，不使用 adversarial loss。

推荐 baseline：

\[
L
=
L_{rec}
+
\lambda_f L_{freq}
\]

可增加 cycle consistency 作为第二个 paired baseline：

\[
L
=
L_{rec}
+
\lambda_cL_{cycle}
+
\lambda_fL_{freq}
\]

比较：

| 方法 | GAN | Cycle | Paired reconstruction |
|---|---:|---:|---:|
| Paired-L1 | 否 | 否 | 是 |
| Paired-L1+Cycle | 否 | 是 | 是 |
| CycleGAN-v2 | 是 | 是 | 是 |

实验编号：

```text
C2-PAIR-P1
C2-PAIR-P2
C2-CYC-V2
```

如果 paired supervised mapping 与 CycleGAN 性能相同或更好，则论文中必须重新评估 CycleGAN 的必要性。

---

## 7. 第六阶段：模型容量 Ablation

只有在前五阶段完成后，再考虑模型结构。

当前结构：

```text
hidden_dim = 64
ResBlocks = 6
Axial Attention = on
dropout = 0
```

建议比较：

| 编号 | hidden_dim | ResBlocks | Attention | Dropout |
|---|---:|---:|---|---:|
| M0 | 64 | 6 | on | 0.0 |
| M1 | 48 | 4 | on | 0.1 |
| M2 | 32 | 4 | on | 0.1 |
| M3 | 48 | 4 | off | 0.1 |

优先观察是否存在：

> 简化模型后困难用户泛化反而更好。

实验编号：

```text
C2-MODEL-M0
C2-MODEL-M1
C2-MODEL-M2
C2-MODEL-M3
```

---

## 8. 首轮只跑 3 个代表性用户

为了节约 GPU 时间，前期不直接跑 9 用户。

选择：

- bo：容易用户、数据质量较好；
- user3：困难用户；
- user8：当前最困难用户之一。

筛选标准：

1. 改动必须不能只改善 bo；
2. 至少在 user3 / user8 中一个用户上明显改善；
3. 最好同时降低用户间性能差距；
4. validation 改善必须与独立 authentication test 方向基本一致。

只有通过三用户筛选的配置才扩展到 9 用户。

---

## 9. 推荐执行顺序

严格按以下顺序进行：

```text
C2-BASE-FIX
修 dynamic lambda
        ↓
C2-BESTCKPT
validation best checkpoint
        ↓
C2-NORM
normalization ablation
        ↓
C2-LOSS
loss ablation
        ↓
C2-PAIR
paired supervised baseline
        ↓
C2-MODEL
模型容量优化
        ↓
9-user validation
```

禁止一次同时修改：

- loss；
- normalization；
- model size；
- learning rate；
- architecture。

否则无法确定性能提升来源。

---

## 10. 训练与评估协议

所有实验必须使用同一份不可变 split manifest。

### CycleGAN

只能使用：

```text
outer_train
```

进行梯度训练。

可在 outer_train 内划分：

```text
generator_train
generator_validation
```

best checkpoint 只根据 generator validation 选择。

### Authentication

必须保证：

```text
final_test ∩ CycleGAN train = 0
final_test ∩ authentication train = 0
```

最终 authentication test 在所有模型选择完成后一次性评估。

---

## 11. 每个 CycleGAN 实验必须保存的结果

每个用户保存：

```text
config.json
split_manifest.json
train_log.txt
best_generator.pth
last_generator.pth
loss_history.csv
validation_metrics.csv
```

GitHub 只提交：

- 配置；
- 模型 SHA-256；
- split manifest SHA-256；
- 聚合 validation 指标；
- 聚合 authentication 指标；
- 实验结论。

不上传原始用户数据、模型二进制或逐样本隐私数据。

---

## 12. 推荐重点分析指标

### CycleGAN 本身

- validation reconstruction L1；
- cycle L1；
- frequency loss；
- paired correlation；
- best epoch；
- train-val gap。

### Authentication 下游

- EER；
- FAR；
- FRR；
- HTER；
- Balanced Accuracy；
- macro-average；
- worst-user；
- hard-impostor pair。

必须优先以下游身份认证指标决定最终配置。

---

## 13. 关于 bo 的数据质量假设

当前可以提出但不能直接下结论：

> bo 的数据可能具有更高信噪比、更稳定的跨模态对应关系，因此其 CycleGAN 更容易学习。

后续应验证：

1. bo 是否具有更低的 reconstruction validation loss；
2. bo 是否具有更高的 ACC-Audio paired correlation；
3. bo 的 train-val gap 是否更小；
4. user3/user8 是否表现出更高噪声、更低稳定性；
5. normalization 改变后用户间差距是否缩小。

如果 bo 在不同训练策略下始终明显更优，则数据质量解释更可信。

---

## 14. 第一轮建议立即执行的最小实验

第一轮只做以下四个配置：

| 配置 | 说明 |
|---|---|
| A | 当前模型 + 修复 lambda |
| B | A + best-checkpoint |
| C | B + train-set z-score |
| D | B + 简化模型（48通道、4 ResBlocks、dropout 0.1） |

只跑：

```text
bo
user3
user8
```

推荐先训练 200 epoch，并启用 validation best-checkpoint。

如果 B/C/D 中出现明显稳定改善，再扩展 loss ablation 与 9 用户。

---

## 15. 成功标准

CycleGAN v2 不以“生成图更漂亮”为主要成功标准。

至少满足以下之一才值得进入九用户完整实验：

1. user3/user8 的认证 EER 明显下降；
2. worst-user HTER 明显改善；
3. macro 性能不下降，同时用户间方差下降；
4. paired unseen samples 的跨模态映射稳定性提高；
5. 简化模型达到相同性能但训练成本显著降低。

最终目标是：

> 找到对不同用户都更稳定、对数据质量差异更不敏感，并且真正提升身份认证泛化能力的跨模态映射训练方案。
