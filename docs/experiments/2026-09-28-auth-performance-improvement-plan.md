# 认证效果提升实验方案

> 日期：2026-09-28  
> 目标：在不优先改动 CycleGAN 主体的前提下，针对当前认证头和特征融合链路进行结构优化，优先降低 EER / FAR / FRR / HTER，并提升 worst-user 和 hard-impostor 表现。

## 1. 当前依据

根据 PR #1 的诊断结果：

- F1/F2/F3 原始特征的直接 cosine 分离度很低；
- fusion 512d 后身份分离明显增强；
- projection 128d 后分离度下降；
- F2 bottleneck 的特征范数明显高于 F1/F3，存在尺度主导；
- 普通随机 batch 下存在 zero-positive anchor；
- SupCon 相比 frozen linear probe 只有小幅提升；
- bo/user1 的优势部分来自原始 ACC 分布 shortcut，而非更好的跨模态对齐。

因此，下一阶段优先优化身份空间构造、分支归一化、采样和 verification-oriented loss，而不是继续增大 CycleGAN 模型。

---

## 2. 推荐认证头结构

将当前：

```text
F1 + F2 + F3
   ↓
raw concat
   ↓
fusion MLP
   ↓
128-d projector
   ↓
SupCon
```

修改为：

```text
F1 128d ─ Linear(128→128) ─ LayerNorm ─ GELU ─ L2 ┐
                                                    │
F2 512d ─ Linear(512→128) ─ LayerNorm ─ GELU ─ L2 ├─ concat
                                                    │
F3 132d ─ Linear(132→128) ─ LayerNorm ─ GELU ─ L2 ┘
                                                    ↓
                                                  384d
                                                    ↓
                                               Fusion MLP
                                                    ↓
                                               256/512d
                                                    ↓
                                                 L2 norm
                                                    ↓
                                   ArcFace / AAM / GE2E / SupCon
```

第一轮不强制保留现有 128-d projector。

---

## 3. P0：source-wise normalization

当前 F1/F2/F3 直接 concat，且 F2 的 L2 norm 显著更大。

因此先做：

```python
h1 = normalize(gelu(layernorm(linear1(F1))))
h2 = normalize(gelu(layernorm(linear2(F2))))
h3 = normalize(gelu(layernorm(linear3(F3))))
fused_input = torch.cat([h1, h2, h3], dim=1)
```

目标：

- 消除 F2 的尺度主导；
- 让三路 feature 在同一量级进入 fusion；
- 检查 F1/F2/F3 是否真正互补。

对照：

1. raw concat；
2. source-wise L2 normalization；
3. source-wise Linear + LayerNorm + GELU + L2。

---

## 4. P0：去掉当前 projector 做直接对比

由于当前诊断显示：

```text
fusion 512d ΔS ≈ 0.23
projection 128d ΔS ≈ 0.09–0.12
```

需要直接比较：

```text
A. fusion512 → L2 → cosine
B. fusion512 → projector128 → L2 → cosine
C. fusion512 → projector256 → L2 → cosine
```

若 final test 同样显示 A 优于 B，则移除当前 128-d projector，或改为更弱的 256-d 压缩。

---

## 5. P0：PK sampler

当前普通 `shuffle=True` 会出现 zero-positive anchor。

改为：

```text
P identities × K samples per identity
```

优先尝试：

```text
P=9, K=4, batch=36
```

显存不足时：

```text
P=6, K=4, batch=24
```

要求：

- 每个 anchor 至少有 1 个 same-user positive；
- 训练日志记录：
  - identities per batch
  - positive count per anchor
  - zero-positive ratio
  - hardest-negative similarity

---

## 6. P1：verification-oriented loss 对比

统一 feature、统一 split、统一 sampler，依次比较：

```text
A0: Linear CE
A1: ArcFace / AAM-Softmax
A2: GE2E
A3: SupCon
A4: SupCon + ArcFace
```

### ArcFace / AAM

重点检查：

- embedding L2 normalize；
- class weight normalize；
- margin m；
- scale s；
- 小样本条件下避免 margin 过大。

### GE2E

GE2E 直接围绕每个用户 centroid 建立 verification objective，与注册模板 + cosine 的部署方式更一致。

推荐 batch 结构：

```text
P users × K utterances
```

每个 batch 内直接计算 user centroid 和样本到 centroid 的相似度。

---

## 7. P1：hard impostor mining

维护用户级 similarity matrix：

```text
D_ij = mean cosine(user_i, user_j)
```

每隔若干 epoch 更新一次。

训练 batch 中提高 hardest impostor 的采样概率：

```text
target
+ top-2/top-3 hardest impostors
+ random impostors
```

不要把所有 impostor 合并成一个 binary 类，仍保留 9-way identity structure。

可额外加入 margin ranking：

```math
L_{hard} = max(0, m + s_{hard} - s_{genuine})
```

---

## 8. P1：amplitude / gain normalization

为了抑制 ACC shortcut，对输入增加以下对照：

```text
N0: current preprocessing
N1: per-window RMS normalization
N2: per-window mean + variance normalization
N3: train-set z-score
```

目的：

- 判断 bo/user1 优势是否主要来自幅值、增益、佩戴压力、接触状态；
- 验证 bottleneck / cycle-diff 是否仍保留结构性身份信息。

关键观察：

如果 normalization 后 shallow ACC 明显下降，但 F2/F3 仍有效，则更支持“跨模态结构身份信息”。

---

## 9. P2：nuisance-invariant augmentation

对同一 ACC 样本随机模拟：

```math
x' = a x + b
```

其中：

- `a`：gain / contact-pressure perturbation；
- `b`：offset perturbation。

把 `x` 与 `x'` 作为同身份 positive，要求：

```math
f(x) \approx f(x')
```

从而减少模型对简单幅值 shortcut 的依赖。

---

## 10. P2：多单词 / 多窗口 temporal fusion

当前数据天然适合连续认证。

比较：

```text
1-word
3-word
5-word
10-word
```

两种方式：

### Embedding averaging

```math
z_T = \frac{1}{T}\sum_{t=1}^{T} z_t
```

然后 L2 normalize 后 cosine。

### Score averaging

```math
s_T = \frac{1}{T}\sum_{t=1}^{T}s_t
```

报告：

- EER vs T
- FAR vs T
- FRR vs T
- HTER vs T
- decision latency

该实验非常适合 VR continuous authentication 场景。

---

## 11. 第一轮最小实验矩阵

只先跑以下 6 个配置：

| ID | Branch normalization | Projector | Sampler | Loss |
|---|---|---|---|---|
| A0 | current | 128d | random | SupCon |
| A1 | source-wise norm | 128d | random | SupCon |
| A2 | source-wise norm | none / fusion direct | random | SupCon |
| A3 | source-wise norm | none / fusion direct | PK | SupCon |
| A4 | source-wise norm | none / fusion direct | PK | ArcFace/AAM |
| A5 | source-wise norm | none / fusion direct | PK | GE2E |

只要 A1/A2/A3 能明显提升，就说明当前瓶颈主要在 fusion/projector/sampling，而不是 CycleGAN。

---

## 12. 第二轮

对第一轮最优模型增加：

```text
B1: hard impostor mining
B2: amplitude normalization
B3: gain/contact augmentation
B4: SupCon + ArcFace
B5: 3-word temporal fusion
B6: 5-word temporal fusion
B7: 10-word temporal fusion
```

---

## 13. 统一评价指标

所有实验固定报告：

- EER
- FAR
- FRR
- HTER
- balanced accuracy
- AUC
- TAR@FAR
- per-user EER/FAR/FRR
- worst-user HTER
- hard-impostor FAR
- 5 seeds mean ± std

阈值必须在 dev / calibration 上选择，final test 只做冻结评估。

---

## 14. 当前优先级

```text
P0  source-wise normalization
P0  fusion512 direct verification / projector ablation
P0  PK sampler
P1  ArcFace / AAM
P1  GE2E
P1  hard impostor mining
P1  amplitude normalization
P2  nuisance augmentation
P2  multi-word temporal fusion
P3  再考虑是否修改 CycleGAN 主体
```

当前阶段不优先：

- 继续加深 CycleGAN；
- 增加更多 attention；
- 大规模搜索 SupCon temperature；
- 直接切换 diffusion；
- 在三路 feature 尺度问题未解决前增加复杂 fusion。

---

## 15. 最终目标结构

推荐最终主线：

```text
ACC
 ↓
CycleGAN multi-level feature extraction
 ↓
F1 / F2 / F3
 ↓
branch-wise normalization + projection
 ↓
balanced fusion
 ↓
GE2E / AAM identity embedding
 ↓
hard-impostor-aware training
 ↓
multi-utterance temporal aggregation
 ↓
cosine verification + frozen threshold
```

原则：

> 先提升身份空间质量，再优化认证损失；先稳定单窗口 embedding，再利用 VR 连续交互做时序证据累积。
