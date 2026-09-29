# 创新点1探索：个体特异跨模态一致性差异身份表征

> 日期：2026-09-29  
> 当前目标：基于 user1/bo 的现有结果，沿开题报告“CycleGAN 跨模态循环一致性身份认证”设计，进一步验证身份信息是否真正来自用户特异的 ACC↔Audio 映射关系。

## 1. 当前实验对开题假设提出的新问题

开题报告的创新点1原本通过 CycleGAN 建模 ACC 与 Audio 的双向映射，并联合浅层纹理、瓶颈语义和 ACC→Audio→ACC 循环重建差异构造身份特征。

当前 user1 结果：

| 特征 | AUC |
|---|---:|
| ACC shallow | 0.874 |
| CycleGAN bottleneck | **0.917** |
| cycle-diff | 0.859 |
| combined frozen probe | 0.912 |
| 第一轮优化后 fusion512 | **0.967** |

当前证据说明：CycleGAN 内部确实存在身份信息，但 user1 上 bottleneck 比 absolute cycle residual 更强。因此不应继续预设“绝对循环重建误差一定是最具判别性的特征”，而应进一步研究：

> **用户专属跨模态模型对输入样本的响应模式，是否能够刻画个体特异的声振传播关系。**

## 2. 核心假设

将创新点1从“CycleGAN + 三类特征”进一步收敛为：

> **个体特异的 ACC–Audio 双向跨模态映射关系本身可以作为身份生物特征。**

对于目标用户 u 的专属模型，传统方法只看绝对循环残差：

```text
r_u(x) = |x - G_u(Audio→ACC)(G_u(ACC→Audio)(x))|
```

新的探索同时让同一个输入经过目标用户模型和非目标用户模型：

```text
r_target(x)
r_mismatch(x)
```

研究相对一致性差异：

```text
Δr(x) = φ(r_mismatch(x)) - φ(r_target(x))
```

其中 φ 可以使用 MAE、MSE、RMS、频带残差能量、残差统计量或残差 embedding。

核心问题：

> user1 genuine sample 是否在 user1 专属跨模态模型下，比在 mismatched 用户模型下表现出更高的跨模态一致性？

如果成立，认证依据就从“重建误差大小”升级为：

> **样本与目标用户特异跨模态传递关系的匹配程度。**

## 3. Bottleneck Response Gap

当前 user1 最强的单一跨模态特征是 bottleneck（AUC 0.917），因此同时研究不同用户模型的深层响应：

```text
z_1(x) = E_1(x)
z_j(x) = E_j(x)
Δz(x) = D(z_1(x), z_j(x))
```

D 可依次测试 cosine、Euclidean、Mahalanobis、prototype distance。

重点不是直接比较不同模型未对齐的 embedding 坐标，而是验证可定义的响应统计、距离或经校准后的相对匹配分数是否具有身份判别力。

## 4. user1 第一轮实验矩阵

固定同一数据划分、同一 final test 和同一阈值协议：

| ID | 特征 | 研究问题 |
|---|---|---|
| C0 | ACC shallow only | 单模态 ACC 本身有多强 |
| C1 | user1 bottleneck | 跨模态 encoder 是否增加身份信息 |
| C2 | user1 absolute cycle residual | 原开题报告假设 |
| C3 | bottleneck + absolute residual | 当前跨模态表示 |
| C4 | relative cycle consistency gap | 个体映射关系是否具有身份特异性 |
| C5 | calibrated bottleneck response gap | 深层跨模态响应是否具有相对匹配信息 |
| C6 | C4 + C5 | 两类相对响应是否互补 |
| C7 | shallow + bottleneck + residual + relative gaps | 完整表示 |

重点判断：

```text
C4 > C2
C6/C7 > C3
```

如果成立，创新点1可以从“循环重建误差身份特征”升级为“个体特异跨模态映射响应身份特征”。

## 5. 必须加入 amplitude-normalized 对照

当前 user1 的 shallow ACC AUC 已达 0.874，且已有分析显示其 ACC 全局统计具有明显区分性，因此必须排除佩戴压力、sensor gain、幅值范围、接触状态等 shortcut。

同一组 C0–C7 同时测试：

```text
N0: current preprocessing
N1: per-window RMS normalization
N2: per-window mean + std normalization
N3: train-set-only z-score
```

理想结果：

```text
ACC shallow        明显下降
bottleneck         保留较多性能
relative gap       保持稳定
full cross-modal   仍优于 ACC-only
```

如果出现该结果，可以更有力地说明跨模态模型利用的不只是绝对幅值/佩戴条件，而包含更稳定的声振传播结构。

## 6. Paired U-Net baseline

由于 ACC 与 Audio 是同步配对数据，创新点1还必须回答：

> 为什么需要 CycleGAN，而不是普通 paired ACC→Audio 网络？

user1 增加公平对照：

```text
ACC-only
Paired U-Net ACC→Audio bottleneck
CycleGAN bottleneck
CycleGAN absolute residual
CycleGAN relative consistency gap
```

要求保持相同 train/validation/test、尽量相同 encoder capacity、相同 authentication backend 和 threshold protocol。

这样才能区分“跨模态监督本身的收益”与“双向循环建模的额外收益”。

## 7. 当前最值得立刻做的单个实验

### user1-specific CycleGAN vs mismatched CycleGAN response-gap

输入：

```text
user1 genuine
+
8-user impostor
```

优先利用已有 user1、user4、user5 CycleGAN 做 pilot，对相同样本导出：

```text
absolute bottleneck
absolute cycle residual
relative cycle consistency gap
calibrated bottleneck response gap
```

并同时运行：

```text
raw preprocessing
vs
amplitude-normalized preprocessing
```

该实验一次回答三个关键问题：

1. Audio 参与训练后的跨模态模型是否真正提供额外身份信息？
2. 身份信息主要来自 absolute reconstruction error，还是用户特异的跨模态映射关系？
3. 去除 ACC 幅值 shortcut 后，该跨模态身份信息是否仍存在？

## 8. 推荐实验顺序

```text
P0  user1 vs user4/user5 relative-response probe（无需重训）
P1  扩展到更多 user-specific CycleGAN，加入 hardest mismatched model
P2  amplitude-normalized ablation
P3  Paired U-Net baseline
P4  relative-gap 有效后再接入 fusion + PK + AAM/GE2E/SupCon
```

在 P0–P3 前不优先继续堆复杂认证头，以免下游监督模型掩盖上游特征是否真正包含跨模态身份信息。

## 9. 创新点1建议表述

不再强调：

> “循环重建误差一定是最具判别性的身份特征。”

建议调整为：

> **提出基于个体特异跨模态映射响应的身份表征方法。通过目标用户专属 ACC–Audio 双向映射模型，联合提取浅层振动纹理、跨模态瓶颈表征与循环一致性响应，并利用样本在匹配与非匹配跨模态映射下的相对响应差异，刻画用户特异的声振传播关系，实现基于发声振动信号的身份认证。**

核心研究命题：

```text
Identity ≈ Sample-to-User Cross-modal Mapping Compatibility
```

而不是：

```text
Identity ≈ Absolute Reconstruction Error Only
```

## 10. 成功判据

本探索最有价值的结果不是单纯降低若干百分点 EER，而是建立以下证据链：

```text
ACC-only
    <
cross-modal representation
    <
user-specific relative cross-modal response
```

并且该关系在 amplitude normalization 后仍成立。

若成立，创新点1即可从“模型组合”提升为更清晰的科学贡献：

> **个体特异的 ACC–Audio 跨模态传递/响应关系能够提供超出单模态 ACC 简单统计特征的身份判别信息。**
