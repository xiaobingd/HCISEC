# 对比学习认证头诊断与创新点 1 特征体检方案

- 日期：2026-09-28
- 状态：待执行
- 适用代码：trainb.py、modelb.py 及 CycleGAN 特征提取链路
- 目标：定位当前对比学习认证头效果较差的原因，并判断问题主要来自 CycleGAN 身份特征、特征融合、projection head、SupCon/ArcFace，还是数据与评估协议。

## 1. 核心问题

当前认证链路：ACC → CycleGAN → F1 shallow / F2 bottleneck / F3 cycle-difference → feature fusion → 512-d fusion → 128-d projection → L2 normalization → SupCon / ArcFace → cosine / threshold。

首先回答：去掉复杂认证头后，CycleGAN 提取的 F1/F2/F3 本身是否具有稳定的身份可分性？

需要区分三种情况：
1. F1/F2/F3 本身可分，但 fusion/projection 后明显变差：认证头或融合设计存在问题。
2. F1/F2/F3 不可分，但 target-specific cross-modal consistency / reconstruction 能区分 genuine 与 impostor：身份信息更可能存在于用户特异性的跨模态映射关系中，而非统一 embedding。
3. 两者均不能稳定区分：重新检查创新点 1 的核心假设和上游 CycleGAN 表征。

## 2. 第一阶段：认证头代码静态审查

重点检查 trainb.py 和 modelb.py。

### 2.1 SupCon

- positive mask 是否正确；
- 同一 identity 在一个 batch 内是否至少 2 个样本；
- 是否存在 anchor 没有 positive；
- self-contrast 是否正确排除；
- temperature 和 logit 数值范围；
- loss 输入是否在正确维度 L2 normalization；
- label shape、device、dtype；
- positive pair 是同用户不同窗口，还是仅同一窗口 augmentation；
- train / validation 是否使用同一 representation space。

### 2.2 Batch / sampler

统计每个 batch 的 identity 数、每个 identity 的样本数、每个 anchor 的有效 positive 数和 zero-positive anchor 数。必要时改用 PK sampler，例如 P=9、K=4、batch=36；实际 P/K 按训练用户数和显存调整。

### 2.3 ArcFace

检查 ArcFace 输入和 class weight 的 normalization、scale s、margin m、与 SupCon 的 loss 权重，以及两个 loss 是否在同一 representation 上产生冲突梯度。先做单 loss 对照，不预设 SupCon+ArcFace 一定优于简单分类头。

## 3. 第二阶段：F1/F2/F3 尺度与融合检查

对 F1、F2、F3 分别记录 dimension、mean、std、L2 norm mean/std、min/max、per-dimension variance。特别检查 F2 是否因维度更高或尺度更大而主导 concat。

至少比较：F1、F2、F3、F1+F2、F1+F3、F2+F3、F1+F2+F3、source-wise L2 normalized F1+F2+F3。不要预设三路融合一定最好。

source-wise normalized fusion 可写为：Fi_hat = Fi / ||Fi||2，然后 F=[α1F1_hat; α2F2_hat; α3F3_hat]。

## 4. 第三阶段：逐级 feature separability 诊断

对同一固定数据划分保存：raw ACC baseline feature、F1、F2、F3、F1+F2+F3、fusion output (512-d)、projection output (128-d)、final normalized embedding。

计算：
- S_intra = E[cos(zi,zj) | yi=yj, i!=j]
- S_inter = E[cos(zi,zj) | yi!=yj]
- ΔS = S_intra - S_inter

同时记录 intra/inter cosine mean/std、hardest impostor cosine、per-user statistics、prototype accuracy、kNN、linear probe，以及协议允许时的 verification EER。

如果 F1/F2/F3 的 ΔS 较好但 projection 后骤降，重点检查认证头；如果 F1/F2/F3 的 ΔS 都接近 0，则不应继续只调 SupCon 超参数。

## 5. 第四阶段：最小认证 baseline

按顺序建立：
1. frozen CycleGAN feature + nearest prototype；
2. frozen CycleGAN feature + kNN；
3. frozen CycleGAN feature + Linear CE；
4. frozen CycleGAN feature + ArcFace；
5. frozen CycleGAN feature + SupCon；
6. frozen CycleGAN feature + SupCon + ArcFace。

如果 Linear CE 都无法在 validation/test 上形成身份可分性，优先检查上游 feature，而不是继续增加 metric-learning 模块。

## 6. 第五阶段：数据与评估协议

采用固定 outer split：outer_train 内进行 CycleGAN training、auth-head training 和 validation/model selection；final_test 仅在全部模型与阈值冻结后使用。

要求：
- final test 不进入 CycleGAN 训练；
- final test 不进入认证头训练；
- final test 不用于 threshold selection；
- validation threshold 冻结后再报告 test FAR/FRR/HTER；
- 检查 sliding-window parent recording 是否跨 split；
- 有多 session 时优先 session-disjoint；
- 保存 sample ID / source recording ID 并自动检查交集。

旧九用户结果可作为历史结果，但存在跨阶段数据泄漏，不能作为最终独立测试结论。2026-09-23 严格 holdout 结果也不能单独证明认证头无效，因为该协议显著减少了认证头训练样本和注册样本。需要建立“无泄漏但训练数据利用合理”的固定协议。

## 7. 第六阶段：CycleGAN 上游特征体检

### 7.1 Normalization

比较 per-sample min-max、train-set global min-max、train-set z-score、log magnitude + train-set z-score。逐样本 min-max 可能消除绝对幅值/能量等潜在身份信息，应作为正式消融变量。

### 7.2 生成目标与身份目标

CycleGAN 的 adversarial、reconstruction、cycle、identity、feature matching、frequency/smoothness 等目标并不自动保证中间 representation 满足 same user close / different users far。因此必须直接验证生成表征的身份可分性，不能仅以生成质量推断。

## 8. 建议结果表

| Feature / Head | S_intra | S_inter | ΔS | Linear Probe | EER | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw ACC baseline | | | | | | | |
| F1 | | | | | | | |
| F2 | | | | | | | |
| F3 | | | | | | | |
| F1+F2 | | | | | | | |
| F1+F3 | | | | | | | |
| F2+F3 | | | | | | | |
| F1+F2+F3 | | | | | | | |
| normalized fusion | | | | | | | |
| + Linear CE | | | | | | | |
| + ArcFace | | | | | | | |
| + SupCon | | | | | | | |
| + SupCon + ArcFace | | | | | | | |

另外必须保存 per-user 结果，不只报告 macro average。

## 9. 决策规则

### A. F1/F2/F3 可分，projection 后变差
CycleGAN 已提取身份相关信息，主要问题位于 fusion / projection / metric-learning head，优先修认证头。

### B. 单个特征一般，但 cycle residual 对 hard impostor 有明显增益
说明跨模态循环一致性包含补充身份信息，可直接支撑创新点 1 中 cycle residual 的增量价值，需要用消融和 hard-impostor FAR 证明。

### C. 统一 embedding 不可分，但 target-specific consistency 有效
身份特异性可能主要体现为用户与跨模态映射模型的匹配关系，而非共享 embedding。此时不要强行把创新点 2 建立在 SupCon embedding 上，可转向少样本用户特异性跨模态模型快速适配 / 个体化认证边界。

### D. 两者均不稳定
重新审查创新点 1 的跨模态身份假设、数据协议和特征定义，不继续堆叠复杂认证模块。

## 10. 当前执行优先级

P0 审查 trainb.py/modelb.py 的 SupCon、ArcFace、sampler、normalization → P1 导出 F1/F2/F3/fusion/projection → P2 计算逐级 intra/inter cosine 与 ΔS → P3 prototype/kNN/Linear CE → P4 单独 ArcFace、单独 SupCon → P5 SupCon+ArcFace → P6 根据结果决定是否修改 CycleGAN representation。

在 P0–P3 完成前，不优先投入大量时间调 temperature、projection 层数或复杂 loss 权重。

## 11. 与创新点 2 的关系

本实验首先服务于创新点 1 的证据链，不预设创新点 2 必须依赖 SupCon embedding。只有共享身份 embedding 的可迁移性得到验证后，创新点 2 才优先采用 shared cross-modal identity prior → unseen-user K-shot enrollment → user-specific distribution/boundary → open-set authentication。

如果共享 embedding 假设不成立，则根据本实验结果重新选择创新点 2 的技术路线，避免把第二创新点建立在未经验证的前提上。