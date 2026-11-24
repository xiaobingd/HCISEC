# Audio 注册条件化 ACC 身份验证器：验证独立跨模态身份增益

> 2026-11-24；**预先固定的实验计划，尚无新实测结果**。此分支独立于前面已经停止的“注册样本排序辅助”路线。只需要既有 10 人×40 词数据，不引入多会话约束。

## 核心目标

让注册阶段的 Audio 身份知识参与**目标用户的 ACC 验证决策函数**，而不是让 ACC 复现完整 Audio embedding、局部频谱，或从 32 条注册候选中选择 12 条。查询阶段仅有 ACC。只有正确 Audio 比纯 ACC 和**不同用户 Audio**的匹配模型在锁定的跨词认证上稳定更好，才可称为独立模态身份增益。

此前证据与动机：

- 50–300 Hz 正确配对局部 Audio 检索胜过错配，但没有稳定的认证收益；70–300 Hz 硬筛频带使认证退化。
- Audio/ACC 注册质量词内 Spearman ≈ -0.0179；注册 Audio 排序与边界样本选取不稳定，后者在 2026-11-24 已按协议停止。
- 因此避免再调频带、蒸馏强度、样本质量排序权重。新实验直接以**跨词本人/冒充验证误差**监督条件化模型。

## 模型：Enrollment-conditioned ACC verifier

冻结 ACC 编码器 E_ACC（stage3 A_ACC，分别 seed=42/43/44）与 Audio 教师 E_Audio（已验证官方 VoxCeleb ECAPA，192 维）。用同一组固定注册 ACC 构建目标原型 p_u；用对应注册 Audio 构建教师身份原型 a_u。**所有对照保留完全相同注册 ACC**；Audio 从不进入查询网络。

一个极小、身份共享的条件化网络 G_theta 读取 (p_u,a_u,ACC 注册离散度)，输出按通道权重 w_u∈[1−ε,1+ε]^192（例如 ε=0.25，参数初始化时 w_u=1）。对 ACC 查询 z_q 计算

    score_u(q) = weighted_cosine(z_q,p_u;w_u)
    weighted_cosine(x,p;w) =
      sum_d w[d]*x[d]*p[d] / sqrt(sum_d w[d]*x[d]^2 * sum_d w[d]*p[d]^2).

这样 Audio 不被强迫进入 ACC 的 192 维语义坐标；它仅预测哪些 ACC 身份特征**对这个注册人更可信**。把 G_theta 限制为小 MLP（隐藏维 32、LayerNorm/SiLU、单隐层）并控制权重范围，避免十人数据上高自由度记忆。可输出标量残差分数，但首轮禁用，以实现可归因的“通道门控”。

## 训练数据隔离

- 身份验证器 G_theta 的训练仅使用既定 16 个 fit 词 [0,1,2,4,5,6,7,10,11,13,14,15,16,17,33,34] 的 ACC+Audio 注册模拟集；需要**额外提取 fit-only 的 1280 条 Audio ECAPA embedding**，禁止从最终查询内容提取教师特征。
- 每个训练 episode 将某些词的 Audio+ACC 作为注册支持，**不同的 fit 词** ACC 作为查询（含本人及其他训练用户）。同一词 8 遍不是独立身份，不允许错遍作为身份负样本。
- 使用 10-fold leave-one-target-out 训练 G_theta：当前测试目标 u 的任何 ACC 或 Audio 不进入对应 G_theta 的优化（但**冻结 E_ACC 此前在 10 人 fit 数据上学习过身份**，故不可声称整套模型对未见用户的开放集泛化）。
- 评估用原定 REG [20,21,22,23] 4 词的 Audio+ACC 仅构建 p_u,a_u；固定同样的 12 条 ACC（参照旧 acc_crossword12）或预先固定的全部 32 条，两种预算不可混杂。主实验固定 **12 条**。
- CAL [8,18,24,30,31] 只使用 ACC 锁定阈值；最终 query 15 词：英文单词 [25..29]，英文长词 [3,9,12,19,32]，中文 [35..39]，**不读取查询 Audio**。主报告单词单条查询，与上一轮同口径；三词平均只能作为额外报告。
- 使用现有 44100 候选解释的官方 ECAPA 缓存与提取管线；真实音频采样率与物理同步仍未独立验证。基于 utterance 级 Audio 身份向量的本实验不要求逐帧 Audio/ACC 物理同步，但需要正确用户/录制归属。

## 必须匹配的对照

A: ACC-only condition network：把 Audio 输入设置为固定零向量，完全相同网络结构、训练 episode、步数、初始种子；与 G_theta 无门控的纯 ACC 原型同时报告。

B: 正确 Audio：目标 u 的真实注册 Audio 身份原型。

C: 不同人 Audio：把目标 u 的注册 Audio 替换为另一个用户同类注册词的 Audio，保留其 ACC 注册数据与真假标签；**在训练和评估中均有对应置换对照**，以区分偶然正则化和真实用户配对信息。

D: 随机/破坏 Audio 表征：与 Audio 均值、方差匹配但打乱用户的 teacher 特征，等结构训练。

E: 同用户不同注册词 Audio：不属于身份负样本，可作鲁棒性控制，应该支持相近的身份证据。

重要：所有 A/B/C/D 模型使用同样的 ACC 注册子集、初始化、训练批次、超参数、负类候选和锁阈值规则。禁止只对正确 Audio 使用更大的模型/更久训练。

## 训练损失与门控检查

以直接验证损失训练，不做 embedding MSE / 全频谱 SmoothL1：

    L = BCEWithLogits(m * (score_u(q)-b), genuine_label)
        + λ * max(0, margin + score_u(negative) - score_u(positive))
        + β * mean((w_u - 1)^2)

其中 impostor 在训练用户中取“相似度高的困难冒充者”，**完全相同的困难负样本索引提供给所有对照**；硬负类挖掘只看 ACC，不能借错误 Audio 形成伪收益。所有超参数由 fit 内部划分固定，不使用 CAL/最终 query 调门控。首轮可仅使用 BCE + 正则，在预先固定的训练条件下做 B/A/C/D，避免损失项搜索。条件网络只训练 G_theta 和共享的标量校准参数，ACC/Audio 两个编码器冻结。

## 成功、失败、诊断

1. 首先在**预定未见词（训练词集外）**的单条查询评估同一已锁阈值规则下 EER、HTER、FAR、FRR；10 人 × 3 冻结编码器种子全保留。
2. 按目标用户整簇 bootstrap 给出 B−A、B−C、B−D 的区间，逐用户显示改善人数，不能把词/8 遍/攻击比较当独立用户。
3. 额外报告门控 w_u 对 Audio 扰动的敏感性、same-user different-word 稳定性、不同用户 Audio 的退化；若 B≈A≈C/D，仅说明该实验没有证实 Audio 增益。
4. 若 B 比 ACC baseline、容量匹配的 A 和错模态 C/D **同时**稳定降低 HTER/EER，才继续研究音频条件化个体模板；若 B 只有轻微提升或换用户 Audio 也提升，则停止**该方法**，不据此宣布所有跨模态身份增益不可能。
5. 目前仅探索性 10 已见身份与多轮查看的查询词；即使成功仍需另一次独立确认后才作为论文核心结论。

## 最小实施顺序

1. 提取 fit-only ECAPA 1280 条（复用 `outputs/vr261124_enrollment_template_quality/source/work/extract_registered_ecapa.py` 的长度感知处理，保持官方权重 SHA），核对所有 sample_id、注册身份和词。
2. 构建严格不同词的 support/query episodes，并固定所有随机种子与负样本抽样；保留缓存 SHA。
3. 训练 A/B/C/D 四组小型 G_theta；验证 C 的 Audio 确为不同人，同时 target 的 ACC 不变。
4. 用 REG 的 12 条相同 ACC 及 Audio 生成注册条件门控，CAL 锁阈值，最后才打开 query ACC，输出完整逐用户指标及匿名汇总。
5. 按上述成功/失败标准决定是否延续这个方法，不循环搜索已有测试词。

## 与旧实验的本质区别

- **不是 Audio→ACC embedding 蒸馏**：不要求两模态全局身份空间相同。
- **不是局部频谱预测**：优化真实 ACC-only 查询的身份验证目标。
- **不是 Audio 选注册样本**：所有方法使用完全相同的 ACC 模板样本；差异来自 Audio 条件化验证度量。
- **仍有可归因的对照**：若乱用他人的 Audio 也能提升，则不能称为跨模态身份知识有效迁移。

文献启发：跨模态语音身份蒸馏（AAAI 2023）；不确定性加权的身份蒸馏 UA-MeMo（IEEE TASLP 2026）；TRKD 关注声纹验证难负类并说明直接 embedding MSE 未必有效。这些是设计动机，不是本数据上已验证的结果。
