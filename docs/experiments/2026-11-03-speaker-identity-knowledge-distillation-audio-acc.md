# Audio → ACC 说话人身份知识蒸馏：独立实验方案（V1.0）

> 日期：2026-11-03  
> 状态：**拟执行；当前尚无本方案对应训练结果**  
> 研究对象：HCISEC 的 VR 发声振动身份认证，**本人/非本人 verification**，不是九人闭集 identification  
> 范围：**只研究 Audio 身份教师 → ACC 身份学生的知识蒸馏**。CycleGAN 属于后续可选模块，不作为本实验的必要步骤。

## 1. 本实验与 AirBone 的关系

参考论文：[Eve Said Yes: AirBone Authentication for Head-Wearable Smart Voice Assistant](https://doi.org/10.1109/TMC.2025.3530962)。

- **AirBone 原文**：使用公开气导（AC）语音预训练 ResNet 类特征提取网络；再用真实 BC 数据及 BC 用户 ID 训练身份分支；利用 BC 运动状态的域对抗减少动作干扰。原文**未明确给出 AC/BC 配对身份 embedding 蒸馏损失**、未使用 AC→BC 生成器建立身份映射。AC 预训练的具体标签/冻结层等实现细节也没有全面披露。
- **本实验**：先利用公开 Audio 训练/获得**说话人身份教师** `E_A`；教师冻结，用同一次发声配对 Audio 和 ACC 指导 ACC 学生 `E_B`，让学生保留用户身份差异，而不是简单复刻音色/语音内容。
- 因此，**参数初始化迁移**（`KD1`）作为 AirBone-style baseline；**显式身份知识蒸馏**（`KD2/KD3`）是拟验证的扩展，不能表述为“AirBone 已经实现了蒸馏”。

核心科学问题：

> 与仅利用公开 Audio 预训练权重初始化 ACC 模型相比，**显式学习 Audio 身份教师的身份表征**，能否使 ACC 在未见合法用户 + 未见冒充者条件下取得更低 FAR、更高 AUC，并支持 K-shot 注册？

## 2. 现有基础与风险

目前 Mei 数据约 **9 用户 × 30 词 × 10 次重复**，每条数据为同一次单词发声的配对 Audio/ACC 时频数组，历史网络使用 80×80 输入，若干用户仅约 297–299 样本。主要为同一采集会话。

关联结果：

- [九用户跨模态认证（2026-10-27）](./2026-10-27-mei-nine-user-crossmodal-authentication.md)：原始 ACC 身份教师在部分用户上优于生成 Audio 特征；不同用户表现差异显著。
- [U3/U8 未见冒充者（2026-10-27）](./2026-10-27-open-set-unseen-u3-u8.md)：现有 positive-only prototype 的宏平均未知冒充 FAR 较高（原始 ACC 77.8%，身份引导生成 λ=0.5 时 75.7%）。这**不是**未见合法用户注册结果。
- [历史数据泄漏审计](./2026-09-23-training-test-leakage-root-cause-audit.md)：旧实验存在跨模型阶段重新划分导致的测试集污染。新实验必须先建立统一的用户、单词、配对样本 manifest。

本项目应避免三个假象：**(1)** 教师只识别单词不识别人；**(2)** 蒸馏损失下降却只保留词语内容；**(3)** 学生在目标用户参与训练时表现好，误认为具备未知用户注册能力。

## 3. 核心模型：先建立可信的身份教师

### 3.1 Audio 教师 `E_A`

**主实验**使用提供 speaker identity 标签的公开语音资料（优先试用 Google Speech Commands 的可核实说话人标识），训练单声道频谱输入的 ResNet18，并输出 128 维 L2-normalized 说话人 embedding。

- **训练标签必须是 `speaker_id`，不是单词 `word_label`**。若某个公共数据版本无法可靠获得身份标签，则不能用于该身份教师训练。
- 主模型输入采用经审计的 log 时频图，可使用 80×80 以便与本地架构兼容；不同信号的物理频率范围允许采用各自合适的前端，但输出 embedding 维度需兼容。
- 参考损失：`L_A = CE(speaker_logits, speaker_id) + 0.1 SupCon(speaker_embedding,speaker_id)`。
- 初始超参数：ResNet18，128-d；AdamW，LR=3e-4，weight decay=1e-4，最多 100 epochs；speaker-balanced batches；在公开**未参与训练的说话人**上选择检查点。
- 增强：轻微时间裁剪/平移、噪声和增益扰动；不使用可能大幅破坏身份特征的频率轴任意变换。

**更强的独立教师对照**：VoxCeleb 预训练的 ECAPA 等说话人模型。但波形输入的 ECAPA 不能直接读取 80×80 数组；必须确认本地原始 Audio 波形可用，或采用有对应频谱输入的预训练教师。不同教师比较须固定学生及评估协议。

### 3.2 教师上岗前的两个诊断

在公共独立说话人及本地仅 **Base users** 的允许开发数据上，检查：

1. **身份能力**：同用户不同词的 Audio embedding 相似度应总体高于不同用户同词；报告验证 AUC/EER、同/异用户得分分布。
2. **内容 shortcut**：在匹配词内容的不同用户负样本上应保持可分性；只会识别词语而无法区分说话人的教师**不得用于身份蒸馏**。

一旦开始学生训练，冻结教师全部参数（`eval()`、`requires_grad=False`、输出 `detach`），不使用外层 unseen target / impostor 对教师做本地调优。

## 4. ACC 学生及蒸馏目标

### 4.1 编码器

`E_B(x_b) = z_b`，ACC 学生以 ResNet18 + 128-d 投影层作为初始架构，输出 L2-normalized embedding。公开 Audio 教师输出 `z_a = E_A(x_a)`，训练数据必须是**同一次发声的正确配对 Audio/ACC**。

如果教师维度非 128，添加**只在 Base 数据训练**的投影头使维数对齐；否则不要比较不一致空间的余弦。

### 4.2 基础身份损失

用 Base users 的用户 ID 训练学生身份分离：

\[
L_{\mathrm{ID}}=\mathrm{CE}(W z_b,y),\qquad
L_{\mathrm{SupCon}}=\mathrm{SupCon}(z_b,y)
\]

- 正例：同用户不同单词/重复；
- 关键难负例：**不同用户说相同单词**；
- batch 按用户均衡，至少包含每个出现身份的多个样本。7 Base users、12 训练词条件下，可以使用 7 身份 × 每身份 4–8 样本的 PK sampler。
- 用户分类头只服务于训练，未知用户验证时不使用该头。

### 4.3 基础配对身份蒸馏（首选最小实现）

\[
z_a=\mathrm{stopgrad}\bigl(\mathrm{norm}(E_A(x_a))\bigr),\qquad
z_b=\mathrm{norm}(E_B(x_b))
\]

\[
L_{\mathrm{pair}}=\frac{1}{N}\sum_i [1-\cos(z_{b,i},z_{a,i})]
\]

总目标：

\[
\boxed{L=L_{\mathrm{ID}}+\alpha L_{\mathrm{SupCon}}+\beta L_{\mathrm{pair}}}
\]

建议初始化 `α=0.1`，`β∈{0,0.1,0.5,1.0}`；**只在内部开发用户上选择 β**。`β=0` 是无蒸馏对照。

**关键解释**：`L_pair` 接近 0 只表示配对表示相近，不代表学生已学习身份。必须结合 `L_ID`、同/异用户得分检验和 word-disjoint/open-set 测试判断其身份意义。

### 4.4 第二种蒸馏：跨单词身份原型（可选增强）

配对蒸馏可能把**同一次说话内容**一起搬给学生。引入按 Base 用户聚合的教师身份原型：

\[
p^A_u=\mathrm{norm}\bigl(\mathbb E_{x_a:y=u}[\mathrm{norm}(E_A(x_a))]\bigr)
\]

均值仅来自 **Base fit** 允许样本；不计算 outer unseen target 的教师原型。学生仍以用户为监督：

\[
L_{\mathrm{proto}}=\frac{1}{N}\sum_i [1-\cos(z_{b,i},p^A_{y_i})]
\]

对照 `L_pair`、`L_proto` 和 `L_pair+λ_pL_proto`。预训练教师原型可在训练阶段缓存，计算来源必须可审计。

该组回答：**身份原型蒸馏能否比同一次发声配对蒸馏更好地抑制单词内容 shortcut？** 不假定一定更好；原型也可能丢失有用的细粒度身份信息。

### 4.5 代码级损失示例（非完整训练脚本）

```python
# x_audio / x_acc: 同一批经过真实 pair_id 核验的配对样本
# y: Base users 的身份标签
with torch.no_grad():
    z_teacher = F.normalize(teacher(x_audio), dim=-1)

z_student = F.normalize(student(x_acc), dim=-1)
speaker_logits = speaker_head(z_student)

loss_id = F.cross_entropy(speaker_logits, y)
loss_kd = 1.0 - (z_student * z_teacher).sum(dim=-1).mean()
loss_sc = supervised_contrastive_loss(z_student, y)  # 必须实现/测试正负对掩码

loss = loss_id + alpha * loss_sc + beta * loss_kd
loss.backward()
optimizer.step()
```

这里的教师为固定参数；`supervised_contrastive_loss` 是待实现的函数，不可把示例当作可直接运行的完整脚本。

## 5. 必做消融：将“预训练”和“蒸馏”分开归因

所有组固定相同的数据划分、ACC Backbone、输入处理、验证后端和超参搜索预算。

| 实验 | ACC 初始化 | KD | 科学问题 |
|---|---|---|---|
| **KD0** | 随机 | 否 | ACC 监督训练基线 |
| **KD1** | Audio 预训练权重 | 否 | AirBone-style 参数迁移的收益 |
| **KD2** | 随机 | 正确配对 `L_pair` | 不依赖参数初始化，蒸馏是否有效 |
| **KD3** | Audio 预训练权重 | 正确配对 `L_pair` | 主方案；增益 `KD3−KD1` |
| **KD4** | 同 KD3 | `L_proto`（不用 `L_pair`） | 原型蒸馏是否比逐样本蒸馏稳健 |
| **KD5** | 同 KD3 | `L_pair+L_proto` | 两种蒸馏是否互补 |
| **NEG1** | 同 KD3 | **同词不同用户的错误 Audio 配对** | 是否在学用户身份而非词内容 |
| **NEG2** | 同 KD3 | 随机/未训练 Audio 教师（同结构） | 是否只得到普通噪声正则化效果 |

主假设：`KD3` 对比 `KD1`；`KD2` 对比 `KD0`；`KD4/KD5` 对比 `KD3`；负控制 `NEG1/NEG2` 应弱于真实身份教师，但如果不是，也要据实分析。

训练配置建议：AdamW；已预训练 backbone LR=1e-4、head LR=1e-3，scratch 在相同候选范围选择；batch 28–56、最多 60–100 epochs、连续 10 次内部开发未改善则早停；训练种子 42/123/3407。冻结教师后保存训练/验证 `L_ID/L_KD/L_SupCon`、身份分数分布与最终 checkpoint 哈希。

## 6. 强制用户隔离与词语隔离

### 6.1 外层：Leave-Two-Users-Out

每轮选定：

- **7 Base users**：学生本地训练、身份监督、超参开发；
- **1 unseen target**：仅 K-shot genuine 注册、可选合法样本阈值校准、最终本人查询；
- **1 unseen impostor**：仅最终攻击查询。

模型、教师本地适配、投影头、分类头与任何蒸馏损失都**不能读取**该轮外层 unseen target/impostor 的训练之外数据。公开教师本身只用独立公共语音库预训练。

全部九人可构成 `9×8=72` 个有方向的 target/impostor 测试对；相同 7 Base 用户模型可复用。开始先做只检查管线正确性的 pilot，但不得据 pilot 的外层最终指标挑选方法；最终采用锁定的开发协议轮换九用户。

### 6.2 内层：只能在 Base users 内部调参

例如先用 5 个 Base identities 训练、2 个 Base identities 模拟未见目标和未知攻击者，轮换以选择 `α/β`、教师和阈值方案。确定配置后重新用 7 Base fit 数据训练正式学生。

### 6.3 词语互斥（核实 manifest 后执行）

| 词 ID（0 起始） | 作用 |
|---|---|
| 0–11 | Base fit：学生身份训练与配对蒸馏 |
| 12–14 | Base develop：训练检查点和早停 |
| 15–20 | unseen target genuine K-shot 注册池 |
| 21–23 | 可选 target genuine-only 阈值校准 |
| 24–29 | 最终本人和未见攻击者独立查询 |

真实词编号与 10 次重复必须核实，不得直接把不完整数组的行号当作准确词语标签。训练只使用 Base fit 样本估计本地归一化统计；如果历史数组在原始阶段已经跨全部用户归一化，必须重建或明确实验结论受限。

**公开教师的词汇可能与本地词汇重叠**：word-disjoint 指的是本地 ACC 学生训练/本地评估词语不重叠，不声称公共教师从未接触这些词。

## 7. 推理：只有 ACC，不需要 Audio 或教师

训练结束后丢弃 Base 分类头；固定 `E_B`。未见用户 `u` 仅注册 `K∈{1,3,5,10,20}` 条 ACC：

\[
p_u=\mathrm{norm}\Big(\frac1K\sum_{i=1}^{K}\mathrm{norm}(E_B(x_i^u))\Big)
\]

\[
s(x,u)=\cos(\mathrm{norm}(E_B(x)),p_u)
\]

若 `s≥τ` 则本人通过，否则拒绝。每种 K 使用至少 10 组预先规定的注册抽样种子，优先跨多个注册词采样。

**主 operating threshold**：只在 Base users 的内层模拟新用户任务中确定，并对外层最终测试冻结。可按预注册方案比较 legitimate-only 校准，但**不能**据 genuine-only 样本声称未知攻击 FAR 已得到保证。测试集 EER 仅作为描述性分数统计，与冻结阈值下的 FAR/FRR 分开报告。

此实验评估的是 **unseen legitimate target + unseen impostor**；用户级平均及置信区间必须按受试者聚合，不把 K-shot 多种子当作额外的独立用户。

## 8. 评价指标与结果记录模板

主指标：每目标 AUC、描述性 EER、**冻结阈值 FAR/FRR/TAR/HTER**、worst-target FAR、K-shot 注册稳定性、推理成本；辅指标：教师身份 AUC、学生 Base-user train/val 分类准确率、同用户跨词/不同用户同词 embedding 分布、蒸馏 loss 收敛和模型参数量。

### 8.1 蒸馏实验主表

| 方法 | Macro AUC | EER | Frozen FAR | Frozen FRR | Worst FAR |
|---|---:|---:|---:|---:|---:|
| KD0 ACC scratch | 待实验 | | | | |
| KD1 AC-init | 待实验 | | | | |
| KD2 scratch+pair KD | 待实验 | | | | |
| KD3 AC-init+pair KD | 待实验 | | | | |
| KD4 prototype KD | 待实验 | | | | |
| KD5 pair+proto | 待实验 | | | | |
| NEG1 same-word wrong-user | 待实验 | | | | |
| NEG2 random teacher | 待实验 | | | | |

### 8.2 少样本注册主表

| K | KD0 AUC/FAR | KD1 AUC/FAR | KD3 AUC/FAR | KD4 AUC/FAR |
|---:|---|---|---|---|
| 1 | 待实验 | | | |
| 3 | | | | |
| 5 | | | | |
| 10 | | | | |
| 20 | | | | |

对九个目标用户单独报告成绩，再进行目标用户宏平均；至少使用 3 个训练种子、K-shot 10 个注册种子。九人规模偏小，统计解释应保守。

### 8.3 成功与失败判据

- **`KD1 > KD0`**：支持公开语音参数初始化有用。
- **`KD3 > KD1`**：支持显式跨模态身份蒸馏有额外增益；重点看 unseen-user open-set，而非 Base 训练准确率。
- **`KD3 > NEG1/NEG2`** 且在跨单词条件有效：更支持学习的是有意义的身份对应，而不只是词语 shortcut 或额外正则化。
- **`KD4/KD5 > KD3`**：支持跨单词身份原型减少配对内容混淆；若不成立，保留简洁配对 KD。
- 未见用户 FAR 仍很高、收益仅局限 U1、教师无法区分不同人、或者错配 KD 与正确 KD 同效，均应如实记录并拒绝“蒸馏成功”结论。

工程候选门槛可在**内层开发开始前**预设，例如用户级 macro AUC 提升约 0.03 或同冻结安全策略下 FAR 有实质性下降、且 worst-user 不明显恶化；最终测试结果绝不可用来事后调门槛。由于样本少，不声称有大规模人口意义上的显著提升。

## 9. 分步执行、门禁和可复现产物

| 步骤 | 任务 | 通过要求 |
|---|---|---|
| **P0** | 审计配对、词 ID、采样率、频谱轴、数据标准化及不可变 manifest | 错配、样本泄漏为 0；无法核实的记录退出正式结论 |
| **P1** | 训练/验证 Audio 身份教师，检查跨词身份判别 | 教师确实主要学习 speaker，不只认 word |
| **P2** | KD0/KD1/KD2/KD3，固定内层开发协议 | 主要检验身份蒸馏增益 |
| **P3** | NEG1/NEG2 + KD4/KD5 | 识别 shortcut、对齐与原型作用 |
| **P4** | 全部有方向 LTUO + K-shot | 生成公平开放集主表、逐用户指标与置信区间 |
| **P5** | 若上一步有效，再选择是否接入 CycleGAN | 单独建立生成消融；不纳入本方案主结论 |

建议实施目录：`experiments/identity_kd/`，文件包含 `audit_pair_manifest.py`、`pretrain_audio_teacher.py`、`train_acc_student.py`、`losses.py`、`eval_unseen_kshot.py`、`configs/` 和仅存汇总的 `results/`。

每次实验需记录：用户/词/样本隔离断言、数据和代码哈希、训练参数和种子、硬件、最优 epoch、用户级聚合指标、异常与负结果、模型权重哈希。**不得提交原始用户数据、教师/学生模型二进制、可逆身份特征及用户凭据到 GitHub。**

## 10. 最简起跑版本

如果只做首轮严谨可行实验，推荐先执行：

1. **P0**：修复/确认正确配对、词语编号和规范化，生成分组清单；
2. **P1**：GSC 的 `speaker_id` 训练 128-d Audio teacher，检查不同用户同词、同用户不同词分布；
3. **P2**：固定 7 Base users，运行 `KD0 / KD1 / KD2 / KD3`；`β=0/0.1/0.5/1.0` 只在 Base 内层开发选；
4. **P4**：模型冻结，外层 target 仅 K 条 ACC，另一个用户作为完全未见攻击者；在冻结阈值下报告 FAR/FRR/AUC，并轮换目标；
5. **若 KD3 有收益**：才运行 KD4/KD5 和 NEG1/NEG2 完成证据链，继而考虑是否用预训练教师引导 CycleGAN。

**最终关键证据不是 `L_KD` 下降，而是 `KD3` 对比 `KD1` 在多名训练阶段未见目标及未见冒充者上，稳定提高身份分离并改善安全工作点。**
