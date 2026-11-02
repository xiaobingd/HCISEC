# 基于 AirBone 思想的 Audio→ACC 预训练与身份蒸馏实验方案（V1.0）

> 创建日期：2026-11-03  
> 状态：**拟执行方案；尚未训练或产生新实验结果**  
> 仓库：HCISEC  
> 任务：面向 VR 的本人 / 非本人身份验证（verification），不做部署端九分类身份识别。  
> 原则：先做非生成式身份迁移的可证伪对照，再检验 CycleGAN 额外价值，最后评估少样本注册及场景鲁棒性。

## 0. 研究问题和结论边界

**主问题**：公开 AC（气导语音）中学习的说话人身份知识，能否在少量真实配对 Audio–ACC 数据支持下迁移到头戴设备振动域，使训练阶段完全未见的新用户只需 K 个 ACC 注册样本就能可靠拒绝未知冒充者？

**对 AirBone 的准确理解**：[Eve Said Yes: AirBone Authentication for Head-Wearable Smart Voice Assistant](https://doi.org/10.1109/TMC.2025.3530962) 的 BC-SR 使用公开 AC 语音数据预训练特征提取网络，再以真实 BC 用户标签训练/微调；其 BC 状态域对抗用于抑制头部运动干扰。论文**没有明确描述通过配对 AC–BC 身份 embedding 蒸馏、AC→BC 生成器或跨模态身份映射损失**；AC 预训练的详细标签策略及冻结层配置亦未充分披露。因此：

- **A1（AC 权重迁移→ACC 微调）**：AirBone 风格的迁移学习基线，不声称是作者精确代码复现；
- **A2/A3（冻结 Audio 身份教师→配对 ACC 身份蒸馏）**：**本项目拟提出并验证的扩展**；
- **G 系列（CycleGAN / 生成身份约束）**：用于验证个体特异映射是否增加超越非生成式身份编码器的判别信息。

四项假设：

- **H1 参数迁移**：A1 优于 A0（同协议的 ACC 随机初始化）。
- **H2 显式身份知识迁移**：A2 优于 A1，且蒸馏增益不是音素内容或错误配对造成的假象。
- **H3 新用户泛化**：在目标用户和冒充者均未参与任何本地表征训练时，K-shot 模板有有效区分能力。
- **H4 生成模型的必要性**：控制特征维数、参数量与认证后端后，G 系列比最强非生成式 ACC 方案仍有稳定增益。

**注意**：训练用户分类准确率、重建 L1 下降、配对余弦提高，均不能单独证明开放集本人认证性能改进。

## 1. 基础事实与现有问题（引用仓库历史记录）

当前 Mei 数据约为 **9 用户 × 30 单词 × 每词 10 次**的配对 Audio/ACC；部分用户保存 297–299 条有效记录；历史输入为 80×80 时频特征。现有数据主要为同一次采集会话。

- [2026-10-27 九用户跨模态身份认证](./2026-10-27-mei-nine-user-crossmodal-authentication.md)：严格内容对齐的 U1/U2/U5/U6 上，四用户 macro AUC 分别为原始 ACC 教师 **0.833**、配对生成 **0.778**、身份引导生成 **0.816**。U3/U8 等困难用户很不稳定；不同用户采用的教师/生成器协议不完全相同，不能把九用户结果当统一最终基准。
- [2026-10-27 未知冒充者 U3/U8](./2026-10-27-open-set-unseen-u3-u8.md)：七目标用户的 positive-only 原始 ACC macro 未知冒充 FAR **77.8%**，身份引导生成（λ=0.5）**75.7%**。这是**未见攻击者**测试，不是“目标合法用户训练阶段完全未见”的注册实验。
- [2026-09-23 数据泄漏根因审计](./2026-09-23-training-test-leakage-root-cause-audit.md)：历史一些 CycleGAN、认证头和最终测试各自重新随机切分，发生跨阶段测试样本污染。新实验必须使用同一不可变外层 manifest。
- [创新点 2：未见用户 K-shot 开放集验证协议](../innovation2_fewshot_open_set_pipeline.md) 与[下一步实验](../下一步实验.md)：目标是新用户少量 genuine 注册、不为每位新用户训练整个 backbone。

本实验首先诊断“身份表征先验不足”的可能性；不预设导致高 FAR 的原因一定是 CycleGAN、数据质量或者网络容量。

## 2. Step D0：数据审计与准入门槛（阻断项）

### 2.1 样本清单

在训练服务器为每条配对记录建立不可变 manifest：

`sample_id, subject_id, word_id, repeat_id, session_id, audio_source, acc_source, pair_status, sample_rate, sensor_axis, feature_version`。

检查：

1. Audio/ACC 是否同一次发声且没有错位、重复或混配；
2. U3/U4/U7/U8/U9 缺样是否真的发生在尾部；不能仅由数组行号推断 word_id；
3. ACC 原始采样率、有效频率范围、振动轴及 80×80 时频图的时间/频率含义；
4. 已存 `.npy` 的归一化参数是否在划分前使用了所有用户/全部样本统计量；
5. 数据文件、manifest、预处理代码均生成 SHA-256 版本标识。

**门禁**：word_id/配对关系无法可靠恢复的用户不得计入正式 word-disjoint 报告；原始采样率/频带缺失时不得声称复现 AirBone 的 2 kHz CQT 处理。若现有数组存在全量统计泄漏，应从原始信号重建，无法重建则相关实验标记为“探索性”。

### 2.2 数据安全

遵守 [实验记录规范](./README.md)：不向 GitHub 提交原始 Audio/ACC、可逆样本特征、身份 embedding、私人绝对路径、凭据或模型二进制；仅提交代码、去标识配置、哈希、聚合评价和分析报告。

## 3. 数据划分与严格开放集定义

### 3.1 外层 Leave-Two-Users-Out（LTUO）

对每个**有方向**目标/攻击者对 `(u,v)`：

- 剩余 **7 人：Base users**，仅用于本地 Audio–ACC 模型训练和内部开发；
- **unseen target u**：训练阶段完全不可见，仅允许 K-shot genuine 注册、事先规定的 genuine-only 校准和最终本人测试；
- **unseen impostor v**：完全不进入训练、注册、阈值选择，仅用于最终冒充测试；
- **公开 AC 预训练**来自外部公共语音库，确认不存在本地 9 名受试者数据。

理论上有 `9×8=72` 个有方向 target/impostor 对（36 个不同的七人 Base 集合，重用模型后交换目标/攻击者角色）；先执行自动化正确性试运行，再以预先锁定的超参数完成全量评估。

**试运行不用于针对外层 final-test 指标挑选模型。**

### 3.2 内层开发划分

在每个外层 7 人 Base 集合内部执行模拟未见用户验证：

- 例如 5 人用于特征模型拟合；
- 另外 2 人用作**内部模拟 target/impostor**，负责选择架构、`β`、早停、阈值规则与 K 策略；
- 多次轮换内部模拟身份，固定获选设置后，再用全部 7 Base users 的允许训练样本训练外层正式模型；
- 所有外层 unseen target/impostor 分数均不能参与调参。

### 3.3 词语维度（前提：真实标签已验证）

建议沿用历史便于审计的互斥单词划分：

| 词语编号（0-based） | 最多每人条数 | 用途 |
|---|---:|---|
| 0–11 | 120 | Base 模型训练 |
| 12–14 | 30 | Base 模型开发 / 早停 |
| 15–20 | 60 | Unseen target 注册候选池 |
| 21–23 | 30 | 可选 genuine-only 校准（不得用于挑方法） |
| 24–29 | 60 | 最终独立 genuine/impostor 测试 |

实际条数以审计后的有效样本为准，不用“截取前 297 行”假装已验证词义。全部同词十次重复必须进入同一数据分区。正式结论区分 **unseen-user + word-disjoint + same-session** 和未来单独采集的 **cross-session**；当前仅一个 session 不能宣称跨佩戴泛化。

**阈值主规则**：在 Base users 内部模拟 unseen-user 的开发试验上确定单一阈值策略（可针对 K 预设）。外层 target 的 genuine-only 数据允许按预注册规则做可选校准，但不能声称由此控制实际未知攻击 FAR。外层最终测试只做报告。

## 4. PRE-AC：公开 AC 说话人预训练

### 4.1 数据选择与任务一致性

**首选低成本方案：Google Speech Commands（GSC）**，约单词级短音频。必须核对所用版本中的可用 speaker_id/文件名身份标识与说话人不交叉划分；若身份标签不适用，不能用“单词 label”冒充 speaker label。其身份预训练性能须在**独立公共说话人**上验证。

**增强对照**：VoxCeleb 说话人预训练的 ECAPA-TDNN（如 SpeechBrain 的公开 checkpoint），先确认本地 Audio 原始波形可获取。ECAPA 的波形输入与本项目 80×80 时频图不可无转换直接混用。

**关键控制**：公开教师能否在本地 Base users 的 Audio 验证任务中形成非退化的说话人分离空间；若只学到词语类别、教师 AUC 接近随机或本地原始音频不可用，停止身份蒸馏并排查，不能强行继续。

### 4.2 主实验网络及初始超参数（不是原论文配置）

- 输入：统一的单通道 log 时频图，暂沿用 80×80，需经过 D0 的轴定义检查；AC 和 ACC 可以使用**各自适配物理频带的前端**，但 backbone、投影层和目标评估协议必须保持公平。
- Backbone：ResNet18；embedding 128-d + L2 normalization；speaker classification head 仅用于训练。
- 损失：`L_pre = L_speaker + 0.1 L_SupCon`，对照单独 `L_speaker`；基于 speaker_id 而不是 word_id。
- AdamW，初始 LR `3e-4`、weight decay `1e-4`、上限 100 epoch、speaker-balanced batch、验证早停；
- 增强：轻微时间平移/裁剪、增益、背景噪声；不要使用可能破坏身份信息的强变调或任意频率轴变形。
- 保存：`teacher_best`（仅训练机）、预训练训练集/验证集身份清单、配置与哈希、公共未见说话人验证 AUC/EER。

不要声称“仅有 GSC 就提供与 VoxCeleb 等价的说话人身份先验”；两者训练目标、信号时长和采集环境不同。

## 5. TR-ACC：四个主要迁移对照 + 一个负控制

统一 ACC Backbone = ResNet18（128-d embedding，L2 normalize），同一七人训练数据、预处理与最终 cosine-prototype 认证后端。所有方案用同样的身份损失/优化预算，仅改变表中指定因素。

记真实配对样本为 `(x_a,x_b,y)`，`E_A` 为预训练 Audio 编码器，`E_B` 为 ACC 编码器。

基础目标：

`L_base = L_id(E_B(x_b),y) + α L_SupCon(E_B(x_b),y)`，初始 `α=0.1`。

| 编号 | 初始化 | 额外监督 | 要验证什么 |
|---|---|---|---|
| **A0** ACC-Scratch | 随机 | 无 | 仅 ACC 小数据基线 |
| **A1** AirBone-style | AC 预训练主干参数迁移 | 无 | 参数初始化增益，A1−A0 |
| **A2** Transfer+KD | 与 A1 同初始化 | 冻结 Audio 教师蒸馏 | 显式身份蒸馏增益，A2−A1 |
| **A3** Scratch+KD | 与 A0 同初始化 | 冻结 Audio 教师蒸馏 | 分离蒸馏与初始化影响，A3−A0 |
| **A4** Wrong-Pair KD | 与 A2 相同 | 训练数据内错配 Audio/ACC | 负控制：防止把词语/域共性误判为正确配对身份迁移 |

A1 只迁移输入形式兼容的 backbone 权重，丢弃公开数据专属分类头；若前端频率分布有根本差异，使用可学习前端适配层并把它作为已登记的参数，在 A0/A1/A2 里保持公平。

A2/A3 训练时冻结教师：

\[
z_a=\mathrm{norm}(E_A(x_a)),\quad z_b=\mathrm{norm}(E_B(x_b))
\]

\[
L_{\mathrm{KD}}=1-\mathrm{cos}(z_a,z_b)
\]

\[
L_{\mathrm{A2/A3}}=L_{\mathrm{base}}+\beta L_{\mathrm{KD}}
\]

先用 `β ∈ {0,0.1,0.5,1.0}` 在内部 Base development 选值；`β=0` 正好等同非蒸馏基线。教师的 embedding 维数若不同，用投影头对齐；推理时只保存/使用 ACC 编码器。

**身份 vs 单词混淆控制**：

1. SupCon 正样本应包含**同一用户不同单词**，负样本应包含**不同用户相同单词**；
2. 开发和最终评估必须存在 word-disjoint 结果；
3. A4 应同时设置“错用户但相同单词”配对对照，不能只对全部随机打乱造成更简单的错误；
4. 如时间允许，比较配对样本级 KD 与**跨单词用户原型 KD**：`1−cos(mean E_A(x_a^u), E_B(x_b^u))`，检验单词内容 shortcut。

**初始训练建议**：AdamW；已初始化 backbone LR `1e-4`，新增 head LR `1e-3`；scratch 可在相同候选 LR 范围选；batch 28–56（按用户平衡）；最多 60–100 epoch；开发早停 10 次无改善。配置冻结前建议种子 42/123/3407。每方法使用同样的内部开发预算，禁止外层最终指标反复驱动修改。

**重要限制**：拥有 7 个 Base users 并不等于拥有大规模本地身份监督。即使 PRE-AC 显著提高公共音频 speaker verification，也可能由于 ACC 有效频带窄、教师偏向声学音色、学生只学词语而无法迁移。此时应报告负结果而非归因成“训练未充分”。

## 6. OPEN-K：未见合法用户、未知冒充者的少样本认证

在外层 LTUO 的七人模型训练完成后：**冻结 `E_B`**，不再为新用户微调任何 backbone、生成器或二分类 head。

注册样本数量：

`K ∈ {1,3,5,10,20}`，从 unseen target 的 15–20 词池抽取 K 条 genuine，优先多单词均衡抽取；每 K 至少 10 个固定注册随机种子。

注册模板：

\[
p_u=\mathrm{norm}\left(\frac{1}{K}\sum_{i=1}^{K}\mathrm{norm}(E_B(x_i^u))\right)
\]

最终查询：

\[
s(x,u)=\mathrm{cos}(\mathrm{norm}(E_B(x)),p_u),\qquad \mathrm{accept}\Leftrightarrow s\ge\tau
\]

主结果使用在内部 Base-user 模拟任务确定并冻结的全局阈值/校准策略；次实验可比较 genuine-only 个体校准，但须明确 K 小时对尾部分布估计不稳定且未知 impostor FAR 无法由 genuine-only 校准保证。

**重点指标**：macro/逐用户 AUC、描述性 EER、冻结阈值 FAR/FRR/TAR/HTER、TAR@指定 FAR（若开发协议支持）、最差目标 FAR、hard-impostor pair、K 与性能曲线、注册耗时。对每一目标-攻击者有方向对保留结果，汇总时先按用户聚合，不把多次 K 采样当成新增受试者。9 人规模较小，应报告不确定性而非凭单一测试集宣称普适部署精度。

## 7. GEN-ID：CycleGAN 是否提供独立额外价值（后续阶段）

**前置门槛**：A0/A1/A2/A3 + OPEN-K 已得到稳定的内部开发结果，再开展生成器对照；不要先假定 CycleGAN 有用。

固定同一个 `E_A` 教师及共享训练集，比较：

| 编号 | 方法 | 输入身份特征 |
|---|---|---|
| G0 | 最优非生成式 ACC（A0–A3） | 直接 ACC embedding |
| G1 | Paired ACC→Audio（重建） | 生成 Audio 的冻结教师 embedding |
| G2 | Paired + 生成身份约束 | 同上 |
| G3 | CycleGAN（不加预训练教师约束） | bottleneck、cycle residual 或生成 embedding |
| G4 | CycleGAN + 生成身份约束 | 与 G3 同结构并明确融合 |

生成身份约束：

\[
\hat x_a=G(x_b),\quad
L_{\mathrm{gen-id}}=1-\mathrm{cos}(E_A(\hat x_a), E_A(x_a))
\]

与配对 L1、频率/时间梯度以及 G3 所用循环/对抗目标组合；所有权重由内层开发决定。

**必要控制**：G1/G2 是配对监督基线；G3/G4 必须相同主干容量；如与 ACC embedding 融合，应有容量相近的非生成式融合基线。强制报告 G4 vs G0 的开放集 FAR/AUC、最差用户、训练/注册计算成本。生成器的频谱输入必须与冻结教师兼容；波形型 ECAPA checkpoint 不能直接接收 80×80 输出。

**退出条件**：若生成器在稳定的未见用户/未知攻击者协议下没有提高认证性能，保留作为负结果而不作为部署必需模块。

## 8. VR 场景扩展（新采集后开展）

当前只有同一会话离线单词，不能直接得出跨佩戴连续认证结论。后续增加：

- 跨日期、不同佩戴/接触压力、转头点头/正常 VR 操作；
- 环境噪声及不同 SNR；需要标记信号与动作条件；
- genuine→impostor 设备接管的**真实连续时间戳**，报告检测延迟、正常用户误锁率与交互打扰；
- 真人冒充、扬声器重放、机器振动耦合分别采集测试；仅使用其他人说话的 ACC 数据不能宣称已验证抗重放攻击；
- 单词级到多次语音证据融合（single、mean-3/5、EMA），同时报告累计发声次数和真实等待时间。

域对抗（DAL）只有在存在可信的佩戴/动作条件标签、且可以验证身份与域信号不被不当混淆时才作为扩展组；不将 AirBone 的 BC 动作域对抗误写成 AC–BC 身份蒸馏。

## 9. 统计、成功判据与失败条件

**主对比（预先确定）**：
- H1：A1 vs A0；H2：A2 vs A1；H3：OPEN-K 的逐用户/未知冒充表现；H4：G4 vs G0/G1/G2/G3。
- 主评价为用户级 macro AUC 与冻结阈值的 macro FAR/FRR，安全性特别看 worst-user FAR。
- EER 作为测试分数的**描述性阈值扫描**，不是部署阈值下的 FAR。

建议将“相对于基线 AUC 增加约 0.03，或同一冻结安全策略下 FAR 明显下降且 worst-user 未明显恶化”作为**内部开发时的工程候选门槛**，不是事后按最终结果改变的统计显著性标准。报告至少 3 个训练种子、K-shot 10 个注册种子和用户层级不确定性；不要把窗口/注册种子当成独立人群样本。

**No-go**：
- D0 身份/word manifest 失败或全量预处理泄漏不可修复；
- 教师仅区分词语、不区分说话人；
- 蒸馏对配对打乱同样有效，且无法排除内容 shortcut；
- A2 仅提高 Base-user 分类指标，不提高外层 unseen verification；
- G4 只改善重建损失而不能稳定改善认证。

## 10. 分阶段执行顺序

| Phase | 内容 | 通过标准 / 产物 |
|---|---|---|
| **P0 数据审计** | 数据/词序/配对/频带/归一化，创建不可变 manifest | 审计报告、数据哈希、阻断问题清单 |
| **P1 教师** | GSC 身份预训练；可选 ECAPA teacher | 公共 unseen-speaker 验证、Base Audio 检查、配置哈希 |
| **P2 迁移** | A0/A1/A2/A3 + A4 负控制 | 内层模型选择、损失曲线、训练成本 |
| **P3 严格认证** | LTUO + K-shot + 固定阈值 | 72 有方向组合聚合、逐用户 FAR/FRR/AUC/EER、K 曲线 |
| **P4 生成消融** | G0–G4 | 生成器增益、必要性与开销报告 |
| **P5 场景研究** | 新采集 session/动作/攻击/连续使用 | 跨会话与接管检测报告 |

先跑程序级 pilot（验证训练和评估流程），但**不能根据外层最终测试挑选方法**；再用统一开发协议选定参数并锁定全量评估。若算力不足，先完成 P0–P3，而不是削弱用户隔离以保留更多训练样本。

## 11. 推荐仓库结构和输出

`experiments/airbone_transfer/`（后续新增代码时使用）：

- `audit_data.py` / `build_manifest.py`：验证配对及不可变数据划分；
- `pretrain_audio.py`：公开 AC speaker 预训练；
- `train_acc_transfer.py`：A0–A4 统一训练；
- `eval_open_set.py`：LTUO、K-shot、冻结阈值与统计；
- `train_generator_id.py` / `eval_generator_ablation.py`：G0–G4；
- `configs/`：训练、超参数和阈值策略；
- `results/`：去标识汇总结果。

每次实验记录：实验 ID、状态、数据清单/代码/模型 SHA-256、外层/内层用户与词语分组、seed、硬件/时长、学习率/loss、最佳 epoch、指标、错误、限制。GitHub 只存可安全公开的实验记录，模型与原始数据保存在授权实验环境。

## 12. 结果表模板

### 12.1 不同迁移机制

| 方法 | Macro AUC | 描述性 EER | 冻结 FAR | 冻结 FRR | Worst FAR |
|---|---:|---:|---:|---:|---:|
| A0 ACC scratch | 待实验 | | | | |
| A1 AC weights → ACC | 待实验 | | | | |
| A2 A1 + paired KD | 待实验 | | | | |
| A3 scratch + paired KD | 待实验 | | | | |
| A4 wrong-pair KD | 待实验 | | | | |

### 12.2 K-shot 未见用户注册

| K | A0 AUC/EER | A1 AUC/EER | A2 AUC/EER | A2 冻结 FAR | A2 冻结 TAR |
|---:|---|---|---|---:|---:|
| 1 | | | | | |
| 3 | | | | | |
| 5 | | | | | |
| 10 | | | | | |
| 20 | | | | | |

### 12.3 生成模型的额外价值

| 方法 | Macro AUC | Macro FAR | Worst FAR | 训练开销 | 新用户注册开销 |
|---|---:|---:|---:|---:|---:|
| G0 非生成式最佳模型 | | | | | |
| G1 Paired | | | | | |
| G2 Paired + identity | | | | | |
| G3 CycleGAN | | | | | |
| G4 CycleGAN + identity | | | | | |

## 13. 最终结论应如何表述

- **A1 > A0**：证明 AC 参数预训练有助于 ACC 身份认证，不证明显式 AC–ACC 身份映射。
- **A2 > A1，且 A4 明显较差**：进一步支持真实配对跨模态身份蒸馏的额外价值；仍需消除词语 shortcut 解释。
- **unseen target + unseen impostor + K-shot 稳定通过**：支持共享模型的未知用户注册潜力；不能由 seen-user CSI 指标替代。
- **G4 > G0/G1/G2/G3 且更稳健**：支持 CycleGAN 跨模态映射产生额外可用的身份信息；否则对生成模型必要性作保守结论。
- **只有同会话测试**：不能声称已验证跨天/跨设备佩戴部署；研究推广仍受九用户小样本限制。

**本轮最先执行的任务：P0 数据/频率审计 → P1 Audio 教师 → P2 A0/A1/A2/A3 公平训练 → P3 严格用户隔离 K-shot 验证。** 先验证身份表征迁移是否成立，再决定是否投入新一轮 CycleGAN。
