# 下一步实验：VibraVox 额骨振动身份教师预训练与头显迁移

日期：2026-11-17。状态：**实验设计/待在有数据和计算资源的机器执行，不代表已完成训练或获得新成绩**。

## 研究目标

利用 VibraVox 的真实额骨振动传感器数据，训练或复核面向**说话人身份**（而不是语音识别、活动识别）的强骨传导教师，检验其知识能否迁移到 VR 头显信号，并降低独立身份认证 EER、固定阈值 HTER 和困难冒充者风险。

部署时只输入头显运动信号；不得要求查询 Audio。

## 已核对的基础事实

- 官方数据集：https://huggingface.co/datasets/Cnam-LMSSC/vibravox；论文 DOI：https://doi.org/10.1016/j.specom.2025.103238。总计 188 位有效参与者、约 45.5 小时/传感器。
- 主输入 **`audio.forehead_accelerometer`** 是贴在额骨的**单轴加速度传感器**，虽然数据 API 将其表示为音频格式；身份标签用 `speaker_id`，句子用 `sentence_id` / `normalized_text`。官方 `speech_clean` 有 speaker-disjoint train/validation/test；不需要用语音内容作为身份标签。
- `audio.temple_vibration_pickup` 是其他类型的传感器。禁止将其或 `audio.headset_microphone` 混入主教师输入，避免声学泄漏。
- 本仓库已引用过私有预训练权重 `/autodl-fs/data/vibravox_dual_teacher_20261105/acc_teacher/best_backbone_private.pt`，以及 `experiments/identity_kd/extract_vibravox_acc_transfer.py`、`finetune_vibravox_acc_open.py`。**先核对权重存在、模型结构、训练身份和训练日志，再决定是否重训，不能称其为本轮新预训练。**
- 先前一折本地微调：冻结教师到微调教师，跨词本人接受率从 33.33% 到 37.50%，仍远不足以证明有效认证；见 `docs/experiments/2026-11-08-vibravox-acc-ft-first-fold-results.md`。此次须避免重复既有实验。
- 旧 Mei9 是三轴 OpenVR `vVelocity` 跟踪速度，插值 1 kHz **不是原始物理加速度**；新采集才是 1 kHz 原始三轴 IMU，但目前存在音频溢出、断档、跨传感器同步未验证，见 `outputs/new_hmd_two_participants_inventory_20261117/sync_repair_report.md`。

## P0：教师审计与公共身份泛化门槛（先做）

1. 核查现有 VibraVox ACC 教师权重的来源、SHA256、训练说话人清单、是否已见官方 validation/test 身份、采样率、前端、声学预训练来源与模型结构。无法证明身份隔离就标为探索模型，不用于严格未见用户结论。
2. 用纯额骨加速度信号评估**公共未见说话人**的同人异句 vs 异人句配对 AUC/EER；只用 train 做拟合、validation 做选择、test 最终锁定报告。记录每人/句覆盖、注册 K 值和阈值来源，不能用公共 test 调参。
3. 如果现有教师缺少可复现训练记录或未通过身份泛化门槛，则重新预训练：
   - 官方 `speech_clean` train，输入仅 `audio.forehead_accelerometer`，按 `speaker_id` 做 CE/ArcFace 类身份损失 + 0.1 跨句 SupCon；批次要在说话人及句子上平衡；
   - 不以 `sentence_id` 或词汇作为身份标签；防止句子与说话人关联造成的 shortcut，使用留句和留人评估，记录文本条件化结果；
   - 同容量 CE-only、CE+SupCon 对照，完整冻结训练配置；先做一次小规模数据通路检查再完整训练。
4. 输出公共数据已见/未见身份的 AUC/EER、完整性审计及教师哈希。只有分类 train accuracy 而没有留身份验证不能视为合格教师。

## P1：频段/模态桥接（决定是否适合 HMD）

VibraVox 额头单轴振动的 WAV 采样为 48 kHz；新 HMD 原始 IMU 约 1 kHz；旧 `vVelocity` 是另一类物理输入，绝不当成等效额头振动。

至少对照：

- T_full：原有宽带 VibraVox 教师（锁定，保留原训练前端）；
- T_shared：VibraVox 额骨加速度经过**抗混叠低通/重采样**后，在 HMD IMU 真实可用频带上的身份教师。候选频带例如 20–350 Hz，最终截止频率需先依据 HMD 原生采样率、噪声底、谱能量和设备有效响应范围确定，不能把 1 kHz 插值当成带宽；
- T_random：与 T_shared 同架构、未做外部身份预训练的容量对照。

记录前端的 STFT 窗、hop、频率网格、时间窗、幅度归一化和源/目标域统计差异。仅把预训练模型适配到相同频谱尺寸不等于已消除域差异。

## P2：隔离的迁移对照

**首选真正原始 HMD IMU**。在采集完整性/原生时间戳质量通过的记录上执行 ACC-only 迁移；这个实验不依赖 Audio–IMU 精确同步（它不是配对 Audio 蒸馏），但不能在丢帧/断档记录上假装得到连续频谱。当前仅两人不能支撑稳健未见用户结论，先保留探索性质，扩大参与者和重复佩戴。

如果先对 Mei9 旧 `vVelocity` 做探索，必须独立成实验，不把速度求导称真实 IMU，也不直接跨协议对比旧 8.55% 与新 15% HTER。

同一批数据、相同后端/阈值、同一注册 K 和训练预算比较：

| ID | 方法 | 目的 |
|---|---|---|
| C0 | HMD 学生从零训练 | 基线 |
| C1 | HMD 学生加载 T_shared 参数再微调 | 预训练权重作用 |
| C2 | C0 + 冻结当前 HMD ACC 自教师关系蒸馏 | 与 E4 思路比较 |
| C3 | C0 + 冻结 VibraVox T_shared 关系蒸馏 | 大骨传导教师的独立增益 |
| C4 | C1 + T_shared 关系蒸馏 | 预训练与蒸馏是否互补 |
| C5 | C3 的教师跨身份错配/相同额外损失容量控制 | 排除正则化假增益 |

如教师和学生输入通道不一致，使用训练身份上的固定三轴融合/适配器，**禁止根据最终目标表现挑轴或挑频段**。外部教师类别头只用于 VibraVox 训练；到 HMD 不直接复用其 speaker_id 分类头。

验证分两层：1）已知身份、未见词；2）严格留目标及留攻击者身份的 K-shot 注册认证。所有外层身份在 HMD 学生训练、选择、归一化拟合和阈值优化中隔离。主指标 EER、锁定阈值 FAR/FRR/HTER、TAR@FAR、最难冒充者、逐人差值；报告至少 3 个训练 seed，并将多次录音依赖按身份聚合。

## P3：明确的继续/停止条件

只有同时满足：
1. 公共留身份验证证明教师确实学到了可泛化的骨传导说话人身份结构；
2. T_shared 在 HMD 开发身份上有可测的非退化区分能力；
3. C3 优于 C0、C2、C5（证明**正确外部教师**的独立贡献），或 C1 优于 C0（仅证明预训练权重迁移），且在新独立测试上改善固定阈值认证；
才可以将“VibraVox 大教师改善 VR 身份认证”作为实验结论。

若只有 C1 增益而 C3 无增益，应如实表述为**参数预训练有效、显式关系蒸馏未证实**；若所有方法都差，先检查物理频段、通道、佩戴与域偏移，不盲目增加 KL 权重。

## 当前实际可执行的第一步

**复核已有 VibraVox ACC 私有教师检查点并运行公共留身份验证**；已有权重合格时不重复大规模训练。随后做共享带宽教师与 HMD ACC 的可观测性试验；在没有合格新 IMU 身份数据时不伪造迁移认证结果。

公开仓库仅保存脚本、匿名汇总、哈希与协议；不上传原始波形、speaker_id 对照、逐样本 embedding、模型权重或机器上的私有路径。
