# `train_core` 旧实验与当前认证结果差异诊断

日期：2026-09-28

## 结论

当前结果看起来比旧实验差，主要因为几个被并列引用的旧数字来自不同任务/评估协议，不能直接对照；在确实可比较的严格用户级认证结果中，user1/bo 有回退，user4 和 user5 则有改善。user1 的回退与 CycleGAN 特征提取器从旧 ResNet epoch 150 更换为 aligned U-Net epoch 50 同时发生，属于首要待验证原因。现有结果不足以把回退单独归因于生成质量或认证损失。

## 逐项核对

### 1. 旧“好结果”里有不等价的指标

- `train_core/cyclegan/eval_auth_discrim.py` 中报告的 CycleGAN 冒充 EER，是 wei 两个采集会话对 ming 单个用户的二人实验：用 wei-s1 注册、wei-s2 做真实用户、ming 做冒充者。它不是当前一个目标用户对其余八个用户的开放集认证。脚本对 600 个样本做 LOO/跨会话分析，开放集部分仅有一个冒充身份。
- `HANDOVER.md` 把域内 diffusion EER 0.013 明确标注为训练含全部会话的上界并注明会话泄漏；六源 PCA32 的 0.015 也注明无泄漏版是 0.020。它们都不是当前 CycleGAN 用户级协议的基线。
- 旧 `train_supcon_9u.py` 的 ACC EER 0.2869 是另一个九用户通用 SupCon 实验。日志显示每 25 epoch 都在同一认证集上选最佳 EER；所以该数是用于模型选择的认证集分数，不能视作冻结模型后的独立测试集结果。

### 2. 严格用户级对比：不是所有用户都退步

`outputs/exact_trainb_user_level_results_20260924.md` 记录的旧 `trainb.py` 严格用户级结果与本轮原始 SupCon 认证头结果如下。两轮认证数据/评估方案并非完全相同，数值应视为方向性参考，而不是严格配对消融。

| 用户 | 旧 exact_trainb HTER | 当前 SupCon HTER | 变化 |
|---|---:|---:|---:|
| user1 / bo | 7.50% | 12.00% | 变差 4.50 个百分点 |
| user4 | 27.75% | 24.40% | 改善 3.35 个百分点 |
| user5 | 40.04% | 33.90% | 改善 6.14 个百分点 |
| 三用户宏平均 | 25.10% | 23.43% | 改善 1.67 个百分点 |

所以现有证据指向 user1 的个别回退，以及旧高分对协议差异的敏感性；不能概括成这轮方法整体失效。

### 3. CycleGAN 特征空间发生了变化

- 旧 exact_trainb 的 checkpoint 元数据指向 per-user ResNet CycleGAN epoch 150；旧代码的 `model_drop.py` 使用 ResNet 生成器和残差块。
- 当前 aligned 流程认证头载入的是 U-Net CycleGAN epoch 50，特征提取分支也按 U-Net 的浅层、瓶颈和 cycle-difference 定义构造。
- 因此，即使认证头算法名称仍是 SupCon，输入表征、可训练融合层看到的特征分布和最优阈值都会变化。user1 旧头上的 7.5% 不能预期在新特征上自动复现。
- 当前 150/150 划分中，每用户约 150 个 GAN 样本和 150 个认证样本；认证头约 90 个/用户用于训练，另有模型验证、注册/阈值开发及最终测试集合。旧 exact_trainb 清单是约 36 个/类训练、12 个/类验证和测试。当前 head 训练样本更多，不能把表现变化归结为“认证头数据不足”。

### 4. 测试覆盖与阈值来源更严格，结果方差更明显

当前协议每个目标用户最终用 15 个 genuine 和 240 个 impostor（其余 8 个来源各 30 个）做冻结评估，注册和阈值从 dev/calibration 集合确定。旧 exact_trainb 汇总是五个注册随机种子，旧测试清单每类样本量较小；旧的 CycleGAN .067 实验只有一个冒充来源。当前 FAR/HTER 因此承受更广的攻击分布，且目标 genuine 数较少，单个窗口即可改变 FRR 约 6.7 个百分点。

当前 five-seed 认证头汇总主要改变注册抽样；训练头 checkpoint 固定。曾运行的多次重训没有完整固定 Torch 随机状态，故不应把它们宣传为严格可复现的五个训练随机种子。

### 5. 已发现的头部风险

- 原 `trainb.py` 和当前 `trainb_target_150.py` 都在 projector 输出上计算 SupCon，但评估入口 `return_embedding=True` 使用 fusion 512 维。这种训练/推理表示不一致在两版均存在，不足以单独解释两轮差距，但应在下一轮配对实验中统一。
- 旧 `trainb.py` 的普通随机 batch 有些 anchor 没有同类正样本；当前诊断实测 zero-positive anchor 约 2.4%。P-K 采样的首轮对 user4 有帮助，user5 仍受 user8/user9 hard impostor 影响。
- 当前 `heads_145` 结果是在独立 dev/calibration 上定阈值、最终集合冻结评估，比旧日志中直接在认证集选 epoch 的流程更接近部署评估；协议更严格会压低表面分数，但提升了结果可信度。

## 最可能原因排序

1. **高置信：旧宣传数字和当前协议不可直接比较。** 尤其 EER 0.067、0.013、0.015 都不是当前九用户开放集、独立阈值校准的严格对照。
2. **中高置信：user1 的 CycleGAN 特征提取器/检查点改变。** ResNet-150 → aligned U-Net-50 改变了输入特征空间，是 user1 相对旧头回退的首要实验假设，但目前尚无配对重跑来证明因果。
3. **中置信：攻击来源更广、阈值/测试集合隔离更严格，且最终 genuine 样本量只有 15。** 这些因素使当前评估更难，FRR 分辨率也较粗。
4. **已知但两版共有：SupCon projector 与推理 embedding 不一致。** 这是实现缺陷，需修复后统一比较；因两版共有，它不是已证实的差异原因。
5. **低置信/仍未知：CycleGAN 图像转换质量是否导致认证下降。** 像素重建好坏不能代替身份可分性，需要在固定数据和认证协议下直接测冻结特征。

## 建议的决定性对照

做一次 2×2 配对实验，而不是继续改损失和阈值：

1. 固定同一份每用户 150 GAN / 150 auth 索引及全部 auth_train、model_validation、dev、final 索引；保存并校验 manifest。
2. 只替换 CycleGAN 特征源：旧 ResNet epoch150、当前 aligned U-Net epoch50。对每个目标用户使用相同认证头代码、初始化 seeds、增强、P-K/普通 sampler、checkpoint 选择规则。
3. 明确训练与评估都使用 fusion512 或都使用 projector128；不混用。阈值仅从同一 dev/calibration 组确定，final 只评估一次。
4. 固定 Torch、NumPy、DataLoader 随机种子，至少五个独立训练 seeds；逐用户报告 HTER、FAR、FRR、AUC 及 user1/4/5 的来源级 FAR。
5. 报告 CycleGAN 冻结特征上的简单 probe 与认证头结果。这样可区分“上游特征变化”与“认证头训练变化”。

此外，现有 sample ID 只记录 `user:index`，尚无证据证明相邻滑窗或同一 parent recording 被分到不同集合时仍保持分离。若录音/session 元数据可用，应按原录音或 session 分组切分；在完成该检查前，不应宣称已做到 session-disjoint。

