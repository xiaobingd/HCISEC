# 创新点2验证方案：未见用户少样本个体化开放集认证

> 日期：2026-09-29
> 前提：假设创新点1已经成立，已经获得能够表征用户特异 ACC-Audio 跨模态关系的稳定身份 embedding。

## 1. 核心研究问题

创新点1解决“身份相关的跨模态声振特征如何提取”。

创新点2进一步解决：

> 一个训练阶段完全没见过的新用户，能否利用 population-level 跨模态身份先验，仅通过 K 条 genuine 注册样本建立个体化开放集认证边界？

核心形式：

Population Cross-modal Prior + K-shot Enrollment -> Personalized Open-set Boundary

推荐表述：

> 基于跨模态共享先验的未见用户少样本个体化开放集身份认证方法。

## 2. 为什么当前9人闭集SupCon不能直接验证

当前认证头是9个已知用户的监督对比学习。只要目标用户参加过认证头训练，就不能证明 unseen-user few-shot generalization。

真正验证创新点2必须满足：

1. target user 在 backbone / representation 训练阶段完全不可见；
2. final impostor 最好也不参与训练和阈值校准；
3. target 只提供 K 条 genuine enrollment；
4. final test 不参与 threshold selection。

## 3. 核心协议：Leave-Two-Users-Out

当前共9个用户。每轮留出两个用户：一个 unseen target u，一个 unseen impostor v，其余7人作为 base users。

流程：

Base users (7 users)
-> 训练创新点1共享跨模态身份表征
-> 冻结 backbone
-> unseen target u 仅提供 K 条 genuine enrollment
-> 建立 personalized template / boundary
-> unseen impostor v 完全不参与训练、注册和阈值选择
-> final verification

K 建议测试：1 / 3 / 5 / 10 / 20。

随后轮换 target 和 impostor，避免只在 user1 上得出结论。

## 4. user1第一轮Pilot

第一轮可以固定：

- Train: user2-user8
- Unseen target: user1 / bo
- Unseen impostor: user9

必须重新训练不包含 user1 的共享模型。

不能直接拿当前已经见过 user1 的9-user SupCon模型，再用 user1 K-shot enrollment 宣称“未见用户少样本认证”。那只能证明 few-shot enrollment，不能证明 unseen-user generalization。

## 5. K-shot注册

最基础方法为 prototype：

z1, z2, ..., zK -> mean -> L2 normalize -> target prototype p_u

验证时：

probe -> innovation1 embedding -> cosine(z, p_u) -> frozen threshold -> accept/reject

核心曲线：

- x-axis: K
- y-axis: EER / HTER / FAR / FRR / TAR@FAR

如果 K=3/5 已明显稳定，并逐渐接近 K=10/20，可支持“小样本注册”。

## 6. 个体化边界方法对比

| ID | 方法 | 目的 |
|---|---|---|
| F0 | ACC-only + prototype | 单模态最低基线 |
| F1 | 创新点1 embedding + prototype | 跨模态表征迁移能力 |
| F2 | F1 + personalized threshold | 个体化阈值 |
| F3 | F1 + covariance-aware / Mahalanobis | 建模用户内部方差 |
| F4 | F1 + one-class Gaussian / Deep SVDD | 单类边界 |
| F5 | F1 + lightweight adapter | 少样本快速适配 |
| F6 | F5 + synthetic hard negatives | 生成式困难负样本增强 |

优先顺序：

Prototype -> covariance-aware one-class -> lightweight adapter -> synthetic hard negatives

第一阶段不建议直接把 Diffusion 作为创新点主体。

## 7. Diffusion的合理位置

如果保留开题报告中的 Diffusion，更适合作为“少样本个体边界增强模块”，而不是“把 CycleGAN 换成 Diffusion”。

可探索：

K-shot genuine -> population prior / conditional generator -> synthetic hard negatives near target boundary -> personalized one-class / margin boundary

比较：

- without synthetic negatives
- random synthetic negatives
- hard-boundary synthetic negatives

重点观察 unknown-impostor FAR 是否下降。如果没有稳定增益，不强行保留 Diffusion。

## 8. Cross-Session验证

创新点2不能只随机拆同一次采集的数据。

如果 user1 存在多次采集：

Session 1 -> K-shot enrollment

Session 2 -> genuine final test

目标是验证注册模板捕获的是身份，而不是一次佩戴状态、sensor gain、接触压力或 session 环境。

分别报告 same-session 和 cross-session，其中 cross-session 是更关键的泛化结果。

## 9. Word-Disjoint验证

当前数据约为30个词、每词多次重复。

建议增加：

Enrollment: 少量词汇

Test: 注册阶段未出现的其他词汇

例如 5 words enrollment / 25 words test。

最终可逐级比较：

- same-word / same-session
- word-disjoint
- session-disjoint
- word + session disjoint

## 10. Unknown-Impostor验证

严格开放集设置：

Base users -> representation learning

Target u -> unseen during training -> K-shot enrollment only

Impostor v -> unseen during training -> unseen during calibration -> final attack only

这样才能回答：面对系统从未见过的冒充者，少样本注册边界是否仍能拒绝攻击。

## 11. 阈值协议

正确流程：

base-user development data -> 确定通用 calibration strategy

target K-shot enrollment -> 可选：仅利用 target genuine 估计个体统计

final genuine + unseen impostor -> 只评估，不调 threshold

禁止在 final-test genuine/impostor 上遍历阈值后再报告部署 FAR/FRR。

Final-test EER 可作为描述性指标，但实际 operating FAR/FRR/HTER 必须使用测试前冻结的阈值。

## 12. 推荐指标

至少报告：

- EER
- FAR
- FRR
- HTER
- AUC
- TAR@fixed FAR
- unknown-impostor FAR
- per-target HTER
- worst-target HTER
- K-shot mean ± std
- registration/adaptation time
- updated parameter count

每个 K 使用多次 enrollment resampling，例如5-20个 enrollment seeds。

## 13. Adapter路线

如果简单 prototype 已有效，下一步研究：

Frozen shared backbone -> small adapter / user latent / LoRA-like module -> K genuine samples -> personalized embedding

对照：

- Prototype only
- Head-only tuning
- Adapter
- Full retraining

比较性能、更新时间、更新参数量、K-shot稳定性和 cross-session generalization。

## 14. 完整验证层级

Level 1：Unseen-user K-shot enrollment

证明 innovation1 embedding + K genuine enrollment 可以支持新用户认证。

Level 2：Cross-word / Cross-session

证明 identity 不只是 word content 或 one-session wearing condition。

Level 3：Unknown-impostor rejection

证明 unseen target vs unseen attacker 仍能建立有效认证边界。

Level 4：Personalized adaptation

证明 population prior + few-shot adaptation 相比简单 prototype 有稳定增益或更好的性能/注册成本权衡。

## 15. 创新点1与创新点2的边界

创新点1：

> 个体特异 ACC-Audio 跨模态传递关系的身份表征。

解决：身份信息在哪里，如何提取？

创新点2：

> 基于跨模态共享先验的未见用户少样本个体化开放集认证。

解决：一个模型训练阶段从未见过的新用户，如何仅使用少量注册样本建立可靠的个人认证边界？

整体关系：

创新点1 Identity Representation
-> 创新点2 Few-shot Personalization
-> Open-set Verification

## 16. 当前最关键实验

如果现在只做一个创新点2实验：

Train on user2-user8
-> Freeze representation
-> user1 completely unseen
-> K = 1 / 3 / 5 / 10 enrollment
-> prototype / personalized boundary
-> user9 completely unseen impostor
-> final verification

随后交换 target / impostor 并轮换用户。

成功的核心证据不是单个 user1 得到一个很低的 EER，而是：

> 多个 unseen target 上，创新点1跨模态表征在 K=3/5 等少样本条件下稳定优于 ACC-only baseline，并能拒绝训练和校准阶段完全未见的 impostor。

如果这一证据成立，创新点2的核心研究假设即可得到较强支持。
