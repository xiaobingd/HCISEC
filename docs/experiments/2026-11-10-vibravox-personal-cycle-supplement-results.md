# VibraVox：注册者单人CycleGAN微调与特征补充

2026-11-10。59人：35 Base预训练、12 Cal选择与校准、12未见Test注册。每个目标12条注册配对，查询仅振动；所有11位攻击者保留。完整公开每位Test与Cal结果。

[技术方案与用户补充要求](2026-11-10-vibravox-cycle-supplement-protocol.md)。这是紧凑共享初始化＋单人微调新增分支，不是旧完整772维重现。VibraVox是单轴额头加速度，不是本地头显速度。

## 训练路线

- 共享G/F只用35位Base fit，4000次更新、batch16，约28.91个训练集等效轮；按70条Base-develop录制选择第4000步。不是200轮。
- G/F为width32、4残差块，PatchGAN；LSGAN＋10配对L1＋10cycleL1＋identity mapping L1。identity mapping不是用户身份损失。
- 每位Cal/Test从同一共享G/F独立重置，仅用自己的12条注册振动–Audio，微调100步、batch6、LR2e-5，保存25/100步；不使用任何其他身份样本参与个人更新。判别器冻结为Base共享训练最终步，G/F初始化为Base-dev选择步。
- 沿用Base-fit的输入归一化和特征StandardScaler，个人注册/查询不重新拟合尺度。其他用户攻击样本经过注册者的同一个模型。
- 原始shape40＋已有个人Audio教师V4学生192=baseline232保持不变。固定个人补充分支权重0.5；自主选择可不补充。匹配shared/random分支与个人选择使用同分支、同权重。
- Cal前6位定阈值、后6位选择25/100步及分支/权重；锁定后才训练Test个人G/F。最终全部12位Cal分别定Cal-EER与Cal-FAR5阈值。注册Audio共288条，查询Audio不参与生成器微调或认证。

## 校准选择

锁定：step=0，branch=none，alpha=0.0。Cal开发HTER=3.33%，AUC=0.9956。不能看到Test后更换该选择。

## 全部Test汇总（Cal-EER）

| 方法 | TAR% | FAR% | HTER% | pooled AUC | 最差目标FAR% |
|---|---:|---:|---:|---:|---:|
| shape40 | 95.14 | 1.96 | 3.41 | 0.9950 | 10.98 |
| baseline232 | 95.49 | 1.58 | 3.05 | 0.9965 | 8.71 |
| cycle_only100 | 88.54 | 15.12 | 13.29 | 0.9439 | 28.79 |
| input_only100 | 91.67 | 10.51 | 9.42 | 0.9703 | 25.38 |
| personal25_input_encoder256 | 95.14 | 1.64 | 3.25 | 0.9964 | 10.23 |
| personal25_cycle_encoder256 | 95.14 | 1.83 | 3.35 | 0.9960 | 10.23 |
| personal25_cycle_statistics240 | 95.14 | 1.77 | 3.31 | 0.9957 | 9.09 |
| personal100_input_encoder256 | 95.49 | 1.61 | 3.06 | 0.9965 | 9.85 |
| personal100_cycle_encoder256 | 95.14 | 1.67 | 3.27 | 0.9962 | 8.33 |
| personal100_cycle_statistics240 | 95.14 | 1.74 | 3.30 | 0.9959 | 9.09 |
| personal_selected | 95.49 | 1.58 | 3.05 | 0.9965 | 8.71 |
| matched_shared_branch | 95.49 | 1.58 | 3.05 | 0.9965 | 8.71 |
| matched_random_branch | 95.49 | 1.58 | 3.05 | 0.9965 | 8.71 |
| shared_selected | 95.49 | 1.64 | 3.08 | 0.9964 | 9.09 |

## 主要观察

个人自主选择相对现baseline232，HTER变化+0.00个百分点，身份bootstrap 95%区间[+0.00,+0.00]个百分点。其结果是TAR=95.49%、FAR=1.58%、HTER=3.05%。

Cal没有选择任何个人CycleGAN补充分支，所以自主选择结果回到原baseline232。这不是个人模型没训练：24人均已训练25/100步，固定分支结果完整列出。此时matched_shared_branch与matched_random_branch也退化为原基线，不能把它们的相同分数解释成随机网络与学习网络等效；共享模型的非零随机对照另见[共享实验逐用户CSV](../../outputs/vibravox_cycle_supplement_20261110/per_user_public.csv)。

特征名：input_encoder256为G对原始振动的编码；cycle_encoder256为G对F(G(振动))的编码；cycle_statistics240为循环频谱统计。input_only100与cycle_only100仅使用个人模型编码，无原始/学生补充。shared_selected是未个人微调的共享模型自主选择参考。

## 全部12位Test逐用户（Cal-EER）

| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |
|---|---|---:|---:|---:|---:|---:|
| Vib59-Test-01 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-01 | baseline232 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-01 | cycle_only100 | 95.83 | 14.39 | 9.28 | 0.9577 | 24/264 |
| Vib59-Test-01 | input_only100 | 100.00 | 7.95 | 3.98 | 0.9942 | 24/264 |
| Vib59-Test-01 | personal25_input_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-01 | personal25_cycle_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-01 | personal25_cycle_statistics240 | 100.00 | 1.52 | 0.76 | 1.0000 | 24/264 |
| Vib59-Test-01 | personal100_input_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-01 | personal100_cycle_encoder256 | 100.00 | 1.52 | 0.76 | 1.0000 | 24/264 |
| Vib59-Test-01 | personal100_cycle_statistics240 | 100.00 | 1.52 | 0.76 | 1.0000 | 24/264 |
| Vib59-Test-01 | personal_selected | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-01 | matched_shared_branch | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-01 | matched_random_branch | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-01 | shared_selected | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-02 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | baseline232 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | cycle_only100 | 66.67 | 1.14 | 17.23 | 0.9744 | 24/264 |
| Vib59-Test-02 | input_only100 | 66.67 | 1.14 | 17.23 | 0.9803 | 24/264 |
| Vib59-Test-02 | personal25_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | personal25_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | personal25_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | personal100_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | personal100_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | personal100_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | personal_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | matched_shared_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | matched_random_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-02 | shared_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-03 | shape40 | 75.00 | 1.52 | 13.26 | 0.9861 | 24/264 |
| Vib59-Test-03 | baseline232 | 66.67 | 0.76 | 17.05 | 0.9912 | 24/264 |
| Vib59-Test-03 | cycle_only100 | 87.50 | 7.58 | 10.04 | 0.9757 | 24/264 |
| Vib59-Test-03 | input_only100 | 83.33 | 3.79 | 10.23 | 0.9688 | 24/264 |
| Vib59-Test-03 | personal25_input_encoder256 | 66.67 | 0.76 | 17.05 | 0.9912 | 24/264 |
| Vib59-Test-03 | personal25_cycle_encoder256 | 66.67 | 0.76 | 17.05 | 0.9904 | 24/264 |
| Vib59-Test-03 | personal25_cycle_statistics240 | 66.67 | 0.76 | 17.05 | 0.9904 | 24/264 |
| Vib59-Test-03 | personal100_input_encoder256 | 70.83 | 0.76 | 14.96 | 0.9912 | 24/264 |
| Vib59-Test-03 | personal100_cycle_encoder256 | 66.67 | 0.76 | 17.05 | 0.9905 | 24/264 |
| Vib59-Test-03 | personal100_cycle_statistics240 | 66.67 | 0.76 | 17.05 | 0.9904 | 24/264 |
| Vib59-Test-03 | personal_selected | 66.67 | 0.76 | 17.05 | 0.9912 | 24/264 |
| Vib59-Test-03 | matched_shared_branch | 66.67 | 0.76 | 17.05 | 0.9912 | 24/264 |
| Vib59-Test-03 | matched_random_branch | 66.67 | 0.76 | 17.05 | 0.9912 | 24/264 |
| Vib59-Test-03 | shared_selected | 66.67 | 0.76 | 17.05 | 0.9907 | 24/264 |
| Vib59-Test-04 | shape40 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-04 | baseline232 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-04 | cycle_only100 | 100.00 | 9.85 | 4.92 | 0.9961 | 24/264 |
| Vib59-Test-04 | input_only100 | 100.00 | 9.09 | 4.55 | 0.9997 | 24/264 |
| Vib59-Test-04 | personal25_input_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-04 | personal25_cycle_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-04 | personal25_cycle_statistics240 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-04 | personal100_input_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-04 | personal100_cycle_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-04 | personal100_cycle_statistics240 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-04 | personal_selected | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-04 | matched_shared_branch | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-04 | matched_random_branch | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-04 | shared_selected | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-05 | shape40 | 100.00 | 1.14 | 0.57 | 0.9997 | 24/264 |
| Vib59-Test-05 | baseline232 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | cycle_only100 | 95.83 | 5.68 | 4.92 | 0.9926 | 24/264 |
| Vib59-Test-05 | input_only100 | 100.00 | 2.65 | 1.33 | 0.9987 | 24/264 |
| Vib59-Test-05 | personal25_input_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | personal25_cycle_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | personal25_cycle_statistics240 | 100.00 | 0.76 | 0.38 | 0.9998 | 24/264 |
| Vib59-Test-05 | personal100_input_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | personal100_cycle_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | personal100_cycle_statistics240 | 100.00 | 0.76 | 0.38 | 0.9998 | 24/264 |
| Vib59-Test-05 | personal_selected | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | matched_shared_branch | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | matched_random_branch | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-05 | shared_selected | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-06 | shape40 | 75.00 | 0.38 | 12.69 | 0.9785 | 24/264 |
| Vib59-Test-06 | baseline232 | 87.50 | 1.14 | 6.82 | 0.9885 | 24/264 |
| Vib59-Test-06 | cycle_only100 | 79.17 | 18.94 | 19.89 | 0.8933 | 24/264 |
| Vib59-Test-06 | input_only100 | 87.50 | 17.05 | 14.77 | 0.9324 | 24/264 |
| Vib59-Test-06 | personal25_input_encoder256 | 87.50 | 0.76 | 6.63 | 0.9893 | 24/264 |
| Vib59-Test-06 | personal25_cycle_encoder256 | 87.50 | 1.52 | 7.01 | 0.9888 | 24/264 |
| Vib59-Test-06 | personal25_cycle_statistics240 | 83.33 | 1.14 | 8.90 | 0.9852 | 24/264 |
| Vib59-Test-06 | personal100_input_encoder256 | 87.50 | 0.76 | 6.63 | 0.9899 | 24/264 |
| Vib59-Test-06 | personal100_cycle_encoder256 | 87.50 | 1.52 | 7.01 | 0.9885 | 24/264 |
| Vib59-Test-06 | personal100_cycle_statistics240 | 83.33 | 0.76 | 8.71 | 0.9855 | 24/264 |
| Vib59-Test-06 | personal_selected | 87.50 | 1.14 | 6.82 | 0.9885 | 24/264 |
| Vib59-Test-06 | matched_shared_branch | 87.50 | 1.14 | 6.82 | 0.9885 | 24/264 |
| Vib59-Test-06 | matched_random_branch | 87.50 | 1.14 | 6.82 | 0.9885 | 24/264 |
| Vib59-Test-06 | shared_selected | 87.50 | 1.52 | 7.01 | 0.9886 | 24/264 |
| Vib59-Test-07 | shape40 | 100.00 | 2.27 | 1.14 | 1.0000 | 24/264 |
| Vib59-Test-07 | baseline232 | 100.00 | 2.27 | 1.14 | 1.0000 | 24/264 |
| Vib59-Test-07 | cycle_only100 | 100.00 | 22.73 | 11.36 | 0.9921 | 24/264 |
| Vib59-Test-07 | input_only100 | 100.00 | 16.67 | 8.33 | 0.9998 | 24/264 |
| Vib59-Test-07 | personal25_input_encoder256 | 100.00 | 2.65 | 1.33 | 1.0000 | 24/264 |
| Vib59-Test-07 | personal25_cycle_encoder256 | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| Vib59-Test-07 | personal25_cycle_statistics240 | 100.00 | 2.65 | 1.33 | 1.0000 | 24/264 |
| Vib59-Test-07 | personal100_input_encoder256 | 100.00 | 2.65 | 1.33 | 1.0000 | 24/264 |
| Vib59-Test-07 | personal100_cycle_encoder256 | 100.00 | 2.65 | 1.33 | 1.0000 | 24/264 |
| Vib59-Test-07 | personal100_cycle_statistics240 | 100.00 | 2.65 | 1.33 | 1.0000 | 24/264 |
| Vib59-Test-07 | personal_selected | 100.00 | 2.27 | 1.14 | 1.0000 | 24/264 |
| Vib59-Test-07 | matched_shared_branch | 100.00 | 2.27 | 1.14 | 1.0000 | 24/264 |
| Vib59-Test-07 | matched_random_branch | 100.00 | 2.27 | 1.14 | 1.0000 | 24/264 |
| Vib59-Test-07 | shared_selected | 100.00 | 1.89 | 0.95 | 1.0000 | 24/264 |
| Vib59-Test-08 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | baseline232 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | cycle_only100 | 83.33 | 9.85 | 13.26 | 0.9525 | 24/264 |
| Vib59-Test-08 | input_only100 | 83.33 | 8.33 | 12.50 | 0.9620 | 24/264 |
| Vib59-Test-08 | personal25_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | personal25_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | personal25_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | personal100_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | personal100_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | personal100_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | personal_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | matched_shared_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | matched_random_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-08 | shared_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Test-09 | shape40 | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| Vib59-Test-09 | baseline232 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-09 | cycle_only100 | 100.00 | 28.41 | 14.20 | 0.9710 | 24/264 |
| Vib59-Test-09 | input_only100 | 100.00 | 14.39 | 7.20 | 0.9924 | 24/264 |
| Vib59-Test-09 | personal25_input_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-09 | personal25_cycle_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-09 | personal25_cycle_statistics240 | 100.00 | 1.89 | 0.95 | 0.9998 | 24/264 |
| Vib59-Test-09 | personal100_input_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-09 | personal100_cycle_encoder256 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Test-09 | personal100_cycle_statistics240 | 100.00 | 1.89 | 0.95 | 0.9998 | 24/264 |
| Vib59-Test-09 | personal_selected | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-09 | matched_shared_branch | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-09 | matched_random_branch | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-09 | shared_selected | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-10 | shape40 | 91.67 | 10.98 | 9.66 | 0.9650 | 24/264 |
| Vib59-Test-10 | baseline232 | 91.67 | 8.71 | 8.52 | 0.9545 | 24/264 |
| Vib59-Test-10 | cycle_only100 | 66.67 | 27.65 | 30.49 | 0.7634 | 24/264 |
| Vib59-Test-10 | input_only100 | 79.17 | 25.38 | 23.11 | 0.8362 | 24/264 |
| Vib59-Test-10 | personal25_input_encoder256 | 87.50 | 10.23 | 11.36 | 0.9523 | 24/264 |
| Vib59-Test-10 | personal25_cycle_encoder256 | 87.50 | 10.23 | 11.36 | 0.9448 | 24/264 |
| Vib59-Test-10 | personal25_cycle_statistics240 | 91.67 | 9.09 | 8.71 | 0.9482 | 24/264 |
| Vib59-Test-10 | personal100_input_encoder256 | 87.50 | 9.85 | 11.17 | 0.9522 | 24/264 |
| Vib59-Test-10 | personal100_cycle_encoder256 | 87.50 | 8.33 | 10.42 | 0.9454 | 24/264 |
| Vib59-Test-10 | personal100_cycle_statistics240 | 91.67 | 9.09 | 8.71 | 0.9497 | 24/264 |
| Vib59-Test-10 | personal_selected | 91.67 | 8.71 | 8.52 | 0.9545 | 24/264 |
| Vib59-Test-10 | matched_shared_branch | 91.67 | 8.71 | 8.52 | 0.9545 | 24/264 |
| Vib59-Test-10 | matched_random_branch | 91.67 | 8.71 | 8.52 | 0.9545 | 24/264 |
| Vib59-Test-10 | shared_selected | 91.67 | 9.09 | 8.71 | 0.9536 | 24/264 |
| Vib59-Test-11 | shape40 | 100.00 | 1.52 | 0.76 | 1.0000 | 24/264 |
| Vib59-Test-11 | baseline232 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | cycle_only100 | 87.50 | 6.44 | 9.47 | 0.9796 | 24/264 |
| Vib59-Test-11 | input_only100 | 100.00 | 2.27 | 1.14 | 0.9992 | 24/264 |
| Vib59-Test-11 | personal25_input_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-11 | personal25_cycle_encoder256 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | personal25_cycle_statistics240 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | personal100_input_encoder256 | 100.00 | 0.38 | 0.19 | 1.0000 | 24/264 |
| Vib59-Test-11 | personal100_cycle_encoder256 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | personal100_cycle_statistics240 | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | personal_selected | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | matched_shared_branch | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | matched_random_branch | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-11 | shared_selected | 100.00 | 0.76 | 0.38 | 1.0000 | 24/264 |
| Vib59-Test-12 | shape40 | 100.00 | 2.27 | 1.14 | 0.9968 | 24/264 |
| Vib59-Test-12 | baseline232 | 100.00 | 1.89 | 0.95 | 0.9994 | 24/264 |
| Vib59-Test-12 | cycle_only100 | 100.00 | 28.79 | 14.39 | 0.9621 | 24/264 |
| Vib59-Test-12 | input_only100 | 100.00 | 17.42 | 8.71 | 0.9815 | 24/264 |
| Vib59-Test-12 | personal25_input_encoder256 | 100.00 | 1.89 | 0.95 | 0.9992 | 24/264 |
| Vib59-Test-12 | personal25_cycle_encoder256 | 100.00 | 1.89 | 0.95 | 0.9992 | 24/264 |
| Vib59-Test-12 | personal25_cycle_statistics240 | 100.00 | 1.89 | 0.95 | 0.9994 | 24/264 |
| Vib59-Test-12 | personal100_input_encoder256 | 100.00 | 1.89 | 0.95 | 0.9992 | 24/264 |
| Vib59-Test-12 | personal100_cycle_encoder256 | 100.00 | 1.89 | 0.95 | 0.9994 | 24/264 |
| Vib59-Test-12 | personal100_cycle_statistics240 | 100.00 | 1.89 | 0.95 | 0.9994 | 24/264 |
| Vib59-Test-12 | personal_selected | 100.00 | 1.89 | 0.95 | 0.9994 | 24/264 |
| Vib59-Test-12 | matched_shared_branch | 100.00 | 1.89 | 0.95 | 0.9994 | 24/264 |
| Vib59-Test-12 | matched_random_branch | 100.00 | 1.89 | 0.95 | 0.9994 | 24/264 |
| Vib59-Test-12 | shared_selected | 100.00 | 1.89 | 0.95 | 0.9995 | 24/264 |

## 全部12位Cal逐用户（仅校准/开发，不是独立Test）

| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |
|---|---|---:|---:|---:|---:|---:|
| Vib59-Cal-01 | shape40 | 100.00 | 5.68 | 2.84 | 0.9997 | 24/264 |
| Vib59-Cal-01 | baseline232 | 100.00 | 6.06 | 3.03 | 0.9984 | 24/264 |
| Vib59-Cal-01 | cycle_only100 | 87.50 | 9.85 | 11.17 | 0.9437 | 24/264 |
| Vib59-Cal-01 | input_only100 | 91.67 | 13.26 | 10.80 | 0.9542 | 24/264 |
| Vib59-Cal-01 | personal25_input_encoder256 | 100.00 | 4.55 | 2.27 | 0.9983 | 24/264 |
| Vib59-Cal-01 | personal25_cycle_encoder256 | 100.00 | 4.17 | 2.08 | 0.9979 | 24/264 |
| Vib59-Cal-01 | personal25_cycle_statistics240 | 100.00 | 4.92 | 2.46 | 0.9981 | 24/264 |
| Vib59-Cal-01 | personal100_input_encoder256 | 100.00 | 4.92 | 2.46 | 0.9983 | 24/264 |
| Vib59-Cal-01 | personal100_cycle_encoder256 | 100.00 | 4.55 | 2.27 | 0.9981 | 24/264 |
| Vib59-Cal-01 | personal100_cycle_statistics240 | 100.00 | 4.55 | 2.27 | 0.9983 | 24/264 |
| Vib59-Cal-01 | personal_selected | 100.00 | 6.06 | 3.03 | 0.9984 | 24/264 |
| Vib59-Cal-01 | matched_shared_branch | 100.00 | 6.06 | 3.03 | 0.9984 | 24/264 |
| Vib59-Cal-01 | matched_random_branch | 100.00 | 6.06 | 3.03 | 0.9984 | 24/264 |
| Vib59-Cal-01 | shared_selected | 100.00 | 5.68 | 2.84 | 0.9983 | 24/264 |
| Vib59-Cal-02 | shape40 | 95.83 | 9.09 | 6.63 | 0.9929 | 24/264 |
| Vib59-Cal-02 | baseline232 | 95.83 | 7.95 | 6.06 | 0.9943 | 24/264 |
| Vib59-Cal-02 | cycle_only100 | 91.67 | 13.64 | 10.98 | 0.9700 | 24/264 |
| Vib59-Cal-02 | input_only100 | 95.83 | 9.47 | 6.82 | 0.9885 | 24/264 |
| Vib59-Cal-02 | personal25_input_encoder256 | 95.83 | 7.58 | 5.87 | 0.9961 | 24/264 |
| Vib59-Cal-02 | personal25_cycle_encoder256 | 95.83 | 8.33 | 6.25 | 0.9948 | 24/264 |
| Vib59-Cal-02 | personal25_cycle_statistics240 | 95.83 | 7.95 | 6.06 | 0.9931 | 24/264 |
| Vib59-Cal-02 | personal100_input_encoder256 | 95.83 | 7.20 | 5.68 | 0.9957 | 24/264 |
| Vib59-Cal-02 | personal100_cycle_encoder256 | 95.83 | 8.33 | 6.25 | 0.9948 | 24/264 |
| Vib59-Cal-02 | personal100_cycle_statistics240 | 95.83 | 7.95 | 6.06 | 0.9932 | 24/264 |
| Vib59-Cal-02 | personal_selected | 95.83 | 7.95 | 6.06 | 0.9943 | 24/264 |
| Vib59-Cal-02 | matched_shared_branch | 95.83 | 7.95 | 6.06 | 0.9943 | 24/264 |
| Vib59-Cal-02 | matched_random_branch | 95.83 | 7.95 | 6.06 | 0.9943 | 24/264 |
| Vib59-Cal-02 | shared_selected | 95.83 | 6.44 | 5.30 | 0.9948 | 24/264 |
| Vib59-Cal-03 | shape40 | 87.50 | 0.00 | 6.25 | 1.0000 | 24/264 |
| Vib59-Cal-03 | baseline232 | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| Vib59-Cal-03 | cycle_only100 | 87.50 | 0.76 | 6.63 | 0.9774 | 24/264 |
| Vib59-Cal-03 | input_only100 | 87.50 | 0.00 | 6.25 | 0.9899 | 24/264 |
| Vib59-Cal-03 | personal25_input_encoder256 | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| Vib59-Cal-03 | personal25_cycle_encoder256 | 95.83 | 0.00 | 2.08 | 0.9998 | 24/264 |
| Vib59-Cal-03 | personal25_cycle_statistics240 | 87.50 | 0.00 | 6.25 | 0.9998 | 24/264 |
| Vib59-Cal-03 | personal100_input_encoder256 | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| Vib59-Cal-03 | personal100_cycle_encoder256 | 95.83 | 0.00 | 2.08 | 0.9998 | 24/264 |
| Vib59-Cal-03 | personal100_cycle_statistics240 | 87.50 | 0.00 | 6.25 | 0.9998 | 24/264 |
| Vib59-Cal-03 | personal_selected | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| Vib59-Cal-03 | matched_shared_branch | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| Vib59-Cal-03 | matched_random_branch | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| Vib59-Cal-03 | shared_selected | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| Vib59-Cal-04 | shape40 | 100.00 | 3.41 | 1.70 | 0.9998 | 24/264 |
| Vib59-Cal-04 | baseline232 | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| Vib59-Cal-04 | cycle_only100 | 100.00 | 15.15 | 7.58 | 0.9861 | 24/264 |
| Vib59-Cal-04 | input_only100 | 100.00 | 7.20 | 3.60 | 0.9989 | 24/264 |
| Vib59-Cal-04 | personal25_input_encoder256 | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| Vib59-Cal-04 | personal25_cycle_encoder256 | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| Vib59-Cal-04 | personal25_cycle_statistics240 | 100.00 | 3.79 | 1.89 | 1.0000 | 24/264 |
| Vib59-Cal-04 | personal100_input_encoder256 | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| Vib59-Cal-04 | personal100_cycle_encoder256 | 100.00 | 3.41 | 1.70 | 1.0000 | 24/264 |
| Vib59-Cal-04 | personal100_cycle_statistics240 | 100.00 | 4.17 | 2.08 | 1.0000 | 24/264 |
| Vib59-Cal-04 | personal_selected | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| Vib59-Cal-04 | matched_shared_branch | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| Vib59-Cal-04 | matched_random_branch | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| Vib59-Cal-04 | shared_selected | 100.00 | 3.41 | 1.70 | 1.0000 | 24/264 |
| Vib59-Cal-05 | shape40 | 95.83 | 12.50 | 8.33 | 0.9683 | 24/264 |
| Vib59-Cal-05 | baseline232 | 100.00 | 6.82 | 3.41 | 0.9991 | 24/264 |
| Vib59-Cal-05 | cycle_only100 | 91.67 | 21.21 | 14.77 | 0.9055 | 24/264 |
| Vib59-Cal-05 | input_only100 | 87.50 | 15.15 | 13.83 | 0.9291 | 24/264 |
| Vib59-Cal-05 | personal25_input_encoder256 | 100.00 | 6.82 | 3.41 | 0.9978 | 24/264 |
| Vib59-Cal-05 | personal25_cycle_encoder256 | 100.00 | 8.71 | 4.36 | 0.9979 | 24/264 |
| Vib59-Cal-05 | personal25_cycle_statistics240 | 100.00 | 8.33 | 4.17 | 0.9992 | 24/264 |
| Vib59-Cal-05 | personal100_input_encoder256 | 100.00 | 7.20 | 3.60 | 0.9981 | 24/264 |
| Vib59-Cal-05 | personal100_cycle_encoder256 | 100.00 | 7.95 | 3.98 | 0.9978 | 24/264 |
| Vib59-Cal-05 | personal100_cycle_statistics240 | 100.00 | 8.71 | 4.36 | 0.9995 | 24/264 |
| Vib59-Cal-05 | personal_selected | 100.00 | 6.82 | 3.41 | 0.9991 | 24/264 |
| Vib59-Cal-05 | matched_shared_branch | 100.00 | 6.82 | 3.41 | 0.9991 | 24/264 |
| Vib59-Cal-05 | matched_random_branch | 100.00 | 6.82 | 3.41 | 0.9991 | 24/264 |
| Vib59-Cal-05 | shared_selected | 100.00 | 6.06 | 3.03 | 0.9995 | 24/264 |
| Vib59-Cal-06 | shape40 | 91.67 | 10.61 | 9.47 | 0.9629 | 24/264 |
| Vib59-Cal-06 | baseline232 | 91.67 | 6.82 | 7.58 | 0.9762 | 24/264 |
| Vib59-Cal-06 | cycle_only100 | 66.67 | 8.33 | 20.83 | 0.9266 | 24/264 |
| Vib59-Cal-06 | input_only100 | 75.00 | 7.95 | 16.48 | 0.9340 | 24/264 |
| Vib59-Cal-06 | personal25_input_encoder256 | 91.67 | 6.06 | 7.20 | 0.9757 | 24/264 |
| Vib59-Cal-06 | personal25_cycle_encoder256 | 91.67 | 6.06 | 7.20 | 0.9751 | 24/264 |
| Vib59-Cal-06 | personal25_cycle_statistics240 | 91.67 | 5.68 | 7.01 | 0.9763 | 24/264 |
| Vib59-Cal-06 | personal100_input_encoder256 | 91.67 | 6.06 | 7.20 | 0.9762 | 24/264 |
| Vib59-Cal-06 | personal100_cycle_encoder256 | 91.67 | 6.06 | 7.20 | 0.9760 | 24/264 |
| Vib59-Cal-06 | personal100_cycle_statistics240 | 91.67 | 5.68 | 7.01 | 0.9760 | 24/264 |
| Vib59-Cal-06 | personal_selected | 91.67 | 6.82 | 7.58 | 0.9762 | 24/264 |
| Vib59-Cal-06 | matched_shared_branch | 91.67 | 6.82 | 7.58 | 0.9762 | 24/264 |
| Vib59-Cal-06 | matched_random_branch | 91.67 | 6.82 | 7.58 | 0.9762 | 24/264 |
| Vib59-Cal-06 | shared_selected | 91.67 | 6.44 | 7.39 | 0.9766 | 24/264 |
| Vib59-Cal-07 | shape40 | 100.00 | 1.14 | 0.57 | 1.0000 | 24/264 |
| Vib59-Cal-07 | baseline232 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | cycle_only100 | 100.00 | 10.61 | 5.30 | 0.9984 | 24/264 |
| Vib59-Cal-07 | input_only100 | 100.00 | 9.09 | 4.55 | 1.0000 | 24/264 |
| Vib59-Cal-07 | personal25_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | personal25_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | personal25_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | personal100_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | personal100_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | personal100_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | personal_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | matched_shared_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | matched_random_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-07 | shared_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-08 | shape40 | 100.00 | 2.65 | 1.33 | 0.9989 | 24/264 |
| Vib59-Cal-08 | baseline232 | 100.00 | 3.41 | 1.70 | 0.9984 | 24/264 |
| Vib59-Cal-08 | cycle_only100 | 100.00 | 31.44 | 15.72 | 0.9643 | 24/264 |
| Vib59-Cal-08 | input_only100 | 100.00 | 22.35 | 11.17 | 0.9885 | 24/264 |
| Vib59-Cal-08 | personal25_input_encoder256 | 100.00 | 3.41 | 1.70 | 0.9986 | 24/264 |
| Vib59-Cal-08 | personal25_cycle_encoder256 | 100.00 | 4.55 | 2.27 | 0.9986 | 24/264 |
| Vib59-Cal-08 | personal25_cycle_statistics240 | 100.00 | 4.55 | 2.27 | 0.9981 | 24/264 |
| Vib59-Cal-08 | personal100_input_encoder256 | 100.00 | 3.41 | 1.70 | 0.9981 | 24/264 |
| Vib59-Cal-08 | personal100_cycle_encoder256 | 100.00 | 4.92 | 2.46 | 0.9984 | 24/264 |
| Vib59-Cal-08 | personal100_cycle_statistics240 | 100.00 | 4.17 | 2.08 | 0.9983 | 24/264 |
| Vib59-Cal-08 | personal_selected | 100.00 | 3.41 | 1.70 | 0.9984 | 24/264 |
| Vib59-Cal-08 | matched_shared_branch | 100.00 | 3.41 | 1.70 | 0.9984 | 24/264 |
| Vib59-Cal-08 | matched_random_branch | 100.00 | 3.41 | 1.70 | 0.9984 | 24/264 |
| Vib59-Cal-08 | shared_selected | 100.00 | 3.41 | 1.70 | 0.9984 | 24/264 |
| Vib59-Cal-09 | shape40 | 91.67 | 0.38 | 4.36 | 0.9992 | 24/264 |
| Vib59-Cal-09 | baseline232 | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | cycle_only100 | 87.50 | 9.09 | 10.80 | 0.9631 | 24/264 |
| Vib59-Cal-09 | input_only100 | 87.50 | 4.55 | 8.52 | 0.9855 | 24/264 |
| Vib59-Cal-09 | personal25_input_encoder256 | 100.00 | 0.38 | 0.19 | 0.9992 | 24/264 |
| Vib59-Cal-09 | personal25_cycle_encoder256 | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | personal25_cycle_statistics240 | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | personal100_input_encoder256 | 100.00 | 0.38 | 0.19 | 0.9992 | 24/264 |
| Vib59-Cal-09 | personal100_cycle_encoder256 | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | personal100_cycle_statistics240 | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | personal_selected | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | matched_shared_branch | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | matched_random_branch | 100.00 | 0.38 | 0.19 | 0.9991 | 24/264 |
| Vib59-Cal-09 | shared_selected | 100.00 | 0.38 | 0.19 | 0.9992 | 24/264 |
| Vib59-Cal-10 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | baseline232 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | cycle_only100 | 100.00 | 3.41 | 1.70 | 0.9948 | 24/264 |
| Vib59-Cal-10 | input_only100 | 100.00 | 6.06 | 3.03 | 0.9945 | 24/264 |
| Vib59-Cal-10 | personal25_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | personal25_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | personal25_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | personal100_input_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | personal100_cycle_encoder256 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | personal100_cycle_statistics240 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | personal_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | matched_shared_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | matched_random_branch | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-10 | shared_selected | 100.00 | 0.00 | 0.00 | 1.0000 | 24/264 |
| Vib59-Cal-11 | shape40 | 83.33 | 0.00 | 8.33 | 0.9984 | 24/264 |
| Vib59-Cal-11 | baseline232 | 75.00 | 0.00 | 12.50 | 0.9967 | 24/264 |
| Vib59-Cal-11 | cycle_only100 | 54.17 | 1.52 | 23.67 | 0.9077 | 24/264 |
| Vib59-Cal-11 | input_only100 | 66.67 | 3.41 | 18.37 | 0.9380 | 24/264 |
| Vib59-Cal-11 | personal25_input_encoder256 | 79.17 | 0.00 | 10.42 | 0.9962 | 24/264 |
| Vib59-Cal-11 | personal25_cycle_encoder256 | 75.00 | 0.00 | 12.50 | 0.9961 | 24/264 |
| Vib59-Cal-11 | personal25_cycle_statistics240 | 79.17 | 0.00 | 10.42 | 0.9962 | 24/264 |
| Vib59-Cal-11 | personal100_input_encoder256 | 79.17 | 0.00 | 10.42 | 0.9962 | 24/264 |
| Vib59-Cal-11 | personal100_cycle_encoder256 | 70.83 | 0.00 | 14.58 | 0.9961 | 24/264 |
| Vib59-Cal-11 | personal100_cycle_statistics240 | 79.17 | 0.00 | 10.42 | 0.9964 | 24/264 |
| Vib59-Cal-11 | personal_selected | 75.00 | 0.00 | 12.50 | 0.9967 | 24/264 |
| Vib59-Cal-11 | matched_shared_branch | 75.00 | 0.00 | 12.50 | 0.9967 | 24/264 |
| Vib59-Cal-11 | matched_random_branch | 75.00 | 0.00 | 12.50 | 0.9967 | 24/264 |
| Vib59-Cal-11 | shared_selected | 79.17 | 0.00 | 10.42 | 0.9965 | 24/264 |
| Vib59-Cal-12 | shape40 | 100.00 | 7.20 | 3.60 | 0.9997 | 24/264 |
| Vib59-Cal-12 | baseline232 | 100.00 | 9.85 | 4.92 | 0.9986 | 24/264 |
| Vib59-Cal-12 | cycle_only100 | 95.83 | 14.02 | 9.09 | 0.9747 | 24/264 |
| Vib59-Cal-12 | input_only100 | 100.00 | 9.85 | 4.92 | 0.9921 | 24/264 |
| Vib59-Cal-12 | personal25_input_encoder256 | 100.00 | 9.85 | 4.92 | 0.9989 | 24/264 |
| Vib59-Cal-12 | personal25_cycle_encoder256 | 100.00 | 7.95 | 3.98 | 0.9989 | 24/264 |
| Vib59-Cal-12 | personal25_cycle_statistics240 | 100.00 | 10.61 | 5.30 | 0.9978 | 24/264 |
| Vib59-Cal-12 | personal100_input_encoder256 | 100.00 | 9.47 | 4.73 | 0.9989 | 24/264 |
| Vib59-Cal-12 | personal100_cycle_encoder256 | 100.00 | 8.71 | 4.36 | 0.9987 | 24/264 |
| Vib59-Cal-12 | personal100_cycle_statistics240 | 100.00 | 10.23 | 5.11 | 0.9976 | 24/264 |
| Vib59-Cal-12 | personal_selected | 100.00 | 9.85 | 4.92 | 0.9986 | 24/264 |
| Vib59-Cal-12 | matched_shared_branch | 100.00 | 9.85 | 4.92 | 0.9986 | 24/264 |
| Vib59-Cal-12 | matched_random_branch | 100.00 | 9.85 | 4.92 | 0.9986 | 24/264 |
| Vib59-Cal-12 | shared_selected | 100.00 | 9.85 | 4.92 | 0.9986 | 24/264 |

## Cal-FAR5工作点

| 方法 | TAR% | FAR% | HTER% |
|---|---:|---:|---:|
| shape40 | 95.49 | 2.34 | 3.42 |
| baseline232 | 95.49 | 1.64 | 3.08 |
| cycle_only100 | 71.53 | 6.25 | 17.36 |
| input_only100 | 86.11 | 5.65 | 9.77 |
| personal25_input_encoder256 | 96.18 | 2.02 | 2.92 |
| personal25_cycle_encoder256 | 95.14 | 1.83 | 3.35 |
| personal25_cycle_statistics240 | 95.49 | 1.80 | 3.16 |
| personal100_input_encoder256 | 96.18 | 2.08 | 2.95 |
| personal100_cycle_encoder256 | 95.14 | 1.70 | 3.28 |
| personal100_cycle_statistics240 | 95.14 | 1.74 | 3.30 |
| personal_selected | 95.49 | 1.64 | 3.08 |
| matched_shared_branch | 95.49 | 1.64 | 3.08 |
| matched_random_branch | 95.49 | 1.64 | 3.08 |
| shared_selected | 95.49 | 1.70 | 3.11 |

## 限制与复核

每位24本人、264攻击，共288本人/3168攻击比较。攻击录制重复用于不同目标比较，bootstrap按目标身份而非录制。已有VibraVox语料多轮探索，本轮不是封存独立确认。不同用户句子不完全相同，不称同内容认证；全部句子角色通过文本哈希隔离。个人注册允许Audio，最终查询仅振动。生成误差降低不等于身份保留，个人风格可能让攻击者也趋向目标；必须看随机与共享模型对照及认证指标。

[完整逐用户双工作点CSV](../../outputs/vibravox_personal_cycle_supplement_20261110/per_user_public.csv)，[独立检查](../../outputs/vibravox_personal_cycle_supplement_20261110/verification_public.json)。原始数据、源身份映射、模型权重、阈值和样本分数均保持私有。

