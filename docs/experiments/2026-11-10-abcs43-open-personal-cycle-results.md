# ABCS43开放集：个人CycleGAN补充结果

2026-11-10。全部43人固定23 Base / 8 Cal / 12 Test，K=12、seed42。共享训练未见Cal/Test身份，注册阶段各自只用本人12条配对微调；每个目标保留全部11位未见攻击者。

本轮主结果：Cal选择个人25步、input_encoder256、权重0.5；Test HTER由4.67%变为4.96%。该配置未改善原基线。

**通道0为振动、通道1为Audio仍是工作假设，物理归属尚未独立确认。结果不能直接表述为已验证的BC/AC认证。**

[预定方案](2026-11-10-abcs43-open-personal-cycle-protocol.md)。原始shape40＋既有个人ECAPA/V4学生192的232维基线保持不变。这里是紧凑CycleGAN新增分支，不是历史完整772维重现。

## 技术路线与隔离

- 从相同2445条WAV提取统一40×32对数功率谱。共享G/F只用23位Base的1380 fit配对，4000更新、batch16，约46.38等效轮；345 develop选择第3500步及对应判别器，不是200轮。
- 全段去直流、16kHz、4096 Hann/1024 hop/16384 FFT、50–7600Hz几何40频带；时间插值32格。所有人使用同一Base-fit全局模态均值/标准差和3SD截断，不拟合个人尺度。Audio指频谱，非可听波形。
- G/F width32、4残差块、PatchGAN；LSGAN＋10配对L1＋10循环L1＋1恒等映射L1。恒等映射不是身份监督。
- 每位Cal/Test独立重置共享G/F，仅用本人12条注册配对微调100步、batch6、LR2e-5，保存25/100步，固定Base判别器。每个目标模型同时处理本人及全部攻击者。
- 输入G编码256、循环G编码256、生成Audio的F编码256、循环统计240、生成统计80；Scaler仅拟合共享G/F的Base-fit输出后固定。原始统计和已有学生不重训。
- Cal01–04定阈值，Cal05–08选择31个配置之一（两步数×五分支×三权重＋不补充），攻击查询限制在各自4人子群。锁定后才训练Test个人G/F。最终用全部8位Cal定Cal-EER/Cal-FAR5阈值。
- 注册12文本、Cal查询24文本、Test查询24文本在对应群体内真实一致，且与Base fit/develop角色全局互斥。查询Audio不计算特征或前向；读取立体声WAV进行CRC和通道0解码会涉及整文件，不能称未物理读取查询Audio字节。

## Cal选择

step=25，branch=input_encoder256，alpha=0.5；Cal开发HTER=4.17%，AUC=0.9894。该配置在Test个人模型训练前锁定。

## 最终Test汇总（Cal-EER）

| 方法 | TAR% | FAR% | HTER% | pooled AUC | 最差目标FAR% |
|---|---:|---:|---:|---:|---:|
| shape40 | 95.83 | 8.30 | 6.23 | 0.9849 | 19.32 |
| statistics240 | 88.89 | 15.37 | 13.24 | 0.9384 | 36.74 |
| student192 | 90.62 | 6.85 | 8.11 | 0.9704 | 17.05 |
| current_supplement | 96.18 | 5.52 | 4.67 | 0.9869 | 14.02 |
| cycle_only100 | 81.25 | 18.02 | 18.39 | 0.9087 | 36.36 |
| personal25_input_supplement | 95.49 | 5.40 | 4.96 | 0.9868 | 14.39 |
| personal25_cycle_supplement | 95.14 | 5.65 | 5.26 | 0.9863 | 14.39 |
| personal25_statistics_supplement | 94.79 | 6.85 | 6.03 | 0.9836 | 15.15 |
| personal100_input_supplement | 95.49 | 5.21 | 4.86 | 0.9868 | 14.02 |
| personal100_cycle_supplement | 95.14 | 5.15 | 5.00 | 0.9864 | 13.64 |
| personal100_statistics_supplement | 94.79 | 5.97 | 5.59 | 0.9843 | 14.39 |
| shared_input_supplement | 95.83 | 5.27 | 4.72 | 0.9876 | 13.26 |
| shared_cycle_supplement | 94.44 | 4.64 | 5.10 | 0.9872 | 12.12 |
| random_input_supplement | 96.18 | 5.71 | 4.77 | 0.9867 | 14.02 |
| random_cycle_supplement | 95.49 | 5.27 | 4.89 | 0.9867 | 13.64 |
| selected_cycle_supplement | 95.49 | 5.40 | 4.96 | 0.9868 | 14.39 |
| matched_shared_supplement | 95.83 | 5.27 | 4.72 | 0.9876 | 13.26 |
| matched_random_supplement | 96.18 | 5.71 | 4.77 | 0.9867 | 14.02 |

## 主结果与对照解释

Cal开发子群有96条本人、288条攻击比较。选中配置本人接受95条，原基线94条；攻击接受分别21、21条。选择收益很小，不能将其视为稳定的泛化优势。

被选择的是G对输入通道0的编码，不是循环ACC编码；不能把这个选择解释成已验证Audio信息进入最终循环ACC。

Cal锁定的是个人25步的input_encoder256分支，权重0.5；不能根据Test换成其他固定分支。匹配shared/random使用相同分支、权重，固定input/cycle另用0.5权重。

相对原232维基线，主结果HTER变化+0.28个百分点，身份bootstrap95%区间[-0.14,+0.82]，4/12位改善。

## 全部配对HTER差（百分点）

| 比较 | 均值差 | 身份bootstrap95%区间 | 改善人数/12 |
|---|---:|---|---:|
| personal25_input_supplement_minus_current_supplement | +0.28 | [-0.14,+0.84] | 4 |
| personal25_cycle_supplement_minus_current_supplement | +0.58 | [+0.03,+1.34] | 3 |
| personal25_statistics_supplement_minus_current_supplement | +1.36 | [+0.68,+2.11] | 0 |
| personal100_input_supplement_minus_current_supplement | +0.19 | [-0.22,+0.73] | 4 |
| personal100_cycle_supplement_minus_current_supplement | +0.33 | [-0.32,+1.18] | 6 |
| personal100_statistics_supplement_minus_current_supplement | +0.92 | [+0.27,+1.66] | 3 |
| shared_input_supplement_minus_current_supplement | +0.05 | [-0.30,+0.51] | 5 |
| shared_cycle_supplement_minus_current_supplement | +0.43 | [-0.43,+1.63] | 7 |
| random_input_supplement_minus_current_supplement | +0.09 | [-0.47,+0.65] | 4 |
| random_cycle_supplement_minus_current_supplement | +0.22 | [-0.17,+0.77] | 7 |
| selected_cycle_supplement_minus_current_supplement | +0.28 | [-0.14,+0.82] | 4 |
| matched_shared_supplement_minus_current_supplement | +0.05 | [-0.30,+0.51] | 5 |
| matched_random_supplement_minus_current_supplement | +0.09 | [-0.47,+0.63] | 4 |

## 全部12位未见目标：Cal-EER

| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |
|---|---|---:|---:|---:|---:|---:|
| ABCS43-Test-01 | shape40 | 100.00 | 19.32 | 9.66 | 0.9998 | 24/264 |
| ABCS43-Test-01 | statistics240 | 100.00 | 18.56 | 9.28 | 1.0000 | 24/264 |
| ABCS43-Test-01 | student192 | 100.00 | 3.79 | 1.89 | 0.9968 | 24/264 |
| ABCS43-Test-01 | current_supplement | 100.00 | 5.68 | 2.84 | 0.9997 | 24/264 |
| ABCS43-Test-01 | cycle_only100 | 95.83 | 24.62 | 14.39 | 0.9547 | 24/264 |
| ABCS43-Test-01 | personal25_input_supplement | 100.00 | 6.06 | 3.03 | 0.9995 | 24/264 |
| ABCS43-Test-01 | personal25_cycle_supplement | 100.00 | 7.20 | 3.60 | 1.0000 | 24/264 |
| ABCS43-Test-01 | personal25_statistics_supplement | 100.00 | 8.33 | 4.17 | 0.9994 | 24/264 |
| ABCS43-Test-01 | personal100_input_supplement | 100.00 | 5.68 | 2.84 | 0.9997 | 24/264 |
| ABCS43-Test-01 | personal100_cycle_supplement | 100.00 | 6.82 | 3.41 | 1.0000 | 24/264 |
| ABCS43-Test-01 | personal100_statistics_supplement | 100.00 | 7.58 | 3.79 | 0.9995 | 24/264 |
| ABCS43-Test-01 | shared_input_supplement | 100.00 | 6.44 | 3.22 | 0.9995 | 24/264 |
| ABCS43-Test-01 | shared_cycle_supplement | 100.00 | 5.30 | 2.65 | 0.9995 | 24/264 |
| ABCS43-Test-01 | random_input_supplement | 100.00 | 7.20 | 3.60 | 0.9995 | 24/264 |
| ABCS43-Test-01 | random_cycle_supplement | 100.00 | 5.68 | 2.84 | 1.0000 | 24/264 |
| ABCS43-Test-01 | selected_cycle_supplement | 100.00 | 6.06 | 3.03 | 0.9995 | 24/264 |
| ABCS43-Test-01 | matched_shared_supplement | 100.00 | 6.44 | 3.22 | 0.9995 | 24/264 |
| ABCS43-Test-01 | matched_random_supplement | 100.00 | 7.20 | 3.60 | 0.9995 | 24/264 |
| ABCS43-Test-02 | shape40 | 100.00 | 6.44 | 3.22 | 1.0000 | 24/264 |
| ABCS43-Test-02 | statistics240 | 100.00 | 1.89 | 0.95 | 0.9995 | 24/264 |
| ABCS43-Test-02 | student192 | 95.83 | 5.30 | 4.73 | 0.9918 | 24/264 |
| ABCS43-Test-02 | current_supplement | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| ABCS43-Test-02 | cycle_only100 | 100.00 | 9.09 | 4.55 | 0.9956 | 24/264 |
| ABCS43-Test-02 | personal25_input_supplement | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| ABCS43-Test-02 | personal25_cycle_supplement | 100.00 | 3.41 | 1.70 | 1.0000 | 24/264 |
| ABCS43-Test-02 | personal25_statistics_supplement | 100.00 | 6.06 | 3.03 | 0.9998 | 24/264 |
| ABCS43-Test-02 | personal100_input_supplement | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| ABCS43-Test-02 | personal100_cycle_supplement | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| ABCS43-Test-02 | personal100_statistics_supplement | 100.00 | 5.30 | 2.65 | 0.9998 | 24/264 |
| ABCS43-Test-02 | shared_input_supplement | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| ABCS43-Test-02 | shared_cycle_supplement | 100.00 | 2.65 | 1.33 | 1.0000 | 24/264 |
| ABCS43-Test-02 | random_input_supplement | 100.00 | 5.68 | 2.84 | 1.0000 | 24/264 |
| ABCS43-Test-02 | random_cycle_supplement | 100.00 | 3.79 | 1.89 | 1.0000 | 24/264 |
| ABCS43-Test-02 | selected_cycle_supplement | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| ABCS43-Test-02 | matched_shared_supplement | 100.00 | 3.03 | 1.52 | 1.0000 | 24/264 |
| ABCS43-Test-02 | matched_random_supplement | 100.00 | 5.68 | 2.84 | 1.0000 | 24/264 |
| ABCS43-Test-03 | shape40 | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| ABCS43-Test-03 | statistics240 | 66.67 | 0.00 | 16.67 | 0.9949 | 24/264 |
| ABCS43-Test-03 | student192 | 91.67 | 1.14 | 4.73 | 0.9967 | 24/264 |
| ABCS43-Test-03 | current_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | cycle_only100 | 41.67 | 11.74 | 35.04 | 0.8493 | 24/264 |
| ABCS43-Test-03 | personal25_input_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | personal25_cycle_supplement | 87.50 | 0.00 | 6.25 | 1.0000 | 24/264 |
| ABCS43-Test-03 | personal25_statistics_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | personal100_input_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | personal100_cycle_supplement | 87.50 | 0.00 | 6.25 | 1.0000 | 24/264 |
| ABCS43-Test-03 | personal100_statistics_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | shared_input_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | shared_cycle_supplement | 83.33 | 0.00 | 8.33 | 1.0000 | 24/264 |
| ABCS43-Test-03 | random_input_supplement | 91.67 | 0.00 | 4.17 | 1.0000 | 24/264 |
| ABCS43-Test-03 | random_cycle_supplement | 91.67 | 0.00 | 4.17 | 1.0000 | 24/264 |
| ABCS43-Test-03 | selected_cycle_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | matched_shared_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | matched_random_supplement | 91.67 | 0.00 | 4.17 | 1.0000 | 24/264 |
| ABCS43-Test-04 | shape40 | 100.00 | 18.94 | 9.47 | 0.9965 | 24/264 |
| ABCS43-Test-04 | statistics240 | 100.00 | 6.82 | 3.41 | 0.9968 | 24/264 |
| ABCS43-Test-04 | student192 | 100.00 | 8.71 | 4.36 | 0.9834 | 24/264 |
| ABCS43-Test-04 | current_supplement | 100.00 | 13.26 | 6.63 | 0.9983 | 24/264 |
| ABCS43-Test-04 | cycle_only100 | 100.00 | 18.94 | 9.47 | 0.9637 | 24/264 |
| ABCS43-Test-04 | personal25_input_supplement | 100.00 | 12.88 | 6.44 | 0.9994 | 24/264 |
| ABCS43-Test-04 | personal25_cycle_supplement | 100.00 | 14.02 | 7.01 | 0.9995 | 24/264 |
| ABCS43-Test-04 | personal25_statistics_supplement | 100.00 | 14.77 | 7.39 | 0.9975 | 24/264 |
| ABCS43-Test-04 | personal100_input_supplement | 100.00 | 12.88 | 6.44 | 0.9994 | 24/264 |
| ABCS43-Test-04 | personal100_cycle_supplement | 100.00 | 13.64 | 6.82 | 0.9995 | 24/264 |
| ABCS43-Test-04 | personal100_statistics_supplement | 100.00 | 14.39 | 7.20 | 0.9973 | 24/264 |
| ABCS43-Test-04 | shared_input_supplement | 100.00 | 13.26 | 6.63 | 0.9989 | 24/264 |
| ABCS43-Test-04 | shared_cycle_supplement | 100.00 | 12.12 | 6.06 | 0.9995 | 24/264 |
| ABCS43-Test-04 | random_input_supplement | 100.00 | 14.02 | 7.01 | 0.9986 | 24/264 |
| ABCS43-Test-04 | random_cycle_supplement | 100.00 | 13.26 | 6.63 | 0.9981 | 24/264 |
| ABCS43-Test-04 | selected_cycle_supplement | 100.00 | 12.88 | 6.44 | 0.9994 | 24/264 |
| ABCS43-Test-04 | matched_shared_supplement | 100.00 | 13.26 | 6.63 | 0.9989 | 24/264 |
| ABCS43-Test-04 | matched_random_supplement | 100.00 | 14.02 | 7.01 | 0.9986 | 24/264 |
| ABCS43-Test-05 | shape40 | 100.00 | 14.77 | 7.39 | 0.9871 | 24/264 |
| ABCS43-Test-05 | statistics240 | 100.00 | 36.74 | 18.37 | 0.9569 | 24/264 |
| ABCS43-Test-05 | student192 | 95.83 | 17.05 | 10.61 | 0.9648 | 24/264 |
| ABCS43-Test-05 | current_supplement | 100.00 | 14.02 | 7.01 | 0.9853 | 24/264 |
| ABCS43-Test-05 | cycle_only100 | 100.00 | 31.44 | 15.72 | 0.9747 | 24/264 |
| ABCS43-Test-05 | personal25_input_supplement | 100.00 | 14.39 | 7.20 | 0.9872 | 24/264 |
| ABCS43-Test-05 | personal25_cycle_supplement | 100.00 | 14.39 | 7.20 | 0.9894 | 24/264 |
| ABCS43-Test-05 | personal25_statistics_supplement | 100.00 | 15.15 | 7.58 | 0.9845 | 24/264 |
| ABCS43-Test-05 | personal100_input_supplement | 100.00 | 14.02 | 7.01 | 0.9867 | 24/264 |
| ABCS43-Test-05 | personal100_cycle_supplement | 100.00 | 12.88 | 6.44 | 0.9912 | 24/264 |
| ABCS43-Test-05 | personal100_statistics_supplement | 100.00 | 13.26 | 6.63 | 0.9848 | 24/264 |
| ABCS43-Test-05 | shared_input_supplement | 100.00 | 12.88 | 6.44 | 0.9867 | 24/264 |
| ABCS43-Test-05 | shared_cycle_supplement | 100.00 | 11.74 | 5.87 | 0.9866 | 24/264 |
| ABCS43-Test-05 | random_input_supplement | 100.00 | 13.64 | 6.82 | 0.9860 | 24/264 |
| ABCS43-Test-05 | random_cycle_supplement | 100.00 | 13.64 | 6.82 | 0.9869 | 24/264 |
| ABCS43-Test-05 | selected_cycle_supplement | 100.00 | 14.39 | 7.20 | 0.9872 | 24/264 |
| ABCS43-Test-05 | matched_shared_supplement | 100.00 | 12.88 | 6.44 | 0.9867 | 24/264 |
| ABCS43-Test-05 | matched_random_supplement | 100.00 | 13.64 | 6.82 | 0.9860 | 24/264 |
| ABCS43-Test-06 | shape40 | 100.00 | 18.18 | 9.09 | 0.9901 | 24/264 |
| ABCS43-Test-06 | statistics240 | 100.00 | 35.23 | 17.61 | 0.9910 | 24/264 |
| ABCS43-Test-06 | student192 | 95.83 | 13.26 | 8.71 | 0.9749 | 24/264 |
| ABCS43-Test-06 | current_supplement | 100.00 | 10.98 | 5.49 | 0.9924 | 24/264 |
| ABCS43-Test-06 | cycle_only100 | 100.00 | 23.48 | 11.74 | 0.9579 | 24/264 |
| ABCS43-Test-06 | personal25_input_supplement | 100.00 | 11.36 | 5.68 | 0.9940 | 24/264 |
| ABCS43-Test-06 | personal25_cycle_supplement | 100.00 | 11.74 | 5.87 | 0.9937 | 24/264 |
| ABCS43-Test-06 | personal25_statistics_supplement | 100.00 | 14.39 | 7.20 | 0.9918 | 24/264 |
| ABCS43-Test-06 | personal100_input_supplement | 100.00 | 10.98 | 5.49 | 0.9940 | 24/264 |
| ABCS43-Test-06 | personal100_cycle_supplement | 100.00 | 10.61 | 5.30 | 0.9932 | 24/264 |
| ABCS43-Test-06 | personal100_statistics_supplement | 100.00 | 11.74 | 5.87 | 0.9918 | 24/264 |
| ABCS43-Test-06 | shared_input_supplement | 100.00 | 11.36 | 5.68 | 0.9942 | 24/264 |
| ABCS43-Test-06 | shared_cycle_supplement | 100.00 | 10.61 | 5.30 | 0.9942 | 24/264 |
| ABCS43-Test-06 | random_input_supplement | 100.00 | 11.36 | 5.68 | 0.9921 | 24/264 |
| ABCS43-Test-06 | random_cycle_supplement | 100.00 | 10.61 | 5.30 | 0.9929 | 24/264 |
| ABCS43-Test-06 | selected_cycle_supplement | 100.00 | 11.36 | 5.68 | 0.9940 | 24/264 |
| ABCS43-Test-06 | matched_shared_supplement | 100.00 | 11.36 | 5.68 | 0.9942 | 24/264 |
| ABCS43-Test-06 | matched_random_supplement | 100.00 | 11.36 | 5.68 | 0.9921 | 24/264 |
| ABCS43-Test-07 | shape40 | 100.00 | 6.06 | 3.03 | 0.9970 | 24/264 |
| ABCS43-Test-07 | statistics240 | 100.00 | 25.38 | 12.69 | 0.9506 | 24/264 |
| ABCS43-Test-07 | student192 | 91.67 | 7.58 | 7.95 | 0.9798 | 24/264 |
| ABCS43-Test-07 | current_supplement | 100.00 | 5.30 | 2.65 | 0.9970 | 24/264 |
| ABCS43-Test-07 | cycle_only100 | 79.17 | 7.95 | 14.39 | 0.9504 | 24/264 |
| ABCS43-Test-07 | personal25_input_supplement | 100.00 | 5.30 | 2.65 | 0.9983 | 24/264 |
| ABCS43-Test-07 | personal25_cycle_supplement | 100.00 | 5.30 | 2.65 | 0.9979 | 24/264 |
| ABCS43-Test-07 | personal25_statistics_supplement | 100.00 | 6.44 | 3.22 | 0.9965 | 24/264 |
| ABCS43-Test-07 | personal100_input_supplement | 100.00 | 5.30 | 2.65 | 0.9981 | 24/264 |
| ABCS43-Test-07 | personal100_cycle_supplement | 100.00 | 4.55 | 2.27 | 0.9983 | 24/264 |
| ABCS43-Test-07 | personal100_statistics_supplement | 100.00 | 6.44 | 3.22 | 0.9968 | 24/264 |
| ABCS43-Test-07 | shared_input_supplement | 100.00 | 5.68 | 2.84 | 0.9983 | 24/264 |
| ABCS43-Test-07 | shared_cycle_supplement | 100.00 | 5.30 | 2.65 | 0.9976 | 24/264 |
| ABCS43-Test-07 | random_input_supplement | 100.00 | 6.06 | 3.03 | 0.9972 | 24/264 |
| ABCS43-Test-07 | random_cycle_supplement | 100.00 | 5.30 | 2.65 | 0.9968 | 24/264 |
| ABCS43-Test-07 | selected_cycle_supplement | 100.00 | 5.30 | 2.65 | 0.9983 | 24/264 |
| ABCS43-Test-07 | matched_shared_supplement | 100.00 | 5.68 | 2.84 | 0.9983 | 24/264 |
| ABCS43-Test-07 | matched_random_supplement | 100.00 | 6.06 | 3.03 | 0.9972 | 24/264 |
| ABCS43-Test-08 | shape40 | 91.67 | 2.65 | 5.49 | 0.9943 | 24/264 |
| ABCS43-Test-08 | statistics240 | 62.50 | 0.00 | 18.75 | 0.9877 | 24/264 |
| ABCS43-Test-08 | student192 | 79.17 | 1.89 | 11.36 | 0.9495 | 24/264 |
| ABCS43-Test-08 | current_supplement | 83.33 | 1.89 | 9.28 | 0.9811 | 24/264 |
| ABCS43-Test-08 | cycle_only100 | 50.00 | 4.55 | 27.27 | 0.9265 | 24/264 |
| ABCS43-Test-08 | personal25_input_supplement | 83.33 | 1.52 | 9.09 | 0.9822 | 24/264 |
| ABCS43-Test-08 | personal25_cycle_supplement | 83.33 | 1.14 | 8.90 | 0.9811 | 24/264 |
| ABCS43-Test-08 | personal25_statistics_supplement | 83.33 | 1.89 | 9.28 | 0.9792 | 24/264 |
| ABCS43-Test-08 | personal100_input_supplement | 83.33 | 1.14 | 8.90 | 0.9820 | 24/264 |
| ABCS43-Test-08 | personal100_cycle_supplement | 83.33 | 0.76 | 8.71 | 0.9817 | 24/264 |
| ABCS43-Test-08 | personal100_statistics_supplement | 83.33 | 1.52 | 9.09 | 0.9792 | 24/264 |
| ABCS43-Test-08 | shared_input_supplement | 83.33 | 1.52 | 9.09 | 0.9826 | 24/264 |
| ABCS43-Test-08 | shared_cycle_supplement | 83.33 | 0.76 | 8.71 | 0.9817 | 24/264 |
| ABCS43-Test-08 | random_input_supplement | 83.33 | 0.38 | 8.52 | 0.9798 | 24/264 |
| ABCS43-Test-08 | random_cycle_supplement | 83.33 | 1.52 | 9.09 | 0.9806 | 24/264 |
| ABCS43-Test-08 | selected_cycle_supplement | 83.33 | 1.52 | 9.09 | 0.9822 | 24/264 |
| ABCS43-Test-08 | matched_shared_supplement | 83.33 | 1.52 | 9.09 | 0.9826 | 24/264 |
| ABCS43-Test-08 | matched_random_supplement | 83.33 | 0.38 | 8.52 | 0.9798 | 24/264 |
| ABCS43-Test-09 | shape40 | 95.83 | 3.79 | 3.98 | 0.9897 | 24/264 |
| ABCS43-Test-09 | statistics240 | 75.00 | 6.82 | 15.91 | 0.9692 | 24/264 |
| ABCS43-Test-09 | student192 | 75.00 | 8.33 | 16.67 | 0.9542 | 24/264 |
| ABCS43-Test-09 | current_supplement | 91.67 | 4.17 | 6.25 | 0.9864 | 24/264 |
| ABCS43-Test-09 | cycle_only100 | 75.00 | 22.73 | 23.86 | 0.8647 | 24/264 |
| ABCS43-Test-09 | personal25_input_supplement | 87.50 | 4.92 | 8.71 | 0.9848 | 24/264 |
| ABCS43-Test-09 | personal25_cycle_supplement | 91.67 | 4.17 | 6.25 | 0.9864 | 24/264 |
| ABCS43-Test-09 | personal25_statistics_supplement | 87.50 | 4.92 | 8.71 | 0.9848 | 24/264 |
| ABCS43-Test-09 | personal100_input_supplement | 87.50 | 4.55 | 8.52 | 0.9845 | 24/264 |
| ABCS43-Test-09 | personal100_cycle_supplement | 87.50 | 4.17 | 8.33 | 0.9852 | 24/264 |
| ABCS43-Test-09 | personal100_statistics_supplement | 87.50 | 4.55 | 8.52 | 0.9848 | 24/264 |
| ABCS43-Test-09 | shared_input_supplement | 87.50 | 4.17 | 8.33 | 0.9855 | 24/264 |
| ABCS43-Test-09 | shared_cycle_supplement | 91.67 | 4.17 | 6.25 | 0.9864 | 24/264 |
| ABCS43-Test-09 | random_input_supplement | 91.67 | 4.17 | 6.25 | 0.9867 | 24/264 |
| ABCS43-Test-09 | random_cycle_supplement | 87.50 | 4.17 | 8.33 | 0.9867 | 24/264 |
| ABCS43-Test-09 | selected_cycle_supplement | 87.50 | 4.92 | 8.71 | 0.9848 | 24/264 |
| ABCS43-Test-09 | matched_shared_supplement | 87.50 | 4.17 | 8.33 | 0.9855 | 24/264 |
| ABCS43-Test-09 | matched_random_supplement | 91.67 | 4.17 | 6.25 | 0.9867 | 24/264 |
| ABCS43-Test-10 | shape40 | 95.83 | 2.27 | 3.22 | 0.9888 | 24/264 |
| ABCS43-Test-10 | statistics240 | 95.83 | 23.48 | 13.83 | 0.9261 | 24/264 |
| ABCS43-Test-10 | student192 | 75.00 | 7.20 | 16.10 | 0.9347 | 24/264 |
| ABCS43-Test-10 | current_supplement | 91.67 | 2.27 | 5.30 | 0.9856 | 24/264 |
| ABCS43-Test-10 | cycle_only100 | 91.67 | 17.80 | 13.07 | 0.9246 | 24/264 |
| ABCS43-Test-10 | personal25_input_supplement | 91.67 | 1.89 | 5.11 | 0.9871 | 24/264 |
| ABCS43-Test-10 | personal25_cycle_supplement | 91.67 | 1.89 | 5.11 | 0.9874 | 24/264 |
| ABCS43-Test-10 | personal25_statistics_supplement | 87.50 | 6.06 | 9.28 | 0.9773 | 24/264 |
| ABCS43-Test-10 | personal100_input_supplement | 91.67 | 1.52 | 4.92 | 0.9869 | 24/264 |
| ABCS43-Test-10 | personal100_cycle_supplement | 91.67 | 2.27 | 5.30 | 0.9866 | 24/264 |
| ABCS43-Test-10 | personal100_statistics_supplement | 87.50 | 4.17 | 8.33 | 0.9774 | 24/264 |
| ABCS43-Test-10 | shared_input_supplement | 91.67 | 2.65 | 5.49 | 0.9885 | 24/264 |
| ABCS43-Test-10 | shared_cycle_supplement | 91.67 | 1.52 | 4.92 | 0.9896 | 24/264 |
| ABCS43-Test-10 | random_input_supplement | 91.67 | 2.65 | 5.49 | 0.9860 | 24/264 |
| ABCS43-Test-10 | random_cycle_supplement | 91.67 | 1.89 | 5.11 | 0.9847 | 24/264 |
| ABCS43-Test-10 | selected_cycle_supplement | 91.67 | 1.89 | 5.11 | 0.9871 | 24/264 |
| ABCS43-Test-10 | matched_shared_supplement | 91.67 | 2.65 | 5.49 | 0.9885 | 24/264 |
| ABCS43-Test-10 | matched_random_supplement | 91.67 | 2.65 | 5.49 | 0.9860 | 24/264 |
| ABCS43-Test-11 | shape40 | 100.00 | 5.30 | 2.65 | 0.9973 | 24/264 |
| ABCS43-Test-11 | statistics240 | 100.00 | 24.62 | 12.31 | 0.9299 | 24/264 |
| ABCS43-Test-11 | student192 | 95.83 | 3.79 | 3.98 | 0.9897 | 24/264 |
| ABCS43-Test-11 | current_supplement | 100.00 | 3.03 | 1.52 | 0.9984 | 24/264 |
| ABCS43-Test-11 | cycle_only100 | 87.50 | 36.36 | 24.43 | 0.8523 | 24/264 |
| ABCS43-Test-11 | personal25_input_supplement | 100.00 | 3.03 | 1.52 | 0.9978 | 24/264 |
| ABCS43-Test-11 | personal25_cycle_supplement | 100.00 | 4.55 | 2.27 | 0.9979 | 24/264 |
| ABCS43-Test-11 | personal25_statistics_supplement | 100.00 | 4.17 | 2.08 | 0.9970 | 24/264 |
| ABCS43-Test-11 | personal100_input_supplement | 100.00 | 3.03 | 1.52 | 0.9981 | 24/264 |
| ABCS43-Test-11 | personal100_cycle_supplement | 100.00 | 3.03 | 1.52 | 0.9975 | 24/264 |
| ABCS43-Test-11 | personal100_statistics_supplement | 100.00 | 2.65 | 1.33 | 0.9976 | 24/264 |
| ABCS43-Test-11 | shared_input_supplement | 100.00 | 1.89 | 0.95 | 0.9986 | 24/264 |
| ABCS43-Test-11 | shared_cycle_supplement | 95.83 | 1.52 | 2.84 | 0.9984 | 24/264 |
| ABCS43-Test-11 | random_input_supplement | 100.00 | 2.65 | 1.33 | 0.9989 | 24/264 |
| ABCS43-Test-11 | random_cycle_supplement | 100.00 | 2.65 | 1.33 | 0.9986 | 24/264 |
| ABCS43-Test-11 | selected_cycle_supplement | 100.00 | 3.03 | 1.52 | 0.9978 | 24/264 |
| ABCS43-Test-11 | matched_shared_supplement | 100.00 | 1.89 | 0.95 | 0.9986 | 24/264 |
| ABCS43-Test-11 | matched_random_supplement | 100.00 | 2.65 | 1.33 | 0.9989 | 24/264 |
| ABCS43-Test-12 | shape40 | 75.00 | 1.89 | 13.45 | 0.9901 | 24/264 |
| ABCS43-Test-12 | statistics240 | 66.67 | 4.92 | 19.13 | 0.9246 | 24/264 |
| ABCS43-Test-12 | student192 | 91.67 | 4.17 | 6.25 | 0.9613 | 24/264 |
| ABCS43-Test-12 | current_supplement | 91.67 | 1.14 | 4.73 | 0.9877 | 24/264 |
| ABCS43-Test-12 | cycle_only100 | 54.17 | 7.58 | 26.70 | 0.8447 | 24/264 |
| ABCS43-Test-12 | personal25_input_supplement | 87.50 | 0.38 | 6.44 | 0.9875 | 24/264 |
| ABCS43-Test-12 | personal25_cycle_supplement | 87.50 | 0.00 | 6.25 | 0.9872 | 24/264 |
| ABCS43-Test-12 | personal25_statistics_supplement | 83.33 | 0.00 | 8.33 | 0.9812 | 24/264 |
| ABCS43-Test-12 | personal100_input_supplement | 87.50 | 0.38 | 6.44 | 0.9875 | 24/264 |
| ABCS43-Test-12 | personal100_cycle_supplement | 91.67 | 0.00 | 4.17 | 0.9872 | 24/264 |
| ABCS43-Test-12 | personal100_statistics_supplement | 83.33 | 0.00 | 8.33 | 0.9831 | 24/264 |
| ABCS43-Test-12 | shared_input_supplement | 91.67 | 0.38 | 4.36 | 0.9890 | 24/264 |
| ABCS43-Test-12 | shared_cycle_supplement | 87.50 | 0.00 | 6.25 | 0.9893 | 24/264 |
| ABCS43-Test-12 | random_input_supplement | 95.83 | 0.76 | 2.46 | 0.9878 | 24/264 |
| ABCS43-Test-12 | random_cycle_supplement | 91.67 | 0.76 | 4.55 | 0.9878 | 24/264 |
| ABCS43-Test-12 | selected_cycle_supplement | 87.50 | 0.38 | 6.44 | 0.9875 | 24/264 |
| ABCS43-Test-12 | matched_shared_supplement | 91.67 | 0.38 | 4.36 | 0.9890 | 24/264 |
| ABCS43-Test-12 | matched_random_supplement | 95.83 | 0.76 | 2.46 | 0.9878 | 24/264 |

## 全部8位Cal诊断（用于选择与校准，不是独立Test）：Cal-EER

| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |
|---|---|---:|---:|---:|---:|---:|
| ABCS43-Cal-01 | shape40 | 100.00 | 4.76 | 2.38 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | statistics240 | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | student192 | 95.83 | 0.60 | 2.38 | 0.9993 | 24/168 |
| ABCS43-Cal-01 | current_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | cycle_only100 | 91.67 | 4.76 | 6.55 | 0.9834 | 24/168 |
| ABCS43-Cal-01 | personal25_input_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | personal25_cycle_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | personal25_statistics_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | personal100_input_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | personal100_cycle_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | personal100_statistics_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | shared_input_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | shared_cycle_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | random_input_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | random_cycle_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | selected_cycle_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | matched_shared_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | matched_random_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-02 | shape40 | 95.83 | 0.60 | 2.38 | 0.9980 | 24/168 |
| ABCS43-Cal-02 | statistics240 | 100.00 | 7.74 | 3.87 | 0.9983 | 24/168 |
| ABCS43-Cal-02 | student192 | 100.00 | 2.98 | 1.49 | 0.9995 | 24/168 |
| ABCS43-Cal-02 | current_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | cycle_only100 | 95.83 | 5.95 | 5.06 | 0.9896 | 24/168 |
| ABCS43-Cal-02 | personal25_input_supplement | 100.00 | 0.60 | 0.30 | 0.9988 | 24/168 |
| ABCS43-Cal-02 | personal25_cycle_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | personal25_statistics_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | personal100_input_supplement | 100.00 | 0.60 | 0.30 | 0.9988 | 24/168 |
| ABCS43-Cal-02 | personal100_cycle_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | personal100_statistics_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | shared_input_supplement | 100.00 | 0.60 | 0.30 | 0.9988 | 24/168 |
| ABCS43-Cal-02 | shared_cycle_supplement | 100.00 | 0.60 | 0.30 | 0.9993 | 24/168 |
| ABCS43-Cal-02 | random_input_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | random_cycle_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | selected_cycle_supplement | 100.00 | 0.60 | 0.30 | 0.9988 | 24/168 |
| ABCS43-Cal-02 | matched_shared_supplement | 100.00 | 0.60 | 0.30 | 0.9988 | 24/168 |
| ABCS43-Cal-02 | matched_random_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-03 | shape40 | 100.00 | 1.79 | 0.89 | 1.0000 | 24/168 |
| ABCS43-Cal-03 | statistics240 | 100.00 | 15.48 | 7.74 | 0.9712 | 24/168 |
| ABCS43-Cal-03 | student192 | 100.00 | 7.74 | 3.87 | 0.9918 | 24/168 |
| ABCS43-Cal-03 | current_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | cycle_only100 | 87.50 | 17.86 | 15.18 | 0.8929 | 24/168 |
| ABCS43-Cal-03 | personal25_input_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | personal25_cycle_supplement | 100.00 | 4.17 | 2.08 | 0.9990 | 24/168 |
| ABCS43-Cal-03 | personal25_statistics_supplement | 100.00 | 5.36 | 2.68 | 0.9993 | 24/168 |
| ABCS43-Cal-03 | personal100_input_supplement | 100.00 | 2.38 | 1.19 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | personal100_cycle_supplement | 100.00 | 2.98 | 1.49 | 0.9995 | 24/168 |
| ABCS43-Cal-03 | personal100_statistics_supplement | 100.00 | 4.17 | 2.08 | 0.9993 | 24/168 |
| ABCS43-Cal-03 | shared_input_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | shared_cycle_supplement | 100.00 | 1.79 | 0.89 | 0.9993 | 24/168 |
| ABCS43-Cal-03 | random_input_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | random_cycle_supplement | 100.00 | 2.98 | 1.49 | 1.0000 | 24/168 |
| ABCS43-Cal-03 | selected_cycle_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | matched_shared_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | matched_random_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-04 | statistics240 | 62.50 | 0.00 | 18.75 | 0.9983 | 24/168 |
| ABCS43-Cal-04 | student192 | 91.67 | 2.38 | 5.36 | 0.9968 | 24/168 |
| ABCS43-Cal-04 | current_supplement | 91.67 | 0.60 | 4.46 | 0.9993 | 24/168 |
| ABCS43-Cal-04 | cycle_only100 | 87.50 | 2.98 | 7.74 | 0.9824 | 24/168 |
| ABCS43-Cal-04 | personal25_input_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | personal25_cycle_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | personal25_statistics_supplement | 87.50 | 0.60 | 6.55 | 0.9993 | 24/168 |
| ABCS43-Cal-04 | personal100_input_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | personal100_cycle_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | personal100_statistics_supplement | 91.67 | 0.60 | 4.46 | 0.9993 | 24/168 |
| ABCS43-Cal-04 | shared_input_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | shared_cycle_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | random_input_supplement | 91.67 | 0.00 | 4.17 | 1.0000 | 24/168 |
| ABCS43-Cal-04 | random_cycle_supplement | 91.67 | 0.00 | 4.17 | 0.9995 | 24/168 |
| ABCS43-Cal-04 | selected_cycle_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | matched_shared_supplement | 91.67 | 0.00 | 4.17 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | matched_random_supplement | 91.67 | 0.00 | 4.17 | 1.0000 | 24/168 |
| ABCS43-Cal-05 | shape40 | 95.83 | 2.98 | 3.57 | 0.9968 | 24/168 |
| ABCS43-Cal-05 | statistics240 | 100.00 | 6.55 | 3.27 | 0.9968 | 24/168 |
| ABCS43-Cal-05 | student192 | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-05 | current_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | cycle_only100 | 95.83 | 20.83 | 12.50 | 0.9400 | 24/168 |
| ABCS43-Cal-05 | personal25_input_supplement | 100.00 | 1.79 | 0.89 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | personal25_cycle_supplement | 100.00 | 2.38 | 1.19 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | personal25_statistics_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | personal100_input_supplement | 100.00 | 1.79 | 0.89 | 0.9990 | 24/168 |
| ABCS43-Cal-05 | personal100_cycle_supplement | 100.00 | 2.38 | 1.19 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | personal100_statistics_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | shared_input_supplement | 100.00 | 1.79 | 0.89 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | shared_cycle_supplement | 100.00 | 2.38 | 1.19 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | random_input_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | random_cycle_supplement | 100.00 | 1.79 | 0.89 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | selected_cycle_supplement | 100.00 | 1.79 | 0.89 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | matched_shared_supplement | 100.00 | 1.79 | 0.89 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | matched_random_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-06 | shape40 | 91.67 | 10.71 | 9.52 | 0.9680 | 24/168 |
| ABCS43-Cal-06 | statistics240 | 75.00 | 11.90 | 18.45 | 0.9343 | 24/168 |
| ABCS43-Cal-06 | student192 | 87.50 | 13.69 | 13.10 | 0.9472 | 24/168 |
| ABCS43-Cal-06 | current_supplement | 95.83 | 10.71 | 7.44 | 0.9697 | 24/168 |
| ABCS43-Cal-06 | cycle_only100 | 91.67 | 11.90 | 10.12 | 0.9541 | 24/168 |
| ABCS43-Cal-06 | personal25_input_supplement | 91.67 | 10.71 | 9.52 | 0.9702 | 24/168 |
| ABCS43-Cal-06 | personal25_cycle_supplement | 91.67 | 10.71 | 9.52 | 0.9683 | 24/168 |
| ABCS43-Cal-06 | personal25_statistics_supplement | 91.67 | 10.71 | 9.52 | 0.9623 | 24/168 |
| ABCS43-Cal-06 | personal100_input_supplement | 91.67 | 10.71 | 9.52 | 0.9705 | 24/168 |
| ABCS43-Cal-06 | personal100_cycle_supplement | 91.67 | 10.71 | 9.52 | 0.9707 | 24/168 |
| ABCS43-Cal-06 | personal100_statistics_supplement | 87.50 | 10.71 | 11.61 | 0.9628 | 24/168 |
| ABCS43-Cal-06 | shared_input_supplement | 91.67 | 10.71 | 9.52 | 0.9697 | 24/168 |
| ABCS43-Cal-06 | shared_cycle_supplement | 91.67 | 11.31 | 9.82 | 0.9668 | 24/168 |
| ABCS43-Cal-06 | random_input_supplement | 95.83 | 10.71 | 7.44 | 0.9683 | 24/168 |
| ABCS43-Cal-06 | random_cycle_supplement | 95.83 | 10.71 | 7.44 | 0.9685 | 24/168 |
| ABCS43-Cal-06 | selected_cycle_supplement | 91.67 | 10.71 | 9.52 | 0.9702 | 24/168 |
| ABCS43-Cal-06 | matched_shared_supplement | 91.67 | 10.71 | 9.52 | 0.9697 | 24/168 |
| ABCS43-Cal-06 | matched_random_supplement | 95.83 | 10.71 | 7.44 | 0.9683 | 24/168 |
| ABCS43-Cal-07 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | statistics240 | 100.00 | 18.45 | 9.23 | 0.9948 | 24/168 |
| ABCS43-Cal-07 | student192 | 95.83 | 2.38 | 3.27 | 0.9990 | 24/168 |
| ABCS43-Cal-07 | current_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | cycle_only100 | 87.50 | 18.45 | 15.48 | 0.9400 | 24/168 |
| ABCS43-Cal-07 | personal25_input_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | personal25_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | personal25_statistics_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | personal100_input_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | personal100_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | personal100_statistics_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | shared_input_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | shared_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | random_input_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | random_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | selected_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | matched_shared_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | matched_random_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | shape40 | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | statistics240 | 100.00 | 1.79 | 0.89 | 0.9998 | 24/168 |
| ABCS43-Cal-08 | student192 | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | current_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | cycle_only100 | 75.00 | 4.76 | 14.88 | 0.9673 | 24/168 |
| ABCS43-Cal-08 | personal25_input_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | personal25_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | personal25_statistics_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | personal100_input_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | personal100_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | personal100_statistics_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | shared_input_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | shared_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | random_input_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | random_cycle_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | selected_cycle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | matched_shared_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | matched_random_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |

## Cal-FAR5工作点汇总

| 方法 | TAR% | FAR% | HTER% |
|---|---:|---:|---:|
| shape40 | 97.22 | 11.14 | 6.96 |
| statistics240 | 81.94 | 10.67 | 14.36 |
| student192 | 92.36 | 8.18 | 7.91 |
| current_supplement | 96.53 | 6.50 | 4.99 |
| cycle_only100 | 66.67 | 9.06 | 21.20 |
| personal25_input_supplement | 96.88 | 6.91 | 5.02 |
| personal25_cycle_supplement | 96.88 | 8.27 | 5.70 |
| personal25_statistics_supplement | 96.88 | 9.82 | 6.47 |
| personal100_input_supplement | 96.53 | 6.66 | 5.07 |
| personal100_cycle_supplement | 97.22 | 7.83 | 5.30 |
| personal100_statistics_supplement | 96.88 | 8.90 | 6.01 |
| shared_input_supplement | 96.53 | 6.60 | 5.03 |
| shared_cycle_supplement | 95.83 | 6.38 | 5.27 |
| random_input_supplement | 96.18 | 6.06 | 4.94 |
| random_cycle_supplement | 96.53 | 6.12 | 4.80 |
| selected_cycle_supplement | 96.88 | 6.91 | 5.02 |
| matched_shared_supplement | 96.53 | 6.60 | 5.03 |
| matched_random_supplement | 96.18 | 6.06 | 4.94 |

## 边界与完整性

每位Test 24本人/264攻击，共288本人/3168攻击比较，攻击录制用于多个目标，不是独立3168条录制；bootstrap按目标身份，不按录制。仍是单种子、已探索语料，不是封存独立确认，多个候选未做多重比较校正。Cal-FAR5不保证Test FAR≤5%。23位Base不伪报未见用户认证结果，全部Test/Cal均报告。

原教师的Base回放与个人注册内容不同，此基线限制仍存在，本轮未重新训练该教师。生成或循环误差下降不等于身份保留，个人风格也可能让攻击输入趋向注册者。应同时看原始形状、232维、个人模型、共享模型和随机特征，并验证新的身份划分和更多种子。

[全部逐用户双工作点CSV](../../outputs/abcs43_open_personal_cycle_20261110/per_user_public.csv)，[检查记录](../../outputs/abcs43_open_personal_cycle_20261110/verification_public.json)。原始信号、私有频谱包、模型权重、姓名/文本映射、阈值、样本分数不发布。

独立复核40个个人G/F检查点、720个工作点通过；原四个参考结果复现，查询Audio特征/前向为0，所有查询接受/拒绝决定在GPU重算后不变，最大特征/分数差1.43e-05。
