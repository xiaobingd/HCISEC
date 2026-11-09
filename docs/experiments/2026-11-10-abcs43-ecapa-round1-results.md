# ABCS43 ECAPA探索第1轮：逐用户完整结果

2026-11-10。23 Base / 8 Cal / 12未见Test，K=12、seed42。使用全部43位可用参与者，沿用固定划分和句子隔离。通道0为振动、通道1为Audio仍是未确认假设。查询不使用Audio。

[两轮预定方案](2026-11-10-abcs43-ecapa-two-round-protocol.md)。shape40固定保留；student为192维；supplement是二者等权拼接232维。各个人学生100步，相同初始化与批次规则；current复用上一实验结果。

## 汇总：Cal-EER（百分比）

| 方法 | TAR | FAR | HTER | pooled AUC | 最差用户FAR |
|---|---:|---:|---:|---:|---:|
| shape40 | 95.83 | 8.30 | 6.23 | 0.9849 | 19.32 |
| current_student | 90.62 | 6.85 | 8.11 | 0.9704 | 17.05 |
| current_supplement | 96.18 | 5.52 | 4.67 | 0.9869 | 14.02 |
| no_audio_student | 88.19 | 6.63 | 9.22 | 0.9697 | 12.88 |
| no_audio_supplement | 95.14 | 5.52 | 5.19 | 0.9859 | 14.02 |
| frozen_student | 90.97 | 7.01 | 8.02 | 0.9703 | 17.42 |
| frozen_supplement | 96.53 | 5.62 | 4.55 | 0.9869 | 14.02 |
| wrong_identity_student | 88.89 | 7.01 | 9.06 | 0.9687 | 15.53 |
| wrong_identity_supplement | 95.83 | 5.62 | 4.89 | 0.9865 | 14.77 |
| same_identity_shuffle_student | 90.62 | 6.69 | 8.03 | 0.9702 | 16.67 |
| same_identity_shuffle_supplement | 96.53 | 5.68 | 4.58 | 0.9869 | 14.02 |

## 解释

冻结教师融合HTER=4.55%，现有全主体微调=4.67%，无Audio个人适配=5.19%。这次30步微调没有超过冻结教师。不能只凭超过原始形状6.23%就断言Audio收益：无Audio适配本身也有改善。

同身份录制错配融合HTER=4.58%，异身份错配=4.89%，都与正确目标接近。当前证据不能证明精确逐录制跨模态对应是主要收益来源；结果可能主要依赖身份级关系、正则化和融合。异身份随机错配也未大幅崩溃，进一步削弱强因果解释；需要更多种子和新的身份划分。

## 配对差异：HTER百分点（身份bootstrap）

| 比较 | 均值差 | 95%区间 | 改善人数/12 |
|---|---:|---|---:|
| current_supplement_minus_shape40 | -1.56 | [-3.50, +0.39] | 9 |
| current_supplement_minus_no_audio_supplement | -0.52 | [-1.06, -0.03] | 7 |
| current_supplement_minus_frozen_supplement | +0.13 | [-0.08, +0.51] | 3 |
| no_audio_supplement_minus_shape40 | -1.04 | [-2.90, +0.80] | 8 |
| no_audio_supplement_minus_current_supplement | +0.52 | [+0.05, +1.06] | 3 |
| no_audio_supplement_minus_frozen_supplement | +0.65 | [+0.00, +1.40] | 4 |
| frozen_supplement_minus_shape40 | -1.69 | [-4.02, +0.32] | 9 |
| frozen_supplement_minus_no_audio_supplement | -0.65 | [-1.39, +0.02] | 7 |
| frozen_supplement_minus_current_supplement | -0.13 | [-0.51, +0.09] | 1 |
| wrong_identity_supplement_minus_shape40 | -1.34 | [-3.11, +0.38] | 9 |
| wrong_identity_supplement_minus_no_audio_supplement | -0.30 | [-0.80, +0.11] | 5 |
| wrong_identity_supplement_minus_current_supplement | +0.22 | [-0.02, +0.54] | 1 |
| wrong_identity_supplement_minus_frozen_supplement | +0.35 | [-0.05, +1.01] | 2 |
| same_identity_shuffle_supplement_minus_shape40 | -1.66 | [-3.95, +0.35] | 8 |
| same_identity_shuffle_supplement_minus_no_audio_supplement | -0.62 | [-1.39, +0.02] | 6 |
| same_identity_shuffle_supplement_minus_current_supplement | -0.09 | [-0.47, +0.14] | 1 |
| same_identity_shuffle_supplement_minus_frozen_supplement | +0.03 | [+0.00, +0.09] | 0 |

## 全部12位未见用户：Cal-EER

| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |
|---|---|---:|---:|---:|---:|---:|
| ABCS43-Test-01 | shape40 | 100.00 | 19.32 | 9.66 | 0.9998 | 24/264 |
| ABCS43-Test-01 | current_student | 100.00 | 3.79 | 1.89 | 0.9968 | 24/264 |
| ABCS43-Test-01 | current_supplement | 100.00 | 5.68 | 2.84 | 0.9997 | 24/264 |
| ABCS43-Test-01 | no_audio_student | 100.00 | 4.92 | 2.46 | 0.9970 | 24/264 |
| ABCS43-Test-01 | no_audio_supplement | 100.00 | 7.20 | 3.60 | 0.9997 | 24/264 |
| ABCS43-Test-01 | frozen_student | 100.00 | 3.79 | 1.89 | 0.9967 | 24/264 |
| ABCS43-Test-01 | frozen_supplement | 100.00 | 6.06 | 3.03 | 0.9997 | 24/264 |
| ABCS43-Test-01 | wrong_identity_student | 100.00 | 4.17 | 2.08 | 0.9967 | 24/264 |
| ABCS43-Test-01 | wrong_identity_supplement | 100.00 | 6.44 | 3.22 | 0.9997 | 24/264 |
| ABCS43-Test-01 | same_identity_shuffle_student | 100.00 | 3.41 | 1.70 | 0.9965 | 24/264 |
| ABCS43-Test-01 | same_identity_shuffle_supplement | 100.00 | 6.06 | 3.03 | 0.9998 | 24/264 |
| ABCS43-Test-02 | shape40 | 100.00 | 6.44 | 3.22 | 1.0000 | 24/264 |
| ABCS43-Test-02 | current_student | 95.83 | 5.30 | 4.73 | 0.9918 | 24/264 |
| ABCS43-Test-02 | current_supplement | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| ABCS43-Test-02 | no_audio_student | 95.83 | 6.06 | 5.11 | 0.9927 | 24/264 |
| ABCS43-Test-02 | no_audio_supplement | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| ABCS43-Test-02 | frozen_student | 95.83 | 5.30 | 4.73 | 0.9916 | 24/264 |
| ABCS43-Test-02 | frozen_supplement | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| ABCS43-Test-02 | wrong_identity_student | 95.83 | 5.68 | 4.92 | 0.9923 | 24/264 |
| ABCS43-Test-02 | wrong_identity_supplement | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| ABCS43-Test-02 | same_identity_shuffle_student | 95.83 | 4.92 | 4.55 | 0.9916 | 24/264 |
| ABCS43-Test-02 | same_identity_shuffle_supplement | 100.00 | 4.55 | 2.27 | 1.0000 | 24/264 |
| ABCS43-Test-03 | shape40 | 91.67 | 0.00 | 4.17 | 0.9998 | 24/264 |
| ABCS43-Test-03 | current_student | 91.67 | 1.14 | 4.73 | 0.9967 | 24/264 |
| ABCS43-Test-03 | current_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | no_audio_student | 70.83 | 0.38 | 14.77 | 0.9957 | 24/264 |
| ABCS43-Test-03 | no_audio_supplement | 91.67 | 0.00 | 4.17 | 1.0000 | 24/264 |
| ABCS43-Test-03 | frozen_student | 91.67 | 1.14 | 4.73 | 0.9968 | 24/264 |
| ABCS43-Test-03 | frozen_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | wrong_identity_student | 75.00 | 0.38 | 12.69 | 0.9973 | 24/264 |
| ABCS43-Test-03 | wrong_identity_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-03 | same_identity_shuffle_student | 91.67 | 1.14 | 4.73 | 0.9967 | 24/264 |
| ABCS43-Test-03 | same_identity_shuffle_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/264 |
| ABCS43-Test-04 | shape40 | 100.00 | 18.94 | 9.47 | 0.9965 | 24/264 |
| ABCS43-Test-04 | current_student | 100.00 | 8.71 | 4.36 | 0.9834 | 24/264 |
| ABCS43-Test-04 | current_supplement | 100.00 | 13.26 | 6.63 | 0.9983 | 24/264 |
| ABCS43-Test-04 | no_audio_student | 100.00 | 10.98 | 5.49 | 0.9828 | 24/264 |
| ABCS43-Test-04 | no_audio_supplement | 100.00 | 14.02 | 7.01 | 0.9984 | 24/264 |
| ABCS43-Test-04 | frozen_student | 100.00 | 9.47 | 4.73 | 0.9833 | 24/264 |
| ABCS43-Test-04 | frozen_supplement | 100.00 | 13.64 | 6.82 | 0.9983 | 24/264 |
| ABCS43-Test-04 | wrong_identity_student | 100.00 | 10.23 | 5.11 | 0.9823 | 24/264 |
| ABCS43-Test-04 | wrong_identity_supplement | 100.00 | 14.77 | 7.39 | 0.9981 | 24/264 |
| ABCS43-Test-04 | same_identity_shuffle_student | 100.00 | 8.33 | 4.17 | 0.9831 | 24/264 |
| ABCS43-Test-04 | same_identity_shuffle_supplement | 100.00 | 13.64 | 6.82 | 0.9984 | 24/264 |
| ABCS43-Test-05 | shape40 | 100.00 | 14.77 | 7.39 | 0.9871 | 24/264 |
| ABCS43-Test-05 | current_student | 95.83 | 17.05 | 10.61 | 0.9648 | 24/264 |
| ABCS43-Test-05 | current_supplement | 100.00 | 14.02 | 7.01 | 0.9853 | 24/264 |
| ABCS43-Test-05 | no_audio_student | 91.67 | 12.88 | 10.61 | 0.9646 | 24/264 |
| ABCS43-Test-05 | no_audio_supplement | 100.00 | 13.26 | 6.63 | 0.9853 | 24/264 |
| ABCS43-Test-05 | frozen_student | 95.83 | 17.42 | 10.80 | 0.9648 | 24/264 |
| ABCS43-Test-05 | frozen_supplement | 100.00 | 14.02 | 7.01 | 0.9850 | 24/264 |
| ABCS43-Test-05 | wrong_identity_student | 91.67 | 15.53 | 11.93 | 0.9602 | 24/264 |
| ABCS43-Test-05 | wrong_identity_supplement | 100.00 | 13.64 | 6.82 | 0.9848 | 24/264 |
| ABCS43-Test-05 | same_identity_shuffle_student | 95.83 | 16.67 | 10.42 | 0.9643 | 24/264 |
| ABCS43-Test-05 | same_identity_shuffle_supplement | 100.00 | 14.02 | 7.01 | 0.9856 | 24/264 |
| ABCS43-Test-06 | shape40 | 100.00 | 18.18 | 9.09 | 0.9901 | 24/264 |
| ABCS43-Test-06 | current_student | 95.83 | 13.26 | 8.71 | 0.9749 | 24/264 |
| ABCS43-Test-06 | current_supplement | 100.00 | 10.98 | 5.49 | 0.9924 | 24/264 |
| ABCS43-Test-06 | no_audio_student | 95.83 | 12.88 | 8.52 | 0.9762 | 24/264 |
| ABCS43-Test-06 | no_audio_supplement | 100.00 | 10.98 | 5.49 | 0.9927 | 24/264 |
| ABCS43-Test-06 | frozen_student | 95.83 | 13.64 | 8.90 | 0.9749 | 24/264 |
| ABCS43-Test-06 | frozen_supplement | 100.00 | 11.36 | 5.68 | 0.9924 | 24/264 |
| ABCS43-Test-06 | wrong_identity_student | 95.83 | 14.39 | 9.28 | 0.9733 | 24/264 |
| ABCS43-Test-06 | wrong_identity_supplement | 100.00 | 10.98 | 5.49 | 0.9926 | 24/264 |
| ABCS43-Test-06 | same_identity_shuffle_student | 95.83 | 13.26 | 8.71 | 0.9755 | 24/264 |
| ABCS43-Test-06 | same_identity_shuffle_supplement | 100.00 | 11.36 | 5.68 | 0.9924 | 24/264 |
| ABCS43-Test-07 | shape40 | 100.00 | 6.06 | 3.03 | 0.9970 | 24/264 |
| ABCS43-Test-07 | current_student | 91.67 | 7.58 | 7.95 | 0.9798 | 24/264 |
| ABCS43-Test-07 | current_supplement | 100.00 | 5.30 | 2.65 | 0.9970 | 24/264 |
| ABCS43-Test-07 | no_audio_student | 95.83 | 7.95 | 6.06 | 0.9781 | 24/264 |
| ABCS43-Test-07 | no_audio_supplement | 100.00 | 5.68 | 2.84 | 0.9970 | 24/264 |
| ABCS43-Test-07 | frozen_student | 95.83 | 7.58 | 5.87 | 0.9798 | 24/264 |
| ABCS43-Test-07 | frozen_supplement | 100.00 | 5.30 | 2.65 | 0.9970 | 24/264 |
| ABCS43-Test-07 | wrong_identity_student | 91.67 | 7.95 | 8.14 | 0.9804 | 24/264 |
| ABCS43-Test-07 | wrong_identity_supplement | 100.00 | 5.30 | 2.65 | 0.9970 | 24/264 |
| ABCS43-Test-07 | same_identity_shuffle_student | 91.67 | 7.58 | 7.95 | 0.9795 | 24/264 |
| ABCS43-Test-07 | same_identity_shuffle_supplement | 100.00 | 6.06 | 3.03 | 0.9973 | 24/264 |
| ABCS43-Test-08 | shape40 | 91.67 | 2.65 | 5.49 | 0.9943 | 24/264 |
| ABCS43-Test-08 | current_student | 79.17 | 1.89 | 11.36 | 0.9495 | 24/264 |
| ABCS43-Test-08 | current_supplement | 83.33 | 1.89 | 9.28 | 0.9811 | 24/264 |
| ABCS43-Test-08 | no_audio_student | 79.17 | 1.89 | 11.36 | 0.9531 | 24/264 |
| ABCS43-Test-08 | no_audio_supplement | 83.33 | 0.76 | 8.71 | 0.9812 | 24/264 |
| ABCS43-Test-08 | frozen_student | 79.17 | 2.27 | 11.55 | 0.9490 | 24/264 |
| ABCS43-Test-08 | frozen_supplement | 83.33 | 1.89 | 9.28 | 0.9811 | 24/264 |
| ABCS43-Test-08 | wrong_identity_student | 79.17 | 3.79 | 12.31 | 0.9449 | 24/264 |
| ABCS43-Test-08 | wrong_identity_supplement | 83.33 | 1.89 | 9.28 | 0.9812 | 24/264 |
| ABCS43-Test-08 | same_identity_shuffle_student | 79.17 | 1.89 | 11.36 | 0.9489 | 24/264 |
| ABCS43-Test-08 | same_identity_shuffle_supplement | 83.33 | 1.89 | 9.28 | 0.9812 | 24/264 |
| ABCS43-Test-09 | shape40 | 95.83 | 3.79 | 3.98 | 0.9897 | 24/264 |
| ABCS43-Test-09 | current_student | 75.00 | 8.33 | 16.67 | 0.9542 | 24/264 |
| ABCS43-Test-09 | current_supplement | 91.67 | 4.17 | 6.25 | 0.9864 | 24/264 |
| ABCS43-Test-09 | no_audio_student | 75.00 | 7.58 | 16.29 | 0.9553 | 24/264 |
| ABCS43-Test-09 | no_audio_supplement | 87.50 | 4.17 | 8.33 | 0.9863 | 24/264 |
| ABCS43-Test-09 | frozen_student | 75.00 | 8.33 | 16.67 | 0.9539 | 24/264 |
| ABCS43-Test-09 | frozen_supplement | 91.67 | 4.17 | 6.25 | 0.9864 | 24/264 |
| ABCS43-Test-09 | wrong_identity_student | 75.00 | 8.33 | 16.67 | 0.9525 | 24/264 |
| ABCS43-Test-09 | wrong_identity_supplement | 91.67 | 4.17 | 6.25 | 0.9867 | 24/264 |
| ABCS43-Test-09 | same_identity_shuffle_student | 75.00 | 8.33 | 16.67 | 0.9539 | 24/264 |
| ABCS43-Test-09 | same_identity_shuffle_supplement | 91.67 | 4.17 | 6.25 | 0.9864 | 24/264 |
| ABCS43-Test-10 | shape40 | 95.83 | 2.27 | 3.22 | 0.9888 | 24/264 |
| ABCS43-Test-10 | current_student | 75.00 | 7.20 | 16.10 | 0.9347 | 24/264 |
| ABCS43-Test-10 | current_supplement | 91.67 | 2.27 | 5.30 | 0.9856 | 24/264 |
| ABCS43-Test-10 | no_audio_student | 79.17 | 7.95 | 14.39 | 0.9394 | 24/264 |
| ABCS43-Test-10 | no_audio_supplement | 91.67 | 2.65 | 5.49 | 0.9850 | 24/264 |
| ABCS43-Test-10 | frozen_student | 75.00 | 7.20 | 16.10 | 0.9347 | 24/264 |
| ABCS43-Test-10 | frozen_supplement | 91.67 | 2.27 | 5.30 | 0.9856 | 24/264 |
| ABCS43-Test-10 | wrong_identity_student | 75.00 | 6.44 | 15.72 | 0.9323 | 24/264 |
| ABCS43-Test-10 | wrong_identity_supplement | 91.67 | 2.27 | 5.30 | 0.9855 | 24/264 |
| ABCS43-Test-10 | same_identity_shuffle_student | 75.00 | 7.20 | 16.10 | 0.9323 | 24/264 |
| ABCS43-Test-10 | same_identity_shuffle_supplement | 91.67 | 2.27 | 5.30 | 0.9850 | 24/264 |
| ABCS43-Test-11 | shape40 | 100.00 | 5.30 | 2.65 | 0.9973 | 24/264 |
| ABCS43-Test-11 | current_student | 95.83 | 3.79 | 3.98 | 0.9897 | 24/264 |
| ABCS43-Test-11 | current_supplement | 100.00 | 3.03 | 1.52 | 0.9984 | 24/264 |
| ABCS43-Test-11 | no_audio_student | 91.67 | 3.41 | 5.87 | 0.9893 | 24/264 |
| ABCS43-Test-11 | no_audio_supplement | 100.00 | 2.65 | 1.33 | 0.9983 | 24/264 |
| ABCS43-Test-11 | frozen_student | 95.83 | 3.79 | 3.98 | 0.9897 | 24/264 |
| ABCS43-Test-11 | frozen_supplement | 100.00 | 3.03 | 1.52 | 0.9984 | 24/264 |
| ABCS43-Test-11 | wrong_identity_student | 95.83 | 3.79 | 3.98 | 0.9890 | 24/264 |
| ABCS43-Test-11 | wrong_identity_supplement | 100.00 | 3.03 | 1.52 | 0.9984 | 24/264 |
| ABCS43-Test-11 | same_identity_shuffle_student | 95.83 | 3.79 | 3.98 | 0.9899 | 24/264 |
| ABCS43-Test-11 | same_identity_shuffle_supplement | 100.00 | 3.03 | 1.52 | 0.9984 | 24/264 |
| ABCS43-Test-12 | shape40 | 75.00 | 1.89 | 13.45 | 0.9901 | 24/264 |
| ABCS43-Test-12 | current_student | 91.67 | 4.17 | 6.25 | 0.9613 | 24/264 |
| ABCS43-Test-12 | current_supplement | 91.67 | 1.14 | 4.73 | 0.9877 | 24/264 |
| ABCS43-Test-12 | no_audio_student | 83.33 | 2.65 | 9.66 | 0.9635 | 24/264 |
| ABCS43-Test-12 | no_audio_supplement | 87.50 | 0.38 | 6.44 | 0.9883 | 24/264 |
| ABCS43-Test-12 | frozen_student | 91.67 | 4.17 | 6.25 | 0.9613 | 24/264 |
| ABCS43-Test-12 | frozen_supplement | 95.83 | 1.14 | 2.65 | 0.9877 | 24/264 |
| ABCS43-Test-12 | wrong_identity_student | 91.67 | 3.41 | 5.87 | 0.9651 | 24/264 |
| ABCS43-Test-12 | wrong_identity_supplement | 87.50 | 0.38 | 6.44 | 0.9902 | 24/264 |
| ABCS43-Test-12 | same_identity_shuffle_student | 91.67 | 3.79 | 6.06 | 0.9618 | 24/264 |
| ABCS43-Test-12 | same_identity_shuffle_supplement | 95.83 | 1.14 | 2.65 | 0.9877 | 24/264 |

## 全部8位校准用户（用于校准，不是独立Test）：Cal-EER

| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |
|---|---|---:|---:|---:|---:|---:|
| ABCS43-Cal-01 | shape40 | 100.00 | 4.76 | 2.38 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | current_student | 95.83 | 0.60 | 2.38 | 0.9993 | 24/168 |
| ABCS43-Cal-01 | current_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | no_audio_student | 95.83 | 0.60 | 2.38 | 0.9990 | 24/168 |
| ABCS43-Cal-01 | no_audio_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | frozen_student | 95.83 | 0.60 | 2.38 | 0.9993 | 24/168 |
| ABCS43-Cal-01 | frozen_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | wrong_identity_student | 95.83 | 0.60 | 2.38 | 0.9988 | 24/168 |
| ABCS43-Cal-01 | wrong_identity_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-01 | same_identity_shuffle_student | 95.83 | 0.60 | 2.38 | 0.9990 | 24/168 |
| ABCS43-Cal-01 | same_identity_shuffle_supplement | 100.00 | 0.60 | 0.30 | 1.0000 | 24/168 |
| ABCS43-Cal-02 | shape40 | 95.83 | 0.60 | 2.38 | 0.9980 | 24/168 |
| ABCS43-Cal-02 | current_student | 100.00 | 2.98 | 1.49 | 0.9995 | 24/168 |
| ABCS43-Cal-02 | current_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | no_audio_student | 100.00 | 2.98 | 1.49 | 0.9995 | 24/168 |
| ABCS43-Cal-02 | no_audio_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | frozen_student | 100.00 | 3.57 | 1.79 | 0.9995 | 24/168 |
| ABCS43-Cal-02 | frozen_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | wrong_identity_student | 100.00 | 2.98 | 1.49 | 0.9995 | 24/168 |
| ABCS43-Cal-02 | wrong_identity_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-02 | same_identity_shuffle_student | 100.00 | 3.57 | 1.79 | 0.9995 | 24/168 |
| ABCS43-Cal-02 | same_identity_shuffle_supplement | 100.00 | 0.60 | 0.30 | 0.9990 | 24/168 |
| ABCS43-Cal-03 | shape40 | 100.00 | 1.79 | 0.89 | 1.0000 | 24/168 |
| ABCS43-Cal-03 | current_student | 100.00 | 7.74 | 3.87 | 0.9918 | 24/168 |
| ABCS43-Cal-03 | current_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | no_audio_student | 100.00 | 7.14 | 3.57 | 0.9918 | 24/168 |
| ABCS43-Cal-03 | no_audio_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | frozen_student | 100.00 | 7.74 | 3.87 | 0.9918 | 24/168 |
| ABCS43-Cal-03 | frozen_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | wrong_identity_student | 100.00 | 7.14 | 3.57 | 0.9926 | 24/168 |
| ABCS43-Cal-03 | wrong_identity_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-03 | same_identity_shuffle_student | 100.00 | 7.74 | 3.87 | 0.9921 | 24/168 |
| ABCS43-Cal-03 | same_identity_shuffle_supplement | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-04 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-04 | current_student | 91.67 | 2.38 | 5.36 | 0.9968 | 24/168 |
| ABCS43-Cal-04 | current_supplement | 91.67 | 0.60 | 4.46 | 0.9993 | 24/168 |
| ABCS43-Cal-04 | no_audio_student | 95.83 | 1.79 | 2.98 | 0.9975 | 24/168 |
| ABCS43-Cal-04 | no_audio_supplement | 100.00 | 0.60 | 0.30 | 0.9995 | 24/168 |
| ABCS43-Cal-04 | frozen_student | 91.67 | 2.38 | 5.36 | 0.9968 | 24/168 |
| ABCS43-Cal-04 | frozen_supplement | 91.67 | 0.60 | 4.46 | 0.9993 | 24/168 |
| ABCS43-Cal-04 | wrong_identity_student | 95.83 | 4.17 | 4.17 | 0.9945 | 24/168 |
| ABCS43-Cal-04 | wrong_identity_supplement | 95.83 | 0.60 | 2.38 | 0.9993 | 24/168 |
| ABCS43-Cal-04 | same_identity_shuffle_student | 91.67 | 2.38 | 5.36 | 0.9965 | 24/168 |
| ABCS43-Cal-04 | same_identity_shuffle_supplement | 91.67 | 0.60 | 4.46 | 0.9993 | 24/168 |
| ABCS43-Cal-05 | shape40 | 95.83 | 2.98 | 3.57 | 0.9968 | 24/168 |
| ABCS43-Cal-05 | current_student | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-05 | current_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | no_audio_student | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-05 | no_audio_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | frozen_student | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-05 | frozen_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | wrong_identity_student | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-05 | wrong_identity_supplement | 100.00 | 1.79 | 0.89 | 0.9995 | 24/168 |
| ABCS43-Cal-05 | same_identity_shuffle_student | 100.00 | 2.98 | 1.49 | 0.9998 | 24/168 |
| ABCS43-Cal-05 | same_identity_shuffle_supplement | 100.00 | 1.79 | 0.89 | 0.9993 | 24/168 |
| ABCS43-Cal-06 | shape40 | 91.67 | 10.71 | 9.52 | 0.9680 | 24/168 |
| ABCS43-Cal-06 | current_student | 87.50 | 13.69 | 13.10 | 0.9472 | 24/168 |
| ABCS43-Cal-06 | current_supplement | 95.83 | 10.71 | 7.44 | 0.9697 | 24/168 |
| ABCS43-Cal-06 | no_audio_student | 83.33 | 11.90 | 14.29 | 0.9494 | 24/168 |
| ABCS43-Cal-06 | no_audio_supplement | 87.50 | 10.71 | 11.61 | 0.9697 | 24/168 |
| ABCS43-Cal-06 | frozen_student | 87.50 | 13.69 | 13.10 | 0.9469 | 24/168 |
| ABCS43-Cal-06 | frozen_supplement | 95.83 | 10.71 | 7.44 | 0.9697 | 24/168 |
| ABCS43-Cal-06 | wrong_identity_student | 83.33 | 13.10 | 14.88 | 0.9516 | 24/168 |
| ABCS43-Cal-06 | wrong_identity_supplement | 91.67 | 10.71 | 9.52 | 0.9695 | 24/168 |
| ABCS43-Cal-06 | same_identity_shuffle_student | 87.50 | 13.10 | 12.80 | 0.9477 | 24/168 |
| ABCS43-Cal-06 | same_identity_shuffle_supplement | 95.83 | 10.71 | 7.44 | 0.9692 | 24/168 |
| ABCS43-Cal-07 | shape40 | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | current_student | 95.83 | 2.38 | 3.27 | 0.9990 | 24/168 |
| ABCS43-Cal-07 | current_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | no_audio_student | 95.83 | 1.79 | 2.98 | 0.9988 | 24/168 |
| ABCS43-Cal-07 | no_audio_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | frozen_student | 95.83 | 2.38 | 3.27 | 0.9990 | 24/168 |
| ABCS43-Cal-07 | frozen_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | wrong_identity_student | 95.83 | 1.79 | 2.98 | 0.9988 | 24/168 |
| ABCS43-Cal-07 | wrong_identity_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-07 | same_identity_shuffle_student | 95.83 | 2.38 | 3.27 | 0.9990 | 24/168 |
| ABCS43-Cal-07 | same_identity_shuffle_supplement | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | shape40 | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | current_student | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | current_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | no_audio_student | 100.00 | 0.00 | 0.00 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | no_audio_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | frozen_student | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | frozen_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | wrong_identity_student | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | wrong_identity_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | same_identity_shuffle_student | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |
| ABCS43-Cal-08 | same_identity_shuffle_supplement | 95.83 | 0.00 | 2.08 | 1.0000 | 24/168 |

## Cal-FAR5工作点汇总

| 方法 | TAR% | FAR% | HTER% |
|---|---:|---:|---:|
| shape40 | 97.22 | 11.14 | 6.96 |
| current_student | 92.36 | 8.18 | 7.91 |
| current_supplement | 96.53 | 6.50 | 4.99 |
| no_audio_student | 92.71 | 9.69 | 8.49 |
| no_audio_supplement | 95.49 | 6.16 | 5.33 |
| frozen_student | 92.36 | 8.18 | 7.91 |
| frozen_supplement | 96.53 | 6.50 | 4.99 |
| wrong_identity_student | 91.67 | 8.62 | 8.48 |
| wrong_identity_supplement | 96.53 | 7.01 | 5.24 |
| same_identity_shuffle_student | 92.01 | 8.18 | 8.08 |
| same_identity_shuffle_supplement | 96.53 | 6.38 | 4.92 |

## 限制与可复现性

这是同一已探索数据划分上的单种子多方法探索，不是独立确认。多项比较的区间未做多重比较校正；不能选择Test最好者作为正式最终方案。Cal-FAR5不保证未知Test FAR≤5%。每位24本人、264攻击，总288本人/3168攻击比较；同一录制参与多目标比较，不能按独立录制计算置信区间。bootstrap单位是目标身份。

重要的训练限制：个人适配使用本人12条注册句子，而Base回放使用另一组4句；个人分类头的目标身份与背景身份没有同句困难负样本。虽然最终跨句查询和跨用户同句比较保持隔离，这仍可能让个人适配学习到内容差异；不能称适配阶段已完全消除文本混杂。此问题两轮都相同，未引入Test泄漏，但可能限制教师微调收益，下一步应补充同注册句子的Base背景录制后再做独立对照。

Base23位仅用于通用模型训练，不伪报未见用户结果。全部20个注册目标均报告，未移除难用户或难攻击。完整双工作点逐用户数据见[逐用户CSV](../../outputs/abcs43_ecapa_round1_20261110/per_user_public.csv)。教师、学生权重、阈值、样本分数、源身份映射不发布。检查记录见verification_public.json。

