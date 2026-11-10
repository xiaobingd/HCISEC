# mei九人闭集：个人CycleGAN特征补充实验

2026-11-10。真正闭集：九位用户均参与共享身份训练，九个独立本人/非本人二分类RBF头均使用全部八位已知负类；不把开放模型all_eight_mixed测试改名为闭集。全部九人报告，不剔除难用户或攻击来源。

[完整预定方案](2026-11-10-mei9-closed-personal-cycle-protocol.md)。本报告的编号是raw9索引，不保证对应历史trainb U编号。输入为同一头显采集的三轴OpenVR tracking vVelocity，不是原始硬件加速度。词序未逐条转写核验，因此同词/跨词为名义条件。

## 数据与模型

- 共享ACC编码器使用九人fit身份监督700步；Audio ECAPA使用九人fit回放＋注册共144条Audio微调100步，再冻结并给ACC学生蒸馏100步。这里Audio教师是九人共享闭集教师，个人CycleGAN才是逐注册者微调。
- 原始形状120维保持不变，与学生192维构成baseline312；保留原始720维统计、学生单支路。
- 共享G/F预训练4000次更新，batch16、约44.51等效轮，按36条develop录制选择第3250步。G为3→1、F为1→3，LSGAN＋配对L1＋循环L1，因通道不同不加恒等映射损失。
- 九个个人G/F各从共享模型重置，仅用本人的12条注册配对微调100步，保存25/100步，判别器冻结；其余用户注册负类由当前目标的G/F处理。
- 每人12注册（4词×3遍），用于二分类头训练；CalA、CalB各4条（每词下一遍），仅用于阈值与分支选择。最终阈值用CalA+B的8条本人/64条负类选择。
- 名义同词查询每人16条、128条攻击；跨词每人24条、192条攻击。所有样本索引互斥；同词与跨词共用同一阈值。查询Audio不参与模型前向。缓存归档可能整体解压，不能声称从未物理读取查询Audio字节。
- 后端固定balanced RBF SVM，C=10、gamma=scale；只在108条注册特征上拟合，不在拼接后重做StandardScaler。所有方法使用相同后端和数据。

## 校准选择

九目标平均CalB选择：step=0，branch=none，alpha=0.0；CalA选阈值。CalB HTER=9.55%，AUC=0.9922。配置在提取最终个人G/F查询特征前锁定。

## 宏平均结果（Cal-EER，百分比）

| 方法 | 条件 | TAR | FAR | HTER | 用户平均AUC |
|---|---|---:|---:|---:|---:|
| raw120 | 名义同词 | 81.94 | 10.33 | 14.19 | 0.9289 |
| raw120 | 名义跨词 | 79.63 | 11.57 | 15.97 | 0.9177 |
| statistics720 | 名义同词 | 83.33 | 7.47 | 12.07 | 0.9657 |
| statistics720 | 名义跨词 | 81.02 | 7.47 | 13.22 | 0.9371 |
| student192 | 名义同词 | 84.72 | 7.73 | 11.50 | 0.9687 |
| student192 | 名义跨词 | 83.80 | 6.89 | 11.55 | 0.9607 |
| baseline312 | 名义同词 | 83.33 | 3.91 | 10.29 | 0.9749 |
| baseline312 | 名义跨词 | 78.70 | 4.40 | 12.85 | 0.9701 |
| personal25_input256 | 名义同词 | 85.42 | 4.25 | 9.42 | 0.9764 |
| personal25_input256 | 名义跨词 | 81.94 | 5.21 | 11.63 | 0.9709 |
| personal25_cycle256 | 名义同词 | 86.11 | 5.12 | 9.51 | 0.9795 |
| personal25_cycle256 | 名义跨词 | 82.41 | 6.19 | 11.89 | 0.9709 |
| personal25_cycle_stats720 | 名义同词 | 85.42 | 5.38 | 9.98 | 0.9802 |
| personal25_cycle_stats720 | 名义跨词 | 81.94 | 6.42 | 12.24 | 0.9710 |
| personal100_input256 | 名义同词 | 84.72 | 4.60 | 9.94 | 0.9765 |
| personal100_input256 | 名义跨词 | 82.41 | 5.09 | 11.34 | 0.9710 |
| personal100_cycle256 | 名义同词 | 86.81 | 5.21 | 9.20 | 0.9782 |
| personal100_cycle256 | 名义跨词 | 82.87 | 5.84 | 11.49 | 0.9715 |
| personal100_cycle_stats720 | 名义同词 | 85.42 | 4.43 | 9.51 | 0.9795 |
| personal100_cycle_stats720 | 名义跨词 | 80.56 | 5.84 | 12.64 | 0.9729 |
| selected_cycle | 名义同词 | 83.33 | 3.91 | 10.29 | 0.9749 |
| selected_cycle | 名义跨词 | 78.70 | 4.40 | 12.85 | 0.9701 |
| cycle_only100 | 名义同词 | 80.56 | 21.96 | 20.70 | 0.8740 |
| cycle_only100 | 名义跨词 | 80.56 | 20.95 | 20.20 | 0.8646 |

## 自主选择相对baseline312的配对差异

| 条件 | HTER差百分点 | 身份bootstrap95%区间 | 改善人数/9 |
|---|---:|---|---:|
| sameword | +0.00 | [+0.00,+0.00] | 0 |
| crossword | +0.00 | [+0.00,+0.00] | 0 |

Cal选择不增加CycleGAN分支，自主选择结果等于baseline312；九个个人G/F都确实完成了25/100步训练，固定补充结果仍完整报告。不能在看到Test后用表现更好的固定分支替换该主结果。

## 逐用户完整结果（Cal-EER）

| 用户 | 条件 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |
|---|---|---|---:|---:|---:|---:|---:|
| Mei9-U01 | sameword | raw120 | 81.25 | 0.78 | 9.77 | 0.9922 | 16/128 |
| Mei9-U01 | sameword | statistics720 | 75.00 | 1.56 | 13.28 | 0.9907 | 16/128 |
| Mei9-U01 | sameword | student192 | 75.00 | 1.56 | 13.28 | 0.9932 | 16/128 |
| Mei9-U01 | sameword | baseline312 | 75.00 | 0.78 | 12.89 | 0.9946 | 16/128 |
| Mei9-U01 | sameword | personal25_input256 | 68.75 | 0.78 | 16.02 | 0.9937 | 16/128 |
| Mei9-U01 | sameword | personal25_cycle256 | 62.50 | 0.00 | 18.75 | 0.9946 | 16/128 |
| Mei9-U01 | sameword | personal25_cycle_stats720 | 68.75 | 0.00 | 15.62 | 0.9951 | 16/128 |
| Mei9-U01 | sameword | personal100_input256 | 68.75 | 0.78 | 16.02 | 0.9941 | 16/128 |
| Mei9-U01 | sameword | personal100_cycle256 | 68.75 | 0.78 | 16.02 | 0.9951 | 16/128 |
| Mei9-U01 | sameword | personal100_cycle_stats720 | 68.75 | 0.78 | 16.02 | 0.9946 | 16/128 |
| Mei9-U01 | sameword | selected_cycle | 75.00 | 0.78 | 12.89 | 0.9946 | 16/128 |
| Mei9-U01 | sameword | cycle_only100 | 100.00 | 9.38 | 4.69 | 0.9624 | 16/128 |
| Mei9-U01 | crossword | raw120 | 79.17 | 1.04 | 10.94 | 0.9909 | 24/192 |
| Mei9-U01 | crossword | statistics720 | 75.00 | 1.56 | 13.28 | 0.9859 | 24/192 |
| Mei9-U01 | crossword | student192 | 83.33 | 0.00 | 8.33 | 0.9980 | 24/192 |
| Mei9-U01 | crossword | baseline312 | 70.83 | 0.00 | 14.58 | 0.9996 | 24/192 |
| Mei9-U01 | crossword | personal25_input256 | 75.00 | 0.00 | 12.50 | 0.9996 | 24/192 |
| Mei9-U01 | crossword | personal25_cycle256 | 62.50 | 0.00 | 18.75 | 1.0000 | 24/192 |
| Mei9-U01 | crossword | personal25_cycle_stats720 | 66.67 | 0.00 | 16.67 | 0.9985 | 24/192 |
| Mei9-U01 | crossword | personal100_input256 | 70.83 | 0.00 | 14.58 | 0.9996 | 24/192 |
| Mei9-U01 | crossword | personal100_cycle256 | 66.67 | 0.00 | 16.67 | 1.0000 | 24/192 |
| Mei9-U01 | crossword | personal100_cycle_stats720 | 66.67 | 0.00 | 16.67 | 0.9989 | 24/192 |
| Mei9-U01 | crossword | selected_cycle | 70.83 | 0.00 | 14.58 | 0.9996 | 24/192 |
| Mei9-U01 | crossword | cycle_only100 | 79.17 | 14.58 | 17.71 | 0.9234 | 24/192 |
| Mei9-U02 | sameword | raw120 | 100.00 | 3.91 | 1.95 | 0.9976 | 16/128 |
| Mei9-U02 | sameword | statistics720 | 100.00 | 6.25 | 3.12 | 0.9937 | 16/128 |
| Mei9-U02 | sameword | student192 | 81.25 | 0.78 | 9.77 | 0.9873 | 16/128 |
| Mei9-U02 | sameword | baseline312 | 81.25 | 0.00 | 9.38 | 0.9917 | 16/128 |
| Mei9-U02 | sameword | personal25_input256 | 81.25 | 0.00 | 9.38 | 0.9922 | 16/128 |
| Mei9-U02 | sameword | personal25_cycle256 | 81.25 | 0.00 | 9.38 | 0.9951 | 16/128 |
| Mei9-U02 | sameword | personal25_cycle_stats720 | 87.50 | 0.00 | 6.25 | 0.9966 | 16/128 |
| Mei9-U02 | sameword | personal100_input256 | 81.25 | 0.00 | 9.38 | 0.9932 | 16/128 |
| Mei9-U02 | sameword | personal100_cycle256 | 81.25 | 0.00 | 9.38 | 0.9941 | 16/128 |
| Mei9-U02 | sameword | personal100_cycle_stats720 | 81.25 | 0.00 | 9.38 | 0.9966 | 16/128 |
| Mei9-U02 | sameword | selected_cycle | 81.25 | 0.00 | 9.38 | 0.9917 | 16/128 |
| Mei9-U02 | sameword | cycle_only100 | 93.75 | 21.09 | 13.67 | 0.9414 | 16/128 |
| Mei9-U02 | crossword | raw120 | 100.00 | 8.33 | 4.17 | 0.9922 | 24/192 |
| Mei9-U02 | crossword | statistics720 | 95.83 | 2.60 | 3.39 | 0.9980 | 24/192 |
| Mei9-U02 | crossword | student192 | 91.67 | 0.00 | 4.17 | 0.9993 | 24/192 |
| Mei9-U02 | crossword | baseline312 | 83.33 | 0.00 | 8.33 | 0.9998 | 24/192 |
| Mei9-U02 | crossword | personal25_input256 | 91.67 | 0.00 | 4.17 | 0.9996 | 24/192 |
| Mei9-U02 | crossword | personal25_cycle256 | 95.83 | 0.00 | 2.08 | 1.0000 | 24/192 |
| Mei9-U02 | crossword | personal25_cycle_stats720 | 91.67 | 0.00 | 4.17 | 1.0000 | 24/192 |
| Mei9-U02 | crossword | personal100_input256 | 95.83 | 0.00 | 2.08 | 0.9996 | 24/192 |
| Mei9-U02 | crossword | personal100_cycle256 | 95.83 | 0.00 | 2.08 | 0.9996 | 24/192 |
| Mei9-U02 | crossword | personal100_cycle_stats720 | 91.67 | 0.00 | 4.17 | 1.0000 | 24/192 |
| Mei9-U02 | crossword | selected_cycle | 83.33 | 0.00 | 8.33 | 0.9998 | 24/192 |
| Mei9-U02 | crossword | cycle_only100 | 100.00 | 11.46 | 5.73 | 0.9839 | 24/192 |
| Mei9-U03 | sameword | raw120 | 87.50 | 10.16 | 11.33 | 0.9502 | 16/128 |
| Mei9-U03 | sameword | statistics720 | 93.75 | 7.81 | 7.03 | 0.9707 | 16/128 |
| Mei9-U03 | sameword | student192 | 87.50 | 7.81 | 10.16 | 0.9761 | 16/128 |
| Mei9-U03 | sameword | baseline312 | 87.50 | 2.34 | 7.42 | 0.9800 | 16/128 |
| Mei9-U03 | sameword | personal25_input256 | 87.50 | 0.78 | 6.64 | 0.9805 | 16/128 |
| Mei9-U03 | sameword | personal25_cycle256 | 87.50 | 0.78 | 6.64 | 0.9834 | 16/128 |
| Mei9-U03 | sameword | personal25_cycle_stats720 | 87.50 | 0.78 | 6.64 | 0.9839 | 16/128 |
| Mei9-U03 | sameword | personal100_input256 | 87.50 | 0.78 | 6.64 | 0.9800 | 16/128 |
| Mei9-U03 | sameword | personal100_cycle256 | 87.50 | 0.78 | 6.64 | 0.9800 | 16/128 |
| Mei9-U03 | sameword | personal100_cycle_stats720 | 87.50 | 0.78 | 6.64 | 0.9814 | 16/128 |
| Mei9-U03 | sameword | selected_cycle | 87.50 | 2.34 | 7.42 | 0.9800 | 16/128 |
| Mei9-U03 | sameword | cycle_only100 | 81.25 | 17.19 | 17.97 | 0.9136 | 16/128 |
| Mei9-U03 | crossword | raw120 | 83.33 | 7.29 | 11.98 | 0.9757 | 24/192 |
| Mei9-U03 | crossword | statistics720 | 100.00 | 4.69 | 2.34 | 0.9822 | 24/192 |
| Mei9-U03 | crossword | student192 | 100.00 | 6.25 | 3.12 | 0.9961 | 24/192 |
| Mei9-U03 | crossword | baseline312 | 100.00 | 3.12 | 1.56 | 0.9959 | 24/192 |
| Mei9-U03 | crossword | personal25_input256 | 100.00 | 2.60 | 1.30 | 0.9963 | 24/192 |
| Mei9-U03 | crossword | personal25_cycle256 | 100.00 | 2.60 | 1.30 | 0.9978 | 24/192 |
| Mei9-U03 | crossword | personal25_cycle_stats720 | 91.67 | 1.56 | 4.95 | 0.9952 | 24/192 |
| Mei9-U03 | crossword | personal100_input256 | 100.00 | 2.60 | 1.30 | 0.9963 | 24/192 |
| Mei9-U03 | crossword | personal100_cycle256 | 100.00 | 2.60 | 1.30 | 0.9976 | 24/192 |
| Mei9-U03 | crossword | personal100_cycle_stats720 | 91.67 | 2.08 | 5.21 | 0.9950 | 24/192 |
| Mei9-U03 | crossword | selected_cycle | 100.00 | 3.12 | 1.56 | 0.9959 | 24/192 |
| Mei9-U03 | crossword | cycle_only100 | 83.33 | 10.94 | 13.80 | 0.9568 | 24/192 |
| Mei9-U04 | sameword | raw120 | 75.00 | 16.41 | 20.70 | 0.8789 | 16/128 |
| Mei9-U04 | sameword | statistics720 | 68.75 | 4.69 | 17.97 | 0.9497 | 16/128 |
| Mei9-U04 | sameword | student192 | 87.50 | 3.91 | 8.20 | 0.9639 | 16/128 |
| Mei9-U04 | sameword | baseline312 | 81.25 | 3.91 | 11.33 | 0.9663 | 16/128 |
| Mei9-U04 | sameword | personal25_input256 | 87.50 | 4.69 | 8.59 | 0.9683 | 16/128 |
| Mei9-U04 | sameword | personal25_cycle256 | 87.50 | 4.69 | 8.59 | 0.9766 | 16/128 |
| Mei9-U04 | sameword | personal25_cycle_stats720 | 87.50 | 5.47 | 8.98 | 0.9854 | 16/128 |
| Mei9-U04 | sameword | personal100_input256 | 81.25 | 4.69 | 11.72 | 0.9668 | 16/128 |
| Mei9-U04 | sameword | personal100_cycle256 | 87.50 | 4.69 | 8.59 | 0.9785 | 16/128 |
| Mei9-U04 | sameword | personal100_cycle_stats720 | 87.50 | 5.47 | 8.98 | 0.9810 | 16/128 |
| Mei9-U04 | sameword | selected_cycle | 81.25 | 3.91 | 11.33 | 0.9663 | 16/128 |
| Mei9-U04 | sameword | cycle_only100 | 100.00 | 18.75 | 9.38 | 0.9585 | 16/128 |
| Mei9-U04 | crossword | raw120 | 79.17 | 13.02 | 16.93 | 0.9021 | 24/192 |
| Mei9-U04 | crossword | statistics720 | 79.17 | 4.17 | 12.50 | 0.9555 | 24/192 |
| Mei9-U04 | crossword | student192 | 87.50 | 6.77 | 9.64 | 0.9737 | 24/192 |
| Mei9-U04 | crossword | baseline312 | 79.17 | 3.12 | 11.98 | 0.9855 | 24/192 |
| Mei9-U04 | crossword | personal25_input256 | 87.50 | 4.69 | 8.59 | 0.9842 | 24/192 |
| Mei9-U04 | crossword | personal25_cycle256 | 83.33 | 5.73 | 11.20 | 0.9805 | 24/192 |
| Mei9-U04 | crossword | personal25_cycle_stats720 | 91.67 | 7.29 | 7.81 | 0.9898 | 24/192 |
| Mei9-U04 | crossword | personal100_input256 | 87.50 | 3.12 | 7.81 | 0.9846 | 24/192 |
| Mei9-U04 | crossword | personal100_cycle256 | 83.33 | 6.77 | 11.72 | 0.9813 | 24/192 |
| Mei9-U04 | crossword | personal100_cycle_stats720 | 91.67 | 6.77 | 7.55 | 0.9896 | 24/192 |
| Mei9-U04 | crossword | selected_cycle | 79.17 | 3.12 | 11.98 | 0.9855 | 24/192 |
| Mei9-U04 | crossword | cycle_only100 | 87.50 | 21.35 | 16.93 | 0.9069 | 24/192 |
| Mei9-U05 | sameword | raw120 | 81.25 | 14.06 | 16.41 | 0.9512 | 16/128 |
| Mei9-U05 | sameword | statistics720 | 87.50 | 14.84 | 13.67 | 0.9717 | 16/128 |
| Mei9-U05 | sameword | student192 | 87.50 | 10.94 | 11.72 | 0.9458 | 16/128 |
| Mei9-U05 | sameword | baseline312 | 87.50 | 7.81 | 10.16 | 0.9648 | 16/128 |
| Mei9-U05 | sameword | personal25_input256 | 87.50 | 7.03 | 9.77 | 0.9644 | 16/128 |
| Mei9-U05 | sameword | personal25_cycle256 | 93.75 | 10.94 | 8.59 | 0.9644 | 16/128 |
| Mei9-U05 | sameword | personal25_cycle_stats720 | 93.75 | 14.06 | 10.16 | 0.9644 | 16/128 |
| Mei9-U05 | sameword | personal100_input256 | 87.50 | 7.03 | 9.77 | 0.9648 | 16/128 |
| Mei9-U05 | sameword | personal100_cycle256 | 93.75 | 9.38 | 7.81 | 0.9595 | 16/128 |
| Mei9-U05 | sameword | personal100_cycle_stats720 | 93.75 | 9.38 | 7.81 | 0.9668 | 16/128 |
| Mei9-U05 | sameword | selected_cycle | 87.50 | 7.81 | 10.16 | 0.9648 | 16/128 |
| Mei9-U05 | sameword | cycle_only100 | 68.75 | 29.69 | 30.47 | 0.7432 | 16/128 |
| Mei9-U05 | crossword | raw120 | 83.33 | 13.54 | 15.10 | 0.9303 | 24/192 |
| Mei9-U05 | crossword | statistics720 | 83.33 | 11.98 | 14.32 | 0.9418 | 24/192 |
| Mei9-U05 | crossword | student192 | 62.50 | 5.21 | 21.35 | 0.8869 | 24/192 |
| Mei9-U05 | crossword | baseline312 | 58.33 | 2.60 | 22.14 | 0.9447 | 24/192 |
| Mei9-U05 | crossword | personal25_input256 | 58.33 | 3.12 | 22.40 | 0.9440 | 24/192 |
| Mei9-U05 | crossword | personal25_cycle256 | 75.00 | 9.38 | 17.19 | 0.9373 | 24/192 |
| Mei9-U05 | crossword | personal25_cycle_stats720 | 79.17 | 9.90 | 15.36 | 0.9342 | 24/192 |
| Mei9-U05 | crossword | personal100_input256 | 58.33 | 2.60 | 22.14 | 0.9440 | 24/192 |
| Mei9-U05 | crossword | personal100_cycle256 | 70.83 | 5.73 | 17.45 | 0.9425 | 24/192 |
| Mei9-U05 | crossword | personal100_cycle_stats720 | 62.50 | 6.25 | 21.88 | 0.9412 | 24/192 |
| Mei9-U05 | crossword | selected_cycle | 58.33 | 2.60 | 22.14 | 0.9447 | 24/192 |
| Mei9-U05 | crossword | cycle_only100 | 66.67 | 33.85 | 33.59 | 0.7359 | 24/192 |
| Mei9-U06 | sameword | raw120 | 62.50 | 7.81 | 22.66 | 0.8350 | 16/128 |
| Mei9-U06 | sameword | statistics720 | 81.25 | 12.50 | 15.62 | 0.9453 | 16/128 |
| Mei9-U06 | sameword | student192 | 75.00 | 7.81 | 16.41 | 0.9492 | 16/128 |
| Mei9-U06 | sameword | baseline312 | 68.75 | 5.47 | 18.36 | 0.9521 | 16/128 |
| Mei9-U06 | sameword | personal25_input256 | 81.25 | 8.59 | 13.67 | 0.9507 | 16/128 |
| Mei9-U06 | sameword | personal25_cycle256 | 93.75 | 13.28 | 9.77 | 0.9561 | 16/128 |
| Mei9-U06 | sameword | personal25_cycle_stats720 | 87.50 | 12.50 | 12.50 | 0.9595 | 16/128 |
| Mei9-U06 | sameword | personal100_input256 | 81.25 | 11.72 | 15.23 | 0.9507 | 16/128 |
| Mei9-U06 | sameword | personal100_cycle256 | 87.50 | 14.06 | 13.28 | 0.9536 | 16/128 |
| Mei9-U06 | sameword | personal100_cycle_stats720 | 87.50 | 10.16 | 11.33 | 0.9561 | 16/128 |
| Mei9-U06 | sameword | selected_cycle | 68.75 | 5.47 | 18.36 | 0.9521 | 16/128 |
| Mei9-U06 | sameword | cycle_only100 | 62.50 | 20.31 | 28.91 | 0.8188 | 16/128 |
| Mei9-U06 | crossword | raw120 | 66.67 | 8.85 | 21.09 | 0.9112 | 24/192 |
| Mei9-U06 | crossword | statistics720 | 87.50 | 11.98 | 12.24 | 0.9579 | 24/192 |
| Mei9-U06 | crossword | student192 | 87.50 | 8.33 | 10.42 | 0.9551 | 24/192 |
| Mei9-U06 | crossword | baseline312 | 91.67 | 8.85 | 8.59 | 0.9666 | 24/192 |
| Mei9-U06 | crossword | personal25_input256 | 100.00 | 11.98 | 5.99 | 0.9698 | 24/192 |
| Mei9-U06 | crossword | personal25_cycle256 | 100.00 | 14.06 | 7.03 | 0.9720 | 24/192 |
| Mei9-U06 | crossword | personal25_cycle_stats720 | 100.00 | 13.02 | 6.51 | 0.9722 | 24/192 |
| Mei9-U06 | crossword | personal100_input256 | 100.00 | 13.02 | 6.51 | 0.9703 | 24/192 |
| Mei9-U06 | crossword | personal100_cycle256 | 100.00 | 13.02 | 6.51 | 0.9705 | 24/192 |
| Mei9-U06 | crossword | personal100_cycle_stats720 | 100.00 | 12.50 | 6.25 | 0.9701 | 24/192 |
| Mei9-U06 | crossword | selected_cycle | 91.67 | 8.85 | 8.59 | 0.9666 | 24/192 |
| Mei9-U06 | crossword | cycle_only100 | 79.17 | 23.44 | 22.14 | 0.8663 | 24/192 |
| Mei9-U07 | sameword | raw120 | 75.00 | 10.94 | 17.97 | 0.9155 | 16/128 |
| Mei9-U07 | sameword | statistics720 | 68.75 | 5.47 | 18.36 | 0.9512 | 16/128 |
| Mei9-U07 | sameword | student192 | 81.25 | 9.38 | 14.06 | 0.9360 | 16/128 |
| Mei9-U07 | sameword | baseline312 | 75.00 | 5.47 | 15.23 | 0.9458 | 16/128 |
| Mei9-U07 | sameword | personal25_input256 | 81.25 | 7.03 | 12.89 | 0.9561 | 16/128 |
| Mei9-U07 | sameword | personal25_cycle256 | 75.00 | 5.47 | 15.23 | 0.9614 | 16/128 |
| Mei9-U07 | sameword | personal25_cycle_stats720 | 68.75 | 4.69 | 17.97 | 0.9546 | 16/128 |
| Mei9-U07 | sameword | personal100_input256 | 81.25 | 7.03 | 12.89 | 0.9561 | 16/128 |
| Mei9-U07 | sameword | personal100_cycle256 | 75.00 | 5.47 | 15.23 | 0.9575 | 16/128 |
| Mei9-U07 | sameword | personal100_cycle_stats720 | 68.75 | 3.91 | 17.58 | 0.9541 | 16/128 |
| Mei9-U07 | sameword | selected_cycle | 75.00 | 5.47 | 15.23 | 0.9458 | 16/128 |
| Mei9-U07 | sameword | cycle_only100 | 68.75 | 20.31 | 25.78 | 0.8887 | 16/128 |
| Mei9-U07 | crossword | raw120 | 95.83 | 21.35 | 12.76 | 0.9592 | 24/192 |
| Mei9-U07 | crossword | statistics720 | 91.67 | 8.33 | 8.33 | 0.9620 | 24/192 |
| Mei9-U07 | crossword | student192 | 91.67 | 7.81 | 8.07 | 0.9694 | 24/192 |
| Mei9-U07 | crossword | baseline312 | 91.67 | 7.29 | 7.81 | 0.9779 | 24/192 |
| Mei9-U07 | crossword | personal25_input256 | 91.67 | 9.90 | 9.11 | 0.9789 | 24/192 |
| Mei9-U07 | crossword | personal25_cycle256 | 91.67 | 8.33 | 8.33 | 0.9768 | 24/192 |
| Mei9-U07 | crossword | personal25_cycle_stats720 | 87.50 | 6.25 | 9.38 | 0.9763 | 24/192 |
| Mei9-U07 | crossword | personal100_input256 | 95.83 | 9.90 | 7.03 | 0.9792 | 24/192 |
| Mei9-U07 | crossword | personal100_cycle256 | 91.67 | 7.29 | 7.81 | 0.9763 | 24/192 |
| Mei9-U07 | crossword | personal100_cycle_stats720 | 87.50 | 5.73 | 9.11 | 0.9766 | 24/192 |
| Mei9-U07 | crossword | selected_cycle | 91.67 | 7.29 | 7.81 | 0.9779 | 24/192 |
| Mei9-U07 | crossword | cycle_only100 | 91.67 | 28.12 | 18.23 | 0.8813 | 24/192 |
| Mei9-U08 | sameword | raw120 | 93.75 | 12.50 | 9.38 | 0.9688 | 16/128 |
| Mei9-U08 | sameword | statistics720 | 100.00 | 2.34 | 1.17 | 0.9976 | 16/128 |
| Mei9-U08 | sameword | student192 | 100.00 | 19.53 | 9.77 | 0.9907 | 16/128 |
| Mei9-U08 | sameword | baseline312 | 100.00 | 5.47 | 2.73 | 0.9927 | 16/128 |
| Mei9-U08 | sameword | personal25_input256 | 100.00 | 4.69 | 2.34 | 0.9946 | 16/128 |
| Mei9-U08 | sameword | personal25_cycle256 | 100.00 | 5.47 | 2.73 | 0.9971 | 16/128 |
| Mei9-U08 | sameword | personal25_cycle_stats720 | 100.00 | 5.47 | 2.73 | 0.9980 | 16/128 |
| Mei9-U08 | sameword | personal100_input256 | 100.00 | 4.69 | 2.34 | 0.9946 | 16/128 |
| Mei9-U08 | sameword | personal100_cycle256 | 100.00 | 5.47 | 2.73 | 0.9966 | 16/128 |
| Mei9-U08 | sameword | personal100_cycle_stats720 | 100.00 | 5.47 | 2.73 | 0.9980 | 16/128 |
| Mei9-U08 | sameword | selected_cycle | 100.00 | 5.47 | 2.73 | 0.9927 | 16/128 |
| Mei9-U08 | sameword | cycle_only100 | 87.50 | 32.81 | 22.66 | 0.8696 | 16/128 |
| Mei9-U08 | crossword | raw120 | 100.00 | 13.02 | 6.51 | 0.9731 | 24/192 |
| Mei9-U08 | crossword | statistics720 | 95.83 | 10.42 | 7.29 | 0.9716 | 24/192 |
| Mei9-U08 | crossword | student192 | 100.00 | 21.35 | 10.68 | 0.9859 | 24/192 |
| Mei9-U08 | crossword | baseline312 | 100.00 | 9.90 | 4.95 | 0.9898 | 24/192 |
| Mei9-U08 | crossword | personal25_input256 | 100.00 | 9.90 | 4.95 | 0.9909 | 24/192 |
| Mei9-U08 | crossword | personal25_cycle256 | 100.00 | 10.42 | 5.21 | 0.9915 | 24/192 |
| Mei9-U08 | crossword | personal25_cycle_stats720 | 100.00 | 14.06 | 7.03 | 0.9924 | 24/192 |
| Mei9-U08 | crossword | personal100_input256 | 100.00 | 9.90 | 4.95 | 0.9909 | 24/192 |
| Mei9-U08 | crossword | personal100_cycle256 | 100.00 | 10.94 | 5.47 | 0.9928 | 24/192 |
| Mei9-U08 | crossword | personal100_cycle_stats720 | 100.00 | 14.06 | 7.03 | 0.9946 | 24/192 |
| Mei9-U08 | crossword | selected_cycle | 100.00 | 9.90 | 4.95 | 0.9898 | 24/192 |
| Mei9-U08 | crossword | cycle_only100 | 83.33 | 29.69 | 23.18 | 0.8346 | 24/192 |
| Mei9-U09 | sameword | raw120 | 81.25 | 16.41 | 17.58 | 0.8711 | 16/128 |
| Mei9-U09 | sameword | statistics720 | 75.00 | 11.72 | 18.36 | 0.9209 | 16/128 |
| Mei9-U09 | sameword | student192 | 87.50 | 7.81 | 10.16 | 0.9761 | 16/128 |
| Mei9-U09 | sameword | baseline312 | 93.75 | 3.91 | 5.08 | 0.9863 | 16/128 |
| Mei9-U09 | sameword | personal25_input256 | 93.75 | 4.69 | 5.47 | 0.9873 | 16/128 |
| Mei9-U09 | sameword | personal25_cycle256 | 93.75 | 5.47 | 5.86 | 0.9868 | 16/128 |
| Mei9-U09 | sameword | personal25_cycle_stats720 | 87.50 | 5.47 | 8.98 | 0.9844 | 16/128 |
| Mei9-U09 | sameword | personal100_input256 | 93.75 | 4.69 | 5.47 | 0.9883 | 16/128 |
| Mei9-U09 | sameword | personal100_cycle256 | 100.00 | 6.25 | 3.12 | 0.9888 | 16/128 |
| Mei9-U09 | sameword | personal100_cycle_stats720 | 93.75 | 3.91 | 5.08 | 0.9868 | 16/128 |
| Mei9-U09 | sameword | selected_cycle | 93.75 | 3.91 | 5.08 | 0.9863 | 16/128 |
| Mei9-U09 | sameword | cycle_only100 | 62.50 | 28.12 | 32.81 | 0.7695 | 16/128 |
| Mei9-U09 | crossword | raw120 | 29.17 | 17.71 | 44.27 | 0.6243 | 24/192 |
| Mei9-U09 | crossword | statistics720 | 20.83 | 11.46 | 45.31 | 0.6788 | 24/192 |
| Mei9-U09 | crossword | student192 | 50.00 | 6.25 | 28.12 | 0.8815 | 24/192 |
| Mei9-U09 | crossword | baseline312 | 33.33 | 4.69 | 35.68 | 0.8709 | 24/192 |
| Mei9-U09 | crossword | personal25_input256 | 33.33 | 4.69 | 35.68 | 0.8746 | 24/192 |
| Mei9-U09 | crossword | personal25_cycle256 | 33.33 | 5.21 | 35.94 | 0.8824 | 24/192 |
| Mei9-U09 | crossword | personal25_cycle_stats720 | 29.17 | 5.73 | 38.28 | 0.8806 | 24/192 |
| Mei9-U09 | crossword | personal100_input256 | 33.33 | 4.69 | 35.68 | 0.8746 | 24/192 |
| Mei9-U09 | crossword | personal100_cycle256 | 37.50 | 6.25 | 34.38 | 0.8830 | 24/192 |
| Mei9-U09 | crossword | personal100_cycle_stats720 | 33.33 | 5.21 | 35.94 | 0.8902 | 24/192 |
| Mei9-U09 | crossword | selected_cycle | 33.33 | 4.69 | 35.68 | 0.8709 | 24/192 |
| Mei9-U09 | crossword | cycle_only100 | 54.17 | 15.10 | 30.47 | 0.6923 | 24/192 |

## Cal-FAR5宏平均

| 方法 | 条件 | TAR% | FAR% | HTER% |
|---|---|---:|---:|---:|
| raw120 | sameword | 56.25 | 2.86 | 23.31 |
| raw120 | crossword | 58.80 | 3.94 | 22.57 |
| statistics720 | sameword | 71.53 | 3.12 | 15.80 |
| statistics720 | crossword | 68.06 | 3.47 | 17.71 |
| student192 | sameword | 74.31 | 3.82 | 14.76 |
| student192 | crossword | 75.93 | 3.65 | 13.86 |
| baseline312 | sameword | 73.61 | 2.52 | 14.45 |
| baseline312 | crossword | 70.37 | 3.30 | 16.46 |
| personal25_input256 | sameword | 72.92 | 2.26 | 14.67 |
| personal25_input256 | crossword | 72.69 | 3.53 | 15.42 |
| personal25_cycle256 | sameword | 71.53 | 2.34 | 15.41 |
| personal25_cycle256 | crossword | 71.76 | 3.01 | 15.62 |
| personal25_cycle_stats720 | sameword | 76.39 | 2.69 | 13.15 |
| personal25_cycle_stats720 | crossword | 72.22 | 3.65 | 15.71 |
| personal100_input256 | sameword | 72.92 | 2.26 | 14.67 |
| personal100_input256 | crossword | 73.61 | 3.53 | 14.96 |
| personal100_cycle256 | sameword | 73.61 | 2.52 | 14.45 |
| personal100_cycle256 | crossword | 72.22 | 2.95 | 15.36 |
| personal100_cycle_stats720 | sameword | 75.69 | 2.60 | 13.45 |
| personal100_cycle_stats720 | crossword | 73.15 | 3.65 | 15.25 |
| selected_cycle | sameword | 73.61 | 2.52 | 14.45 |
| selected_cycle | crossword | 70.37 | 3.30 | 16.46 |
| cycle_only100 | sameword | 32.64 | 3.04 | 35.20 |
| cycle_only100 | crossword | 39.35 | 3.88 | 32.26 |

## 解读边界

本轮是已见身份闭集，并不证明未见用户注册泛化。与此前开放集比较还同时改变了身份训练、后端和查询数量，不能把差异全部归因闭集。只有本轮方法之间是固定数据/固定后端的配对比较。原始统计与baseline312还同时改变了身份监督训练和维度，缺少同预算纯ACC编码器对照，因此不能把其差异全部归因Audio蒸馏。

名义跨词仍取决于推断词序，未独立完成逐录制语义核验。数据来自同一头显、缺少重戴/跨天/跨设备条件标记；不能排除佩戴、设备或录制条件身份线索。单种子、多方法探索，bootstrap按用户身份，未经多重比较校正。最终Cal-FAR5不保证Test FAR≤5%。没有加入随机未训练GAN对照，因此即使个别分支改善，也不能单凭本轮证明跨模态生成是唯一增益来源。

[全部九人双工作点CSV](../../outputs/mei9_closed_personal_cycle_20261110/per_user_public.csv)，[独立检查记录](../../outputs/mei9_closed_personal_cycle_20261110/verification_public.json)。原始信号、源姓名、个人权重、阈值及样本分数保持私有。

复核通过：18个个人G/F检查点、432个逐用户工作点；划分互斥、全九人训练/八个已知负类、Cal选择重放均通过。原缓存特征重训SVM一致；GPU重算最大特征/分数差为5.39e-05，所有查询接受/拒绝决定不变。查询Audio模型前向次数为0。
