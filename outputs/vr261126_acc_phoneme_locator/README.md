# 从Audio音素位置到ACC局部定位器

完整方法、同预算控制和逐用户结果见 [report.md](report.md)。本轮在前述EH组局部关系信号后追加，属于已查看历史词上的探索，不是独立盲测。

```powershell
python experiments/continuous_acc_activity_20261124/train_acc_phoneme_locator.py
python experiments/continuous_acc_activity_20261124/verify_report_acc_locator.py
```

需要上轮授权私有缓存、音素边界和定位控制实验的支持词选择记录。使用此前相同Python依赖。

定位器只用训练身份/训练词的ACC候选窗及Audio软标签拟合；目标身份与查询词排除在训练之外。注册和查询的局部位置由ACC决定。Audio仅在离线诊断中评价查询定位误差，不参与查询位置选择或身份评分。

原始3秒缓存已经依赖历史词分段，因此这里只证明局部读出不需要查询Audio，不能称为端到端连续信号认证。没有训练神经身份编码器，没有EER/HTER锁阈值实验。

同结构ACC能量自监督定位器是在首次读出后追加的补充控制，未调alpha、sigma或筛选有利用户。公开代码和匿名逐用户汇总，不包含原始信号、局部特征、模型、边界、身份映射或逐样本分数。
