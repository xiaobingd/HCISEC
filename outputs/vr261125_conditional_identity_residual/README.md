# 训练缓存内的条件化身份残差检验

完整方法、实际结果与局限见 [report.md](report.md)。这是可预测性审计，不是新的认证精度实验。

运行需要已授权的本地私有缓存，以及 NumPy、SciPy、scikit-learn、threadpoolctl。

```powershell
python experiments/continuous_acc_activity_20261124/test_conditional_residual.py
python experiments/continuous_acc_activity_20261124/audit_conditional_residual.py
```

只读取历史16个训练词的 ECAPA 与局部频谱缓存。前12词拟合、后4词验证；逐身份对留出双方，不读取最终测试 Audio，不修改冻结的创新点一或已有模型。固定 Ridge 正则，不依验证结果调整权重。词均值只来自基础八用户；验证词背景中心化属于离线诊断，不能称为无背景部署。

199次全用户身份置换重跑所有预测器；逐用户结果为关联身份对的汇总，并非180个独立人。未公开私人信号、向量、身份映射或逐样本分数。
