# 原始发声周期性跨模态身份审计

结果与逐用户误差见 [report.md](report.md)。只使用历史16训练词、10用户的1280窗及160条训练词Audio录制。无最终测试Audio、无神经模型训练、无EER改善声明。

需要授权私有数据、Python、NumPy、SciPy、scikit-learn、threadpoolctl。依次运行：

```powershell
python experiments/continuous_acc_activity_20261124/test_raw_periodicity.py
python experiments/continuous_acc_activity_20261124/extract_raw_periodicity.py
python experiments/continuous_acc_activity_20261124/audit_periodicity_mechanism.py
python experiments/continuous_acc_activity_20261124/audit_periodicity_identity.py
python experiments/continuous_acc_activity_20261124/report_verify_periodicity.py
```

提取缓存写入私有工作目录，不属于公开交付。公开仅包含代码、参数、匿名汇总及验证记录；不含波形、周期轨迹、embedding、身份映射或模型权重。

结果JSON沿用计算引擎的候选键：shape表示周期性6维、envelope表示谐波4维、combined表示两个关系特征组合；不是上一轮的宽频带shape/envelope。

结果为历史已看过四词的探索复核。周期检测低覆盖、物理同步与实际采样率尚未验证；条件预测的小幅改善不能解释为已经获得身份认证收益。
