# 联合身份与 Audio 活动监督实验

完整中文方法、逐用户结果与限制见 [report.md](report.md)。
公开内容仅为匿名汇总和代码；不含原始信号、embedding、权重、身份映射、个人阈值或逐样本分数。

## 运行

需要 Python、PyTorch、NumPy、SciPy、scikit-learn、threadpoolctl，以及授权私有缓存。
数据准备与缓存字段沿用此前连续 ACC 实验，公开目录不足以重新生成私人数据。

```powershell
python experiments/continuous_acc_activity_20261124/test_joint_identity.py
python experiments/continuous_acc_activity_20261124/run_joint_identity.py --root outputs/vr261119_10x40_raw_acc
python experiments/continuous_acc_activity_20261124/verify_joint_identity.py --root outputs/vr261119_10x40_raw_acc
python experiments/continuous_acc_activity_20261124/report_joint_identity.py --root outputs/vr261119_10x40_raw_acc
```

训练固定400步、三个种子；联合组共享开发选择的辅助权重。测试阶段只用 ACC。
本轮测试词曾被历史实验查看，属于探索复核。未改变冻结的创新点一。
