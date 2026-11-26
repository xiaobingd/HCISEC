# 局部身份损失回传查询注意力

完整结果、逐用户/逐词/逐种子对照和梯度诊断见 [report.md](report.md)。

```powershell
python experiments/continuous_acc_activity_20261124/test_connected_attention_identity.py
python experiments/continuous_acc_activity_20261124/train_connected_attention_identity.py
python experiments/continuous_acc_activity_20261124/verify_report_connected_attention.py
```

复用上轮授权私有输入、角色、Audio训练标签与采样；不修改其文件。依赖既有Python科学计算/PyTorch/PocketSphinx环境。

六组同5057参数、400步、三个种子。原始三词为主要比较，上一轮已查看的追加三词固定保留。所有查询只输入ACC；Audio教师仅训练角色可用。全部180阈值由校准集锁定后才评分查询。

闭集跨词，历史内容探索复核；原始输入已依赖历史分段，不能称为端到端连续无Audio认证。公开只含代码和匿名汇总；原始数据、片段特征、边界、权重、身份映射、用户阈值及逐样本分数不公开。
