# 音素条件化身份学习：闭集跨词神经网络对照

完整方法、全部逐用户/逐词/逐种子结果及追加冻结验证见 [report.md](report.md)。

```powershell
python experiments/continuous_acc_activity_20261124/test_phoneme_conditioned_identity.py
python experiments/continuous_acc_activity_20261124/train_phoneme_conditioned_identity.py
python experiments/continuous_acc_activity_20261124/verify_report_phoneme_learning.py
python experiments/continuous_acc_activity_20261124/evaluate_frozen_new_eh_words.py
python experiments/continuous_acc_activity_20261124/verify_report_phoneme_learning.py
```

需要已授权的私有ACC、历史训练词自动音素边界、Python科学计算/PyTorch/PocketSphinx环境；公开目录不含这些私人输入。

四组相同5057参数、400步、3种子；全部10身份参与表示训练，训练/注册/校准录制不交叉，3词留出查询。音素类别来自词典，Audio只为训练样本提供局部软区域；注册/校准/查询没有Audio教师，推理只输入ACC。原始缓存本身已使用历史分段，因此不是连续端到端无Audio系统。

初始查询之后冻结模型与阈值，按词典规则追加所有尚未用于本轮的EH英语单词（Reset、Select、seven），不读这些词Audio，也不重训或重新校准。这是事后探索复核，历史数据不能称为新盲测。

私人目录只保留本地参数、输入、教师软标签、采样索引与阈值；GitHub仅发布代码和匿名汇总，不发布波形、片段特征、模型权重、身份映射或逐样本分数。
