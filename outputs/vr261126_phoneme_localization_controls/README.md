# 局部身份信息：Audio边界与固定位置/ACC能量对照

完整方法、逐用户结果和隔离核验见 [report.md](report.md)。这是历史训练词内的探索诊断，不是新的盲测或最终ACC-only部署验证。

依赖此前私有原始ACC缓存和训练Audio自动音素边界；本轮不重新读取任何Audio文件。需要此前Python科学计算环境及本地PocketSphinx依赖（复用前端模块）。

```powershell
python experiments/continuous_acc_activity_20261124/explore_phoneme_localization.py
python experiments/continuous_acc_activity_20261124/verify_report_localization.py
```

所有组使用固定120ms和24维ACC统计。固定位置与Audio位置各5个候选，排除外层查询词和目标身份后选择；错遍Audio保留本人/同词标签。完整公开7种对照，不能只挑最优偏移。

公开匿名逐用户/逐词汇总、支持集选择分数和代码；不含信号、局部特征、边界、私人身份映射或逐样本认证分数。
