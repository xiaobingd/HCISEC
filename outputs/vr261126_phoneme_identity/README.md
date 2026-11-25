# 共享音素跨词ACC身份诊断

方法、全部窗口对照和逐用户AUC见 [report.md](report.md)。这是训练词内使用Audio定位的诊断实验，不是最终ACC-only部署验证。

使用本地PocketSphinx5.0.4及其自带英语模型，私人音频不传出。公开安装包的本次Windows wheel SHA256见protocol_public.json。

```powershell
python -m pip install --target .runtime/phoneme --no-deps pocketsphinx==5.0.4
python experiments/continuous_acc_activity_20261124/test_phoneme_identity.py
python experiments/continuous_acc_activity_20261124/audit_phoneme_identity.py
python experiments/continuous_acc_activity_20261124/report_verify_phoneme.py
```

需要此前授权私有原始ACC缓存、Audio原始录音、时间映射清单和Python科学计算依赖。公开代码不足以重建私人数据。

不上传音频、局部特征、自动边界、身份映射或模型。所有用户保留，3个元音组/6个窗口方法完整报告；不得从偏移敏感性结果挑最佳值后声称独立验证。
