# Mei9：闭集跨内容逐用户报告

全部用户均参与身份学习，个人CycleGAN仅用本人12条注册配对微调。全部其他身份作为训练和测试负类。

用户数9；每用户已知攻击来源8；seed42。

校准锁定的CycleGAN配置：`{"step": 0, "branch": "none", "alpha": 0.0, "hter": 0.0954861111111111, "auc": 0.9921875}`。不根据最终查询结果改选配置。

|方法|工作点|TAR%|FAR%|HTER%|AUC|
|---|---|---:|---:|---:|---:|
|raw_shape|Cal_EER|79.63|11.57|15.97|0.9177|
|raw_shape|Cal_FAR5|58.80|3.94|22.57|0.9177|
|raw_statistics|Cal_EER|81.02|7.47|13.22|0.9371|
|raw_statistics|Cal_FAR5|68.06|3.47|17.71|0.9371|
|audio_encoder|Cal_EER|83.80|6.89|11.55|0.9607|
|audio_encoder|Cal_FAR5|75.93|3.65|13.86|0.9607|
|audio_baseline|Cal_EER|78.70|4.40|12.85|0.9701|
|audio_baseline|Cal_FAR5|70.37|3.30|16.46|0.9701|
|selected_cycle|Cal_EER|78.70|4.40|12.85|0.9701|
|selected_cycle|Cal_FAR5|70.37|3.30|16.46|0.9701|
|cycle_only100|Cal_EER|80.56|20.95|20.20|0.8646|
|cycle_only100|Cal_FAR5|39.35|3.88|32.26|0.8646|
|personal25_input256|Cal_EER|81.94|5.21|11.63|0.9709|
|personal25_input256|Cal_FAR5|72.69|3.53|15.42|0.9709|
|personal25_cycle256|Cal_EER|82.41|6.19|11.89|0.9709|
|personal25_cycle256|Cal_FAR5|71.76|3.01|15.62|0.9709|
|personal25_cycle_statistics|Cal_EER|81.94|6.42|12.24|0.9710|
|personal25_cycle_statistics|Cal_FAR5|72.22|3.65|15.71|0.9710|
|personal100_input256|Cal_EER|82.41|5.09|11.34|0.9710|
|personal100_input256|Cal_FAR5|73.61|3.53|14.96|0.9710|
|personal100_cycle256|Cal_EER|82.87|5.84|11.49|0.9715|
|personal100_cycle256|Cal_FAR5|72.22|2.95|15.36|0.9715|
|personal100_cycle_statistics|Cal_EER|80.56|5.84|12.64|0.9729|
|personal100_cycle_statistics|Cal_FAR5|73.15|3.65|15.25|0.9729|
|no_audio_encoder|Cal_EER|80.56|5.79|12.62|0.9631|
|no_audio_encoder|Cal_FAR5|72.69|2.84|15.08|0.9631|
|no_audio_baseline|Cal_EER|79.17|4.80|12.82|0.9701|
|no_audio_baseline|Cal_FAR5|70.37|3.70|16.67|0.9701|

## 全部用户：Cal-EER工作点

每格为TAR / FAR / HTER（%）；完整14方法及两个工作点见同目录CSV。

|用户|原始统计|纯ACC基线|Audio辅助|校准选择CycleGAN|
|---|---|---|---|---|
|Mei9-C3-U01|75.00 / 1.56 / 13.28|70.83 / 0.00 / 14.58|70.83 / 0.00 / 14.58|70.83 / 0.00 / 14.58|
|Mei9-C3-U02|95.83 / 2.60 / 3.39|83.33 / 0.00 / 8.33|83.33 / 0.00 / 8.33|83.33 / 0.00 / 8.33|
|Mei9-C3-U03|100.00 / 4.69 / 2.34|95.83 / 2.60 / 3.39|100.00 / 3.12 / 1.56|100.00 / 3.12 / 1.56|
|Mei9-C3-U04|79.17 / 4.17 / 12.50|83.33 / 3.12 / 9.90|79.17 / 3.12 / 11.98|79.17 / 3.12 / 11.98|
|Mei9-C3-U05|83.33 / 11.98 / 14.32|62.50 / 2.60 / 20.05|58.33 / 2.60 / 22.14|58.33 / 2.60 / 22.14|
|Mei9-C3-U06|87.50 / 11.98 / 12.24|100.00 / 10.42 / 5.21|91.67 / 8.85 / 8.59|91.67 / 8.85 / 8.59|
|Mei9-C3-U07|91.67 / 8.33 / 8.33|91.67 / 7.29 / 7.81|91.67 / 7.29 / 7.81|91.67 / 7.29 / 7.81|
|Mei9-C3-U08|95.83 / 10.42 / 7.29|100.00 / 14.06 / 7.03|100.00 / 9.90 / 4.95|100.00 / 9.90 / 4.95|
|Mei9-C3-U09|20.83 / 11.46 / 45.31|25.00 / 3.12 / 39.06|33.33 / 4.69 / 35.68|33.33 / 4.69 / 35.68|

## 循环特征相对Audio基线的逐身份配对变化

负数表示HTER下降；区间是逐身份bootstrap95%，不能替代训练随机种子。

|方法|HTER变化pp|95%区间pp|改善用户数|
|---|---:|---|---:|
|raw_shape|3.12|-1.22 / 7.29|3|
|raw_statistics|0.38|-2.58 / 3.53|3|
|audio_encoder|-1.30|-3.94 / 1.22|5|
|selected_cycle|0.00|0.00 / 0.00|0|
|cycle_only100|7.35|2.34 / 11.98|2|
|personal25_input256|-1.22|-2.40 / -0.09|5|
|personal25_cycle256|-0.95|-2.84 / 0.87|5|
|personal25_cycle_statistics|-0.61|-2.95 / 1.62|4|
|personal100_input256|-1.50|-3.01 / -0.26|5|
|personal100_cycle256|-1.36|-3.07 / 0.14|6|
|personal100_cycle_statistics|-0.20|-1.97 / 1.48|4|
|no_audio_encoder|-0.23|-2.52 / 2.37|6|
|no_audio_baseline|-0.03|-1.39 / 1.36|3|

## 审核与边界

独立重放审核：`{"state": "PASS", "dataset": "Mei9", "users": 9, "known_attackers_each": 8, "new_matched_identity_only_control_verified": true, "archived_cycle_verification_reused": true, "archived_personal_checkpoints": 18, "workpoint_checks": 252, "max_cached_refit_error": 0, "query_audio_model_forwards": 0}`。

单种子探索结果；Mei为tracking速度而非原始ACC；公开数据为跨句而非词汇互斥。ABCS物理通道归属待确认。查询Audio没有模型前向；注册Audio允许用于教师/个人模型适配。Cal-FAR5是校准目标，不保证查询FAR≤5%。没有跨天/重新佩戴结论，也没有本轮随机GAN对照。

[完整协议](../../docs/experiments/2026-11-10-three-dataset-closed-cross-protocol.md)
