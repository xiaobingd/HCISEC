# 小范围音素辅助权重优化

原研究：[固定五组结果](../../outputs/vr261127_audio_phonetic_cross_content/report.md)，提交 `e6ec6c1f0caefd3caf7a9a1435feeef3c19ac0ea`。

只搜索两套新损失权重（0.01/0、0.03/0.01）。独立开发词上四组各400步，以跨词认证EER/HTER和FAR/FRR门槛选择一套；最终1000步、三个种子，对比既有纯ACC、原参数、匹配无Audio、选择后的正确Audio和同参数时间错配。原实验和私有模型只读复用，不覆盖。

- [完整报告](../../outputs/vr261127_audio_phonetic_tuning/report.md)
- [复现](../../outputs/vr261127_audio_phonetic_tuning/REPRODUCE.md)
- [匿名汇总](../../outputs/vr261127_audio_phonetic_tuning/summary_public.json)

本轮为同一历史查询池的探索性后续；查询不用于参数排序，不声称全新盲测。
