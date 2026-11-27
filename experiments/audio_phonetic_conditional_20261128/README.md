# 去blank有效音素条件蒸馏

按用户给定公式：教师/学生均排除blank后归一化；权重为非blank概率质量×条件最大后验，低于0.001质量帧不参与辅助项，全blank损失为0。全部身份窗口保留，不用Audio定位查询。

固定λphone=.01、λword=.01；同权重完整KL、正确条件KL、条件KL联合时间错配三监督×三种子各1000步，另只读复用纯ACC和匹配无Audio对照。开发四组各400步用于诊断，不选择λ。网络、数据、采样、注册及校准协议沿用前两轮。

- [完整报告](../../outputs/vr261128_audio_phonetic_conditional/report.md)
- [复现](../../outputs/vr261128_audio_phonetic_conditional/REPRODUCE.md)
- [匿名汇总](../../outputs/vr261128_audio_phonetic_conditional/summary_public.json)

原权重探索归档：`53be0e2539dcecfb9637656ae7b0f645946ffc03`。本轮所有信号/后验/模型/逐样本分数私有，公开逐用户及逐词聚合指标。历史查询池复核，不称为新盲测或开放集。
