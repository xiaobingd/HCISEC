# 分层跨内容 ACC 身份表征学习

执行 [2026-11-27 锁定方案](../../docs/experiments/2026-11-27-audio-phonetic-cross-content-acc-plan.md)，源提交 `65f31238c0e3eeed7e19bac79fd6f1bbe3d58ce9`。方案快照见 `PLAN_SOURCE.md`。

五组：纯 ACC、词 GRL、词典伪音素、真实 Audio CTC 软后验、同后验时间错配。12/4 开发后锁定设置，全部16训练词重训，5组×3种子×1000步；10人闭集、ACC-only 单窗口跨词查询。不是 ECAPA 身份 embedding 对齐，不改已冻结 CycleGAN 创新点一。

- [完整中文报告](../../outputs/vr261127_audio_phonetic_cross_content/report.md)
- [复现与隐私说明](../../outputs/vr261127_audio_phonetic_cross_content/REPRODUCE.md)
- [匿名结果汇总](../../outputs/vr261127_audio_phonetic_cross_content/summary_public.json)
- [独立核验](../../outputs/vr261127_audio_phonetic_cross_content/verification_public.json)

仅离线训练/开发读取 Audio，查询不读取 Audio 或其时间戳。所有私有信号、后验、样本边界、模型、注册模板、阈值及认证分数均不公开。
