# 冻结 ACC＋小型 Audio 辅助残差：严格跨词认证

2026-11-24。10人，3种子；沿用16训练词、全部20种注册/校准词划分，全部攻击者。

ACC完全冻结（9个残差模型逐参数比对一致）；仅训练192→32→96→192残差，最后一层零初始化。输出normalize(h+0.1 delta)。400步，正确Audio/无Audio/身份置换Audio同预算。Audio只监督96维辅助分支，梯度权重按残差身份梯度0.1匹配；分类头只训练使用。

表格为三词联合百分比，EER按评估ROC，HTER按校准Q10/Q95中点阈值。每组1词8条注册、1不同词8条校准、3不同词查询；三词校准组合仅8条源录制，组合不独立。

|内容|方法|FAR|FRR|HTER|EER|
|---|---|---:|---:|---:|---:|
|english_word|ACC|3.838|16.125|9.981|3.435|
|english_long|ACC|9.896|27.187|18.542|4.041|
|chinese_long|ACC|4.866|17.750|11.308|2.536|
|english_word|SplitAudio_fused|3.801|16.083|9.942|3.417|
|english_long|SplitAudio_fused|9.750|27.062|18.406|3.951|
|chinese_long|SplitAudio_fused|4.840|17.688|11.264|2.473|
|english_word|SplitNoAudio_fused|3.801|16.083|9.942|3.418|
|english_long|SplitNoAudio_fused|9.757|27.062|18.410|3.949|
|chinese_long|SplitNoAudio_fused|4.843|17.688|11.265|2.475|
|english_word|SplitWrongAudio_fused|3.803|16.083|9.943|3.417|
|english_long|SplitWrongAudio_fused|9.757|27.062|18.410|3.950|
|chinese_long|SplitWrongAudio_fused|4.843|17.708|11.275|2.473|

结论：保护原ACC后不再出现上一轮明显退化，但收益很小。正确Audio、无Audio和错误Audio几乎相当，不能将改善归于正确Audio身份知识。小残差改善部分指标，但当前Audio关系监督没有提供清晰独立贡献。

shared读出为辅助96维；private读出仅冻结原ACC前96维诊断，不称独有空间，不用于主结论。原ACC baseline不再额外训练。

36000工作点计数及独立重放通过；逐用户和逐种子结果见JSON。已见用户闭集，不是开放集；离线前端沿用Audio元数据发声边界，不是已验证的纯ACC在线分段。没有读取查询Audio教师特征，没有删除用户或按查询选择权重。

仅公开匿名汇总和代码，不上传原始数据、embedding、阈值或模型。