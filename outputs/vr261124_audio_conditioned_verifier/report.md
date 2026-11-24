# 注册 Audio 条件化 ACC 验证器：实际实验报告

执行来源：`experiment/audio-conditioned-verifier-20261124`，固定提交 `b2953752d7156c9b54d62a7adb16df5d2b5add46`。

10 人全部保留，3 个种子（42/43/44），10 个留一目标验证器折，120 个小模型，每模型 400 步。所有查询均为一条 ACC 记录；没有三词联合。注册包含 4 个词的 12 条相同 ACC，Audio 取相同记录。注册、校准、查询词互不重叠，属于已见身份的跨内容验证。

冻结 ACC 身份编码器与官方 ECAPA。576→32→192 小网络根据 ACC 原型、Audio 原型、ACC 注册方差输出 [0.75,1.25] 通道权重；使用加权余弦验证。A 为零 Audio，B 为正确 Audio，C 为异人 Audio，D 为训练统计匹配的随机 Audio。优化 BCE 与门控正则，所有组共享 ACC 难负类和 episode。

G 只优化前 12 个 fit 词，另外 4 个 fit 词记录内部验证损失，不用于挑选 checkpoint；固定第 400 步。每折目标完全不进入 G 训练或 Audio 统计。冻结 ACC 编码器曾在全 10 人 fit 词上学习，因此这不是整套系统严格未见用户开放集。

校准规则预先固定为本人 Q10 与冒充者 Q95 的中点；240 个阈值全部落盘后才评分查询 ACC。B 的错 Audio 和同人异词扰动沿用 B 原阈值。没有查询 Audio。

## 三种查询内容的宏平均

数值为百分比，AUC 除外；每行先按目标与种子平均。

|方法|查询内容|EER|HTER|FAR|FRR|AUC|
|---|---|---:|---:|---:|---:|---:|
|ungated_ACC|english_word|6.898|7.412|6.241|8.583|0.98024|
|ungated_ACC|english_long|4.546|7.282|5.398|9.167|0.98619|
|ungated_ACC|chinese_long|6.926|11.440|6.296|16.583|0.97055|
|A_zero_audio|english_word|7.000|7.657|6.315|9.000|0.97987|
|A_zero_audio|english_long|4.556|7.324|5.398|9.250|0.98616|
|A_zero_audio|chinese_long|6.801|11.458|6.333|16.583|0.96966|
|B_correct_audio|english_word|7.009|7.616|6.315|8.917|0.97968|
|B_correct_audio|english_long|4.519|7.343|5.352|9.333|0.98595|
|B_correct_audio|chinese_long|6.852|11.542|6.417|16.667|0.96950|
|C_other_audio|english_word|6.972|7.759|6.352|9.167|0.97986|
|C_other_audio|english_long|4.616|7.421|5.426|9.417|0.98596|
|C_other_audio|chinese_long|6.870|11.481|6.296|16.667|0.96939|
|D_destroyed_audio|english_word|7.005|7.773|6.296|9.250|0.97968|
|D_destroyed_audio|english_long|4.560|7.421|5.426|9.417|0.98600|
|D_destroyed_audio|chinese_long|6.880|11.583|6.333|16.833|0.96935|
|B_wrong_audio|english_word|6.940|7.500|6.417|8.583|0.97995|
|B_wrong_audio|english_long|4.449|7.282|5.315|9.250|0.98600|
|B_wrong_audio|chinese_long|6.898|11.477|6.454|16.500|0.96951|
|B_audio_first2words|english_word|7.009|7.583|6.333|8.833|0.97971|
|B_audio_first2words|english_long|4.556|7.380|5.343|9.417|0.98596|
|B_audio_first2words|chinese_long|6.852|11.454|6.407|16.500|0.96948|
|B_audio_last2words|english_word|6.981|7.625|6.333|8.917|0.97971|
|B_audio_last2words|english_long|4.505|7.329|5.324|9.333|0.98598|
|B_audio_last2words|chinese_long|6.847|11.597|6.444|16.750|0.96958|

## 以用户为簇的配对比较

差值 B−对照，负值为改善。每位用户先平均三个种子；bootstrap 重采样 10 位用户，不将重复录制视为独立受试者。95% 区间为探索性百分位区间。

|对照|指标|B−对照/百分点|95%区间|改善用户/10|
|---|---|---:|---|---:|
|ungated_ACC|eer|+0.003|[-0.094, +0.106]|6|
|ungated_ACC|hter|+0.122|[-0.039, +0.290]|4|
|A_zero_audio|eer|+0.008|[-0.083, +0.094]|4|
|A_zero_audio|hter|+0.020|[-0.145, +0.198]|3|
|C_other_audio|eer|-0.026|[-0.114, +0.074]|6|
|C_other_audio|hter|-0.054|[-0.204, +0.076]|4|
|D_destroyed_audio|eer|-0.022|[-0.123, +0.073]|5|
|D_destroyed_audio|hter|-0.093|[-0.210, +0.012]|8|
|B_wrong_audio|eer|+0.031|[-0.025, +0.090]|3|
|B_wrong_audio|hter|+0.080|[-0.031, +0.190]|3|
|B_audio_first2words|eer|-0.012|[-0.048, +0.011]|4|
|B_audio_first2words|hter|+0.028|[-0.003, +0.063]|2|
|B_audio_last2words|eer|+0.015|[-0.008, +0.051]|3|
|B_audio_last2words|hter|-0.017|[-0.049, +0.006]|4|

## 判断

未通过预定独立增益标准；停止当前通道门控方案，不在已查看查询词上继续搜索权重。

本轮直接检验注册 Audio 是否能改善 ACC 验证度量。B 与无 Audio、错 Audio、破坏 Audio 的配对差异才是 Audio 独立贡献的依据；不能只看某一内容组或用户的最好结果。

## 逐用户结果（3 种子平均）

|用户|查询内容|方法|EER%|HTER%|FAR%|FRR%|
|---|---|---|---:|---:|---:|---:|
|U01|english_word|ungated_ACC|8.333|7.731|7.130|8.333|
|U01|english_word|A_zero_audio|8.241|7.685|7.037|8.333|
|U01|english_word|B_correct_audio|8.194|8.333|6.667|10.000|
|U01|english_word|C_other_audio|8.241|8.657|7.315|10.000|
|U01|english_word|D_destroyed_audio|8.287|8.426|6.852|10.000|
|U01|english_long|ungated_ACC|9.213|11.204|6.574|15.833|
|U01|english_long|A_zero_audio|9.074|10.694|6.389|15.000|
|U01|english_long|B_correct_audio|9.074|10.694|6.389|15.000|
|U01|english_long|C_other_audio|9.259|11.111|6.389|15.833|
|U01|english_long|D_destroyed_audio|9.120|11.111|6.389|15.833|
|U01|chinese_long|ungated_ACC|8.380|13.009|5.185|20.833|
|U01|chinese_long|A_zero_audio|8.194|12.176|5.185|19.167|
|U01|chinese_long|B_correct_audio|8.380|13.426|5.185|21.667|
|U01|chinese_long|C_other_audio|8.796|12.454|4.907|20.000|
|U01|chinese_long|D_destroyed_audio|8.287|12.963|5.093|20.833|
|U02|english_word|ungated_ACC|4.028|6.620|0.741|12.500|
|U02|english_word|A_zero_audio|4.074|6.528|0.556|12.500|
|U02|english_word|B_correct_audio|3.981|6.574|0.648|12.500|
|U02|english_word|C_other_audio|3.889|6.528|0.556|12.500|
|U02|english_word|D_destroyed_audio|4.120|6.528|0.556|12.500|
|U02|english_long|ungated_ACC|4.306|6.296|2.593|10.000|
|U02|english_long|A_zero_audio|4.306|6.759|2.685|10.833|
|U02|english_long|B_correct_audio|4.352|6.759|2.685|10.833|
|U02|english_long|C_other_audio|4.352|6.759|2.685|10.833|
|U02|english_long|D_destroyed_audio|4.352|7.130|2.593|11.667|
|U02|chinese_long|ungated_ACC|2.546|3.194|3.056|3.333|
|U02|chinese_long|A_zero_audio|2.500|3.148|2.963|3.333|
|U02|chinese_long|B_correct_audio|2.500|3.148|2.963|3.333|
|U02|chinese_long|C_other_audio|2.500|3.148|2.963|3.333|
|U02|chinese_long|D_destroyed_audio|2.500|3.102|2.870|3.333|
|U03|english_word|ungated_ACC|6.574|5.741|5.648|5.833|
|U03|english_word|A_zero_audio|6.713|6.713|5.926|7.500|
|U03|english_word|B_correct_audio|6.759|5.972|6.111|5.833|
|U03|english_word|C_other_audio|7.269|6.759|6.019|7.500|
|U03|english_word|D_destroyed_audio|6.759|6.713|5.926|7.500|
|U03|english_long|ungated_ACC|6.759|14.352|2.037|26.667|
|U03|english_long|A_zero_audio|6.852|14.444|2.222|26.667|
|U03|english_long|B_correct_audio|6.944|14.444|2.222|26.667|
|U03|english_long|C_other_audio|6.898|14.444|2.222|26.667|
|U03|english_long|D_destroyed_audio|6.852|14.444|2.222|26.667|
|U03|chinese_long|ungated_ACC|9.815|23.426|1.852|45.000|
|U03|chinese_long|A_zero_audio|9.352|23.426|1.852|45.000|
|U03|chinese_long|B_correct_audio|9.352|22.639|1.944|43.333|
|U03|chinese_long|C_other_audio|9.352|23.519|2.037|45.000|
|U03|chinese_long|D_destroyed_audio|9.352|23.426|1.852|45.000|
|U04|english_word|ungated_ACC|3.333|5.278|0.556|10.000|
|U04|english_word|A_zero_audio|3.241|4.907|0.648|9.167|
|U04|english_word|B_correct_audio|3.194|4.861|0.556|9.167|
|U04|english_word|C_other_audio|3.287|4.907|0.648|9.167|
|U04|english_word|D_destroyed_audio|3.241|4.907|0.648|9.167|
|U04|english_long|ungated_ACC|0.231|1.296|2.593|0.000|
|U04|english_long|A_zero_audio|0.231|1.343|2.685|0.000|
|U04|english_long|B_correct_audio|0.231|1.343|2.685|0.000|
|U04|english_long|C_other_audio|0.231|1.343|2.685|0.000|
|U04|english_long|D_destroyed_audio|0.231|1.296|2.593|0.000|
|U04|chinese_long|ungated_ACC|0.046|2.361|4.722|0.000|
|U04|chinese_long|A_zero_audio|0.046|2.315|4.630|0.000|
|U04|chinese_long|B_correct_audio|0.093|2.361|4.722|0.000|
|U04|chinese_long|C_other_audio|0.046|2.361|4.722|0.000|
|U04|chinese_long|D_destroyed_audio|0.046|2.407|4.815|0.000|
|U05|english_word|ungated_ACC|12.500|10.509|6.019|15.000|
|U05|english_word|A_zero_audio|12.500|10.741|5.648|15.833|
|U05|english_word|B_correct_audio|12.500|10.741|5.648|15.833|
|U05|english_word|C_other_audio|12.500|10.741|5.648|15.833|
|U05|english_word|D_destroyed_audio|12.500|10.741|5.648|15.833|
|U05|english_long|ungated_ACC|10.694|11.759|6.019|17.500|
|U05|english_long|A_zero_audio|10.741|12.315|5.463|19.167|
|U05|english_long|B_correct_audio|10.694|12.315|5.463|19.167|
|U05|english_long|C_other_audio|10.694|12.361|5.556|19.167|
|U05|english_long|D_destroyed_audio|10.694|12.454|5.741|19.167|
|U05|chinese_long|ungated_ACC|23.241|33.611|4.722|62.500|
|U05|chinese_long|A_zero_audio|22.778|34.676|4.352|65.000|
|U05|chinese_long|B_correct_audio|22.731|34.769|4.537|65.000|
|U05|chinese_long|C_other_audio|22.778|34.676|4.352|65.000|
|U05|chinese_long|D_destroyed_audio|23.241|34.676|4.352|65.000|
|U06|english_word|ungated_ACC|7.315|7.870|10.741|5.000|
|U06|english_word|A_zero_audio|7.500|7.917|10.833|5.000|
|U06|english_word|B_correct_audio|7.500|7.778|10.556|5.000|
|U06|english_word|C_other_audio|7.407|8.102|11.204|5.000|
|U06|english_word|D_destroyed_audio|7.407|7.963|10.926|5.000|
|U06|english_long|ungated_ACC|0.000|3.380|6.759|0.000|
|U06|english_long|A_zero_audio|0.000|3.333|6.667|0.000|
|U06|english_long|B_correct_audio|0.000|3.056|6.111|0.000|
|U06|english_long|C_other_audio|0.000|3.611|7.222|0.000|
|U06|english_long|D_destroyed_audio|0.000|3.426|6.852|0.000|
|U06|chinese_long|ungated_ACC|0.139|3.241|6.481|0.000|
|U06|chinese_long|A_zero_audio|0.093|3.472|6.944|0.000|
|U06|chinese_long|B_correct_audio|0.139|3.380|6.759|0.000|
|U06|chinese_long|C_other_audio|0.139|3.565|7.130|0.000|
|U06|chinese_long|D_destroyed_audio|0.139|3.519|7.037|0.000|
|U07|english_word|ungated_ACC|4.028|6.852|2.870|10.833|
|U07|english_word|A_zero_audio|4.213|8.148|2.963|13.333|
|U07|english_word|B_correct_audio|4.213|7.778|3.056|12.500|
|U07|english_word|C_other_audio|4.306|8.148|2.963|13.333|
|U07|english_word|D_destroyed_audio|4.213|8.148|2.963|13.333|
|U07|english_long|ungated_ACC|3.056|2.917|3.333|2.500|
|U07|english_long|A_zero_audio|3.102|3.009|3.519|2.500|
|U07|english_long|B_correct_audio|3.102|3.009|3.519|2.500|
|U07|english_long|C_other_audio|3.056|2.963|3.426|2.500|
|U07|english_long|D_destroyed_audio|3.102|3.009|3.519|2.500|
|U07|chinese_long|ungated_ACC|6.019|5.648|11.296|0.000|
|U07|chinese_long|A_zero_audio|6.574|5.648|11.296|0.000|
|U07|chinese_long|B_correct_audio|6.574|5.694|11.389|0.000|
|U07|chinese_long|C_other_audio|6.574|5.648|11.296|0.000|
|U07|chinese_long|D_destroyed_audio|6.574|5.556|11.111|0.000|
|U08|english_word|ungated_ACC|9.954|9.306|16.111|2.500|
|U08|english_word|A_zero_audio|10.694|9.676|16.852|2.500|
|U08|english_word|B_correct_audio|10.093|9.676|16.852|2.500|
|U08|english_word|C_other_audio|10.000|9.537|16.574|2.500|
|U08|english_word|D_destroyed_audio|10.741|10.139|16.944|3.333|
|U08|english_long|ungated_ACC|9.167|10.602|7.870|13.333|
|U08|english_long|A_zero_audio|9.167|10.324|8.148|12.500|
|U08|english_long|B_correct_audio|8.704|10.694|8.056|13.333|
|U08|english_long|C_other_audio|9.630|10.694|8.056|13.333|
|U08|english_long|D_destroyed_audio|9.167|10.370|8.241|12.500|
|U08|chinese_long|ungated_ACC|13.333|13.796|10.926|16.667|
|U08|chinese_long|A_zero_audio|12.685|14.352|11.204|17.500|
|U08|chinese_long|B_correct_audio|12.824|14.352|11.204|17.500|
|U08|chinese_long|C_other_audio|12.731|14.259|11.019|17.500|
|U08|chinese_long|D_destroyed_audio|12.824|14.769|11.204|18.333|
|U09|english_word|ungated_ACC|0.926|1.528|0.556|2.500|
|U09|english_word|A_zero_audio|0.972|1.574|0.648|2.500|
|U09|english_word|B_correct_audio|1.019|1.620|0.741|2.500|
|U09|english_word|C_other_audio|1.019|1.574|0.648|2.500|
|U09|english_word|D_destroyed_audio|0.972|1.574|0.648|2.500|
|U09|english_long|ungated_ACC|1.852|3.426|1.019|5.833|
|U09|english_long|A_zero_audio|1.852|3.426|1.019|5.833|
|U09|english_long|B_correct_audio|1.898|3.426|1.019|5.833|
|U09|english_long|C_other_audio|1.852|3.426|1.019|5.833|
|U09|english_long|D_destroyed_audio|1.852|3.426|1.019|5.833|
|U09|chinese_long|ungated_ACC|2.500|8.981|0.463|17.500|
|U09|chinese_long|A_zero_audio|2.500|8.194|0.556|15.833|
|U09|chinese_long|B_correct_audio|2.500|8.194|0.556|15.833|
|U09|chinese_long|C_other_audio|2.500|8.194|0.556|15.833|
|U09|chinese_long|D_destroyed_audio|2.500|8.194|0.556|15.833|
|U10|english_word|ungated_ACC|11.991|12.685|12.037|13.333|
|U10|english_word|A_zero_audio|11.852|12.685|12.037|13.333|
|U10|english_word|B_correct_audio|12.639|12.824|12.315|13.333|
|U10|english_word|C_other_audio|11.806|12.639|11.944|13.333|
|U10|english_word|D_destroyed_audio|11.806|12.593|11.852|13.333|
|U10|english_long|ungated_ACC|0.185|7.593|15.185|0.000|
|U10|english_long|A_zero_audio|0.231|7.593|15.185|0.000|
|U10|english_long|B_correct_audio|0.185|7.685|15.370|0.000|
|U10|english_long|C_other_audio|0.185|7.500|15.000|0.000|
|U10|english_long|D_destroyed_audio|0.231|7.546|15.093|0.000|
|U10|chinese_long|ungated_ACC|3.241|7.130|14.259|0.000|
|U10|chinese_long|A_zero_audio|3.287|7.176|14.352|0.000|
|U10|chinese_long|B_correct_audio|3.426|7.454|14.907|0.000|
|U10|chinese_long|C_other_audio|3.287|6.991|13.981|0.000|
|U10|chinese_long|D_destroyed_audio|3.333|7.222|14.444|0.000|

## 门控诊断

换成他人 Audio 后，权重平均绝对变化 0.052944；同人两组注册词的变化 0.022448。后一诊断每组只有 6 条 Audio，主实验为 12 条，数量不匹配，仅用于敏感性诊断。

## 检查与边界

独立核验 12000 个 episode：目标隔离、support/query 词无交集、真假标签一致。无门控余弦与历史相同预算基线完全回归；合成测试验证全一初始化、加权余弦、梯度与随机矩匹配。全部逐用户逐种子匿名汇总在 JSON 中。

44100 Hz 为现有时间解释候选，物理延迟未独立标定。ACC 缓存为约 3 秒窗口，可能包含相邻重复发声；本报告的单条查询不等同于经人工确认只含一次孤立发声。历史上已反复查看这些查询词，当前结果为探索性。

仅上传代码、方案、匿名逐用户汇总与审计摘要；私有 Audio、embedding、采样索引、阈值和权重不上传。