# 跨词跨模态身份投影与 Audio 注册原型实验

## 方法与数据

冻结现有纯 ACC 192 维编码器与官方 ECAPA 192 维 Audio 编码器，仅训练两个 192→64→32 投影头（GELU，L2 归一化）。双向身份监督对比学习，温度 0.1。同一批中 Audio 全部来自同一训练词，ACC 全部来自另一个训练词；同人跨词为正、异人同词来源为负。没有复制教师相似度矩阵。

全部 10 人保留、种子 42/43/44。每折留出目标及下一位循环身份；其余 8 人训练投影 400 步。ACC→ACC 对照使用相同两投影头、初始化、episode、优化预算，把 Audio 输入替换为相应词的 ACC。共 60 个模型。

投影优化只使用前 12 个 fit 词；其余 4 个 fit 词用于内部注册、校准和开发：前两词各 6 条支持，第三词 8 条本人＋64 条攻击校准，第四词 8＋64 条选择融合权重。开发本人身份未参与投影优化；最终目标身份不参与此权重选择。权重候选为 0/0.1/0.25/0.5/1，按内部 HTER、EER、更小权重排序。该内部样本量较小，权重选择存在不确定性。

实际注册沿用旧 acc_crossword12：4 个 REG 词、每词 3 条，共 12 条相同 ACC，Audio 也取对应 12 条。没有修改注册样本排序。最终查询为单条 ACC，未读取查询 Audio；注册词、CAL 词与三个查询内容组的词 ID 完全分离。

分数＝原 ACC 余弦＋α×查询 ACC 投影与注册 Audio 投影原型的余弦。异人 Audio、192 坐标打乱 Audio 使用相同正确模型和 α；同时报告分别校准阈值及沿用正确 Audio 阈值的对照。AudioModel_ACCprototype 使用同一 Audio 联合训练的 ACC 投影头，将原型换成 ACC，α 保持相同。

240 个 CAL 阈值全部锁定后才评分最终查询。阈值规则沿用本人 Q10 与冒充者 Q95 中点。ACC 主干此前见过所有 10 人与全部 16 个 fit 词，因此这不是完整未见身份开放集，也不将内部 4 词称为编码器未见词。

## 第一层：跨模态检索

CAL 的目标 ACC 查询在 10 个注册 Audio 原型中 Top1 为 8.08%（随机期望 10%；目标簇 bootstrap 95% 区间 [4.33%, 11.92%]）。目标身份跨模态验证 AUC=0.5802；同一查询在本人/异人原型间检索 AUC=0.6444。CAL 只用于此描述和固定规则阈值，不用于选择投影模型或 α。

|用户|Top1%|目标验证 AUC|同查询检索 AUC|本人−最难冒充原型间隔|
|---|---:|---:|---:|---:|
|U01|19.17|0.8007|0.7967|-0.1715|
|U02|10.00|0.7218|0.7789|-0.1744|
|U03|13.33|0.5731|0.6029|-0.4562|
|U04|0.00|0.3976|0.4664|-0.4513|
|U05|6.67|0.6993|0.7734|-0.2942|
|U06|8.33|0.6671|0.7439|-0.2030|
|U07|0.00|0.2270|0.4308|-0.6847|
|U08|9.17|0.7724|0.7149|-0.2198|
|U09|0.00|0.2040|0.4121|-0.8112|
|U10|14.17|0.7391|0.7242|-0.2337|

### 事后拟合诊断（不用于选择模型或权重）

|范围|Top1|原型候选/支持量|
|---|---:|---|
|留出目标身份、训练词内跨词|19.31%|10 人；每人 48 条 Audio 支持|
|留出开发身份、内部开发词跨词|11.25%|9 人；每人 16 条 Audio 支持|
|投影训练身份、训练词内跨词|100.00%|8 人；每人 48 条 Audio 支持|

训练身份上的跨词匹配达到 100%，但留出身份与保留注册/CAL 词上的匹配弱。这支持小投影头拟合了训练身份对应、泛化不足的解释。上述事后检查的候选人数与支持预算不完全相同，不能将所有差值只归因于词汇；它们也不能替代最终固定 12 条注册的认证结果。

## 第二层：ACC-only 单条跨内容认证

三种子、十目标宏平均。EER 与 HTER 以及 FAR/FRR 均为百分比。

|方法|查询内容|EER|HTER|FAR|FRR|AUC|
|---|---|---:|---:|---:|---:|---:|
|raw_ACC|english_word|6.898|7.412|6.241|8.583|0.98024|
|raw_ACC|english_long|4.546|7.282|5.398|9.167|0.98619|
|raw_ACC|chinese_long|6.926|11.440|6.296|16.583|0.97055|
|ACC_ACC_fusion|english_word|7.019|7.671|6.259|9.083|0.97991|
|ACC_ACC_fusion|english_long|4.806|7.648|5.463|9.833|0.98543|
|ACC_ACC_fusion|chinese_long|6.917|11.662|6.324|17.000|0.96953|
|Audio_correct|english_word|7.352|7.884|6.602|9.167|0.97738|
|Audio_correct|english_long|4.630|7.694|6.056|9.333|0.98577|
|Audio_correct|chinese_long|7.171|11.625|7.250|16.000|0.96986|
|Audio_other|english_word|8.463|9.014|6.944|11.083|0.97189|
|Audio_other|english_long|4.583|7.852|6.537|9.167|0.98598|
|Audio_other|chinese_long|7.120|12.009|7.602|16.417|0.96976|
|Audio_shuffled|english_word|7.884|8.741|6.815|10.667|0.97463|
|Audio_shuffled|english_long|4.644|7.644|6.037|9.250|0.98582|
|Audio_shuffled|chinese_long|7.171|11.727|7.120|16.333|0.96924|
|AudioModel_ACCprototype|english_word|6.991|7.579|6.241|8.917|0.97971|
|AudioModel_ACCprototype|english_long|4.597|7.352|5.454|9.250|0.98608|
|AudioModel_ACCprototype|chinese_long|6.931|11.514|6.278|16.750|0.97027|
|Audio_other_Bthreshold|english_word|8.463|8.949|7.065|10.833|0.97189|
|Audio_other_Bthreshold|english_long|4.583|7.931|6.528|9.333|0.98598|
|Audio_other_Bthreshold|chinese_long|7.120|11.972|7.611|16.333|0.96976|
|Audio_shuffled_Bthreshold|english_word|7.884|8.940|6.796|11.083|0.97463|
|Audio_shuffled_Bthreshold|english_long|4.644|7.713|6.259|9.167|0.98582|
|Audio_shuffled_Bthreshold|chinese_long|7.171|11.718|7.435|16.000|0.96924|

## 配对不确定性

正确 Audio−对照，负值为好；每位目标先平均三种子、三种内容，再按目标重采样。重复发声不视为独立受试者。

|对照|指标|差/百分点|95%区间|改善人数|
|---|---|---:|---|---:|
|raw_ACC|eer|+0.2608|[-0.0926, +0.8627]|1/10|
|raw_ACC|hter|+0.3565|[-0.1343, +1.2191]|2/10|
|ACC_ACC_fusion|eer|+0.1373|[-0.4398, +0.8194]|1/10|
|ACC_ACC_fusion|hter|+0.0741|[-0.7176, +1.1111]|4/10|
|Audio_other|eer|-0.3380|[-0.9660, +0.0031]|2/10|
|Audio_other|hter|-0.5571|[-1.5448, +0.0000]|3/10|
|Audio_shuffled|eer|-0.1821|[-0.5478, +0.0586]|2/10|
|Audio_shuffled|hter|-0.3025|[-0.8765, +0.0309]|2/10|
|AudioModel_ACCprototype|eer|+0.2114|[-0.0370, +0.6435]|1/10|
|AudioModel_ACCprototype|hter|+0.2531|[-0.2130, +1.0123]|2/10|
|Audio_other_Bthreshold|eer|-0.3380|[-0.9660, +0.0031]|2/10|
|Audio_other_Bthreshold|hter|-0.5494|[-1.4707, +0.0000]|3/10|
|Audio_shuffled_Bthreshold|eer|-0.1821|[-0.5478, +0.0586]|2/10|
|Audio_shuffled_Bthreshold|hter|-0.3889|[-1.2160, +0.1111]|2/10|

## 权重选择与判断

30 折权重计数：`{"crossmodal": {"0.0": 23, "0.1": 1, "0.25": 2, "0.5": 2, "1.0": 2}, "acc_only": {"0.0": 24, "0.1": 1, "0.25": 3, "0.5": 1, "1.0": 1}}`。α=0 表示开发数据未选择 Audio 分数，最终与原 ACC 分数完全相同，不能计作 Audio 收益。

本轮未通过 Audio 独立认证增益标准；不继续在现有查询词上搜索融合权重。跨模态检索结果与最终认证结果分开解释。

跨模态检索好，只能说明小投影头能够学习身份对应；认证必须同时超过原 ACC、ACC→ACC 容量对照和错误注册 Audio，才支持独立收益。第一层指标高不替代第二层。

## 逐用户认证结果（三种子平均）

|用户|查询内容|方法|EER%|HTER%|FAR%|FRR%|
|---|---|---|---:|---:|---:|---:|
|U01|english_word|raw_ACC|8.333|7.731|7.130|8.333|
|U01|english_word|ACC_ACC_fusion|9.259|8.565|7.130|10.000|
|U01|english_word|Audio_correct|8.333|7.731|7.130|8.333|
|U01|english_word|Audio_other|8.333|7.731|7.130|8.333|
|U01|english_word|Audio_shuffled|8.333|7.731|7.130|8.333|
|U01|english_word|AudioModel_ACCprototype|8.333|7.731|7.130|8.333|
|U01|english_long|raw_ACC|9.213|11.204|6.574|15.833|
|U01|english_long|ACC_ACC_fusion|11.435|14.491|6.481|22.500|
|U01|english_long|Audio_correct|9.213|11.204|6.574|15.833|
|U01|english_long|Audio_other|9.213|11.204|6.574|15.833|
|U01|english_long|Audio_shuffled|9.213|11.204|6.574|15.833|
|U01|english_long|AudioModel_ACCprototype|9.213|11.204|6.574|15.833|
|U01|chinese_long|raw_ACC|8.380|13.009|5.185|20.833|
|U01|chinese_long|ACC_ACC_fusion|9.861|15.463|5.093|25.833|
|U01|chinese_long|Audio_correct|8.380|13.009|5.185|20.833|
|U01|chinese_long|Audio_other|8.380|13.009|5.185|20.833|
|U01|chinese_long|Audio_shuffled|8.380|13.009|5.185|20.833|
|U01|chinese_long|AudioModel_ACCprototype|8.380|13.009|5.185|20.833|
|U02|english_word|raw_ACC|4.028|6.620|0.741|12.500|
|U02|english_word|ACC_ACC_fusion|4.028|6.620|0.741|12.500|
|U02|english_word|Audio_correct|4.028|6.620|0.741|12.500|
|U02|english_word|Audio_other|4.028|6.620|0.741|12.500|
|U02|english_word|Audio_shuffled|4.028|6.620|0.741|12.500|
|U02|english_word|AudioModel_ACCprototype|4.028|6.620|0.741|12.500|
|U02|english_long|raw_ACC|4.306|6.296|2.593|10.000|
|U02|english_long|ACC_ACC_fusion|4.306|6.296|2.593|10.000|
|U02|english_long|Audio_correct|4.306|6.296|2.593|10.000|
|U02|english_long|Audio_other|4.306|6.296|2.593|10.000|
|U02|english_long|Audio_shuffled|4.306|6.296|2.593|10.000|
|U02|english_long|AudioModel_ACCprototype|4.306|6.296|2.593|10.000|
|U02|chinese_long|raw_ACC|2.546|3.194|3.056|3.333|
|U02|chinese_long|ACC_ACC_fusion|2.546|3.194|3.056|3.333|
|U02|chinese_long|Audio_correct|2.546|3.194|3.056|3.333|
|U02|chinese_long|Audio_other|2.546|3.194|3.056|3.333|
|U02|chinese_long|Audio_shuffled|2.546|3.194|3.056|3.333|
|U02|chinese_long|AudioModel_ACCprototype|2.546|3.194|3.056|3.333|
|U03|english_word|raw_ACC|6.574|5.741|5.648|5.833|
|U03|english_word|ACC_ACC_fusion|7.407|7.454|5.741|9.167|
|U03|english_word|Audio_correct|6.574|5.741|5.648|5.833|
|U03|english_word|Audio_other|6.574|5.741|5.648|5.833|
|U03|english_word|Audio_shuffled|6.574|5.741|5.648|5.833|
|U03|english_word|AudioModel_ACCprototype|6.574|5.741|5.648|5.833|
|U03|english_long|raw_ACC|6.759|14.352|2.037|26.667|
|U03|english_long|ACC_ACC_fusion|6.574|13.194|2.222|24.167|
|U03|english_long|Audio_correct|6.759|14.352|2.037|26.667|
|U03|english_long|Audio_other|6.759|14.352|2.037|26.667|
|U03|english_long|Audio_shuffled|6.759|14.352|2.037|26.667|
|U03|english_long|AudioModel_ACCprototype|6.759|14.352|2.037|26.667|
|U03|chinese_long|raw_ACC|9.815|23.426|1.852|45.000|
|U03|chinese_long|ACC_ACC_fusion|9.167|22.963|1.759|44.167|
|U03|chinese_long|Audio_correct|9.815|23.426|1.852|45.000|
|U03|chinese_long|Audio_other|9.815|23.426|1.852|45.000|
|U03|chinese_long|Audio_shuffled|9.815|23.426|1.852|45.000|
|U03|chinese_long|AudioModel_ACCprototype|9.815|23.426|1.852|45.000|
|U04|english_word|raw_ACC|3.333|5.278|0.556|10.000|
|U04|english_word|ACC_ACC_fusion|3.611|5.278|0.556|10.000|
|U04|english_word|Audio_correct|9.259|10.046|4.259|15.833|
|U04|english_word|Audio_other|18.333|20.000|7.500|32.500|
|U04|english_word|Audio_shuffled|13.241|17.917|5.833|30.000|
|U04|english_word|AudioModel_ACCprototype|4.954|7.037|0.741|13.333|
|U04|english_long|raw_ACC|0.231|1.296|2.593|0.000|
|U04|english_long|ACC_ACC_fusion|0.231|1.389|2.778|0.000|
|U04|english_long|Audio_correct|1.019|4.306|8.611|0.000|
|U04|english_long|Audio_other|1.111|6.806|13.611|0.000|
|U04|english_long|Audio_shuffled|1.065|4.722|9.444|0.000|
|U04|english_long|AudioModel_ACCprototype|0.741|1.574|3.148|0.000|
|U04|chinese_long|raw_ACC|0.046|2.361|4.722|0.000|
|U04|chinese_long|ACC_ACC_fusion|0.046|2.407|4.815|0.000|
|U04|chinese_long|Audio_correct|1.667|6.852|13.704|0.000|
|U04|chinese_long|Audio_other|1.898|9.074|17.315|0.833|
|U04|chinese_long|Audio_shuffled|2.685|6.944|13.056|0.833|
|U04|chinese_long|AudioModel_ACCprototype|0.046|2.269|4.537|0.000|
|U05|english_word|raw_ACC|12.500|10.509|6.019|15.000|
|U05|english_word|ACC_ACC_fusion|11.620|10.556|6.111|15.000|
|U05|english_word|Audio_correct|11.667|10.787|5.741|15.833|
|U05|english_word|Audio_other|11.759|10.694|5.556|15.833|
|U05|english_word|Audio_shuffled|12.500|11.157|6.481|15.833|
|U05|english_word|AudioModel_ACCprototype|11.713|10.417|5.833|15.000|
|U05|english_long|raw_ACC|10.694|11.759|6.019|17.500|
|U05|english_long|ACC_ACC_fusion|10.694|12.731|6.296|19.167|
|U05|english_long|Audio_correct|10.046|12.083|5.833|18.333|
|U05|english_long|Audio_other|10.046|12.361|5.556|19.167|
|U05|english_long|Audio_shuffled|10.648|12.083|5.833|18.333|
|U05|english_long|AudioModel_ACCprototype|10.694|11.759|6.019|17.500|
|U05|chinese_long|raw_ACC|23.241|33.611|4.722|62.500|
|U05|chinese_long|ACC_ACC_fusion|22.315|33.796|5.093|62.500|
|U05|chinese_long|Audio_correct|23.333|32.778|4.722|60.833|
|U05|chinese_long|Audio_other|23.194|34.352|4.537|64.167|
|U05|chinese_long|Audio_shuffled|23.194|31.944|4.722|59.167|
|U05|chinese_long|AudioModel_ACCprototype|23.194|34.074|4.815|63.333|
|U06|english_word|raw_ACC|7.315|7.870|10.741|5.000|
|U06|english_word|ACC_ACC_fusion|7.315|7.870|10.741|5.000|
|U06|english_word|Audio_correct|7.315|7.870|10.741|5.000|
|U06|english_word|Audio_other|7.315|7.870|10.741|5.000|
|U06|english_word|Audio_shuffled|7.315|7.870|10.741|5.000|
|U06|english_word|AudioModel_ACCprototype|7.315|7.870|10.741|5.000|
|U06|english_long|raw_ACC|0.000|3.380|6.759|0.000|
|U06|english_long|ACC_ACC_fusion|0.000|3.380|6.759|0.000|
|U06|english_long|Audio_correct|0.000|3.380|6.759|0.000|
|U06|english_long|Audio_other|0.000|3.380|6.759|0.000|
|U06|english_long|Audio_shuffled|0.000|3.380|6.759|0.000|
|U06|english_long|AudioModel_ACCprototype|0.000|3.380|6.759|0.000|
|U06|chinese_long|raw_ACC|0.139|3.241|6.481|0.000|
|U06|chinese_long|ACC_ACC_fusion|0.139|3.241|6.481|0.000|
|U06|chinese_long|Audio_correct|0.139|3.241|6.481|0.000|
|U06|chinese_long|Audio_other|0.139|3.241|6.481|0.000|
|U06|chinese_long|Audio_shuffled|0.139|3.241|6.481|0.000|
|U06|chinese_long|AudioModel_ACCprototype|0.139|3.241|6.481|0.000|
|U07|english_word|raw_ACC|4.028|6.852|2.870|10.833|
|U07|english_word|ACC_ACC_fusion|4.028|6.852|2.870|10.833|
|U07|english_word|Audio_correct|4.028|6.852|2.870|10.833|
|U07|english_word|Audio_other|4.028|6.852|2.870|10.833|
|U07|english_word|Audio_shuffled|4.028|6.852|2.870|10.833|
|U07|english_word|AudioModel_ACCprototype|4.028|6.852|2.870|10.833|
|U07|english_long|raw_ACC|3.056|2.917|3.333|2.500|
|U07|english_long|ACC_ACC_fusion|3.056|2.917|3.333|2.500|
|U07|english_long|Audio_correct|3.056|2.917|3.333|2.500|
|U07|english_long|Audio_other|3.056|2.917|3.333|2.500|
|U07|english_long|Audio_shuffled|3.056|2.917|3.333|2.500|
|U07|english_long|AudioModel_ACCprototype|3.056|2.917|3.333|2.500|
|U07|chinese_long|raw_ACC|6.019|5.648|11.296|0.000|
|U07|chinese_long|ACC_ACC_fusion|6.019|5.648|11.296|0.000|
|U07|chinese_long|Audio_correct|6.019|5.648|11.296|0.000|
|U07|chinese_long|Audio_other|6.019|5.648|11.296|0.000|
|U07|chinese_long|Audio_shuffled|6.019|5.648|11.296|0.000|
|U07|chinese_long|AudioModel_ACCprototype|6.019|5.648|11.296|0.000|
|U08|english_word|raw_ACC|9.954|9.306|16.111|2.500|
|U08|english_word|ACC_ACC_fusion|9.954|9.306|16.111|2.500|
|U08|english_word|Audio_correct|9.954|9.306|16.111|2.500|
|U08|english_word|Audio_other|9.954|9.306|16.111|2.500|
|U08|english_word|Audio_shuffled|9.954|9.306|16.111|2.500|
|U08|english_word|AudioModel_ACCprototype|9.954|9.306|16.111|2.500|
|U08|english_long|raw_ACC|9.167|10.602|7.870|13.333|
|U08|english_long|ACC_ACC_fusion|9.167|10.602|7.870|13.333|
|U08|english_long|Audio_correct|9.167|10.602|7.870|13.333|
|U08|english_long|Audio_other|9.167|10.602|7.870|13.333|
|U08|english_long|Audio_shuffled|9.167|10.602|7.870|13.333|
|U08|english_long|AudioModel_ACCprototype|9.167|10.602|7.870|13.333|
|U08|chinese_long|raw_ACC|13.333|13.796|10.926|16.667|
|U08|chinese_long|ACC_ACC_fusion|13.333|13.796|10.926|16.667|
|U08|chinese_long|Audio_correct|13.333|13.796|10.926|16.667|
|U08|chinese_long|Audio_other|13.333|13.796|10.926|16.667|
|U08|chinese_long|Audio_shuffled|13.333|13.796|10.926|16.667|
|U08|chinese_long|AudioModel_ACCprototype|13.333|13.796|10.926|16.667|
|U09|english_word|raw_ACC|0.926|1.528|0.556|2.500|
|U09|english_word|ACC_ACC_fusion|0.972|1.528|0.556|2.500|
|U09|english_word|Audio_correct|0.370|1.204|0.741|1.667|
|U09|english_word|Audio_other|2.315|2.639|1.111|4.167|
|U09|english_word|Audio_shuffled|0.880|1.528|0.556|2.500|
|U09|english_word|AudioModel_ACCprototype|1.019|1.528|0.556|2.500|
|U09|english_long|raw_ACC|1.852|3.426|1.019|5.833|
|U09|english_long|ACC_ACC_fusion|2.407|3.889|1.111|6.667|
|U09|english_long|Audio_correct|2.546|4.213|1.759|6.667|
|U09|english_long|Audio_other|1.991|3.009|1.852|4.167|
|U09|english_long|Audio_shuffled|2.037|3.287|0.741|5.833|
|U09|english_long|AudioModel_ACCprototype|1.852|3.843|1.019|6.667|
|U09|chinese_long|raw_ACC|2.500|8.981|0.463|17.500|
|U09|chinese_long|ACC_ACC_fusion|2.500|8.981|0.463|17.500|
|U09|chinese_long|Audio_correct|3.241|7.176|1.019|13.333|
|U09|chinese_long|Audio_other|2.639|7.222|1.111|13.333|
|U09|chinese_long|Audio_shuffled|2.361|8.935|0.370|17.500|
|U09|chinese_long|AudioModel_ACCprototype|2.593|9.352|0.370|18.333|
|U10|english_word|raw_ACC|11.991|12.685|12.037|13.333|
|U10|english_word|ACC_ACC_fusion|11.991|12.685|12.037|13.333|
|U10|english_word|Audio_correct|11.991|12.685|12.037|13.333|
|U10|english_word|Audio_other|11.991|12.685|12.037|13.333|
|U10|english_word|Audio_shuffled|11.991|12.685|12.037|13.333|
|U10|english_word|AudioModel_ACCprototype|11.991|12.685|12.037|13.333|
|U10|english_long|raw_ACC|0.185|7.593|15.185|0.000|
|U10|english_long|ACC_ACC_fusion|0.185|7.593|15.185|0.000|
|U10|english_long|Audio_correct|0.185|7.593|15.185|0.000|
|U10|english_long|Audio_other|0.185|7.593|15.185|0.000|
|U10|english_long|Audio_shuffled|0.185|7.593|15.185|0.000|
|U10|english_long|AudioModel_ACCprototype|0.185|7.593|15.185|0.000|
|U10|chinese_long|raw_ACC|3.241|7.130|14.259|0.000|
|U10|chinese_long|ACC_ACC_fusion|3.241|7.130|14.259|0.000|
|U10|chinese_long|Audio_correct|3.241|7.130|14.259|0.000|
|U10|chinese_long|Audio_other|3.241|7.130|14.259|0.000|
|U10|chinese_long|Audio_shuffled|3.241|7.130|14.259|0.000|
|U10|chinese_long|AudioModel_ACCprototype|3.241|7.130|14.259|0.000|

## 校验与公开边界

独立检查 12000 个训练任务的目标/开发身份隔离、跨词采样与标签对应，复核 60 次 α 选择、240 阈值和 720 个评估工作点；原 ACC 对照必须与历史同口径完全一致。所有匿名逐用户逐种子数据附在 JSON。

当前为反复查看过的既有队列。44100 Hz Audio 解释与物理延迟未独立确认；约 3 秒 ACC 窗口可能含相邻重复发声。两个编码器、创新点一和历史结果均保持不变。只发布代码与匿名汇总，不上传 Audio、embedding、模型权重、身份映射、采样索引、逐样本分数或实际阈值。