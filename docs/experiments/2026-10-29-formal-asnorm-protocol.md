# 正式AS-Norm1对照协议

采用Matejka等Interspeech2017公式7的AS-Norm1：

`s_AS = 0.5*((s(e,t)-mean(topK s(e,C)))/std(topK s(e,C)) + (s(e,t)-mean(topK s(t,C)))/std(topK s(t,C)))`

注册端和测试端分别独立选择最高分Top-K cohort，不是仅减最大背景分数，也不是可训练TAS-Norm或交叉选择AS-Norm2。样本标准差ddof=1，与公开asv-subtools的pandas标准差约定一致；标准差下限固定1e-4。零方差数值保护不通过测试调参。

参考：[原始论文](https://www.fit.vut.cz/research/group/speech/public/publi/2017/matejka_interspeech2017_IS170803.pdf)；[公开AS-Norm实现](https://github.com/Snowdar/asv-subtools/blob/master/score/ScoreNormalization.py)，cross_select=false。

保持此前限定用户群实验的模型和数据：每身份认证训练80条；完整772维和identity_acc64维各用同一StandardScaler及归一化中心；cohort为每个保留背景身份一个训练均值原型，排除目标身份。比较不删人、删1人、删2人。背景共8/7/6人。上游模型及预处理完全不变。

比较固定K=4、训练内容五折选择K=2/3/4/5以及全cohort的S-Norm。K只按保留认证训练内的内容折平均AUC选择，同分选择较大K。阈值仍由对应独立内容校准数据确定，配置锁定后才读取最终特征缓存。

限制：背景仅6至8个身份，远小于常见大cohort系统，统计稳定性有限。删人名单已根据旧测试FAR选定，所以删人版本只能称测试知情的限定群探索；不得当成独立测试提升。不删人版本仍复用了已观察的历史最终测试。全部八个攻击来源另保留诊断成绩。

代码work/run_formal_asnorm_20261029.py，正式输出/autodl-fs/data/mei_formal_asnorm_20261029_f64。本轮包含对称性和手工公式校验。初始float32评分在小cohort标准差下产生放大的独立重算差异，因此正式相似度、Top-K统计量和标准化评分均使用float64重新计算；模型不重训。效果以独立复核汇总为准。

