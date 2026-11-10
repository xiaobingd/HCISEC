# 三数据集：闭集跨内容认证

所有身份参与学习与负类注册，查询录制独立；这是已知身份的新内容认证，不是未见用户开放集。每用户注册12条、校准8条、查询24条，保留全部其他身份攻击。

Mei按名义词组隔离（未逐条核验转写）；ABCS/VibraVox按完整句子隔离，不保证词汇不重叠或不同用户说相同句子。ABCS通道0=振动仍未经独立证实。

主工作点：仅由校准数据设置每用户Cal-EER阈值；下表为逐用户宏平均百分比。

|数据集|方法|TAR|FAR|HTER|
|---|---|---:|---:|---:|
|Mei9|raw_statistics|81.02|7.47|13.22|
|Mei9|no_audio_baseline|79.17|4.80|12.82|
|Mei9|audio_baseline|78.70|4.40|12.85|
|Mei9|selected_cycle|78.70|4.40|12.85|
|ABCS43|raw_statistics|87.98|2.10|7.06|
|ABCS43|no_audio_baseline|87.69|1.32|6.81|
|ABCS43|audio_baseline|87.79|1.33|6.77|
|ABCS43|selected_cycle|88.47|1.26|6.40|
|VibraVox59|raw_statistics|88.28|0.66|6.19|
|VibraVox59|no_audio_baseline|89.12|0.54|5.71|
|VibraVox59|audio_baseline|89.05|0.52|5.73|
|VibraVox59|selected_cycle|90.82|0.91|5.05|

## 当前结论

三套均未显示Audio蒸馏相对同预算纯ACC的明确增益。Mei校准选择不加入循环；ABCS选择25步循环编码、权重0.5，HTER下降0.37个百分点，但逐身份95%区间[-1.25,+0.53]跨零。VibraVox的循环选择虽使HTER下降0.69个百分点，区间[-1.84,+0.41]也跨零。

VibraVox需特别谨慎：Audio基线到循环补充，TAR为89.05%→90.82%，FAR为0.52%→0.91%，AUC为0.9985→0.9969，最差目标FAR仍为16.09%。这说明工作点有变化，不能将HTER改善直接等同为排序或安全性改善。ABCS补充后的最差目标FAR仍为12.10%，平均值不能代替逐用户评估。

本轮支持“已知身份、不同录制/内容可做认证”，尚不支持“Audio辅助稳定改善”或“循环生成增加身份信息”。公开数据只用文本的身份识别仍高于随机（ABCS4.46%对2.33%，VibraVox3.95%对1.69%），不能声称彻底排除内容分配混杂。

当前路线是统计残差编码器＋RKD Audio蒸馏＋个人CycleGAN＋RBF，并非历史完整772维SupCon基线。

## 解读规则

首先比较同预算纯ACC与Audio辅助；再比较Audio辅助与仅校准选择的CycleGAN。固定分支的查询表现可作诊断，不能替代预先锁定方案。小幅差异不代表稳定提升；逐用户置信区间不能替代多训练种子。闭集优于开放集也不能单归因于内容，因为身份训练覆盖和负类协议同时不同。

## 逐数据集完整结果

- [Mei9全部用户及方法](../../outputs/mei9_closed_cross3_20261110/report.md)
- [ABCS43全部用户及方法](../../outputs/abcs43_closed_cross_20261110/report.md)
- [VibraVox59全部用户及方法](../../outputs/vibravox59_closed_cross_20261110/report.md)
