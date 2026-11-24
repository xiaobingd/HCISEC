# 低容量 Audio–ACC 共同身份子空间审计

## 核心问题与范围

不计算最终认证 EER/HTER，不做分数融合。本轮只检查是否存在能够在映射未见身份上保持正确对应的低维共同结构。仅使用 16 个 fit 词的 1280 条 ECAPA Audio 与纯 ACC embedding，所有原始编码器、历史模型和创新点一不变。

10 位用户、3 个 ACC 种子、45 组两身份留出，共 135 个外层身份折。每折只用其余 8 位身份训练映射；内部逐个留出 1 位，用 7 人训练、1 人开发，选择维度与正则。两种词方向均检查。重复录制首先汇成每人原型，映射训练有效身份数是 8，最大中心化秩是 7，不是上千个独立身份。

## 方法与固定协议

- 正则化 CCA：训练身份均值中心化、按非零特征值均值缩放 ridge 的协方差白化，再做跨协方差 SVD；无交叉关系的零空间丢弃。
- 低秩线性映射：ridge ACC→Audio 回归，对预测 Audio 截断 SVD；Audio 与 ACC 映射到同一低维坐标后计算余弦。
- 候选维度 1/2/4，ridge 0.1/1/10。只根据 8 个内部身份开发折的平均匹配 AUC 选择；开发 AUC 统一保留 12 位小数消除浮点尾数，再按同分优先低维、更强正则。
- 正确映射与身份错配映射使用相同维度和正则，错配组在外层训练 8 人内部做无固定点身份置换后重新拟合；不把外层身份引入训练。
- 原 ACC、原 Audio 各自的跨词余弦作为可分性参照。Audio-only 只使用 fit 词缓存，不代表部署时使用查询 Audio。

12 个映射训练词分为前 6/后 6，构建跨词身份原型；每模态每身份 48 条记录。其余 4 词分为前 2/后 2，每模态每身份 16 条，作为映射保留词。保留词在内部开发身份上参与选超参，因此不是对超参完全盲的词；外层两身份在全部选择中隔离。原 ACC 主干此前见过全部 10 人和 16 词，所以留出只针对线性映射，不是整套系统的开放集。

主要匹配任务只在外层两位未见身份中选择（机会 50%）；同时报告十人候选（含 8 位映射见过的身份）的 Top1（机会 10%）。另把每词 8 遍拆为偶数 4 遍与奇数 4 遍检查重复稳定性。1 维余弦可能大量并列，所有 Top1 与 AUC 都按并列均分处理。

## 留出身份匹配结果

全重复原型；数值为十目标宏平均。这里的 AUC 是原型匹配 AUC，不是最终认证 AUC。

|方法|词范围|两人 Top1%|两人匹配 AUC|十人 Top1%|本人−最难冒充原型间隔|
|---|---|---:|---:|---:|---:|
|raw_ACC|fit|100.00|1.0000|100.00|+1.0330|
|raw_ACC|newwords|100.00|1.0000|100.00|+0.9834|
|raw_Audio|fit|100.00|1.0000|100.00|+0.4602|
|raw_Audio|newwords|97.78|0.9778|85.00|+0.2360|
|rcca_correct|fit|50.37|0.5037|7.56|+0.0445|
|rcca_correct|newwords|54.81|0.5481|7.04|+0.0393|
|rcca_identity_wrong|fit|49.44|0.4944|8.70|-0.0325|
|rcca_identity_wrong|newwords|48.52|0.4852|8.17|-0.0140|
|lowrank_ridge_correct|fit|47.59|0.4759|8.04|-0.0283|
|lowrank_ridge_correct|newwords|51.67|0.5167|7.71|+0.0112|
|lowrank_ridge_identity_wrong|fit|52.59|0.5259|9.18|+0.0018|
|lowrank_ridge_identity_wrong|newwords|50.00|0.5000|7.64|+0.0381|

## 用户簇不确定性

先将每个用户在不同搭档、方向、种子上的表现平均，再重采样十个用户，未把 45 对身份、多个词和录制当独立受试者。身份搭档仍有共享依赖，区间用于探索性描述。

|方法|词范围|Top1 95%区间|正确−错配/百分点|差异95%区间/百分点|
|---|---|---|---:|---|
|rcca|fit|[38.33%, 62.41%]|+0.93|[-13.70, +15.37]|
|rcca|newwords|[46.11%, 62.87%]|+6.30|[-4.81, +16.85]|
|lowrank_ridge|fit|[34.54%, 59.63%]|-5.00|[-20.00, +9.35]|
|lowrank_ridge|newwords|[41.48%, 61.11%]|+1.67|[-11.11, +13.15]|

## 跨重复稳定性

|方法|词范围|偶数重复 Top1%|奇数重复 Top1%|偶/奇身份关系 Spearman|
|---|---|---:|---:|---:|
|rcca_correct|fit|48.89|51.48|0.9493|
|rcca_correct|newwords|52.22|57.59|0.8504|
|rcca_identity_wrong|fit|49.81|47.78|0.9266|
|rcca_identity_wrong|newwords|48.89|48.70|0.8654|
|lowrank_ridge_correct|fit|45.19|48.15|0.9457|
|lowrank_ridge_correct|newwords|51.67|53.52|0.8771|
|lowrank_ridge_identity_wrong|fit|51.48|52.78|0.9140|
|lowrank_ridge_identity_wrong|newwords|51.11|49.44|0.8728|

重复关系稳定本身不证明身份对应正确，错误映射也可能稳定；必须同时比较正确身份匹配与错配控制。常数关系的 Spearman 无定义，脚本将其记作 0，不作为相关证据。

## 超参数选择

- rcca：`{"d=2,ridge=1.0": 14, "d=4,ridge=0.1": 58, "d=1,ridge=1.0": 26, "d=4,ridge=10.0": 49, "d=2,ridge=10.0": 17, "d=1,ridge=10.0": 24, "d=2,ridge=0.1": 30, "d=1,ridge=0.1": 22, "d=4,ridge=1.0": 30}`。
- lowrank_ridge：`{"d=2,ridge=10.0": 26, "d=1,ridge=10.0": 38, "d=1,ridge=1.0": 20, "d=4,ridge=0.1": 58, "d=2,ridge=0.1": 28, "d=4,ridge=10.0": 38, "d=4,ridge=1.0": 31, "d=2,ridge=1.0": 18, "d=1,ridge=0.1": 13}`。

## 逐用户结果（映射保留词、全部重复）

|用户|方法|两人 Top1%|十人 Top1%|
|---|---|---:|---:|
|U01|raw_ACC|100.00|100.00|
|U01|raw_Audio|94.44|50.00|
|U01|rcca_correct|67.59|7.94|
|U01|rcca_identity_wrong|44.44|8.23|
|U01|lowrank_ridge_correct|64.81|15.13|
|U01|lowrank_ridge_identity_wrong|48.15|8.23|
|U02|raw_ACC|100.00|100.00|
|U02|raw_Audio|94.44|50.00|
|U02|rcca_correct|42.59|0.68|
|U02|rcca_identity_wrong|41.67|9.24|
|U02|lowrank_ridge_correct|43.52|3.04|
|U02|lowrank_ridge_identity_wrong|50.00|6.58|
|U03|raw_ACC|100.00|100.00|
|U03|raw_Audio|100.00|100.00|
|U03|rcca_correct|62.04|5.34|
|U03|rcca_identity_wrong|51.85|6.65|
|U03|lowrank_ridge_correct|58.33|10.71|
|U03|lowrank_ridge_identity_wrong|51.85|9.54|
|U04|raw_ACC|100.00|100.00|
|U04|raw_Audio|100.00|100.00|
|U04|rcca_correct|31.48|0.26|
|U04|rcca_identity_wrong|58.33|14.51|
|U04|lowrank_ridge_correct|20.37|0.00|
|U04|lowrank_ridge_identity_wrong|60.19|10.57|
|U05|raw_ACC|100.00|100.00|
|U05|raw_Audio|100.00|100.00|
|U05|rcca_correct|72.22|6.12|
|U05|rcca_identity_wrong|48.15|3.34|
|U05|lowrank_ridge_correct|74.07|8.81|
|U05|lowrank_ridge_identity_wrong|48.15|4.17|
|U06|raw_ACC|100.00|100.00|
|U06|raw_Audio|88.89|50.00|
|U06|rcca_correct|61.11|15.46|
|U06|rcca_identity_wrong|52.78|13.75|
|U06|lowrank_ridge_correct|63.89|19.04|
|U06|lowrank_ridge_identity_wrong|47.22|11.10|
|U07|raw_ACC|100.00|100.00|
|U07|raw_Audio|100.00|100.00|
|U07|rcca_correct|36.11|3.70|
|U07|rcca_identity_wrong|52.78|8.36|
|U07|lowrank_ridge_correct|35.19|0.00|
|U07|lowrank_ridge_identity_wrong|48.15|4.26|
|U08|raw_ACC|100.00|100.00|
|U08|raw_Audio|100.00|100.00|
|U08|rcca_correct|63.89|12.75|
|U08|rcca_identity_wrong|46.30|5.48|
|U08|lowrank_ridge_correct|51.85|8.84|
|U08|lowrank_ridge_identity_wrong|48.15|8.21|
|U09|raw_ACC|100.00|100.00|
|U09|raw_Audio|100.00|100.00|
|U09|rcca_correct|64.81|6.64|
|U09|rcca_identity_wrong|36.11|4.61|
|U09|lowrank_ridge_correct|64.81|3.36|
|U09|lowrank_ridge_identity_wrong|42.59|3.94|
|U10|raw_ACC|100.00|100.00|
|U10|raw_Audio|100.00|100.00|
|U10|rcca_correct|46.30|11.52|
|U10|rcca_identity_wrong|52.78|7.48|
|U10|lowrank_ridge_correct|39.81|8.18|
|U10|lowrank_ridge_identity_wrong|55.56|9.83|

## 判断

当前未建立可跨身份推广的低维共同身份结构证据；不进入最终认证或扩大投影网络。该结论限于现有冻结表征与小样本线性协议，不等于原始信号不含共同信息。

这里寻找的是可迁移的统计子空间，未把某条轴解释为基频、组织传导或其他生理因素；这种物理解释需要独立测量。若仅在映射训练词上成立、保留词不成立，应先解决跨内容稳定性；若两者都不成立，不再通过增大网络将训练身份拟合率当作共享身份信息。

无论是否发现线性共同结构，本轮都不声称蒸馏有效或最终身份认证改善。当前数据规模只允许探索性判断；需要真正与编码器训练隔离的新身份确认。

## 校验与公开范围

合成测试先验证已知共享潜变量在真正留出身份上的恢复、身份错配负控制、训练集中心化以及并列处理。实际数据检查 sample_id、词 ID、重复编号及所有外层/内层身份隔离。公开脚本、协议、匿名逐用户汇总；不上传音频、embedding、身份映射、逐样本分数或私有线性投影矩阵。