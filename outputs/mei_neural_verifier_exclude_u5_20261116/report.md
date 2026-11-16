# Mei：8名注册者的Audio决策蒸馏与跨词神经验证器结果

本轮为8名已知注册身份、单seed42、独立ACC认证方法，不使用CycleGAN。注册/查询均仅ACC；Audio教师为外部冻结ReDimNet2-B3，只使用合法训练词特征。词编号按历史顺序推断，尚未逐条语义核验。

最终每人12条注册，同词/跨词各12条本人及96条攻击查询；保留全部八个攻击来源。训练词0–17，开发18–20，校准21–23，注册/同词24–26，跨词27–29。新协议不能与旧12/8/16/24协议的绝对数值直接比较。

所有神经配置相同1000步预算，开发宏AUC选检查点。每人阈值由Cal确定，并在同词和跨词测试之间保持一致。EER由测试分数作描述性诊断，HTER/FAR/FRR使用锁定阈值。

|方法|条件|TAR%|FAR%|FRR%|HTER%|EER%|AUC|
|---|---|---:|---:|---:|---:|---:|---:|
|audio|sameword|90.62|10.55|9.38|9.96|9.83|0.9652|
|audio|crossword|88.54|8.85|11.46|10.16|9.18|0.9703|
|global|sameword|88.54|11.07|11.46|11.26|10.87|0.9494|
|global|crossword|89.58|10.94|10.42|10.68|7.55|0.9616|
|local|sameword|89.58|10.03|10.42|10.22|9.31|0.9660|
|local|crossword|89.58|8.59|10.42|9.51|8.33|0.9705|
|raw_pair_score|sameword|88.54|11.33|11.46|11.39|10.87|0.9493|
|raw_pair_score|crossword|89.58|10.94|10.42|10.68|7.62|0.9614|
|raw_rbf|sameword|95.83|10.94|4.17|7.55|6.25|0.9786|
|raw_rbf|crossword|95.83|10.55|4.17|7.36|6.05|0.9842|
|shuffle|sameword|90.62|10.81|9.38|10.09|10.09|0.9638|
|shuffle|crossword|90.62|9.38|9.37|9.38|8.46|0.9694|

## 相同八名目标：九人训练与排除U5训练

两边只统计相同八名目标，攻击查询均仍含U5。此表避免仅删除U5目标就改变宏平均；训练任务与Cal具体采样随用户集合变化，不等于严格固定批次的单变量实验。

|方法|条件|旧九人训练HTER%|新八人训练HTER%|差值(pp)|
|---|---|---:|---:|---:|
|audio|sameword|7.75|9.96|+2.21|
|audio|crossword|8.92|10.16|+1.24|
|global|sameword|10.42|11.26|+0.85|
|global|crossword|9.31|10.68|+1.37|
|local|sameword|7.62|10.22|+2.60|
|local|crossword|9.05|9.51|+0.46|
|raw_pair_score|sameword|10.55|11.39|+0.85|
|raw_pair_score|crossword|9.31|10.68|+1.37|
|raw_rbf|sameword|8.07|7.55|-0.52|
|raw_rbf|crossword|6.45|7.36|+0.91|
|shuffle|sameword|7.94|10.09|+2.15|
|shuffle|crossword|8.40|9.38|+0.98|

## 成对比较

|前者减后者|条件|HTER变化(pp)|FAR变化(pp)|TAR变化(pp)|HTER改善用户数|
|---|---|---:|---:|---:|---:|
|local − global|sameword|-1.04|-1.04|+1.04|5/8|
|local − global|crossword|-1.17|-2.34|+0.00|5/8|
|audio − local|sameword|-0.26|+0.52|+1.04|1/8|
|audio − local|crossword|+0.65|+0.26|-1.04|2/8|
|shuffle − audio|sameword|+0.13|+0.26|+0.00|1/8|
|shuffle − audio|crossword|-0.78|+0.52|+2.08|4/8|

## 训练排除用户诊断

排除身份编号[5]不进入编码器、Scaler、教师评分器、神经验证器训练、开发或Cal，也不建立该身份注册目标。最终保留其查询为未见攻击；这是已知目标对未见攻击者的验证，不是未见目标注册实验。

教师仅复用外部冻结主干的逐录制独立特征，训练数组中排除用户的所有特征置零；未复用九人ACC编码器、标准化或验证器。

|方法|条件|仅七名已知攻击FAR%|含本人FRR的限定八人HTER%|U5未见攻击FAR%|
|---|---|---:|---:|---:|
|audio|sameword|10.27|9.82|12.50|
|audio|crossword|9.37|10.42|5.21|
|global|sameword|11.31|11.38|9.38|
|global|crossword|11.76|11.09|5.21|
|local|sameword|9.97|10.19|10.42|
|local|crossword|8.93|9.67|6.25|
|raw_pair_score|sameword|11.61|11.53|9.38|
|raw_pair_score|crossword|11.76|11.09|5.21|
|raw_rbf|sameword|9.67|6.92|19.79|
|raw_rbf|crossword|10.42|7.29|11.46|
|shuffle|sameword|10.57|9.97|12.50|
|shuffle|crossword|9.97|9.67|5.21|

|用户|条件|方法|已知攻击FAR%|U5攻击FAR%|
|---|---|---|---:|---:|
|U1|crossword|audio|2.38|0.00|
|U1|crossword|global|5.95|0.00|
|U1|crossword|local|3.57|0.00|
|U1|crossword|raw_pair_score|5.95|0.00|
|U1|crossword|raw_rbf|4.76|0.00|
|U1|crossword|shuffle|3.57|0.00|
|U1|sameword|audio|0.00|0.00|
|U1|sameword|global|1.19|0.00|
|U1|sameword|local|0.00|0.00|
|U1|sameword|raw_pair_score|1.19|0.00|
|U1|sameword|raw_rbf|1.19|0.00|
|U1|sameword|shuffle|0.00|0.00|
|U2|crossword|audio|2.38|0.00|
|U2|crossword|global|2.38|0.00|
|U2|crossword|local|2.38|0.00|
|U2|crossword|raw_pair_score|2.38|0.00|
|U2|crossword|raw_rbf|3.57|0.00|
|U2|crossword|shuffle|2.38|0.00|
|U2|sameword|audio|1.19|0.00|
|U2|sameword|global|1.19|0.00|
|U2|sameword|local|1.19|0.00|
|U2|sameword|raw_pair_score|1.19|0.00|
|U2|sameword|raw_rbf|4.76|0.00|
|U2|sameword|shuffle|1.19|0.00|
|U3|crossword|audio|10.71|0.00|
|U3|crossword|global|9.52|0.00|
|U3|crossword|local|10.71|0.00|
|U3|crossword|raw_pair_score|9.52|0.00|
|U3|crossword|raw_rbf|5.95|0.00|
|U3|crossword|shuffle|9.52|0.00|
|U3|sameword|audio|14.29|0.00|
|U3|sameword|global|13.10|0.00|
|U3|sameword|local|14.29|0.00|
|U3|sameword|raw_pair_score|13.10|0.00|
|U3|sameword|raw_rbf|5.95|0.00|
|U3|sameword|shuffle|14.29|0.00|
|U4|crossword|audio|7.14|8.33|
|U4|crossword|global|7.14|8.33|
|U4|crossword|local|4.76|8.33|
|U4|crossword|raw_pair_score|7.14|8.33|
|U4|crossword|raw_rbf|11.90|8.33|
|U4|crossword|shuffle|11.90|8.33|
|U4|sameword|audio|16.67|33.33|
|U4|sameword|global|10.71|16.67|
|U4|sameword|local|15.48|16.67|
|U4|sameword|raw_pair_score|10.71|16.67|
|U4|sameword|raw_rbf|11.90|33.33|
|U4|sameword|shuffle|17.86|33.33|
|U6|crossword|audio|13.10|33.33|
|U6|crossword|global|20.24|33.33|
|U6|crossword|local|11.90|33.33|
|U6|crossword|raw_pair_score|20.24|33.33|
|U6|crossword|raw_rbf|10.71|33.33|
|U6|crossword|shuffle|14.29|33.33|
|U6|sameword|audio|9.52|41.67|
|U6|sameword|global|21.43|41.67|
|U6|sameword|local|8.33|41.67|
|U6|sameword|raw_pair_score|23.81|41.67|
|U6|sameword|raw_rbf|8.33|58.33|
|U6|sameword|shuffle|10.71|41.67|
|U7|crossword|audio|25.00|0.00|
|U7|crossword|global|27.38|0.00|
|U7|crossword|local|25.00|0.00|
|U7|crossword|raw_pair_score|27.38|0.00|
|U7|crossword|raw_rbf|19.05|0.00|
|U7|crossword|shuffle|25.00|0.00|
|U7|sameword|audio|23.81|0.00|
|U7|sameword|global|25.00|0.00|
|U7|sameword|local|23.81|0.00|
|U7|sameword|raw_pair_score|25.00|0.00|
|U7|sameword|raw_rbf|16.67|0.00|
|U7|sameword|shuffle|23.81|0.00|
|U8|crossword|audio|5.95|0.00|
|U8|crossword|global|9.52|0.00|
|U8|crossword|local|5.95|8.33|
|U8|crossword|raw_pair_score|9.52|0.00|
|U8|crossword|raw_rbf|5.95|8.33|
|U8|crossword|shuffle|5.95|0.00|
|U8|sameword|audio|9.52|16.67|
|U8|sameword|global|5.95|8.33|
|U8|sameword|local|9.52|16.67|
|U8|sameword|raw_pair_score|5.95|8.33|
|U8|sameword|raw_rbf|4.76|8.33|
|U8|sameword|shuffle|8.33|16.67|
|U9|crossword|audio|8.33|0.00|
|U9|crossword|global|11.90|0.00|
|U9|crossword|local|7.14|0.00|
|U9|crossword|raw_pair_score|11.90|0.00|
|U9|crossword|raw_rbf|21.43|41.67|
|U9|crossword|shuffle|7.14|0.00|
|U9|sameword|audio|7.14|8.33|
|U9|sameword|global|11.90|8.33|
|U9|sameword|local|7.14|8.33|
|U9|sameword|raw_pair_score|11.90|8.33|
|U9|sameword|raw_rbf|23.81|58.33|
|U9|sameword|shuffle|8.33|8.33|

## 逐用户完整工作点

|用户|条件|方法|TAR%|FAR%|FRR%|HTER%|EER%|AUC|
|---|---|---|---:|---:|---:|---:|---:|---:|
|U1|crossword|audio|91.67|2.08|8.33|5.21|7.81|0.9887|
|U1|crossword|global|91.67|5.21|8.33|6.77|7.29|0.9844|
|U1|crossword|local|91.67|3.12|8.33|5.73|7.81|0.9887|
|U1|crossword|raw_pair_score|91.67|5.21|8.33|6.77|7.29|0.9844|
|U1|crossword|raw_rbf|91.67|4.17|8.33|6.25|2.08|0.9931|
|U1|crossword|shuffle|91.67|3.12|8.33|5.73|7.81|0.9878|
|U1|sameword|audio|83.33|0.00|16.67|8.33|6.77|0.9948|
|U1|sameword|global|83.33|1.04|16.67|8.85|6.77|0.9948|
|U1|sameword|local|83.33|0.00|16.67|8.33|1.56|0.9974|
|U1|sameword|raw_pair_score|83.33|1.04|16.67|8.85|6.77|0.9948|
|U1|sameword|raw_rbf|91.67|1.04|8.33|4.69|0.52|0.9991|
|U1|sameword|shuffle|83.33|0.00|16.67|8.33|6.77|0.9948|
|U2|crossword|audio|91.67|2.08|8.33|5.21|1.04|0.9983|
|U2|crossword|global|100.00|2.08|0.00|1.04|0.00|1.0000|
|U2|crossword|local|100.00|2.08|0.00|1.04|1.04|0.9983|
|U2|crossword|raw_pair_score|100.00|2.08|0.00|1.04|0.00|1.0000|
|U2|crossword|raw_rbf|100.00|3.12|0.00|1.56|0.00|1.0000|
|U2|crossword|shuffle|100.00|2.08|0.00|1.04|0.52|0.9991|
|U2|sameword|audio|100.00|1.04|0.00|0.52|0.52|0.9939|
|U2|sameword|global|100.00|1.04|0.00|0.52|0.52|0.9939|
|U2|sameword|local|100.00|1.04|0.00|0.52|0.52|0.9931|
|U2|sameword|raw_pair_score|100.00|1.04|0.00|0.52|0.52|0.9939|
|U2|sameword|raw_rbf|100.00|4.17|0.00|2.08|0.52|0.9991|
|U2|sameword|shuffle|100.00|1.04|0.00|0.52|0.52|0.9931|
|U3|crossword|audio|100.00|9.38|0.00|4.69|6.77|0.9905|
|U3|crossword|global|100.00|8.33|0.00|4.17|1.56|0.9878|
|U3|crossword|local|100.00|9.38|0.00|4.69|2.08|0.9922|
|U3|crossword|raw_pair_score|100.00|8.33|0.00|4.17|1.56|0.9878|
|U3|crossword|raw_rbf|100.00|5.21|0.00|2.60|1.04|0.9974|
|U3|crossword|shuffle|100.00|8.33|0.00|4.17|2.08|0.9887|
|U3|sameword|audio|100.00|12.50|0.00|6.25|7.81|0.9748|
|U3|sameword|global|100.00|11.46|0.00|5.73|8.33|0.9618|
|U3|sameword|local|100.00|12.50|0.00|6.25|7.29|0.9774|
|U3|sameword|raw_pair_score|100.00|11.46|0.00|5.73|8.33|0.9618|
|U3|sameword|raw_rbf|91.67|5.21|8.33|6.77|7.29|0.9852|
|U3|sameword|shuffle|100.00|12.50|0.00|6.25|7.81|0.9740|
|U4|crossword|audio|83.33|7.29|16.67|11.98|9.90|0.9635|
|U4|crossword|global|75.00|7.29|25.00|16.15|8.85|0.9575|
|U4|crossword|local|83.33|5.21|16.67|10.94|8.33|0.9731|
|U4|crossword|raw_pair_score|75.00|7.29|25.00|16.15|8.85|0.9566|
|U4|crossword|raw_rbf|83.33|11.46|16.67|14.06|14.58|0.9635|
|U4|crossword|shuffle|91.67|11.46|8.33|9.90|9.90|0.9627|
|U4|sameword|audio|91.67|18.75|8.33|13.54|16.15|0.9227|
|U4|sameword|global|66.67|11.46|33.33|22.40|16.67|0.9210|
|U4|sameword|local|83.33|15.62|16.67|16.15|16.67|0.9280|
|U4|sameword|raw_pair_score|66.67|11.46|33.33|22.40|16.67|0.9210|
|U4|sameword|raw_rbf|91.67|14.58|8.33|11.46|14.58|0.9488|
|U4|sameword|shuffle|91.67|19.79|8.33|14.06|16.67|0.9193|
|U6|crossword|audio|91.67|15.62|8.33|11.98|8.33|0.9653|
|U6|crossword|global|100.00|21.88|0.00|10.94|8.33|0.9583|
|U6|crossword|local|91.67|14.58|8.33|11.46|8.33|0.9679|
|U6|crossword|raw_pair_score|100.00|21.88|0.00|10.94|8.33|0.9575|
|U6|crossword|raw_rbf|91.67|13.54|8.33|10.94|8.33|0.9679|
|U6|crossword|shuffle|91.67|16.67|8.33|12.50|8.33|0.9601|
|U6|sameword|audio|83.33|13.54|16.67|15.10|15.10|0.9479|
|U6|sameword|global|83.33|23.96|16.67|20.31|16.67|0.8984|
|U6|sameword|local|83.33|12.50|16.67|14.58|15.10|0.9540|
|U6|sameword|raw_pair_score|83.33|26.04|16.67|21.35|16.67|0.8984|
|U6|sameword|raw_rbf|91.67|14.58|8.33|11.46|9.38|0.9470|
|U6|sameword|shuffle|83.33|14.58|16.67|15.62|16.15|0.9436|
|U7|crossword|audio|83.33|21.88|16.67|19.27|16.67|0.9245|
|U7|crossword|global|83.33|23.96|16.67|20.31|16.67|0.8958|
|U7|crossword|local|83.33|21.88|16.67|19.27|16.67|0.9141|
|U7|crossword|raw_pair_score|83.33|23.96|16.67|20.31|17.19|0.8950|
|U7|crossword|raw_rbf|100.00|16.67|0.00|8.33|14.58|0.9653|
|U7|crossword|shuffle|83.33|21.88|16.67|19.27|16.67|0.9262|
|U7|sameword|audio|100.00|20.83|0.00|10.42|14.58|0.9601|
|U7|sameword|global|100.00|21.88|0.00|10.94|15.10|0.9106|
|U7|sameword|local|100.00|20.83|0.00|10.42|14.58|0.9583|
|U7|sameword|raw_pair_score|100.00|21.88|0.00|10.94|15.10|0.9097|
|U7|sameword|raw_rbf|100.00|14.58|0.00|7.29|7.29|0.9844|
|U7|sameword|shuffle|100.00|20.83|0.00|10.42|14.58|0.9592|
|U8|crossword|audio|100.00|5.21|0.00|2.60|6.77|0.9878|
|U8|crossword|global|100.00|8.33|0.00|4.17|1.04|0.9965|
|U8|crossword|local|100.00|6.25|0.00|3.12|7.29|0.9844|
|U8|crossword|raw_pair_score|100.00|8.33|0.00|4.17|1.04|0.9965|
|U8|crossword|raw_rbf|100.00|6.25|0.00|3.12|0.00|1.0000|
|U8|crossword|shuffle|100.00|5.21|0.00|2.60|6.77|0.9878|
|U8|sameword|audio|83.33|10.42|16.67|13.54|9.38|0.9609|
|U8|sameword|global|91.67|6.25|8.33|7.29|8.33|0.9748|
|U8|sameword|local|83.33|10.42|16.67|13.54|9.90|0.9583|
|U8|sameword|raw_pair_score|91.67|6.25|8.33|7.29|8.33|0.9748|
|U8|sameword|raw_rbf|100.00|5.21|0.00|2.60|2.08|0.9931|
|U8|sameword|shuffle|83.33|9.38|16.67|13.02|9.38|0.9609|
|U9|crossword|audio|66.67|7.29|33.33|20.31|16.15|0.9436|
|U9|crossword|global|66.67|10.42|33.33|21.88|16.67|0.9123|
|U9|crossword|local|66.67|6.25|33.33|19.79|15.10|0.9453|
|U9|crossword|raw_pair_score|66.67|10.42|33.33|21.88|16.67|0.9132|
|U9|crossword|raw_rbf|100.00|23.96|0.00|11.98|7.81|0.9861|
|U9|crossword|shuffle|66.67|6.25|33.33|19.79|15.62|0.9427|
|U9|sameword|audio|83.33|7.29|16.67|11.98|8.33|0.9661|
|U9|sameword|global|83.33|11.46|16.67|14.06|14.58|0.9401|
|U9|sameword|local|83.33|7.29|16.67|11.98|8.85|0.9618|
|U9|sameword|raw_pair_score|83.33|11.46|16.67|14.06|14.58|0.9401|
|U9|sameword|raw_rbf|100.00|28.12|0.00|14.06|8.33|0.9722|
|U9|sameword|shuffle|83.33|8.33|16.67|12.50|8.85|0.9653|

## 模型选择与核查

|方法|开发选择步数|开发宏AUC|
|---|---:|---:|
|global|1000|0.9419|
|local|1000|0.9615|
|audio|1000|0.9602|
|shuffle|1000|0.9587|

独立重载编码器、四个验证器和两种原始统计后端，重建合法Cal阈值与全部96个工作点。核查结论：PASS；最大指标误差0。

教师评分器按训练查询词三折交叉拟合，主干冻结；不是未见身份教师交叉验证。打乱对照保留身份，仅破坏具体词/录制配对。

raw_rbf在最终合法注册数据拟合，并用Cal查询校准；神经验证器在Cal内部模拟注册校准，校准支持构造不同。主要方法消融看global/local/audio/shuffle，不能把RBF与神经组差异全部归因于表示。

global仅使用三种全局分数；local增加注意力及更宽的局部输入，实际参与反传的参数更多。因此local对global是完整验证器结构比较，尚未通过等参数MLP控制单独归因于注意力。local/audio/shuffle结构与预算相同，可用于比较Audio监督。

本轮没有完成ECAPA替换、传统特征蒸馏同结构对照或多种子验证；不将强教师本身的识别能力等同于ACC增益。历史同会话数据已探索，未验证重戴/跨天，词标签为名义映射。

协议：docs/experiments/2026-11-16-neural-verifier-protocol.md。私有数据、模型、身份映射、分数和阈值不公开。
