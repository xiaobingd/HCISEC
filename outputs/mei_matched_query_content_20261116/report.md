# 固定相同查询的同词／跨词认证对照

九人训练模型冻结；统计除U5外八名目标，保留U5攻击。K4注册＋K4查询，两组录制交换，共六词30个有向异词组合。每个组合的同词与跨词条件读取完全相同的查询和标签，只改变注册词。

注册每词只有约10次重复，因此本轮K4，不与之前K12的绝对指标直接比较。固定Cal词21–23，共享同词/跨词阈值；最终六词24–29，不使用最终结果调参。

|方法|条件|TAR%|FAR%|HTER%|EER%|AUC|
|---|---|---:|---:|---:|---:|---:|
|raw_rbf|sameword|93.75|13.25|9.75|6.49|0.9659|
|raw_rbf|crossword|88.85|14.15|12.65|9.75|0.9494|
|local|sameword|92.71|12.70|9.99|9.20|0.9568|
|local|crossword|84.43|12.42|13.99|11.56|0.9378|
|audio|sameword|92.19|12.73|10.27|9.33|0.9549|
|audio|crossword|84.17|12.25|14.04|11.61|0.9376|

## 查询词分解

|方法|查询词|条件|HTER%|TAR%|FAR%|
|---|---|---|---:|---:|---:|
|raw_rbf|Down|sameword|11.13|90.62|12.89|
|raw_rbf|Up|sameword|10.35|92.19|12.89|
|raw_rbf|Zoom|sameword|7.71|96.88|12.30|
|raw_rbf|Record|sameword|6.93|98.44|12.30|
|raw_rbf|Play|sameword|10.45|95.31|16.21|
|raw_rbf|Reset|sameword|11.91|89.06|12.89|
|raw_rbf|Down|crossword|12.27|89.69|14.22|
|raw_rbf|Up|crossword|14.06|86.88|15.00|
|raw_rbf|Zoom|crossword|11.56|90.31|13.44|
|raw_rbf|Record|crossword|12.17|89.38|13.71|
|raw_rbf|Play|crossword|11.04|90.94|13.01|
|raw_rbf|Reset|crossword|14.80|85.94|15.55|
|local|Down|sameword|9.57|93.75|12.89|
|local|Up|sameword|11.52|90.62|13.67|
|local|Zoom|sameword|8.89|93.75|11.52|
|local|Record|sameword|8.59|93.75|10.94|
|local|Play|sameword|9.96|92.19|12.11|
|local|Reset|sameword|11.43|92.19|15.04|
|local|Down|crossword|13.14|85.31|11.60|
|local|Up|crossword|16.66|79.38|12.70|
|local|Zoom|crossword|14.06|84.38|12.50|
|local|Record|crossword|13.44|84.69|11.56|
|local|Play|crossword|12.48|87.50|12.46|
|local|Reset|crossword|14.18|85.31|13.67|
|audio|Down|sameword|9.47|93.75|12.70|
|audio|Up|sameword|12.50|89.06|14.06|
|audio|Zoom|sameword|9.08|93.75|11.91|
|audio|Record|sameword|8.50|93.75|10.74|
|audio|Play|sameword|10.06|92.19|12.30|
|audio|Reset|sameword|12.01|90.62|14.65|
|audio|Down|crossword|13.50|84.38|11.37|
|audio|Up|crossword|16.86|79.06|12.77|
|audio|Zoom|crossword|13.98|84.38|12.34|
|audio|Record|crossword|13.50|84.69|11.68|
|audio|Play|crossword|12.29|87.50|12.07|
|audio|Reset|crossword|14.12|85.00|13.24|

## 原两组三词交换方向

|方法|方向|条件|HTER%|TAR%|FAR%|
|---|---|---|---:|---:|---:|
|raw_rbf|A_to_B|sameword|9.77|94.27|13.80|
|raw_rbf|B_to_A|sameword|9.73|93.23|12.70|
|raw_rbf|A_to_B|crossword|12.79|88.02|13.61|
|raw_rbf|B_to_A|crossword|13.49|87.85|14.82|
|local|A_to_B|sameword|9.99|92.71|12.70|
|local|B_to_A|sameword|9.99|92.71|12.70|
|local|A_to_B|crossword|14.52|83.51|12.54|
|local|B_to_A|crossword|14.58|82.99|12.15|
|audio|A_to_B|sameword|10.19|92.19|12.57|
|audio|B_to_A|sameword|10.35|92.19|12.89|
|audio|A_to_B|crossword|14.39|83.51|12.28|
|audio|B_to_A|crossword|14.78|82.47|12.02|

A=Down/Up/Zoom，B=Record/Play/Reset。A_to_B中跨词注册A、查询B，同词条件也查询相同B录制；B_to_A相反。词名是名义顺序映射，未独立转写核验。

## 逐用户

|用户|方法|条件|HTER%|TAR%|FAR%|AUC|
|---|---|---|---:|---:|---:|---:|
|U1|audio|crossword|5.78|91.67|3.23|0.9854|
|U1|audio|sameword|3.78|95.83|3.39|0.9889|
|U1|local|crossword|5.36|92.92|3.65|0.9844|
|U1|local|sameword|3.91|95.83|3.65|0.9876|
|U1|raw_rbf|crossword|8.36|87.50|4.22|0.9820|
|U1|raw_rbf|sameword|3.78|95.83|3.39|0.9896|
|U2|audio|crossword|7.45|91.67|6.56|0.9740|
|U2|audio|sameword|5.99|93.75|5.73|0.9870|
|U2|local|crossword|7.40|92.08|6.88|0.9746|
|U2|local|sameword|5.08|95.83|5.99|0.9896|
|U2|raw_rbf|crossword|4.69|98.75|8.12|0.9896|
|U2|raw_rbf|sameword|4.04|97.92|5.99|0.9974|
|U3|audio|crossword|10.94|87.92|9.79|0.9594|
|U3|audio|sameword|6.38|97.92|10.68|0.9870|
|U3|local|crossword|10.96|87.92|9.84|0.9605|
|U3|local|sameword|5.60|97.92|9.11|0.9889|
|U3|raw_rbf|crossword|10.21|96.25|16.67|0.9719|
|U3|raw_rbf|sameword|8.07|100.00|16.15|0.9902|
|U4|audio|crossword|22.06|70.00|14.11|0.8892|
|U4|audio|sameword|21.22|75.00|17.45|0.8639|
|U4|local|crossword|21.22|71.25|13.70|0.8930|
|U4|local|sameword|20.70|75.00|16.41|0.8685|
|U4|raw_rbf|crossword|17.55|82.92|18.02|0.9069|
|U4|raw_rbf|sameword|19.66|79.17|18.49|0.8802|
|U6|audio|crossword|20.76|76.25|17.76|0.8909|
|U6|audio|sameword|17.97|83.33|19.27|0.9232|
|U6|local|crossword|21.35|75.83|18.54|0.8885|
|U6|local|sameword|17.19|85.42|19.79|0.9264|
|U6|raw_rbf|crossword|16.02|87.08|19.11|0.9154|
|U6|raw_rbf|sameword|11.85|91.67|15.36|0.9512|
|U7|audio|crossword|15.29|91.25|21.82|0.9264|
|U7|audio|sameword|11.20|97.92|20.31|0.9551|
|U7|local|crossword|15.42|90.83|21.67|0.9268|
|U7|local|sameword|10.94|97.92|19.79|0.9570|
|U7|raw_rbf|crossword|16.61|89.58|22.81|0.9227|
|U7|raw_rbf|sameword|14.06|95.83|23.96|0.9688|
|U8|audio|crossword|9.51|94.58|13.59|0.9617|
|U8|audio|sameword|9.38|95.83|14.58|0.9551|
|U8|local|crossword|9.97|93.75|13.70|0.9620|
|U8|local|sameword|9.77|95.83|15.36|0.9590|
|U8|raw_rbf|crossword|9.82|91.25|10.89|0.9733|
|U8|raw_rbf|sameword|7.81|93.75|9.38|0.9740|
|U9|audio|crossword|20.55|70.00|11.09|0.9135|
|U9|audio|sameword|6.25|97.92|10.42|0.9792|
|U9|local|crossword|20.26|70.83|11.35|0.9125|
|U9|local|sameword|6.77|97.92|11.46|0.9772|
|U9|raw_rbf|crossword|17.94|77.50|13.39|0.9336|
|U9|raw_rbf|sameword|8.72|95.83|13.28|0.9759|

主问题看原始ACC＋RBF。local/audio为已有冻结验证器的附加诊断；它们原先按K12训练，本轮K4没有重训。阈值从同等权重的Cal同词/跨词分数拟合，每目标/方法/折固定共用，不是每个最终词单独调阈值。

核查PASS：2880工作点，保存分数重算误差0；注册/查询折内互斥、Cal与最终互斥。同一查询在五个异词组合中重复，组合不是独立试次；两折交换录制也不产生新会话。

本轮控制的是固定查询条件下的注册内容变化，注册录制随条件改变属于该操作的一部分。没有新的Audio提取、生成模型训练或教师微调。原始波形、特征、分数、阈值及模型保持私有。
