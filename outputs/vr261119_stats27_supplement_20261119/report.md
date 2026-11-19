# 原始ACC统计特征补充对照

10用户40词，3200片段。逐条sample_id核对通过，复用既有注册、校准、同词与跨词划分。27维由三轴10–400 Hz八个频带对数能量及三轴对数标准差构成，沿用既有去均值、发声区间外补零前端。统计量标准化仅拟合注册数据，L2归一化后与312维等权拼接。balanced RBF SVM C=10，gamma=scale，阈值由校准集EER锁定。复用现有教师学生检查点。本轮为单次固定划分探索。

|方法|条件|EER %|HTER %|FAR %|FRR %|AUC|
|---|---|---:|---:|---:|---:|---:|
|stats27|sameword|11.25|15.79|9.91|21.67|0.9145|
|stats27|crossword|15.25|22.38|12.27|32.50|0.9022|
|baseline312|sameword|14.12|15.09|11.85|18.33|0.9291|
|baseline312|crossword|18.31|17.31|14.63|20.00|0.9001|
|baseline_plus_stats339|sameword|9.26|13.01|8.52|17.50|0.9538|
|baseline_plus_stats339|crossword|13.87|18.66|9.40|27.92|0.9248|

拼接改善同词与跨词EER。跨词HTER从17.31%升至18.66%，伴随FAR下降、FRR上升，说明校准阈值迁移仍有问题。相同切片下27维跨词EER优于312维，支持原始分带能量与幅度统计具有现有表示遗漏的身份信息。

## 逐用户

|用户|方法|条件|EER %|HTER %|FAR %|FRR %|AUC|
|---|---|---|---:|---:|---:|---:|---:|
|U1|stats27|sameword|8.80|12.96|9.26|16.67|0.9221|
|U1|stats27|crossword|14.58|16.90|12.96|20.83|0.9068|
|U1|baseline312|sameword|8.33|7.41|14.81|0.00|0.9614|
|U1|baseline312|crossword|12.73|14.35|20.37|8.33|0.9147|
|U1|baseline_plus_stats339|sameword|7.41|6.48|4.63|8.33|0.9807|
|U1|baseline_plus_stats339|crossword|14.12|17.13|13.43|20.83|0.9263|
|U2|stats27|sameword|16.67|39.81|12.96|66.67|0.8727|
|U2|stats27|crossword|16.20|27.31|8.80|45.83|0.8879|
|U2|baseline312|sameword|37.96|28.24|23.15|33.33|0.8071|
|U2|baseline312|crossword|19.91|19.68|22.69|16.67|0.9153|
|U2|baseline_plus_stats339|sameword|25.00|19.91|14.81|25.00|0.9059|
|U2|baseline_plus_stats339|crossword|16.20|19.44|9.72|29.17|0.9431|
|U3|stats27|sameword|2.78|5.56|11.11|0.00|0.9776|
|U3|stats27|crossword|17.36|17.36|18.06|16.67|0.9273|
|U3|baseline312|sameword|7.87|10.19|12.04|8.33|0.9792|
|U3|baseline312|crossword|16.20|14.35|16.20|12.50|0.9396|
|U3|baseline_plus_stats339|sameword|6.94|6.48|12.96|0.00|0.9907|
|U3|baseline_plus_stats339|crossword|16.20|12.96|17.59|8.33|0.9481|
|U4|stats27|sameword|1.39|3.24|6.48|0.00|0.9977|
|U4|stats27|crossword|12.27|20.14|6.94|33.33|0.9109|
|U4|baseline312|sameword|3.70|8.80|17.59|0.00|0.9877|
|U4|baseline312|crossword|17.36|17.82|18.98|16.67|0.8926|
|U4|baseline_plus_stats339|sameword|0.46|2.78|5.56|0.00|0.9992|
|U4|baseline_plus_stats339|crossword|11.57|10.88|5.09|16.67|0.9277|
|U5|stats27|sameword|33.33|35.65|4.63|66.67|0.7508|
|U5|stats27|crossword|15.97|13.19|9.72|16.67|0.9443|
|U5|baseline312|sameword|23.61|29.17|8.33|50.00|0.8796|
|U5|baseline312|crossword|20.60|24.54|7.41|41.67|0.8872|
|U5|baseline_plus_stats339|sameword|19.44|33.33|8.33|58.33|0.8773|
|U5|baseline_plus_stats339|crossword|12.96|18.75|4.17|33.33|0.9497|
|U6|stats27|sameword|0.00|8.33|0.00|16.67|1.0000|
|U6|stats27|crossword|3.47|39.81|0.46|79.17|0.9877|
|U6|baseline312|sameword|10.19|10.19|12.04|8.33|0.9900|
|U6|baseline312|crossword|15.74|13.66|14.81|12.50|0.9242|
|U6|baseline_plus_stats339|sameword|0.00|8.33|0.00|16.67|1.0000|
|U6|baseline_plus_stats339|crossword|4.17|31.48|0.46|62.50|0.9878|
|U7|stats27|sameword|8.80|6.48|12.96|0.00|0.9869|
|U7|stats27|crossword|4.63|9.49|18.98|0.00|0.9813|
|U7|baseline312|sameword|9.26|8.80|9.26|8.33|0.9823|
|U7|baseline312|crossword|8.10|7.64|11.11|4.17|0.9832|
|U7|baseline_plus_stats339|sameword|1.39|4.17|8.33|0.00|0.9946|
|U7|baseline_plus_stats339|crossword|3.70|6.02|12.04|0.00|0.9958|
|U8|stats27|sameword|31.94|33.33|33.33|33.33|0.6520|
|U8|stats27|crossword|30.09|32.64|36.11|29.17|0.6993|
|U8|baseline312|sameword|29.63|36.11|13.89|58.33|0.7361|
|U8|baseline312|crossword|47.45|36.11|13.89|58.33|0.6485|
|U8|baseline_plus_stats339|sameword|25.46|31.02|20.37|41.67|0.8071|
|U8|baseline_plus_stats339|crossword|31.48|34.49|18.98|50.00|0.7045|
|U9|stats27|sameword|1.39|4.17|8.33|0.00|0.9923|
|U9|stats27|crossword|12.50|16.20|7.41|25.00|0.9506|
|U9|baseline312|sameword|0.46|1.85|3.70|0.00|0.9961|
|U9|baseline312|crossword|7.87|4.63|9.26|0.00|0.9776|
|U9|baseline_plus_stats339|sameword|0.46|5.09|10.19|0.00|0.9977|
|U9|baseline_plus_stats339|crossword|7.41|8.33|12.50|4.17|0.9743|
|U10|stats27|sameword|7.41|8.33|0.00|16.67|0.9931|
|U10|stats27|crossword|25.46|30.79|3.24|58.33|0.8258|
|U10|baseline312|sameword|10.19|10.19|3.70|16.67|0.9715|
|U10|baseline312|crossword|17.13|20.37|11.57|29.17|0.9182|
|U10|baseline_plus_stats339|sameword|6.02|12.50|0.00|25.00|0.9846|
|U10|baseline_plus_stats339|crossword|20.83|27.08|0.00|54.17|0.8904|
