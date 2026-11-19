# 原始ACC适配编码器与个人CycleGAN

固定10人40词划分，12条注册、8条校准、12条同词和24条跨词查询每用户。复用新硬件ACC上已训练的共享和个人CycleGAN，以及适配后的192维ACC编码器。个人分支取G编码器input256，共享训练集拟合标准化，L2归一化，固定0.5权重拼接为448维。认证使用原RBF及Cal EER阈值。查询只输入ACC。

|用户|方法|条件|EER %|HTER %|FAR %|FRR %|
|---|---|---|---:|---:|---:|---:|
|Macro|adapted192|sameword|6.99|11.67|7.50|15.83|
|Macro|adapted192|crossword|11.50|14.28|6.90|21.67|
|Macro|adapted_cycle448|sameword|6.85|10.79|8.24|13.33|
|Macro|adapted_cycle448|crossword|12.75|14.10|8.61|19.58|
|U1|adapted192|sameword|4.63|7.87|7.41|8.33|
|U1|adapted192|crossword|16.67|15.74|10.65|20.83|
|U1|adapted_cycle448|sameword|5.56|8.80|9.26|8.33|
|U1|adapted_cycle448|crossword|16.44|15.05|9.26|20.83|
|U2|adapted192|sameword|12.96|19.44|22.22|16.67|
|U2|adapted192|crossword|15.97|15.51|14.35|16.67|
|U2|adapted_cycle448|sameword|12.04|19.91|23.15|16.67|
|U2|adapted_cycle448|crossword|21.53|21.30|21.76|20.83|
|U3|adapted192|sameword|0.00|9.72|19.44|0.00|
|U3|adapted192|crossword|13.19|13.89|19.44|8.33|
|U3|adapted_cycle448|sameword|0.46|7.87|15.74|0.00|
|U3|adapted_cycle448|crossword|16.20|15.97|23.61|8.33|
|U4|adapted192|sameword|6.94|8.80|0.93|16.67|
|U4|adapted192|crossword|7.87|16.90|0.46|33.33|
|U4|adapted_cycle448|sameword|6.48|6.48|4.63|8.33|
|U4|adapted_cycle448|crossword|12.04|8.80|5.09|12.50|
|U5|adapted192|sameword|15.28|18.06|11.11|25.00|
|U5|adapted192|crossword|11.11|15.51|6.02|25.00|
|U5|adapted_cycle448|sameword|16.67|15.74|14.81|16.67|
|U5|adapted_cycle448|crossword|10.42|11.34|10.19|12.50|
|U6|adapted192|sameword|0.00|4.17|0.00|8.33|
|U6|adapted192|crossword|11.11|16.90|0.46|33.33|
|U6|adapted_cycle448|sameword|0.00|4.17|0.00|8.33|
|U6|adapted_cycle448|crossword|15.97|18.75|0.00|37.50|
|U7|adapted192|sameword|1.85|5.09|1.85|8.33|
|U7|adapted192|crossword|4.17|3.01|1.85|4.17|
|U7|adapted_cycle448|sameword|1.39|4.63|0.93|8.33|
|U7|adapted_cycle448|crossword|3.70|3.01|1.85|4.17|
|U8|adapted192|sameword|26.85|24.54|7.41|41.67|
|U8|adapted192|crossword|16.20|14.58|12.50|16.67|
|U8|adapted_cycle448|sameword|24.54|21.30|9.26|33.33|
|U8|adapted_cycle448|crossword|13.43|15.05|9.26|20.83|
|U9|adapted192|sameword|1.39|2.31|4.63|0.00|
|U9|adapted192|crossword|2.55|3.70|3.24|4.17|
|U9|adapted_cycle448|sameword|0.46|2.31|4.63|0.00|
|U9|adapted_cycle448|crossword|0.69|2.55|5.09|0.00|
|U10|adapted192|sameword|0.00|16.67|0.00|33.33|
|U10|adapted192|crossword|16.20|27.08|0.00|54.17|
|U10|adapted_cycle448|sameword|0.93|16.67|0.00|33.33|
|U10|adapted_cycle448|crossword|17.13|29.17|0.00|58.33|
