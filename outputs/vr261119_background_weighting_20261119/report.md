# 连续背景校正与Audio频带软加权

10人40词3200片段，sample_id与原始27维逐条一致。沿用12条注册、8条校准、12条同词查询和24条跨词查询/人。训练词26个，Audio仅训练词参与权重估计。原始27维保留，新增24维连续录制频带能量减去安静帧对数背景能量，等权归一化拼接51维。安静帧排除全部发声边界前后100ms，背景为多个安静STFT帧中位数。发声段时频特征从连续STFT切出。现有近似时间映射保留，未实施时钟斜率修正和F0谐波权重。

Audio与ACC在8个频带计算归一化对数能量包络相关，32点相对时长对齐。每训练词前8遍正确配对与7种同人同词循环错配比较，跨训练用户和词平均，正差转为0.5–1.5软权重。错配对照权重由错配相关构成。48k Audio使用2048点窗及16384点FFT，低频真实分辨率受约43ms窗长限制，插值后的低频频带不独立。当前方法强调包络变化，其统计尚不能证明精细低频谐波恢复。

balanced RBF C10。StandardScaler只拟合注册样本，阈值校准集EER锁定。ROC保留全部分数点，最接近FAR=FRR点取均值。此EER离散点取法与旧脚本细节可能造成约0.3–0.5个百分点差异，原始27维AUC及锁定阈值HTER与旧结果一致。

|方法|条件|EER %|HTER %|AUC|
|---|---|---:|---:|---:|
|raw27|sameword|11.76|15.79|0.9145|
|raw27|crossword|15.53|22.38|0.9022|
|raw_background51|sameword|12.22|14.07|0.9460|
|raw_background51|crossword|18.33|22.64|0.9036|
|audio_weighted51|sameword|14.03|15.42|0.9308|
|audio_weighted51|crossword|17.11|22.45|0.9021|
|wrong_audio_weighted51|sameword|10.97|12.82|0.9448|
|wrong_audio_weighted51|crossword|17.29|22.27|0.9064|

背景校正改善同词AUC和HTER，跨词没有稳定收益。正确配对权重未优于错配，当前包络相关加权无法证明Audio独立贡献。部分同步和低频分辨率限制仍存在，本轮完成背景校正与包络软权重验证。

|用户|方法|条件|EER %|FAR %|FRR %|HTER %|
|---|---|---|---:|---:|---:|---:|
|U1|raw27|sameword|8.80|9.26|16.67|12.96|
|U1|raw27|crossword|16.67|12.96|20.83|16.90|
|U1|raw_background51|sameword|7.87|6.48|8.33|7.41|
|U1|raw_background51|crossword|12.50|8.80|16.67|12.73|
|U1|audio_weighted51|sameword|8.80|5.56|16.67|11.11|
|U1|audio_weighted51|crossword|12.27|6.48|25.00|15.74|
|U1|wrong_audio_weighted51|sameword|8.33|5.56|16.67|11.11|
|U1|wrong_audio_weighted51|crossword|9.03|4.17|20.83|12.50|
|U2|raw27|sameword|16.67|12.96|66.67|39.81|
|U2|raw27|crossword|16.67|8.80|45.83|27.31|
|U2|raw_background51|sameword|24.07|19.44|25.00|22.22|
|U2|raw_background51|crossword|20.83|23.61|16.67|20.14|
|U2|audio_weighted51|sameword|32.41|31.48|25.00|28.24|
|U2|audio_weighted51|crossword|17.59|37.50|8.33|22.92|
|U2|wrong_audio_weighted51|sameword|25.46|27.78|25.00|26.39|
|U2|wrong_audio_weighted51|crossword|20.83|27.31|16.67|21.99|
|U3|raw27|sameword|6.94|11.11|0.00|5.56|
|U3|raw27|crossword|17.36|18.06|16.67|17.36|
|U3|raw_background51|sameword|8.33|12.04|0.00|6.02|
|U3|raw_background51|crossword|16.67|13.43|25.00|19.21|
|U3|audio_weighted51|sameword|8.33|15.74|8.33|12.04|
|U3|audio_weighted51|crossword|16.67|16.67|20.83|18.75|
|U3|wrong_audio_weighted51|sameword|8.33|12.96|0.00|6.48|
|U3|wrong_audio_weighted51|crossword|16.67|20.37|12.50|16.44|
|U4|raw27|sameword|1.39|6.48|0.00|3.24|
|U4|raw27|crossword|12.50|6.94|33.33|20.14|
|U4|raw_background51|sameword|0.93|10.19|0.00|5.09|
|U4|raw_background51|crossword|16.67|9.26|29.17|19.21|
|U4|audio_weighted51|sameword|0.46|8.33|0.00|4.17|
|U4|audio_weighted51|crossword|15.74|8.33|29.17|18.75|
|U4|wrong_audio_weighted51|sameword|0.46|5.56|0.00|2.78|
|U4|wrong_audio_weighted51|crossword|12.50|5.09|33.33|19.21|
|U5|raw27|sameword|33.33|4.63|66.67|35.65|
|U5|raw27|crossword|15.97|9.72|16.67|13.19|
|U5|raw_background51|sameword|23.15|7.41|25.00|16.20|
|U5|raw_background51|crossword|16.67|14.81|20.83|17.82|
|U5|audio_weighted51|sameword|25.00|12.04|25.00|18.52|
|U5|audio_weighted51|crossword|16.67|16.20|16.67|16.44|
|U5|wrong_audio_weighted51|sameword|16.67|5.56|33.33|19.44|
|U5|wrong_audio_weighted51|crossword|16.67|10.65|25.00|17.82|
|U6|raw27|sameword|0.00|0.00|16.67|8.33|
|U6|raw27|crossword|3.47|0.46|79.17|39.81|
|U6|raw_background51|sameword|1.85|0.00|33.33|16.67|
|U6|raw_background51|crossword|12.50|0.46|62.50|31.48|
|U6|audio_weighted51|sameword|6.94|0.00|16.67|8.33|
|U6|audio_weighted51|crossword|8.33|0.46|45.83|23.15|
|U6|wrong_audio_weighted51|sameword|6.94|0.00|16.67|8.33|
|U6|wrong_audio_weighted51|crossword|13.43|0.46|58.33|29.40|
|U7|raw27|sameword|8.33|12.96|0.00|6.48|
|U7|raw27|crossword|4.63|18.98|0.00|9.49|
|U7|raw_background51|sameword|6.48|4.63|8.33|6.48|
|U7|raw_background51|crossword|12.50|6.48|12.50|9.49|
|U7|audio_weighted51|sameword|8.33|5.56|16.67|11.11|
|U7|audio_weighted51|crossword|12.50|9.26|12.50|10.88|
|U7|wrong_audio_weighted51|sameword|7.41|4.63|8.33|6.48|
|U7|wrong_audio_weighted51|crossword|12.50|8.80|12.50|10.65|
|U8|raw27|sameword|33.33|33.33|33.33|33.33|
|U8|raw27|crossword|30.09|36.11|29.17|32.64|
|U8|raw_background51|sameword|40.74|25.93|50.00|37.96|
|U8|raw_background51|crossword|29.17|26.39|33.33|29.86|
|U8|audio_weighted51|sameword|41.67|26.85|50.00|38.43|
|U8|audio_weighted51|crossword|29.40|27.78|33.33|30.56|
|U8|wrong_audio_weighted51|sameword|33.33|17.59|50.00|33.80|
|U8|wrong_audio_weighted51|crossword|25.00|19.44|37.50|28.47|
|U9|raw27|sameword|1.39|8.33|0.00|4.17|
|U9|raw27|crossword|12.50|7.41|25.00|16.20|
|U9|raw_background51|sameword|1.39|2.78|8.33|5.56|
|U9|raw_background51|crossword|12.50|0.46|70.83|35.65|
|U9|audio_weighted51|sameword|1.39|1.85|8.33|5.09|
|U9|audio_weighted51|crossword|8.56|0.46|70.83|35.65|
|U9|wrong_audio_weighted51|sameword|0.93|1.85|0.00|0.93|
|U9|wrong_audio_weighted51|crossword|12.96|0.46|70.83|35.65|
|U10|raw27|sameword|7.41|0.00|16.67|8.33|
|U10|raw27|crossword|25.46|3.24|58.33|30.79|
|U10|raw_background51|sameword|7.41|0.93|33.33|17.13|
|U10|raw_background51|crossword|33.33|3.24|58.33|30.79|
|U10|audio_weighted51|sameword|6.94|0.93|33.33|17.13|
|U10|audio_weighted51|crossword|33.33|5.09|58.33|31.71|
|U10|wrong_audio_weighted51|sameword|1.85|0.00|25.00|12.50|
|U10|wrong_audio_weighted51|crossword|33.33|2.78|58.33|30.56|
