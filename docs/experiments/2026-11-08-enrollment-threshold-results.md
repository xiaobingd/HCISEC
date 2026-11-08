# 注册条件个体化阈值对照结果

采用图片建议的任务边界：Base模拟注册学习g，独立Cal选验证策略，目标仅使用合法注册Audio/HMD生成阈值，查询HMD-only。ECAPA/ERes2NetV2分别24折。固定E0/E2模板和编码器，对照阈值策略。

## 主要结论

当前低维Ridge版g没有稳定改善。ECAPA E2固定阈值TAR75.35%、FAR20.31%、HTER22.48%；个体化阈值为74.65%、20.66%、23.00%。ERes E2个体化仅小幅降低HTER（23.87%→23.70%），仍劣于原E0的23.09%，且FAR上升。

Cal约束可使校准集FAR不恶化，但不能保证外层未见攻击FAR。所有折均未采用背景下界候选；不能宣称背景下界已有效。多数折回退到全局阈值，反映该g/校准规则尚未获得可靠个体化策略。此结论不否定所有注册条件校准方案。

## ecapa

|攻击口径|内容|方法|TAR|FAR|HTER|原余弦AUC|校准margin AUC|
|---|---|---|---:|---:|---:|---:|---:|
|unseen_attacker|sameword|E0_fixed|75.00±4.63|21.53±3.05|23.26±3.84|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_adaptive|75.52±4.54|21.70±3.35|23.09±3.94|85.21±0.96|85.05±1.22|
|unseen_attacker|sameword|E0_adaptive_guarded|75.52±4.54|21.70±3.35|23.09±3.94|85.21±0.96|85.05±1.22|
|unseen_attacker|sameword|E2_fixed|75.52±4.63|21.70±3.35|23.09±3.99|85.25±0.99|85.25±0.99|
|unseen_attacker|sameword|E2_adaptive|75.69±3.98|21.88±3.65|23.09±3.79|85.25±0.99|85.10±1.25|
|unseen_attacker|sameword|E2_adaptive_guarded|75.69±3.98|21.88±3.65|23.09±3.79|85.25±0.99|85.10±1.25|
|unseen_attacker|crossword|E0_fixed|73.96±6.31|20.14±2.46|23.09±4.37|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_adaptive|74.31±5.71|20.49±3.05|23.09±4.37|85.61±1.13|85.59±1.17|
|unseen_attacker|crossword|E0_adaptive_guarded|74.31±5.71|20.49±3.05|23.09±4.37|85.61±1.13|85.59±1.17|
|unseen_attacker|crossword|E2_fixed|75.35±5.81|20.31±2.76|22.48±4.28|85.63±1.18|85.63±1.18|
|unseen_attacker|crossword|E2_adaptive|74.65±5.85|20.66±3.35|23.00±4.59|85.63±1.18|85.60±1.21|
|unseen_attacker|crossword|E2_adaptive_guarded|74.65±5.85|20.66±3.35|23.00±4.59|85.63±1.18|85.60±1.21|
|all_seven_sources|sameword|E0_fixed|75.00±4.63|23.09±0.04|24.05±2.30|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_adaptive|75.52±4.54|23.19±0.09|23.83±2.31|85.14±0.33|85.07±0.38|
|all_seven_sources|sameword|E0_adaptive_guarded|75.52±4.54|23.19±0.09|23.83±2.31|85.14±0.33|85.07±0.38|
|all_seven_sources|sameword|E2_fixed|75.52±4.63|23.26±0.23|23.87±2.27|85.18±0.44|85.18±0.44|
|all_seven_sources|sameword|E2_adaptive|75.69±3.98|23.41±0.38|23.86±2.09|85.18±0.44|85.04±0.46|
|all_seven_sources|sameword|E2_adaptive_guarded|75.69±3.98|23.41±0.38|23.86±2.09|85.18±0.44|85.04±0.46|
|all_seven_sources|crossword|E0_fixed|73.96±6.31|22.40±0.13|24.22±3.13|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_adaptive|74.31±5.71|22.57±0.21|24.13±2.91|84.90±0.86|84.93±0.81|
|all_seven_sources|crossword|E0_adaptive_guarded|74.31±5.71|22.57±0.21|24.13±2.91|84.90±0.86|84.93±0.81|
|all_seven_sources|crossword|E2_fixed|75.35±5.81|22.74±0.19|23.70±2.90|84.96±0.91|84.96±0.91|
|all_seven_sources|crossword|E2_adaptive|74.65±5.85|22.84±0.27|24.09±2.99|84.96±0.91|84.95±0.84|
|all_seven_sources|crossword|E2_adaptive_guarded|74.65±5.85|22.84±0.27|24.09±2.99|84.96±0.91|84.95±0.84|

E0策略分布：{"eta=0.0,floor=False": 19, "eta=1.0,floor=False": 2, "eta=0.5,floor=False": 2, "eta=0.25,floor=False": 1}

E2策略分布：{"eta=0.25,floor=False": 2, "eta=0.0,floor=False": 18, "eta=1.0,floor=False": 3, "eta=0.5,floor=False": 1}

|目标|E2固定TAR|E2固定FAR|E2个体化TAR|E2个体化FAR|E2个体化减固定HTER(pp)|
|---|---:|---:|---:|---:|---:|
|bo|95.83|0.00|87.50|0.00|4.17|
|bro|88.89|76.39|88.89|76.39|0.00|
|jin|95.83|0.00|95.83|0.00|0.00|
|ke|95.83|44.44|95.83|44.44|0.00|
|mei|50.00|34.72|52.78|37.50|0.00|
|qi|19.44|2.78|19.44|2.78|0.00|
|qis|83.33|4.17|83.33|4.17|0.00|
|xin|73.61|0.00|73.61|0.00|0.00|
## eres

|攻击口径|内容|方法|TAR|FAR|HTER|原余弦AUC|校准margin AUC|
|---|---|---|---:|---:|---:|---:|---:|
|unseen_attacker|sameword|E0_fixed|75.00±4.63|21.53±3.05|23.26±3.84|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_adaptive|75.17±4.34|21.70±3.35|23.26±3.84|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_adaptive_guarded|75.17±4.34|21.70±3.35|23.26±3.84|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E2_fixed|75.17±3.35|21.88±3.17|23.35±3.26|85.19±1.02|85.19±1.02|
|unseen_attacker|sameword|E2_adaptive|75.87±3.01|22.05±3.47|23.09±3.24|85.19±1.02|85.18±1.04|
|unseen_attacker|sameword|E2_adaptive_guarded|75.87±3.01|22.05±3.47|23.09±3.24|85.19±1.02|85.18±1.04|
|unseen_attacker|crossword|E0_fixed|73.96±6.31|20.14±2.46|23.09±4.37|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_adaptive|74.31±5.71|20.49±3.05|23.09±4.37|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_adaptive_guarded|74.31±5.71|20.49±3.05|23.09±4.37|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E2_fixed|72.74±5.74|20.49±2.67|23.87±4.19|85.75±1.36|85.75±1.36|
|unseen_attacker|crossword|E2_adaptive|73.44±5.41|20.83±3.25|23.70±4.30|85.75±1.36|85.75±1.36|
|unseen_attacker|crossword|E2_adaptive_guarded|73.44±5.41|20.83±3.25|23.70±4.30|85.75±1.36|85.75±1.36|
|all_seven_sources|sameword|E0_fixed|75.00±4.63|23.09±0.04|24.05±2.30|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_adaptive|75.17±4.34|23.16±0.11|24.00±2.22|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_adaptive_guarded|75.17±4.34|23.16±0.11|24.00±2.22|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E2_fixed|75.17±3.35|23.21±0.07|24.02±1.70|85.08±0.45|85.08±0.45|
|all_seven_sources|sameword|E2_adaptive|75.87±3.01|23.24±0.04|23.69±1.52|85.08±0.45|85.06±0.48|
|all_seven_sources|sameword|E2_adaptive_guarded|75.87±3.01|23.24±0.04|23.69±1.52|85.08±0.45|85.06±0.48|
|all_seven_sources|crossword|E0_fixed|73.96±6.31|22.40±0.13|24.22±3.13|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_adaptive|74.31±5.71|22.52±0.19|24.11±2.93|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_adaptive_guarded|74.31±5.71|22.52±0.19|24.11±2.93|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E2_fixed|72.74±5.74|22.69±0.34|24.98±2.73|84.77±0.74|84.77±0.74|
|all_seven_sources|crossword|E2_adaptive|73.44±5.41|22.87±0.23|24.71±2.73|84.77±0.74|84.76±0.77|
|all_seven_sources|crossword|E2_adaptive_guarded|73.44±5.41|22.87±0.23|24.71±2.73|84.77±0.74|84.76±0.77|

E0策略分布：{"eta=0.0,floor=False": 20, "eta=0.25,floor=False": 3, "eta=1.0,floor=False": 1}

E2策略分布：{"eta=0.0,floor=False": 20, "eta=0.25,floor=False": 2, "eta=1.0,floor=False": 2}

|目标|E2固定TAR|E2固定FAR|E2个体化TAR|E2个体化FAR|E2个体化减固定HTER(pp)|
|---|---:|---:|---:|---:|---:|
|bo|75.00|0.00|77.78|0.00|-1.39|
|bro|87.50|76.39|87.50|76.39|0.00|
|jin|95.83|0.00|95.83|0.00|0.00|
|ke|95.83|44.44|95.83|44.44|0.00|
|mei|50.00|34.72|52.78|37.50|0.00|
|qi|18.06|4.17|18.06|4.17|0.00|
|qis|83.33|4.17|83.33|4.17|0.00|
|xin|76.39|0.00|76.39|0.00|0.00|

## 解释边界

模板未变，原余弦AUC不变；margin AUC变化只说明注册条件阈值导致的方向间分数校准变化，不是多提取了身份信息。固定分支逐折复现P1的TAR/FAR/HTER。

g使用6维注册embedding一致性及3维相对Base背景原型统计，Ridge正则20，非高维任意阈值网络。注册端预测相对Cal注册预测中位数的偏移限制±0.1，Cal从0/.25/.5/1选收缩幅度；背景下界为可选候选。阈值策略针对Cal EER工作点，不是经过验证的低FAR安全保证。尚未新增正式分数归一化对照。

当前只有4位Base身份；96个模拟注册任务不等于96名独立用户。内层映射器留人而固定编码器已见Base、内外背景人数3→4、目标与Cal注册质量分布偏移等均限制泛化。词标签未逐条语义核验，八人群体此前排除yue属探索性选择。不能将该版g的失败扩大为所有个体化校准不可行。

[执行协议](2026-11-08-enrollment-threshold-protocol.md) · [E3与阈值迁移诊断](2026-11-08-e3-quality-gating-results.md)