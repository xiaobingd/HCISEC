# 注册背景分数标准化、原余弦与Ridge严格对照

两种冻结Audio模型分别8目标×3种子，各24折。固定P1 E0/E2模板、12次注册和HMD查询；只改变分数校准方法。背景均值/标准差仅来自4个Base身份的fit HMD中心与注册模板相似度，目标/Cal/攻击者不进入背景拟合。

采用图片公式 (s-mu)/(sigma+1e-6)，sigma为4中心相似度的总体标准差，不按最终效果调epsilon或筛背景。各方法每折在相同独立Cal上校准共享阈值，同词/跨词共享各方法阈值。原余弦和Ridge在EER协议逐折复现旧指标；新增Cal FAR<=5%工作点，所有方法使用相同校准池，Ridge在Cal重新选择收缩/背景下界策略，未访问目标测试。

## 主要结论

- 在原EER校准规则下，ECAPA E2原余弦HTER22.48%，背景标准化23.61%，Ridge23.00%。背景标准化未超过两项对照。
- 在Cal FAR<=5%工作点，ECAPA E2原余弦TAR38.37%、FAR6.60%、HTER34.11%；标准化39.06%、6.25%、33.59%，有约0.52个百分点HTER改善，但接受率仍很低，外层FAR仍超过5%，不能称为可用安全认证。
- ERes2NetV2同样在EER点变差（E2 HTER23.87%→24.65%），在Cal5%点小幅改善（34.64%→34.11%）。该归一化不具有跨工作点普遍优势。
- qi和xin在EER点的等效余弦阈值反而升高，TAR下降；bro的高重叠没有改善。4个训练中心提供的背景统计没有实现预期的新用户分数纠偏，不能将训练中心相似度分布等同于真实未知攻击分布。

## ecapa

|攻击口径|内容|方法/校准规则|TAR|FAR|HTER|原余弦AUC|判定margin AUC|
|---|---|---|---:|---:|---:|---:|---:|
|unseen_attacker|sameword|E0_raw_eer|75.00±4.63|21.53±3.05|23.26±3.84|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_zscore_eer|73.09±7.09|21.53±3.01|24.22±5.02|85.21±0.96|85.24±0.96|
|unseen_attacker|sameword|E0_ridge_eer|75.52±4.54|21.70±3.35|23.09±3.94|85.21±0.96|85.05±1.22|
|unseen_attacker|sameword|E0_raw_cal_far05|38.89±4.34|5.73±1.88|33.42±3.11|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_zscore_cal_far05|40.10±6.34|6.60±2.35|33.25±4.28|85.21±0.96|85.24±0.96|
|unseen_attacker|sameword|E0_ridge_cal_far05|31.08±9.85|4.51±2.46|36.72±5.64|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E2_raw_eer|75.52±4.63|21.70±3.35|23.09±3.99|85.25±0.99|85.25±0.99|
|unseen_attacker|sameword|E2_zscore_eer|73.61±6.72|21.53±3.01|23.96±4.82|85.25±0.99|85.29±1.00|
|unseen_attacker|sameword|E2_ridge_eer|75.69±3.98|21.88±3.65|23.09±3.79|85.25±0.99|85.10±1.25|
|unseen_attacker|sameword|E2_raw_cal_far05|39.76±5.81|5.73±1.88|32.99±3.84|85.25±0.99|85.25±0.99|
|unseen_attacker|sameword|E2_zscore_cal_far05|38.89±6.92|6.60±2.35|33.85±4.54|85.25±0.99|85.29±1.00|
|unseen_attacker|sameword|E2_ridge_cal_far05|32.81±12.01|4.51±2.46|35.85±6.71|85.25±0.99|85.25±0.99|
|unseen_attacker|crossword|E0_raw_eer|73.96±6.31|20.14±2.46|23.09±4.37|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_zscore_eer|72.40±8.57|19.97±2.57|23.78±5.57|85.61±1.13|85.58±1.10|
|unseen_attacker|crossword|E0_ridge_eer|74.31±5.71|20.49±3.05|23.09±4.37|85.61±1.13|85.59±1.17|
|unseen_attacker|crossword|E0_raw_cal_far05|37.85±5.14|6.60±2.17|34.38±3.65|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_zscore_cal_far05|38.72±8.44|6.25±1.38|33.77±4.67|85.61±1.13|85.58±1.10|
|unseen_attacker|crossword|E0_ridge_cal_far05|29.86±10.24|5.38±1.67|37.76±5.71|85.61±1.13|85.60±1.13|
|unseen_attacker|crossword|E2_raw_eer|75.35±5.81|20.31±2.76|22.48±4.28|85.63±1.18|85.63±1.18|
|unseen_attacker|crossword|E2_zscore_eer|72.74±7.97|19.97±2.57|23.61±5.27|85.63±1.18|85.59±1.14|
|unseen_attacker|crossword|E2_ridge_eer|74.65±5.85|20.66±3.35|23.00±4.59|85.63±1.18|85.60±1.21|
|unseen_attacker|crossword|E2_raw_cal_far05|38.37±5.57|6.60±2.17|34.11±3.85|85.63±1.18|85.63±1.18|
|unseen_attacker|crossword|E2_zscore_cal_far05|39.06±9.04|6.25±1.38|33.59±4.97|85.63±1.18|85.59±1.14|
|unseen_attacker|crossword|E2_ridge_cal_far05|31.60±11.53|5.38±1.67|36.89±6.40|85.63±1.18|85.61±1.18|
|all_seven_sources|sameword|E0_raw_eer|75.00±4.63|23.09±0.04|24.05±2.30|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_zscore_eer|73.09±7.09|22.47±0.39|24.69±3.39|85.14±0.33|85.11±0.28|
|all_seven_sources|sameword|E0_ridge_eer|75.52±4.54|23.19±0.09|23.83±2.31|85.14±0.33|85.07±0.38|
|all_seven_sources|sameword|E0_raw_cal_far05|38.89±4.34|8.11±0.45|34.61±2.32|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_zscore_cal_far05|40.10±6.34|8.38±0.69|34.14±3.07|85.14±0.33|85.11±0.28|
|all_seven_sources|sameword|E0_ridge_cal_far05|31.08±9.85|6.47±1.45|37.70±4.61|85.14±0.33|85.14±0.35|
|all_seven_sources|sameword|E2_raw_eer|75.52±4.63|23.26±0.23|23.87±2.27|85.18±0.44|85.18±0.44|
|all_seven_sources|sameword|E2_zscore_eer|73.61±6.72|22.47±0.39|24.43±3.25|85.18±0.44|85.15±0.39|
|all_seven_sources|sameword|E2_ridge_eer|75.69±3.98|23.41±0.38|23.86±2.09|85.18±0.44|85.04±0.46|
|all_seven_sources|sameword|E2_raw_cal_far05|39.76±5.81|8.09±0.48|34.16±3.04|85.18±0.44|85.18±0.44|
|all_seven_sources|sameword|E2_zscore_cal_far05|38.89±6.92|8.21±0.69|34.66±3.39|85.18±0.44|85.15±0.39|
|all_seven_sources|sameword|E2_ridge_cal_far05|32.81±12.01|6.65±1.45|36.92±5.62|85.18±0.44|85.10±0.36|
|all_seven_sources|crossword|E0_raw_eer|73.96±6.31|22.40±0.13|24.22±3.13|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_zscore_eer|72.40±8.57|22.02±0.13|24.81±4.25|84.90±0.86|84.88±0.84|
|all_seven_sources|crossword|E0_ridge_eer|74.31±5.71|22.57±0.21|24.13±2.91|84.90±0.86|84.93±0.81|
|all_seven_sources|crossword|E0_raw_cal_far05|37.85±5.14|7.84±0.62|35.00±2.73|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_zscore_cal_far05|38.72±8.44|8.06±0.60|34.67±4.06|84.90±0.86|84.88±0.84|
|all_seven_sources|crossword|E0_ridge_cal_far05|29.86±10.24|6.23±1.48|38.18±4.72|84.90±0.86|84.94±0.80|
|all_seven_sources|crossword|E2_raw_eer|75.35±5.81|22.74±0.19|23.70±2.90|84.96±0.91|84.96±0.91|
|all_seven_sources|crossword|E2_zscore_eer|72.74±7.97|22.15±0.04|24.70±3.96|84.96±0.91|84.92±0.88|
|all_seven_sources|crossword|E2_ridge_eer|74.65±5.85|22.84±0.27|24.09±2.99|84.96±0.91|84.95±0.84|
|all_seven_sources|crossword|E2_raw_cal_far05|38.37±5.57|7.94±0.55|34.78±2.97|84.96±0.91|84.96±0.91|
|all_seven_sources|crossword|E2_zscore_cal_far05|39.06±9.04|8.11±0.45|34.52±4.41|84.96±0.91|84.92±0.88|
|all_seven_sources|crossword|E2_ridge_cal_far05|31.60±11.53|6.37±1.44|37.39±5.38|84.96±0.91|84.94±0.85|

### qi / xin / bro：E2跨词、严格未见攻击者、原EER校准

|目标|方法|TAR|FAR|HTER|等效余弦阈值均值|背景mu|背景sigma|
|---|---|---:|---:|---:|---:|---:|---:|
|qi|raw|19.44%|2.78%|41.67%|0.9331|0.0155|0.5850|
|qi|zscore|5.56%|1.39%|47.92%|0.9721|0.0155|0.5850|
|qi|ridge|19.44%|2.78%|41.67%|0.9331|0.0155|0.5850|
|xin|raw|73.61%|0.00%|13.19%|0.6904|0.0347|0.5650|
|xin|zscore|65.28%|0.00%|17.36%|0.7241|0.0347|0.5650|
|xin|ridge|73.61%|0.00%|13.19%|0.6904|0.0347|0.5650|
|bro|raw|88.89%|76.39%|43.75%|0.5043|-0.0037|0.5731|
|bro|zscore|88.89%|76.39%|43.75%|0.5115|-0.0037|0.5731|
|bro|ridge|88.89%|76.39%|43.75%|0.5126|-0.0037|0.5731|
## eres

|攻击口径|内容|方法/校准规则|TAR|FAR|HTER|原余弦AUC|判定margin AUC|
|---|---|---|---:|---:|---:|---:|---:|
|unseen_attacker|sameword|E0_raw_eer|75.00±4.63|21.53±3.05|23.26±3.84|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_zscore_eer|73.09±7.09|21.53±3.01|24.22±5.02|85.21±0.96|85.24±0.96|
|unseen_attacker|sameword|E0_ridge_eer|75.17±4.34|21.70±3.35|23.26±3.84|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_raw_cal_far05|38.89±4.34|5.73±1.88|33.42±3.11|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E0_zscore_cal_far05|40.10±6.34|6.60±2.35|33.25±4.28|85.21±0.96|85.24±0.96|
|unseen_attacker|sameword|E0_ridge_cal_far05|32.29±10.40|5.03±1.97|36.37±6.04|85.21±0.96|85.21±0.96|
|unseen_attacker|sameword|E2_raw_eer|75.17±3.35|21.88±3.17|23.35±3.26|85.19±1.02|85.19±1.02|
|unseen_attacker|sameword|E2_zscore_eer|73.09±7.37|21.53±3.01|24.22±5.19|85.19±1.02|85.21±1.01|
|unseen_attacker|sameword|E2_ridge_eer|75.87±3.01|22.05±3.47|23.09±3.24|85.19±1.02|85.18±1.04|
|unseen_attacker|sameword|E2_raw_cal_far05|37.50±4.13|5.38±2.46|33.94±3.24|85.19±1.02|85.19±1.02|
|unseen_attacker|sameword|E2_zscore_cal_far05|38.02±8.76|6.08±3.18|34.03±5.80|85.19±1.02|85.21±1.01|
|unseen_attacker|sameword|E2_ridge_cal_far05|30.90±9.86|4.69±2.39|36.89±5.96|85.19±1.02|85.19±1.02|
|unseen_attacker|crossword|E0_raw_eer|73.96±6.31|20.14±2.46|23.09±4.37|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_zscore_eer|72.40±8.57|19.97±2.57|23.78±5.57|85.61±1.13|85.58±1.10|
|unseen_attacker|crossword|E0_ridge_eer|74.31±5.71|20.49±3.05|23.09±4.37|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_raw_cal_far05|37.85±5.14|6.60±2.17|34.38±3.65|85.61±1.13|85.61±1.13|
|unseen_attacker|crossword|E0_zscore_cal_far05|38.72±8.44|6.25±1.38|33.77±4.67|85.61±1.13|85.58±1.10|
|unseen_attacker|crossword|E0_ridge_cal_far05|31.08±11.05|5.56±1.59|37.24±6.20|85.61±1.13|85.60±1.13|
|unseen_attacker|crossword|E2_raw_eer|72.74±5.74|20.49±2.67|23.87±4.19|85.75±1.36|85.75±1.36|
|unseen_attacker|crossword|E2_zscore_eer|70.49±8.78|19.79±2.76|24.65±5.77|85.75±1.36|85.76±1.40|
|unseen_attacker|crossword|E2_ridge_eer|73.44±5.41|20.83±3.25|23.70±4.30|85.75±1.36|85.75±1.36|
|unseen_attacker|crossword|E2_raw_cal_far05|36.63±3.70|5.90±3.35|34.64±3.50|85.75±1.36|85.75±1.36|
|unseen_attacker|crossword|E2_zscore_cal_far05|37.15±8.67|5.38±2.62|34.11±5.44|85.75±1.36|85.76±1.40|
|unseen_attacker|crossword|E2_ridge_cal_far05|29.86±9.34|4.86±2.62|37.50±5.92|85.75±1.36|85.73±1.36|
|all_seven_sources|sameword|E0_raw_eer|75.00±4.63|23.09±0.04|24.05±2.30|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_zscore_eer|73.09±7.09|22.47±0.39|24.69±3.39|85.14±0.33|85.11±0.28|
|all_seven_sources|sameword|E0_ridge_eer|75.17±4.34|23.16±0.11|24.00±2.22|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_raw_cal_far05|38.89±4.34|8.11±0.45|34.61±2.32|85.14±0.33|85.14±0.33|
|all_seven_sources|sameword|E0_zscore_cal_far05|40.10±6.34|8.38±0.69|34.14±3.07|85.14±0.33|85.11±0.28|
|all_seven_sources|sameword|E0_ridge_cal_far05|32.29±10.40|6.67±1.40|37.19±4.88|85.14±0.33|85.14±0.35|
|all_seven_sources|sameword|E2_raw_eer|75.17±3.35|23.21±0.07|24.02±1.70|85.08±0.45|85.08±0.45|
|all_seven_sources|sameword|E2_zscore_eer|73.09±7.37|22.25±0.27|24.58±3.65|85.08±0.45|85.05±0.41|
|all_seven_sources|sameword|E2_ridge_eer|75.87±3.01|23.24±0.04|23.69±1.52|85.08±0.45|85.06±0.48|
|all_seven_sources|sameword|E2_raw_cal_far05|37.50±4.13|7.96±0.52|35.23±2.20|85.08±0.45|85.08±0.45|
|all_seven_sources|sameword|E2_zscore_cal_far05|38.02±8.76|8.09±0.64|35.03±4.38|85.08±0.45|85.05±0.41|
|all_seven_sources|sameword|E2_ridge_cal_far05|30.90±9.86|6.52±1.31|37.81±4.56|85.08±0.45|85.07±0.47|
|all_seven_sources|crossword|E0_raw_eer|73.96±6.31|22.40±0.13|24.22±3.13|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_zscore_eer|72.40±8.57|22.02±0.13|24.81±4.25|84.90±0.86|84.88±0.84|
|all_seven_sources|crossword|E0_ridge_eer|74.31±5.71|22.52±0.19|24.11±2.93|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_raw_cal_far05|37.85±5.14|7.84±0.62|35.00±2.73|84.90±0.86|84.90±0.86|
|all_seven_sources|crossword|E0_zscore_cal_far05|38.72±8.44|8.06±0.60|34.67±4.06|84.90±0.86|84.88±0.84|
|all_seven_sources|crossword|E0_ridge_cal_far05|31.08±11.05|6.40±1.32|37.66±5.16|84.90±0.86|84.94±0.80|
|all_seven_sources|crossword|E2_raw_eer|72.74±5.74|22.69±0.34|24.98±2.73|84.77±0.74|84.77±0.74|
|all_seven_sources|crossword|E2_zscore_eer|70.49±8.78|21.88±0.27|25.69±4.28|84.77±0.74|84.77±0.73|
|all_seven_sources|crossword|E2_ridge_eer|73.44±5.41|22.87±0.23|24.71±2.73|84.77±0.74|84.76±0.77|
|all_seven_sources|crossword|E2_raw_cal_far05|36.63±3.70|7.81±0.72|35.59±1.99|84.77±0.74|84.77±0.74|
|all_seven_sources|crossword|E2_zscore_cal_far05|37.15±8.67|7.89±0.52|35.37±4.38|84.77±0.74|84.77±0.73|
|all_seven_sources|crossword|E2_ridge_cal_far05|29.86±9.34|6.37±1.28|38.26±4.33|84.77±0.74|84.81±0.68|

### qi / xin / bro：E2跨词、严格未见攻击者、原EER校准

|目标|方法|TAR|FAR|HTER|等效余弦阈值均值|背景mu|背景sigma|
|---|---|---:|---:|---:|---:|---:|---:|
|qi|raw|18.06%|4.17%|43.06%|0.9232|0.0140|0.5826|
|qi|zscore|6.94%|1.39%|47.22%|0.9676|0.0140|0.5826|
|qi|ridge|18.06%|4.17%|43.06%|0.9232|0.0140|0.5826|
|xin|raw|76.39%|0.00%|11.81%|0.6757|0.0311|0.5619|
|xin|zscore|62.50%|0.00%|18.75%|0.7209|0.0311|0.5619|
|xin|ridge|76.39%|0.00%|11.81%|0.6698|0.0311|0.5619|
|bro|raw|87.50%|76.39%|44.44%|0.4970|-0.0026|0.5683|
|bro|zscore|87.50%|75.00%|43.75%|0.5089|-0.0026|0.5683|
|bro|ridge|87.50%|76.39%|44.44%|0.4970|-0.0026|0.5683|

## 聚焦解释

ECAPA E2中qi的等效余弦阈值均值由0.9331升至0.9721，本人接受率19.44%→5.56%；xin阈值0.6904→0.7241，TAR73.61%→65.28%；bro TAR88.89%、FAR76.39%基本不变。这里阈值均值只是同一目标三个种子、两个方向的描述统计，不是部署时使用平均阈值。

背景mu接近0、sigma在这些例子约0.57–0.59，未提供与实际新用户分数偏移相匹配的校正。这提示4个Base分类中心的几何关系可能不足以代表未知攻击分布；不是已证明的唯一失败原因，也不应靠最终查询重选背景或epsilon修饰结果。bro的原分数高度重叠，单调线性标准化不能增加同一注册任务内的身份排序信息。

同词/跨词共享标准化后的阈值与统计生成规则，原余弦等效阈值可随注册模板变化；属于新评分协议。原始余弦AUC不变，margin AUC可能因两个注册方向的条件标准化而变化，不代表编码器新提取身份信息。

Cal仅2位身份、背景仅4位训练身份；经验Cal FAR<=5%不是未见攻击FAR保证。多个随机种子不是独立会话，词语编号未逐条语义核验，八人群体此前排除yue具有探索性选择限制。没有根据目标测试挑选校准规则，两个工作点均完整保留。

[个体化阈值对照](2026-11-08-enrollment-threshold-results.md) · [P1固定模型路线](2026-11-08-p1-enrollment-results.md)