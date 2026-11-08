# 注册期个人线性头、微型残差适配与 Audio 辅助约束

日期：2026-11-08。代码和协议已在实验启动前上传。两套冻结 Audio 教师（ECAPA、ERes2NetV2），各 8 个目标 × 3 个种子，全部完成。

## 结论

当前固定配置没有证明个人适配优于冻结编码器的余弦原型。线性头小幅提高接受率，同时提高 FAR；192 参数适配器和注册 Audio 辅助约束均未产生明确额外收益。不能将旧混合校准下的小幅 HTER 下降解释为安全性提升。

## 严格协议与技术路线

每折 4 Base 身份训练通用 HMD 编码器、2 Cal 身份校准、1 目标、1 未见攻击者，四类身份互斥。通用编码器全程冻结。各个人任务使用 12 条本人注册正样本、48 条 Base 背景负样本训练，最终查询记录不参与适配。跨词条件注册与查询使用不同的三个词；同词条件使用相同词的不同重复。两方向合并后每折 24 本人查询、24 未见攻击查询。

个人线性头有 193 参数；增加的通道残差适配器有 192 参数：normalize(z*(1+0.1*tanh(gamma)))，零初始化，通道增益限于 ±10%。固定训练 200 步，不使用目标验证或测试选择步数、权重。Audio 辅助仅使用本人同步注册 Audio，经冻结 Audio 教师和仅 Base 训练的映射器约束适配后 ACC。ACC 自一致性是额外正则化对照。部署查询只需 ACC。

主要校准使用两个独立 Cal 用户互相攻击，避免将个人头已见的 Base 负类作为主要校准攻击者。每种方法按预定规则在 Cal 上确定一个跨同词/跨词共享阈值；并列报告 Cal-EER 与 Cal-FAR≤5% 两种规则。旧 Base+Cal 混合校准为预先保留的次要对照，禁止根据最终测试结果切换规则。

原始 HMD 记录实际为三轴速度；本实验复用既有统一 10–300 Hz STFT 输入及冻结编码器，不将其误称为硬件原始加速度。

## 详细宏平均结果

所有数值为百分比。HTER=(FRR+FAR)/2，越低越好；表中 EER 指校准阈值规则，最终 HTER 不是测试 EER。每行是 8 用户 × 3 种子的宏平均，并非 24 个独立用户。

### ecapa / eer_unseen

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|72.74|24.31|25.78|85.21|
|sameword|unseen_attacker|个人线性头|73.96|25.52|25.78|85.05|
|sameword|unseen_attacker|线性头+残差|73.61|26.56|26.48|85.07|
|sameword|unseen_attacker|残差+ACC自一致|73.61|26.56|26.48|85.05|
|sameword|unseen_attacker|残差+注册Audio|73.44|26.04|26.30|85.11|
|sameword|all_seven_sources|余弦原型|72.74|22.79|25.02|85.14|
|sameword|all_seven_sources|个人线性头|73.96|18.38|22.21|89.70|
|sameword|all_seven_sources|线性头+残差|73.61|18.33|22.36|89.74|
|sameword|all_seven_sources|残差+ACC自一致|73.61|18.30|22.35|89.73|
|sameword|all_seven_sources|残差+注册Audio|73.44|18.28|22.42|89.75|
|crossword|unseen_attacker|余弦原型|72.22|23.44|25.61|85.61|
|crossword|unseen_attacker|个人线性头|73.44|24.83|25.69|84.81|
|crossword|unseen_attacker|线性头+残差|73.26|24.65|25.69|84.80|
|crossword|unseen_attacker|残差+ACC自一致|73.26|24.65|25.69|84.80|
|crossword|unseen_attacker|残差+注册Audio|73.26|24.65|25.69|84.79|
|crossword|all_seven_sources|余弦原型|72.22|22.74|25.26|84.90|
|crossword|all_seven_sources|个人线性头|73.44|17.93|22.25|89.13|
|crossword|all_seven_sources|线性头+残差|73.26|17.63|22.19|89.20|
|crossword|all_seven_sources|残差+ACC自一致|73.26|17.63|22.19|89.19|
|crossword|all_seven_sources|残差+注册Audio|73.26|17.68|22.21|89.18|

### ecapa / cal_far05_unseen

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|40.97|13.37|36.20|85.21|
|sameword|unseen_attacker|个人线性头|25.17|11.28|43.06|85.05|
|sameword|unseen_attacker|线性头+残差|26.04|11.81|42.88|85.07|
|sameword|unseen_attacker|残差+ACC自一致|26.04|11.81|42.88|85.05|
|sameword|unseen_attacker|残差+注册Audio|26.04|11.63|42.80|85.11|
|sameword|all_seven_sources|余弦原型|40.97|12.03|35.53|85.14|
|sameword|all_seven_sources|个人线性头|25.17|6.99|40.91|89.70|
|sameword|all_seven_sources|线性头+残差|26.04|7.19|40.58|89.74|
|sameword|all_seven_sources|残差+ACC自一致|26.04|7.19|40.58|89.73|
|sameword|all_seven_sources|残差+注册Audio|26.04|7.24|40.60|89.75|
|crossword|unseen_attacker|余弦原型|42.53|12.15|34.81|85.61|
|crossword|unseen_attacker|个人线性头|26.04|9.72|41.84|84.81|
|crossword|unseen_attacker|线性头+残差|26.74|10.24|41.75|84.80|
|crossword|unseen_attacker|残差+ACC自一致|26.56|10.24|41.84|84.80|
|crossword|unseen_attacker|残差+注册Audio|26.56|10.24|41.84|84.79|
|crossword|all_seven_sources|余弦原型|42.53|11.71|34.59|84.90|
|crossword|all_seven_sources|个人线性头|26.04|6.87|40.41|89.13|
|crossword|all_seven_sources|线性头+残差|26.74|6.87|40.07|89.20|
|crossword|all_seven_sources|残差+ACC自一致|26.56|6.89|40.17|89.19|
|crossword|all_seven_sources|残差+注册Audio|26.56|6.92|40.18|89.18|

### ecapa / eer_mixed_legacy

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|75.00|21.53|23.26|85.21|
|sameword|unseen_attacker|个人线性头|81.94|26.91|22.48|85.05|
|sameword|unseen_attacker|线性头+残差|82.12|27.08|22.48|85.07|
|sameword|unseen_attacker|残差+ACC自一致|82.12|27.08|22.48|85.05|
|sameword|unseen_attacker|残差+注册Audio|82.29|27.08|22.40|85.11|
|sameword|all_seven_sources|余弦原型|75.00|23.09|24.05|85.14|
|sameword|all_seven_sources|个人线性头|81.94|20.39|19.22|89.70|
|sameword|all_seven_sources|线性头+残差|82.12|20.44|19.16|89.74|
|sameword|all_seven_sources|残差+ACC自一致|82.12|20.39|19.13|89.73|
|sameword|all_seven_sources|残差+注册Audio|82.29|20.36|19.04|89.75|
|crossword|unseen_attacker|余弦原型|73.96|20.14|23.09|85.61|
|crossword|unseen_attacker|个人线性头|79.86|25.69|22.92|84.81|
|crossword|unseen_attacker|线性头+残差|80.21|25.69|22.74|84.80|
|crossword|unseen_attacker|残差+ACC自一致|80.21|25.69|22.74|84.80|
|crossword|unseen_attacker|残差+注册Audio|80.21|25.69|22.74|84.79|
|crossword|all_seven_sources|余弦原型|73.96|22.40|24.22|84.90|
|crossword|all_seven_sources|个人线性头|79.86|19.97|20.05|89.13|
|crossword|all_seven_sources|线性头+残差|80.21|20.09|19.94|89.20|
|crossword|all_seven_sources|残差+ACC自一致|80.21|20.09|19.94|89.19|
|crossword|all_seven_sources|残差+注册Audio|80.21|20.06|19.93|89.18|

### ecapa / cal_far05_mixed_legacy

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|38.89|5.73|33.42|85.21|
|sameword|unseen_attacker|个人线性头|47.57|14.24|33.33|85.05|
|sameword|unseen_attacker|线性头+残差|47.92|14.06|33.07|85.07|
|sameword|unseen_attacker|残差+ACC自一致|47.92|14.06|33.07|85.05|
|sameword|unseen_attacker|残差+注册Audio|46.88|13.89|33.51|85.11|
|sameword|all_seven_sources|余弦原型|38.89|8.11|34.61|85.14|
|sameword|all_seven_sources|个人线性头|47.57|8.11|30.27|89.70|
|sameword|all_seven_sources|线性头+残差|47.92|8.01|30.05|89.74|
|sameword|all_seven_sources|残差+ACC自一致|47.92|8.01|30.05|89.73|
|sameword|all_seven_sources|残差+注册Audio|46.88|7.99|30.56|89.75|
|crossword|unseen_attacker|余弦原型|37.85|6.60|34.38|85.61|
|crossword|unseen_attacker|个人线性头|44.79|14.58|34.90|84.81|
|crossword|unseen_attacker|线性头+残差|44.44|14.58|35.07|84.80|
|crossword|unseen_attacker|残差+ACC自一致|44.44|14.58|35.07|84.80|
|crossword|unseen_attacker|残差+注册Audio|44.79|14.58|34.90|84.79|
|crossword|all_seven_sources|余弦原型|37.85|7.84|35.00|84.90|
|crossword|all_seven_sources|个人线性头|44.79|8.28|31.75|89.13|
|crossword|all_seven_sources|线性头+残差|44.44|8.28|31.92|89.20|
|crossword|all_seven_sources|残差+ACC自一致|44.44|8.26|31.91|89.19|
|crossword|all_seven_sources|残差+注册Audio|44.79|8.28|31.75|89.18|

### eres / eer_unseen

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|72.74|24.31|25.78|85.21|
|sameword|unseen_attacker|个人线性头|73.96|25.52|25.78|85.05|
|sameword|unseen_attacker|线性头+残差|73.61|26.56|26.48|85.07|
|sameword|unseen_attacker|残差+ACC自一致|73.61|26.56|26.48|85.05|
|sameword|unseen_attacker|残差+注册Audio|73.26|26.22|26.48|85.11|
|sameword|all_seven_sources|余弦原型|72.74|22.79|25.02|85.14|
|sameword|all_seven_sources|个人线性头|73.96|18.38|22.21|89.70|
|sameword|all_seven_sources|线性头+残差|73.61|18.33|22.36|89.74|
|sameword|all_seven_sources|残差+ACC自一致|73.61|18.30|22.35|89.73|
|sameword|all_seven_sources|残差+注册Audio|73.26|18.30|22.52|89.74|
|crossword|unseen_attacker|余弦原型|72.22|23.44|25.61|85.61|
|crossword|unseen_attacker|个人线性头|73.44|24.83|25.69|84.81|
|crossword|unseen_attacker|线性头+残差|73.26|24.65|25.69|84.80|
|crossword|unseen_attacker|残差+ACC自一致|73.26|24.65|25.69|84.80|
|crossword|unseen_attacker|残差+注册Audio|73.09|24.65|25.78|84.79|
|crossword|all_seven_sources|余弦原型|72.22|22.74|25.26|84.90|
|crossword|all_seven_sources|个人线性头|73.44|17.93|22.25|89.13|
|crossword|all_seven_sources|线性头+残差|73.26|17.63|22.19|89.20|
|crossword|all_seven_sources|残差+ACC自一致|73.26|17.63|22.19|89.19|
|crossword|all_seven_sources|残差+注册Audio|73.09|17.66|22.28|89.19|

### eres / cal_far05_unseen

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|40.97|13.37|36.20|85.21|
|sameword|unseen_attacker|个人线性头|25.17|11.28|43.06|85.05|
|sameword|unseen_attacker|线性头+残差|26.04|11.81|42.88|85.07|
|sameword|unseen_attacker|残差+ACC自一致|26.04|11.81|42.88|85.05|
|sameword|unseen_attacker|残差+注册Audio|26.04|11.63|42.80|85.11|
|sameword|all_seven_sources|余弦原型|40.97|12.03|35.53|85.14|
|sameword|all_seven_sources|个人线性头|25.17|6.99|40.91|89.70|
|sameword|all_seven_sources|线性头+残差|26.04|7.19|40.58|89.74|
|sameword|all_seven_sources|残差+ACC自一致|26.04|7.19|40.58|89.73|
|sameword|all_seven_sources|残差+注册Audio|26.04|7.17|40.56|89.74|
|crossword|unseen_attacker|余弦原型|42.53|12.15|34.81|85.61|
|crossword|unseen_attacker|个人线性头|26.04|9.72|41.84|84.81|
|crossword|unseen_attacker|线性头+残差|26.74|10.24|41.75|84.80|
|crossword|unseen_attacker|残差+ACC自一致|26.56|10.24|41.84|84.80|
|crossword|unseen_attacker|残差+注册Audio|26.56|10.24|41.84|84.79|
|crossword|all_seven_sources|余弦原型|42.53|11.71|34.59|84.90|
|crossword|all_seven_sources|个人线性头|26.04|6.87|40.41|89.13|
|crossword|all_seven_sources|线性头+残差|26.74|6.87|40.07|89.20|
|crossword|all_seven_sources|残差+ACC自一致|26.56|6.89|40.17|89.19|
|crossword|all_seven_sources|残差+注册Audio|26.56|6.89|40.17|89.19|

### eres / eer_mixed_legacy

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|75.00|21.53|23.26|85.21|
|sameword|unseen_attacker|个人线性头|81.94|26.91|22.48|85.05|
|sameword|unseen_attacker|线性头+残差|82.12|27.08|22.48|85.07|
|sameword|unseen_attacker|残差+ACC自一致|82.12|27.08|22.48|85.05|
|sameword|unseen_attacker|残差+注册Audio|82.29|26.91|22.31|85.11|
|sameword|all_seven_sources|余弦原型|75.00|23.09|24.05|85.14|
|sameword|all_seven_sources|个人线性头|81.94|20.39|19.22|89.70|
|sameword|all_seven_sources|线性头+残差|82.12|20.44|19.16|89.74|
|sameword|all_seven_sources|残差+ACC自一致|82.12|20.39|19.13|89.73|
|sameword|all_seven_sources|残差+注册Audio|82.29|20.41|19.06|89.74|
|crossword|unseen_attacker|余弦原型|73.96|20.14|23.09|85.61|
|crossword|unseen_attacker|个人线性头|79.86|25.69|22.92|84.81|
|crossword|unseen_attacker|线性头+残差|80.21|25.69|22.74|84.80|
|crossword|unseen_attacker|残差+ACC自一致|80.21|25.69|22.74|84.80|
|crossword|unseen_attacker|残差+注册Audio|80.21|25.69|22.74|84.79|
|crossword|all_seven_sources|余弦原型|73.96|22.40|24.22|84.90|
|crossword|all_seven_sources|个人线性头|79.86|19.97|20.05|89.13|
|crossword|all_seven_sources|线性头+残差|80.21|20.09|19.94|89.20|
|crossword|all_seven_sources|残差+ACC自一致|80.21|20.09|19.94|89.19|
|crossword|all_seven_sources|残差+注册Audio|80.21|20.01|19.90|89.19|

### eres / cal_far05_mixed_legacy

|条件|攻击范围|方法|TAR|FAR|HTER|AUC|
|---|---|---|---:|---:|---:|---:|
|sameword|unseen_attacker|余弦原型|38.89|5.73|33.42|85.21|
|sameword|unseen_attacker|个人线性头|47.57|14.24|33.33|85.05|
|sameword|unseen_attacker|线性头+残差|47.92|14.06|33.07|85.07|
|sameword|unseen_attacker|残差+ACC自一致|47.92|14.06|33.07|85.05|
|sameword|unseen_attacker|残差+注册Audio|47.22|13.72|33.25|85.11|
|sameword|all_seven_sources|余弦原型|38.89|8.11|34.61|85.14|
|sameword|all_seven_sources|个人线性头|47.57|8.11|30.27|89.70|
|sameword|all_seven_sources|线性头+残差|47.92|8.01|30.05|89.74|
|sameword|all_seven_sources|残差+ACC自一致|47.92|8.01|30.05|89.73|
|sameword|all_seven_sources|残差+注册Audio|47.22|7.96|30.37|89.74|
|crossword|unseen_attacker|余弦原型|37.85|6.60|34.38|85.61|
|crossword|unseen_attacker|个人线性头|44.79|14.58|34.90|84.81|
|crossword|unseen_attacker|线性头+残差|44.44|14.58|35.07|84.80|
|crossword|unseen_attacker|残差+ACC自一致|44.44|14.58|35.07|84.80|
|crossword|unseen_attacker|残差+注册Audio|44.44|14.58|35.07|84.79|
|crossword|all_seven_sources|余弦原型|37.85|7.84|35.00|84.90|
|crossword|all_seven_sources|个人线性头|44.79|8.28|31.75|89.13|
|crossword|all_seven_sources|线性头+残差|44.44|8.28|31.92|89.20|
|crossword|all_seven_sources|残差+ACC自一致|44.44|8.26|31.91|89.19|
|crossword|all_seven_sources|残差+注册Audio|44.44|8.28|31.92|89.19|

## 解释与边界

主要跨词 ECAPA 结果：余弦 72.22% TAR / 23.44% FAR / 25.61% HTER；线性头 73.44 / 24.83 / 25.69；残差 73.26 / 24.65 / 25.69；加入 Audio 仍为 73.26 / 24.65 / 25.69。Audio AUC 从残差的 84.80% 轻微降至 84.79%，没有额外收益。ERes Audio 的 HTER 为 25.78%。

校准 FAR≤5% 不保证最终未知攻击 FAR≤5%。ECAPA 跨词余弦最终 FAR 为 12.15%，线性头为 9.72%，但 TAR 从 42.53% 降到 26.04%，因此这也是安全性与可用性的权衡。

旧混合校准的余弦 HTER 精确复现 23.09%；残差为 22.74%，同时 FAR 从 20.14% 升到 25.69%。不能把该 0.35 个百分点变化称为整体提升。主要校准下余弦自身 HTER 为 25.61%，两种校准池之间的差异不能归因于适配方法。

个人头仅见 4 个 Base 负身份，可能学习了针对这些背景身份的边界而不能覆盖新攻击者。当前结果与这一风险一致，但不能仅凭本实验确定因果。受限通道适配器和 0.1 Audio 权重下无收益，不代表所有残差结构或 Audio 方法无效。

all_seven_sources 包含 4 个个人训练负身份，不能称为全未见攻击。词标签依据录制顺序推断，尚未逐录音语义核验；yue 是此前探索后排除的，当前只有 8 人。跨种子复用相同查询，样本数和背景身份数有限。没有测试挑人、删难样本或选择最优测试阈值。

验证：每折个人训练/查询记录无交集；Cal-FAR05 校准约束通过；旧混合余弦与既有 P1 逐折逐指标精确一致；无 Audio 的四组结果在两教师运行间精确一致。私有特征、逐记录审计、个人参数不上传公共仓库。