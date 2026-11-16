# 创新点一正式定版 V1：跨模态循环一致性特征身份认证

- 版本：`I1-CYCLE-FROZEN-20261116-V1`。
- 状态：用户确认定版；模型、特征位置、融合权重和本页评估口径固定。
- 定版结果：同词HTER **5.970982%**，跨词HTER **10.431548%**。
- 方法：**既有312维身份基线＋个人CycleGAN的input256特征＋二分类RBF SVM**。
- 结果范围：**九人参与原训练；最终只统计八名目标，并移除U5的最终攻击查询**。U5仍参与原训练、RBF负类注册及阈值校准。不能将本版本写成“全部阶段仅使用八人”。

## 1. 定版结果与方法名称

|方法|同词HTER%|跨词HTER%|用途|
|---|---:|---:|---|
|原始ACC statistics720＋RBF|9.728423|12.183780|原始统计对照|
|原baseline312＋RBF|6.268601|10.524554|不补充CycleGAN特征的直接对照|
|**baseline312＋个人input256，alpha=0.5**|**5.970982**|**10.431548**|**正式定版**|
|baseline312＋个人cycle256，alpha=0.5|6.138393|10.323661|循环后编码诊断，不替换定版|
|cycle256单独认证|22.581845|24.404762|单分支诊断|

定版方法对应逐用户文件中的`selected_input256`。`input256`是受循环一致性训练的G编码器表示；`cycle256`是ACC→Audio→ACC之后再经G编码器提取的表示。两者不是同一个特征位置。

|定版指标|同词|跨词|
|---|---:|---:|
|TAR%|95.572917|88.072917|
|FAR%|7.514881|8.936012|
|FRR%|4.427083|11.927083|
|HTER%|5.970982|10.431548|
|描述性Test-EER%|3.348214|6.216518|
|宏AUC|0.9799107143|0.9643229167|

相对baseline312，HTER降低约0.30/0.09个百分点（同词/跨词）。这组数字描述限定评估范围的完整系统表现，不把完整系统相对原始统计的全部收益归给新增CycleGAN分支。

## 2. 数据与处理

原始清单按时间戳及载荷哈希去重后，统一前端有2695条配对记录、九个身份。名义30词、每词约10次重复：前10词是英文数字，后20词为VR指令。缺失录制不补造；词标签按采集顺序映射，未逐条独立转写核验。

本文的ACC输入来自三轴OpenVR tracking `vVelocity`，不是硬件原始加速度。保留XYZ独立通道，时间戳排序去重，插值为1kHz、2秒，逐轴移除DC；STFT窗256、hop64、FFT1024，插值到10–300Hz的40个几何频率点，得到3×40×28频谱。所有身份处理方法一致。

CycleGAN前端将频谱时间轴插值到32帧，得到3×40×32。按原fit数据计算每轴全局均值/标准差，除以3倍标准差并截断到[-1,1]。插值率不等于设备模拟传感器带宽。

Audio使用原48kHz双通道缓存的左通道，下采样16kHz、DC移除；生成目标采用4096窗、hop1024、FFT16384，50–7600Hz的40个几何频率点及32帧时间轴，再使用原fit全局均值/标准差归一化。该频带仅为Audio目标，不是ACC分析频带。

## 3. 原模型训练：保持原方案

### 3.1 312维基础身份表示

从统一ACC频谱构造738维输入：120维去轴内整体电平的频谱均值、120维时间标准差、480维四段相对动态、3维整体电平和15维幅度统计。Scaler仅拟合原fit。

ACC学生为738→192投影，加192→64→192残差，输出L2归一化192维身份向量。先训练700步CE＋0.1 SupCon；再使用既有ECAPA教师的训练/旧注册Audio目标进行100步身份蒸馏。教师本地主体微调100步后冻结，学生使用身份损失、锚定和梯度平衡RKD，目标梯度比rho=0.3。教师/学生均为既有定版检查点，本次没有更新。

取原标准化输入的前120维作为形状分支，L2归一化；与192维学生拼接：

\[
b(x)=\frac{[s_{120}(x);e_{192}(x)]}{\sqrt{2}}\in\mathbb R^{312}.
\]

原教师使用144条合法训练回放/旧注册Audio目标，每人16条。训练细节以归档`mei9_closed_personal_cycle.py`及CPU适配入口为准；不存在最终查询Audio模型前向。

### 3.2 共享CycleGAN

G：3通道ACC→1通道Audio；F：1通道Audio→3通道ACC。编码通道32/64/128、两次下采样、4个128通道残差块；解码使用两次上采样，输出Tanh。两个Patch判别器采用LSGAN。

\[
L_G=L_{GAN}+10L_{paired}+10L_{cycle},
\]

\[
L_{paired}=\|G(a)-m\|_1+\|F(m)-a\|_1,
\qquad
L_{cycle}=\|F(G(a))-a\|_1+\|G(F(m))-m\|_1.
\]

原fit共1438条，名义词0–7及10–17；原dev共36条，为每人词8/9/18/19各一条。共享训练4000步、batch16、Adam lr2e-4、betas=(0.5,0.999)，后2000步线性衰减，梯度裁剪5。按原dev四项配对/循环误差之和选择检查点，实际选择4000步。seed42。

### 3.3 个人微调与模型绑定

每个用户从共享权重独立初始化G/F，只使用本人旧12条配对注册数据微调100步：词20–23，每词前三条。Adam lr2e-5、betas=(0.5,0.999)、batch6，判别器冻结，保留配对/循环/生成目标。

验证声称身份u时，**本人和所有攻击者都通过同一个G_u/F_u**。不根据查询的真实身份选择生成器。

本次新的K4注册只用于RBF拟合，不再更新个人G/F，也未重新微调Audio教师。个人100步模型为固定历史适配结果。

## 4. 定版568维融合特征

G编码器末端128通道的二维特征图，分别对频率和时间做均值、标准差池化，形成256维：

\[
z_u(x)=\operatorname{L2Norm}\left(\operatorname{Scaler}_{fit}
\left([\mu(G_{u,enc}(x));\sigma(G_{u,enc}(x))]\right)\right).
\]

Scaler使用归档共享G在原fit上的统计，不根据目标测试数据重新拟合。最终特征：

\[
\boxed{v_u(x)=\frac{[b(x);0.5z_u(x)]}{\sqrt{1+0.5^2}}\in\mathbb R^{568}.}
\]

alpha=0.5、个人100步、input256均沿用原CalA/CalB选定配置。本次没有根据新测试更换配置。生成器没有新增身份CE/SupCon分支，ReDimNet2等第二点方法不属于此定版。

## 5. 固定查询的同词／跨词评估

最终词为24 Down、25 Up、26 Zoom、27 Record、28 Play、29 Reset。每词约10次重复，严格逐词使用K4注册、4条查询：

- 折0：重复0–3注册，4–7查询；
- 折1：重复4–7注册，0–3查询。

对每个查询词q，遍历其余五个注册词r，共30个有向异词组合：

- 同词：用q的独立录制注册；
- 跨词：用r的独立录制注册；
- 两条件读取完全相同的q查询、标签和攻击来源。

不先平均五个异词分数再认证，而是每词对计算指标后等权平均，避免给跨词增加额外集成优势。相同查询重复用于五个比较，不能计成新的独立录制。

### 用户范围必须按阶段区分

|阶段|身份范围|U5是否参与|
|---|---|---|
|共享模型、基础学生、Scaler原训练|九人|参与|
|个人G/F历史微调|九人各本人12条|参与其个人模型|
|RBF注册正/负类|每注册词九人各4条|仍作为其他目标的负类|
|Cal任务和阈值确定|九人|仍保留|
|最终目标宏平均|U1/U2/U3/U4/U6/U7/U8/U9|排除U5目标|
|最终攻击查询|上述八人中除当前目标外的七人|排除U5攻击|

每个词对、折、目标在最终汇总中有4条本人、28条攻击查询。xin/yue对应的U8/U9参与本定版目标统计。仅从最终统计删除U5，不是重新训练八人模型。

## 6. RBF与阈值

对每个目标及注册词拟合独立二分类SVC：RBF核、C=10、gamma=scale、class_weight=balanced。训练为目标4条正注册及其余八人各4条负注册，共36条。融合后不另拟合Scaler。

Cal固定使用词21–23，注册重复0–3、查询重复4–7，Cal不交换重复折。旧个人微调使用过0–2号重复，交换Cal会把旧训练样本放入校准查询，因此本版本明确禁止。

每目标/方法在Cal的同词与跨词等权分数池上，以FAR/FRR最接近、并列时选更低FAR的规则确定一个阈值。该阈值在全部最终词、两折、同词/跨词及移除U5攻击后保持不变。删除U5攻击没有重新校准阈值。

HTER=(FAR+FRR)/2，TAR=1−FRR；先在目标/词对/折内计算，再等权宏平均。Test-EER与AUC用于分数区分度描述；部署决策使用Cal阈值，不使用Test-EER阈值。

## 7. 核查结果

1. 旧同词注册/查询词集合一致，录制互斥；旧跨词查询与注册词集合不同。旧同词HTER9.98%可复现，但其与旧跨词比较混入了查询词差异。
2. 原54项工作点重算最大误差1.11e-16。
3. 新固定查询实验5400项保存分数重算误差0，Cal查询和最终录制不在原fit/dev/register或个人微调索引中。
4. 源模型和Scaler按SHA256校验；U1新提特征与旧缓存432条重叠记录的最大绝对误差1.4544e-5，小于1e-4容差。
5. 最终移除U5目标及攻击后重新计算4800项，移除前的重算聚合误差0；所有本人接受率保持不变。

分数重算核查不等于独立实现第二套网络。该结果是历史已观察数据上的限定人群评估，单seed42、同会话、名义词标签；不代表未见用户开放集或跨佩戴验证。旧K12与当前K4结果不直接比较绝对提升。

## 8. 复现入口与版本

执行需要私有统一前端、归档G/F/学生、Scaler及分数文件；公开仓库不包含这些二进制数据。

```text
原训练：
python experiments/identity_kd/run_mei_device_fallback.py experiments/identity_kd/mei9_closed_personal_cycle.py \
  --frontend <private-common-frontend> --raw <private-raw-data> --out <private-original-run>

固定模型、固定查询对照：
python experiments/identity_kd/cycle_matched_content_control.py \
  --frontend <private-common-frontend> --assets <private-frozen-assets> \
  --bundle <private-frozen-bundle> --out outputs/mei_cycle_matched_content_20261116

最终U5统计口径：在项目根目录运行
python experiments/identity_kd/summarize_cycle_without_u5_attacks.py
```

固定查询实现版本：`71e05e377c7a30a4c45a276cff1fbfeb2d3f83f6`；最终统计实现版本：`e1d90322d907f706db1c0f2f02d428eb2d1cff32`。原训练曾因实例无GPU使用CPU适配器执行，保留原损失/批次并以激活重算节约内存；本轮固定查询和汇总在本地CPU完成。没有修改服务器源模型。

## 9. 归档链接

- [定版逐用户报告](../../outputs/mei_cycle_matched_without_u5_attacks_20261116/report.md)
- [精确指标JSON](../../outputs/mei_cycle_matched_without_u5_attacks_20261116/results_public.json)
- [全部八名目标明细](../../outputs/mei_cycle_matched_without_u5_attacks_20261116/per_user_public.json)
- [最终统计核查](../../outputs/mei_cycle_matched_without_u5_attacks_20261116/verification_public.json)
- [完整九人及六人固定查询核查](../../outputs/mei_cycle_matched_content_20261116/report.md)
- [原始九人K12训练报告](../../outputs/mei9_closed_rerun_20261110/report.md)
- [原训练代码](../../experiments/identity_kd/mei9_closed_personal_cycle.py)
- [CPU适配代码](../../experiments/identity_kd/run_mei_device_fallback.py)
- [固定查询代码](../../experiments/identity_kd/cycle_matched_content_control.py)
- [U5统计过滤代码](../../experiments/identity_kd/summarize_cycle_without_u5_attacks.py)
- [机器可读定版配置](../../configs/innovation1_cycle_frozen_v1.json)

后续新实验另建版本，不覆盖V1模型、指标或人群口径。cycle256诊断分支、第二点神经验证器以及排除U5重训均不替代本页定版。
