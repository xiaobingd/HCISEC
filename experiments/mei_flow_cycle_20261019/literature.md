# 从2025–2026语音恢复研究寻找完整跨模态循环的改进

检索核验日期：2026-10-19。只以原论文、会议/出版社页面和作者官方仓库为依据。重点是与ACC输入对应的细节和可验证的信息保留。未采用检索返回的2027卷期文章作为已正式发表的2026成果。

|研究及时间|真正解决的问题|可迁移的思路|本项目必须区别的条件|
|---|---|---|---|
|[Real-time Speech Restoration using Data Prediction Mean Flows](https://arxiv.org/abs/2605.16251)，2026-05预印本|低计算量的非线性退化语音恢复，比较速度预测、数据预测及Mean Flow|从有条件的初始估计出发，多步预测目标数据；在小模型上测试DP目标|原文处理复数压缩STFT和退化语音，不是ACC；本轮只实现普通DP-FM，不实现Mean Flow的JVP恒等式，也不声称实时|
|[FRSE: Flow-matching with residual-decoupled learning for speech enhancement](https://doi.org/10.1016/j.specom.2026.103449)，Speech Communication183，2026-09卷期|把语音残差恢复和Gaussian扰动去除分开学习|用两个输出场分别预测真实Mel与条件估计之差、初始随机扰动|已核验出版社摘要和引言，未取得全部方法正文；本轮双头实现是依据“残差/噪声解耦”思想自行设计，不能标作FRSE原算法复现|
|[Stream.FM](https://arxiv.org/abs/2512.19442)，2025首发、2026修订；[作者代码](https://github.com/sp-uhh/streamfm)标注TASLP2026|流式生成语音恢复，包括带宽扩展、Mel声码器等任务|以生成过程替代一次确定性回归；专门检查少步推理和条件依赖|其语音先验与大规模训练、波形评估不能由单人几分钟数据的小试验替代|
|[PASE](https://ojs.aaai.org/index.php/AAAI/article/view/40562)，AAAI2026；2025预印本|减少语音增强中的语言和声学幻觉|内容先验与低层声学信息分路；不要用语言内容代替身份和韵律|原模型用WavLM及大规模语音训练；本轮没有加载这些权重，也没有复现PASE或声码器。后续可将MIC仅作训练期教师，推理仍仅ACC|
|[Bone-conduction Guided Multimodal Speech Enhancement with Conditional Diffusion Models](https://arxiv.org/abs/2601.12354)，2026-01预印本|用骨传导和空气传导麦克风共同增强噪声语音|生成模型可以接收骨传导条件；条件设计比只匹配纹理重要|原方法推理时需要MIC+骨传导，不能直接声称是ACC-only生成；不据其结果推算本项目效果|
|[Sensing Bone-Conducted Speech with Earbuds](https://arxiv.org/abs/2609.02165)，2026-09预印本，作者注明期刊已接收|测量自声振动的空间方向和有效频带|把轴间响应、传导带宽和噪声底分开分析，避免不分信噪比地强迫恢复所有细节|耳机专用传感器、17被试、不同采样和佩戴条件；其约400Hz转折不能套作HTC VIVE统一截止频率，也不能据此删除本项目频带|
|[Enhancing bone-conducted speech with spectrum similarity metric in adversarial learning](https://doi.org/10.1016/j.specom.2025.103223)，Speech Communication2025-05|骨传导语音的失真、清辅音缺失和精细空气声谱预测|源激励/滤波响应分解、成对频谱相似性评价|依赖骨传导语音及WORLD声码器；现有VR ACC并非同等质量音频输入。不能把个体响应直接归一化抹去|
|[LAU-Net](https://www.isca-archive.org/interspeech_2025/song25d_interspeech.html)，Interspeech2025|皮肤ACC辅助MIC的轻量谐波注意力和相位融合|让真实发声相关条件参与细节恢复，而不是无条件添加条纹|原模型也需要MIC；不预设两个模态必须具有同一基频结构。本项目旧显式F0路径已有失败记录，不优先重复|
|[Contrastive Knowledge Distillation for Embedding Refinement in Personalized Speech Enhancement](https://arxiv.org/abs/2601.16235)，2026-01预印本|个性化语音增强中的轻量说话人嵌入更新|有可用身份教师时，直接约束输入与重建的说话人表示，区别内容匹配与身份匹配|单用户本轮生成实验不能证明身份保持；不能强制任意攻击者输入都生成注册者声纹。需要未来严格隔离的注册/攻击验证|

## 本轮先实际验证什么

先验证“有条件先验＋分布生成”是否比继续压像素误差有效。冻结前轮相同种子2秒基础生成器作为条件估计，比较相同主干的直接回归、速度预测FM、数据预测DP-FM；追加双输出头的残差/噪声解耦FM。FM在约2秒Mel域做5步采样；四种方法均保留ACC→生成Mel→ACC及反向音频循环。F只读取生成Mel，没有ACC/潜在旁路。

本轮不增加公开/LLM/扩散伪负样本，不导入外部预训练权重。训练仍只用原来的180个独立mei训练录制；60个验证录制已被反复探索过，只能作探索比较。所有人的数据处理规则保持一致，未调整源时间戳或逐人做增益/滤波。

对每个随机生成模型固定3个噪声抽样，平均每次的指标，不把3张频谱平均成一张图，也不从中挑最像真实音频的结果。比较错录制ACC和训练均值ACC时使用相同噪声；检查零噪声、1步与5步，以及数值Mel量化/平滑后循环ACC的变化。图像有纹理而配对对应更差、误发声增加或依赖噪声而不是ACC，都视为不成功。

这只是迁移研究思想的小规模试验。改善Mel图不等于有可听语音、正确语音内容或保留身份；本轮不报告声码器听感、PESQ/STOI及身份准确率。
