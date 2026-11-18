# 超越 CycleGAN 的研究与首轮实验（2026-10-17）

## 现有证据

之前四组纹理/配对判别实验没有通过质量门槛：验证频率细节相关约0.405–0.426，低于训练集平均音频模板0.536；循环相关仅0.429–0.449。不能把较低像素误差或更多纹理当成恢复成功。当前原始IMU数据来自同一个人，不能从这组数据证明身份认证效果。

## 文献筛选与迁移条件

1. **Fogerty et al., JASA Express Letters, 2026-02，Parameter efficient speaker adaptation for enhancement of bone-conducted speech**。已核验期刊信息与作者机构摘要，未完整复现正文。研究小数据条件下的说话人个性化适配，适合借鉴“公共语音先验固定、小型个性化部分训练”的分工；摘要不能支持断言其具体采用LoRA。
   https://experts.illinois.edu/en/publications/parameter-efficient-speaker-adaptation-for-enhancement-of-bone-co/
   DOI:10.1121/10.0042462

2. **Hauret et al., WASPAA 2025 demo，Real-time speech enhancement in noise for throat microphone using neural audio codec as foundation model**。已读三页论文。使用预训练Mimi，冻结干净音频参考编码器、微调另一编码器，进行潜变量L1匹配。其喉麦输入、38小时Vibravox、188人和我们的1kHz IMU/单人少量录制差异很大，论文成绩不能搬过来。论文也报告困难片段可能出现音素幻觉。因此先做真实音频编码解码上限，再做ACC条件恢复，检查是否只是语音先验生成了好看的结构。
   https://www.waspaa.com/waspaa25/proceedings/WASPAA2025-292.pdf
   https://arxiv.org/abs/2508.02974

3. **Zheng et al., CycleCodec, Interspeech 2026**。作者机构页面标记2026-06-04接收/待刊，会议2026-09-27至10-01。核验摘要，未获取完整正文。把内容和说话人因素分开，采用说话人交换循环、Transformer聚合器和说话人对比损失。更贴近身份特征目标；但需要多说话人训练，当前单人数据不能完成这类说话人分离实验。后续应使用公开多用户预训练数据，并严格隔离本地攻击者与注册者保留集。
   https://www.research.ed.ac.uk/en/publications/cyclecodec-distillation-free-factorized-neural-speech-codec-via-c/

4. **Khanagha et al., Bone-conduction Guided Multimodal Speech Enhancement with Conditional Diffusion Models, 2026**。arXiv2026-01，作者机构ICASSP2026列表确认会议收录。推理同时使用带噪空气麦克风和骨导输入，不能直接当ACC-only方案。可作为以后“训练时双模态教师→推理时仅ACC学生”的参考，这一步蒸馏属于我们的改编而非论文原方法。
   https://arxiv.org/abs/2601.12354
   https://www.inf.uni-hamburg.de/en/inst/ab/sp/newsfeed/2026icassp.html

5. **DBMIF, arXiv2603.02877, 2026-03预印本**。空气/骨导/融合三分支与迭代交互。运行条件同样包含麦克风；不是ACC单输入结果。暂不优先。
   https://arxiv.org/abs/2603.02877

6. **VibOmni, arXiv2512.02515, 2025-12预印本版本**。个体骨导响应和合成振动增强值得参考，但完整增强方案也融合麦克风输入。不能把其响应补偿直接用于抹平用户差异；用户传导响应正是本任务希望保留的候选身份线索。
   https://arxiv.org/abs/2512.02515

## 首轮实际实现：冻结预训练解码器的ACC潜变量适配

补充两篇2026年相关已发表研究：

- **Zheng et al., Sensors26(4):1137, 2026-02-10，Enhancing Bone Conduction Sensor Signals via Self-Supervised Acoustic Priors and Key-Value Memory**。出版社检索文本与PubMed摘要说明：Wave-U-Net时间域主干、预训练SSL语音先验、Key-Value Memory和门控融合；骨导查询可以在没有空气麦克风参考的推理阶段检索先验，适合作为后续第二条路线。正文直接访问遇429，不能声称已完整核验/复现全部实现。用于认证时应隔离记忆库与测试集，检查检索是否强行注入库中说话人音色；原论文增强指标不等于身份验证效果。
  https://www.mdpi.com/1424-8220/26/4/1137
  https://pubmed.ncbi.nlm.nih.gov/41755077/
- **Hu et al., IEEE Internet of Things Journal13(10):21596–21612, 2026-05期，A Neural Codec for Bone-Conducted Speech With Joint Denoising and Bandwidth Extension**。出版社摘要可核验MP-SENet/BigVGAN、双向时频Transformer、KalmanNet和幅相交互；尚未获取全文或权重。可借鉴“动态时序＋幅相共同建模”，不能把我们的简单模块称为其复现，也不能认为双向结构适合在线实时推理。
  https://ieeexplore.ieee.org/abstract/document/11411718
  DOI:10.1109/JIOT.2026.3667945

这是自主小规模改编实验，不是上述论文复现。官方Mimi safetensors固定到提交89091b3e466eb6a9d11e537bf26b144f194978f7。完全冻结编解码器，干净麦克风只用于训练目标和标为oracle的解码上限，学生推理只接受ACC。

- 数据：现有个人原始IMU，同一套原有预处理，20个训练录制58窗口、4个验证录制16窗口；未导出测试与28个认证保留录制。原有ACC数组不变。
- 解码上限：真实音频→Mimi码→冻结解码器，不能当ACC生成结果。
- 先验对照：训练集平均潜变量→同一解码器，检查是否只产生平均语音。
- 学生1：ACC局部上下文→岭回归→音频潜变量。
- 学生2：小型时序卷积适配网络→音频潜变量，按录制均衡采样，300轮，验证潜变量MSE选轮。
- 学生3：同一网络加跨录制配对对比损失。负配对来自同一人的其他录制，仅是内容约束，不是身份负类。与学生2使用同随机种子/数据/更新次数。
- 补充学生4/5：训练录制音频潜变量作为固定值库、ACC查询与音频键投影学习，带/不带同人跨录制对比约束；训练时屏蔽查询所属整个录制的所有窗口，验证只访问训练库。借鉴2026键值记忆思路，使用Mimi特征而非论文SSL模型，不是复现。
- 补充学生6：从回归学生初始化，冻结解码器，梯度回传至小适配器，40轮。目标为原尺度Mel L1＋0.3频率差分L1＋0.1包络L1＋0.05潜变量MSE；验证以同频谱损失选轮。避免仅匹配可能包含不可预测相位的高维潜变量。无GAN或CycleGAN损失。
- 模型目标为完整codec量化潜变量，包含全部可用码本，而不是仅语义token。预测连续潜变量直接进入冻结解码器；这不同于原论文微调整个输入编码器。
- 输出按原16kHz、512FFT、160hop、80Mel和固定训练标定作图。24kHz codec带来的重采样误差另设对照；所有图使用相同色标。

## 判定规则

先检查oracle是否保留真实结构；若oracle失败，不把失败归于ACC学生。学生至少应优于平均音频/平均潜变量，正确配对优于错误录制配对，并改善频率细节和时间包络，而非仅降低MAE。验证集已多次使用，本轮为探索性结果；不作独立泛化结论。若这些条件未达到，保留失败结果，不能宣布改进。

重建与身份认证分开：任何改善都须在后续多用户、录制隔离、统一预处理、阈值仅由注册训练/验证数据确定的认证试验中验证。不得为好看删除被误接受的攻击测试样本。

## 执行状态

2026-10-17：本地波形导出完成，原16k参考Mel最大偏差4.22e-6；24k往返重采样平均偏差0.000886，单点最大0.1774，故保留重采样控制。服务器隔离deps安装完成（transformers4.46.3适配Python3.8），Mimi导入通过；不修改原torch环境。官方385MB权重现已完成下载与服务器校验。

后续状态：官方权重已下载/上传，SHA256本地与服务器一致，首轮回归/对比与岭回归均已完成；oracle音频MAE0.1132、包络相关0.9779，但学生输出近似平均，未成功。随后补充记忆网络和解码器频谱监督。初轮采用二阶频率差分衡量细节，留存first_pass_second_difference_results.json；最终统一回原实验九频点反射均值高通与所有跨录制错误配对评价，避免与旧纹理指标混用。

最终状态：六种ACC学生全部完成，结果见report.md/results.json。没有通过纹理门槛的候选；测试与认证保留集未打开。无本轮训练后台任务待完成。
