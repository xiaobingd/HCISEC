# 复现有限参数优化

先具备原五组实验的私有源缓存与参考检查点，见 [原实验复现说明](../vr261127_audio_phonetic_cross_content/REPRODUCE.md)。原提交为 `e6ec6c1f0caefd3caf7a9a1435feeef3c19ac0ea`。本轮不会下载教师或读取Audio，直接复用允许训练/开发词的真实冻结音素后验。

源目录 `outputs/vr261119_10x40_raw_acc/hierarchical_phonetic/` 只读；本轮私有检查点写入其同级 `hierarchical_phonetic_tuning/`。全部公开输出写入当前独立目录。不要向公共仓库提交私有NPZ/PT、身份映射、Audio/ACC、音素边界或逐样本分数。

```powershell
$py = 'C:/Users/windows/AppData/Local/Programs/Python/Python310/python.exe'
$src = 'experiments/audio_phonetic_tuning_20261127'
& $py "$src/test_selection.py"
& $py "$src/train_select.py"
```

单一入口顺序执行：冻结有限候选 → 四个开发模型各400步 → 四开发词循环交叉注册/校准/查询 → 选定一套并锁文件 → 16词1000步三种子正式训练 → 全部150阈值锁定 → 一次历史查询复核 → 独立编码器/阈值/指标重放 → 报告。

开发阶段只训练原12优化词；4个开发词每次2词注册、1词校准、1词查询，循环4次，每人固定K=4。候选A：音素0.01/词对抗0；候选B：音素0.03/词对抗0.01。基线0/0及原参数0.1/0.05也各400步以匹配预算。选择指标 `(EER+HTER)/2`，同时要求EER、HTER不差于开发纯ACC，FAR/FRR各不增加超过0.5个百分点。若均未通过，选择开发分较低的一套仅做探索性正式确认，报告明确保留未过门槛标记。

正式训练使用原16词、三个种子42/43/44、每组1000步、原batch80和初始权重。无Audio匹配对照仅关闭Audio项，保留选中的词对抗参数；词对抗为0时直接复用原纯ACC。原参数真实Audio检查点只读复用。正确Audio与循环移位Audio使用相同后验边际/有效帧。架构、学习率、优化器、全词表KL、时间核、注册K=12及校准分位数全部不改。

所有最终查询词、用户、窗口保留，英文两族等权为主分析，中文单独报告。先落盘150阈值，再计算查询。后验只含原训练/开发角色，无新增Audio读取。参数选择只用开发词；由于原查询池和原结果已被查看，本轮不是独立新盲测。

可分别重跑 `evaluate_tuning.py`、`report.py` 复核完成的检查点；已有阈值锁只允许逐值重现，不重置时间顺序。代码导出不含训练辅助头的私有部署权重，再独立提取查询/校准特征、重算全部指标，且要求既有BASE/ORIGINAL参考的全部指标与原结果一致。公共匿名JSON提供全部逐用户/词/种子聚合指标，不含逐样本分数。

依赖沿用原实验PyTorch/NumPy/SciPy/scikit-learn/threadpoolctl环境。源脚本在Windows上显式将分类标签转换为 `torch.long`。`test_selection.py` 核验不能仅凭EER牺牲HTER、FAR保护条件，以及无候选通过时的明确回退标记。
