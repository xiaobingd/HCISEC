# 复现去blank条件监督

输入来自原五组和权重探索的合法私有缓存；前两轮参考提交分别为 `e6ec6c1f0caefd3caf7a9a1435feeef3c19ac0ea`、`53be0e2539dcecfb9637656ae7b0f645946ffc03`。原始信号/后验/模型不在公共GitHub。

```powershell
$py = 'C:/Users/windows/AppData/Local/Programs/Python/Python310/python.exe'
$src = 'experiments/audio_phonetic_conditional_20261128'
& $py "$src/test_conditional.py"
& $py "$src/run.py"
```

新输入/输出绑定：只读 `outputs/vr261119_10x40_raw_acc/hierarchical_phonetic/` 内acc_private.npz、teacher_private.npz和纯ACC检查点；只读其同级 `hierarchical_phonetic_tuning/` 内MATCHED检查点。新私有文件写入 `hierarchical_phonetic_conditional/`，公开结果写当前新目录。既有文件不覆盖。运行前要求源数据/后验SHA与原公开审计一致。

教师条件目标只用训练/开发词的缓存后验，m=sum(q[...,1:])，q_cond=q[...,1:]/m，confidence=max(q_cond)，w=m*confidence*valid*(m>=.001)。学生取log_softmax(logits[...,1:])。辅助项为sum(w*KL)/sum(w)，sum(w)=0时精确为0。目标及权重冻结，不能将blank当作静音或将条件置信度当人工真值。合成测试覆盖解析加权KL、blank分数不变性、blank梯度0、全blank有限零损失、低质量屏蔽及联合错配。

本轮不扫λ：λphone=.01、λword=.01，完整KL和条件KL使用同一数值。开发4组（MATCHED、完整KL、条件KL、条件错配）均seed42、400步、12词训练；4开发词做固定循环2词注册K4/1词校准/1词查询，仅用于诊断。随后固定配置锁定，16词1000步、seeds42/43/44，新训练9个模型；纯ACC和上轮词对抗.01无Audio只读复用。原始网络、前端、lr、优化器、batch80、时间核、K12注册全部不改。

时间错配先在有效帧内移动完整后验，再重新计算条件目标与权重，保证二者联合置换且padding不进入有效域。只屏蔽不可信辅助帧，全部身份样本保留。查询只使用ACC；不打开任何Audio、Audio时间戳或查询音素标签。

所有15个方法/种子先校准并落盘150阈值，再计算全部913个查询窗口。阈值为本人校准Q10和其余9身份校准Q95中点。保留所有10用户、所有15查询词和攻击源；英文两个词族等权为主分析，中文单独报告。四项预定EER比较为条件KL对纯ACC、匹配无Audio、同参完整KL、条件时间错配；按用户簇bootstrap及精确符号置换、统一Holm4校正。历史查询池属于探索复核，独立新采数据才用于新的盲测确认。

`evaluate_conditional.py`独立重放仅含ACC主干的部署状态、注册模板、全部阈值和全部指标，并要求复用BASE/MATCHED与上轮公开参考逐项一致。`report.py`生成完整逐用户/词/种子匿名聚合JSON与发布清单。只上传清单里的源码、协议、日志、哈希和聚合指标；原始信号、条件后验、时间权重、私有模型、阈值数值及逐样本分数不得公开。
