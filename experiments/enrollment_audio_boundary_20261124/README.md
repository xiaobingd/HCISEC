# Audio 条件化注册样本选择（下一步实验）

**状态：实验协议已固定，私有数据上尚未执行。** 不涉及跨会话划分、全局 ACC 重训或新的 Audio→ACC 蒸馏。

## 研究问题

此前 Audio+ACC 50/50 注册质量排序在部分英文 RBF 认证指标上有改善，但不稳定；注册 Audio/ACC 质量排序 Spearman 平均 -0.0179，余弦模板未复现稳定改善。本实验只检验 **当 ACC 已经完成主要注册样本选择时，正确 Audio 能否改善最后一个边界候选的选择**。

## 固定实验

- 10 位已见用户，3 个冻结 ACC 编码器种子 42/43/44，4 个注册词 [20,21,22,23]，每词 8 条，共 32 条候选。
- ACC 跨词质量分：样本与本人其他 3 个注册词 ACC 均值原型的余弦；从高到低排序。
- 每词保留 ACC 排名第 1、2 条；第 3 条从 ACC 排名 3、4 的两个候选中选取，因此每人保留 12 条 ACC。
- `acc_top3`：取排名 3；`audio_boundary`：在第 3/4 名中选择本人其他三注册词 Audio 原型余弦更高者；`coin_boundary`：随机选择；`shuffled_boundary_0..4`：同人同词 8 遍 Audio 分数无固定点置换后选择。所有规则在查询前固定，禁止按结果调权重。
- Audio 用之前 320 条注册录音提取的冻结 ECAPA 192 维缓存；不读取校准或查询 Audio。
- 同一 ACC 编码器输入、每人 12 条正类、其他 9 人共 288 条负类全部固定。主后端 RBF SVM C=10 balanced；次后端 12 条 ACC 的平均余弦原型。
- 与此前注册实验的单条及三条跨词协议完全匹配：校准词 [8,18,24,30,31]；三组查询 [25..29]、[3,9,12,19,32]、[35..39]。校准阈值为本人 Q10 与攻击者 Q95 中点，先锁定全部阈值后计算测试指标。
- 主报告：HTER、EER、FAR、FRR；十位目标用户簇 bootstrap，逐种子及逐用户完整保留；报告实际被 Audio 换出的注册样本数量及随机对照。
- **必须复现此前 `acc_crossword12` 基线到 1e-8 以内**，否则先修复协议，不能拿不匹配的结果归因。

## 成功/停止标准

只有正确 Audio 同时优于 ACC 排名、随机选择和多数打乱 Audio，且跨用户与种子稳定、不能仅依赖单个用户或 RBF 后端，才认为 Audio 有候选边际注册价值。否则停止当前 Audio 排序辅助注册，不继续搜索融合权重。

局限：10 位已见用户；历史查询词已反复查看；3 秒 ACC 窗口可能重叠并受 VAD 中心影响，物理同步未标定。Audio 方法仍需要采集或读取所有注册 Audio，不能把“同样使用 12 条 ACC”误写为相同传感器采集成本。

## 私有运行条件

在原始实验环境中执行同目录 `enrollment_audio_boundary_tiebreak.py`。指定 `--root` 为含下列数据/模块的目录：

```
phrase_auth/frontend_private.npz
conditional_correspondence/stage3/views_private.npz
conditional_correspondence/enrollment_identity_quality/registered_ecapa_private.npz
conditional_correspondence/enrollment_identity_quality_fixed_background/summary_public.json
calibrate_threeword.py
prepare_phrase_auth.py
run_strict_within_content.py
```

```bash
python experiments/enrollment_audio_boundary_20261124/enrollment_audio_boundary_tiebreak.py --test
python experiments/enrollment_audio_boundary_20261124/enrollment_audio_boundary_tiebreak.py --root /ABS/PRIVATE/ROOT
```

私有选择、阈值、逐样本结果不上传。仅将匿名汇总 `summary_public.json` 经隐私审查后发布。尚无本实验实测性能，不能杜撰 EER/HTER。
