# 2026-09-28 认证头诊断计划执行审计

## 结论

现有实验已经完成了“无交集数据划分”和部分 F1/F2/F3 身份探针，但还没有完成诊断计划要求的逐级可分性、PK 采样、ArcFace 对照和归一化消融。因此，当前结果足以说明 user1/bo 的多源特征有身份判别力，不能完整定位 SupCon、融合层和上游 CycleGAN 各自的贡献。

## P0：代码静态检查

### 已确认

- `SupConLoss` 的 label mask 是同标签，随后乘去掉对角线的 mask；self-contrast 被排除。
- 投影头输出经过 L2 normalization，SupCon 的相似度使用归一化后的向量。
- label 被 reshape 为 `(B, 1)`，维度和 device 处理正常。
- 没有发现正样本只来自同一窗口 augmentation 的问题；当前正样本定义是同一用户的不同样本。
- CycleGAN 生成器被冻结，认证头只训练 fusion 和 projector。

### 尚未满足计划

- 当前使用普通 `shuffle=True` 的 batch，没有 P-K sampler，因此没有保证每个 anchor 都有正样本。
- 代码只计算并跳过 zero-positive anchor，没有记录每个 batch 的 identity 数、有效正样本数和 zero-positive 比例。
- 已训练结果只使用 SupCon；ArcFace 和 SupCon+ArcFace 尚未跑同协议对照。

## P1/P2：特征和融合诊断

已有 frozen probe 给出部分结果：

| target | ACC shallow AUC | bottleneck AUC | cycle-diff AUC | combined AUC |
|---|---:|---:|---:|---:|
| user1/bo | 0.874 | 0.917 | 0.859 | 0.912 |
| user2 | 0.678 | 0.615 | 0.628 | 0.686 |
| user3 | 0.604 | 0.529 | 0.521 | 0.565 |
| user4 | 0.736 | 0.815 | 0.582 | 0.836 |
| user5 | 0.693 | 0.748 | 0.684 | 0.751 |

这支持 F1/F2/F3 中存在身份相关信息，尤其 user1 和 user4；但还缺少计划要求的：

- F1、F2、F3 各自及组合的 `S_intra`、`S_inter`、`ΔS`；
- fusion 512 维、projection 128 维、最终 embedding 的逐级变化；
- F2 是否因 512 维和 `gap_std` 聚合主导未归一化 concat；
- source-wise L2 normalized fusion 和不同权重的对照。

当前 `modelb.py` 直接拼接 `128d + 512d + 128d + 4d`，之后进入 fusion MLP，没有 source-wise normalization，因此计划中关于尺度主导的风险真实存在。

## P3：最小 baseline

目前已有 frozen feature 的线性 probe 和最终 SupCon 认证结果；nearest prototype、kNN、Linear CE、单独 ArcFace、单独 SupCon 与 SupCon+ArcFace 的完整同协议表还没有完成。因此不能判断复杂认证头是否优于简单 prototype/CE。

## P4/P5：数据和测试协议

这部分目前基本符合计划：

- 每用户 150 个样本用于 CycleGAN，剩余样本用于认证；代码显式检查两部分无交集。
- 认证数据进一步拆成 `auth_train`、`model_validation`、`target_dev_pool`、`final_test`；组间有交集检查。
- head 选择使用 `model_validation` separation；注册模板和 threshold 使用 dev 数据；final genuine/impostor 只在冻结后评估。

仍需补充的审计：sample ID 目前是 `user:index`，尚未检查滑窗是否来自同一 parent recording；多 session 是否 session-disjoint 也没有元数据证据。

## P6：CycleGAN 上游

已经发现 user1/bo 的原始 ACC 分布明显偏离其他用户，且 shallow ACC 已有很高 AUC；这解释了它的部分优势。尚未完成计划要求的 per-sample min-max、train-set global min-max、z-score 等统一归一化消融，因此还不能把当前结果完全归因于 CycleGAN 身份表征。

## 当前判断

按照计划的决策规则，当前证据最接近“F1/F2/F3 有一定身份可分性，但融合和原始 ACC shortcut 的贡献尚未拆开”，还不能进入“SupCon 超参数继续调优”阶段。优先级应为：

1. 记录随机 batch 的 zero-positive 比例并加入 P-K sampler；
2. 导出 F1/F2/F3/fusion/projection/final 全链路的 intra/inter cosine；
3. 完成 source-wise normalization、F1/F2/F3 组合和 prototype/kNN/Linear CE baseline；
4. 在同一固定协议上补 ArcFace 与 SupCon+ArcFace；
5. 最后做幅值归一化消融，验证 user1 优势是否是 ACC 全局幅值捷径。


