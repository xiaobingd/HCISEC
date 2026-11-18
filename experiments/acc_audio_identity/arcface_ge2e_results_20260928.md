# ArcFace/AAM 与 GE2E 初筛结果

在与 source-wise normalized + P=9/K=4 相同的特征和数据划分上，比较单次 SupCon 参考、ArcFace 和 GE2E。ArcFace 设置 `m=0.2, s=30`；GE2E 使用 batch 内 speaker centroid 和 leave-one-out genuine centroid。阈值由 dev/calibration 数据选择，final test 只做冻结评估。

## Final test

| 用户 | source-wise SupCon参考 HTER | ArcFace HTER / AUC | GE2E HTER / AUC |
|---|---:|---:|---:|
| user1/bo | 11.5% | 19.4% / 0.862 | **11.3% / 0.972** |
| user4 | 20.0% | 34.8% / 0.683 | **20.2% / 0.841** |
| user5 | **32.5%** | 41.0% / 0.653 | 33.9% / 0.693 |

SupCon 参考来自此前 source-wise normalization、P-K、单 seed 结果；ArcFace 和 GE2E 也各为单 seed，因此本表用于筛查方法，不可视为稳定统计比较。

## 结果判断

- GE2E 对 user1/bo 与 user4 接近 SupCon，user1 AUC 较高；尚未稳定证明能降低 HTER。
- user5 上 GE2E HTER 为 33.9%，略差于 SupCon 参考 32.5%，仍需重点处理 user8/user9 hard impostors。
- 当前 ArcFace 设置明显较差，特别是 user4、user5；小样本下 `m=0.2, s=30` 可能过强，或者 embedding/head 与 ArcFace classifier 联合训练不稳定。下一轮应先做 margin/scale 小范围验证，或暂不选作主方法。
- 当前没有跑 proj256；本任务结果只回答 ArcFace/AAM 与 GE2E 对照。


