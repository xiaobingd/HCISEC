# 扩大最终测试集后的认证结果

日期：2026-09-28。远程实验目录：`/root/autodl-fs/auth_forward_aligned_20260928_test20`。

## 切分与防泄漏

CycleGAN 仍使用原 150 个样本；CycleGAN 未见过的认证池按 50% 训练、10% 模型验证、20% 阈值开发、20% 最终测试重新切分。认证头从头重训，固定 head seed=42。阈值只从 dev/calibration 数据选择，final 数据仅用于冻结评估。每用户 manifest 记录样本索引并检查各组无交集。

因原始可用认证数在用户间略有差别，认证头训练实际为每类约 74–75 个（共 670），模型验证 135 个。目标 genuine 最终测试由 15 增至 29–30；攻击测试由 240 增至 358–359 个，覆盖其余八个来源。

## 冻结测试结果

五个注册随机种子使用同一个已训练头；表中 FAR/FRR/HTER/AUC 为五次注册抽样的均值。

| 目标用户 | Genuine 测试 | Impostor 测试 | FAR | FRR | HTER | AUC |
|---|---:|---:|---:|---:|---:|---:|
| bo / user1 | 30 | 358 | 5.87% | 13.33% | **9.60%** | 0.977 |
| user4 | 29 | 359 | 22.51% | 33.10% | **27.81%** | 0.808 |
| user5 | 30 | 358 | 31.84% | 44.67% | **38.26%** | 0.697 |
| 宏平均 | — | — | 20.07% | 30.37% | **25.22%** | 0.827 |

HTER 相对之前的 15-genuine/240-impostor 结果分别为 user1 12.0% → 9.6%、user4 24.4% → 27.8%、user5 33.9% → 38.3%。这不是只改变测试集大小的配对实验：新方案把头训练数据从约 90/类降到 74–75/类，并更换了 auth split，因此这些变化不能归因于测试样本数本身。user1 FRR=13.33% 是 30 个 genuine 中 4 个拒绝；user4 FRR=33.10% 是 29 个中约 9.6 个拒绝的五种注册种子均值；五种注册模板造成小数均值是预期现象。

较大的测试集让 FAR 估计更稳定，也把 FRR 的单样本分辨率从约 6.7 个百分点提高到约 3.3–3.4 个百分点。user5 仍是当前主要瓶颈。

## ROC

![Expanded final-test ROC curves](roc_expanded_test.png)

阴影显示五种注册抽样的 TPR 标准差；AUC 与阈值无关。HTER 使用开发集选出的冻结阈值。

## 产物

- `evaluation.json`：逐用户、逐注册种子指标和 ROC 点。
- `user{1,4,5}/checkpoints/split_manifest.json`：每组样本索引和隔离检查。
- `user{1,4,5}/train.log`：新头训练日志。
- `auth_test20_results.tar.gz`：上述服务器结果的归档副本。
- 远程脚本在 `work/target_auth_protocol_150_test20.py`、`work/trainb_target_150_test20.py` 和 `work/evaluate_target_150_test20.py`。

