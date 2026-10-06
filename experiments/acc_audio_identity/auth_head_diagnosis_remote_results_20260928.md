# 认证头诊断计划：远程实测结果

本次使用远程服务器上的 user1/bo、user4、user5 固定 CycleGAN checkpoint 和现有认证头，在 `model_validation` 集上导出 F1/F2/F3、fusion 和 projection，并统计随机 batch 的 SupCon 正样本覆盖。

## 1. 逐级身份分离

| target | feature | S_intra | S_inter | ΔS |
|---|---|---:|---:|---:|
| user1/bo | F1 shallow | 0.999819 | 0.999807 | 0.000012 |
| user1/bo | F2 bottleneck | 0.998792 | 0.998781 | 0.000011 |
| user1/bo | F3 cycle-diff | 0.998447 | 0.997654 | 0.000792 |
| user1/bo | fusion 512d | 0.292985 | 0.062600 | 0.230385 |
| user1/bo | projection 128d | 0.581867 | 0.476043 | 0.105824 |
| user4 | F1 shallow | 0.999823 | 0.999810 | 0.000013 |
| user4 | F2 bottleneck | 0.998777 | 0.998761 | 0.000017 |
| user4 | F3 cycle-diff | 0.998302 | 0.997981 | 0.000320 |
| user4 | fusion 512d | 0.250952 | 0.025057 | 0.225895 |
| user4 | projection 128d | 0.521981 | 0.431646 | 0.090335 |
| user5 | F1 shallow | 0.999824 | 0.999811 | 0.000013 |
| user5 | F2 bottleneck | 0.998800 | 0.998784 | 0.000017 |
| user5 | F3 cycle-diff | 0.998669 | 0.998488 | 0.000181 |
| user5 | fusion 512d | 0.306125 | 0.062738 | 0.243388 |
| user5 | projection 128d | 0.476092 | 0.356439 | 0.119654 |

## 2. 解释

F1/F2/F3 的向量都集中在相近方向，直接余弦的 ΔS 几乎为零。此前的 frozen linear probe AUC 较高，说明身份差异主要是小幅度、维度方向或非线性组合差异，不能用未经训练的原始余弦直接证明。

fusion 层的 ΔS 达到 0.226–0.243，说明可训练 fusion 确实提取出了身份分离；但 projection 后只剩 0.090–0.120，说明投影头在压缩信息时损失了较多分离度。由于 fusion checkpoint 是按 model-validation separation 选择的，这个提升还需要在最终测试集逐级复核，防止把 validation 拟合当成泛化能力。

## 3. SupCon batch 检查

在 batch size=32、500 个随机 batch 下，三个 target 的结果相同：

- 平均 identity 数：8.824/9；
- 平均每个 anchor 的正样本数：3.388；
- zero-positive anchor 比例：**2.41875%**；
- 最少正样本数：0。

因此当前普通 shuffle batch 并不能保证 SupCon 的每个 anchor 都有正样本。损失函数会跳过这些 anchor，但有效 batch 大小会随机变化。应改成 P-K sampler 并记录实际覆盖率。

## 4. 融合尺度

三路特征的平均 L2 范数近似为：

- F1 shallow：4.31；
- F2 bottleneck：16.02；
- F3 cycle-diff：4.61–4.66；
- 未归一化 concat：17.22。

F2 维度为 512，F1 为 128，F3 为 132；当前代码直接 concat，没有 source-wise L2 normalization。因此 F2 贡献了大部分输入能量，必须补做 source-wise normalized fusion 才能判断三路特征是否真的互补。

## 5. 诊断结论

当前证据支持：

1. CycleGAN 中间特征包含身份相关变化，但未经认证头处理时直接 cosine 几乎不可分；
2. fusion 是当前真正产生身份分离的环节，projection 反而削弱了分离；
3. 普通随机 batch 存在约 2.4% zero-positive anchor，是 SupCon 训练策略缺陷；
4. F2 的尺度和维度明显主导当前融合，尚不能宣称三路信息等权互补；
5. 下一步应先做 P-K sampler、source-wise normalization、fusion/projection 的最终测试集复核，再比较 prototype/kNN/Linear CE、单独 SupCon 和 ArcFace。


