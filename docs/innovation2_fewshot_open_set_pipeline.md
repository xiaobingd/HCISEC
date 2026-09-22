# 创新点 2：小样本 / 开放集身份认证——当前复现基线与下一步实验协议

> 记录日期：2026-09-22  
> 当前阶段：历史 CycleGAN 身份认证实验已复现；以下结果应作为**单目标用户历史基线**，而不是最终的多用户开放集实验结论。

## 1. 当前已经复现的结果应该怎样解释

当前实验的目标用户是 **bo**，其余 8 名用户作为 impostor。这个实验不是“9 个目标用户的平均结果”，也不是“9 分类准确率”。

复现得到：

| 指标 | 结果 |
|---|---:|
| Genuine accept | 29 / 30 = **96.67%** |
| FRR | 1 / 30 = **3.33%** |
| FAR | 11 / 479 = **2.30%** |
| EER（原脚本计算方式） | **2.81%** |
| Overall binary accuracy | 497 / 509 = **97.64%** |
| Balanced accuracy | **97.19%** |
| 原脚本阈值 | **0.8191** |

因此，现阶段最准确的表述是：

> **已复现的历史单用户认证基线：bo vs. 8 impostors，原脚本 EER = 2.81%。**

不能把 2.81% 直接表述为 9 用户平均 EER，也不能据此宣称已经完成多用户开放集泛化验证。

## 2. 当前模型和数据配置

### CycleGAN

- checkpoint：`models_bo150y20.1/checkpoint_epoch_200.pth`
- SHA256：`7e93d961a11dac58ae7b05fc0169b669ebdbd878d235a71a5b93ddf80c8c0806`

### Authentication model

- checkpoint：`checkpoints_auth-y-cyc-new/best_auth_model.pth`
- epoch：79
- model type：`multi`
- 特征：**shallow + bottleneck + cycle-difference**
- fusion_dim：772
- feat_dim：512
- proj_dim：128
- SHA256：`31507851afd2c1a02b1d02b5989950f45daf189b9cf25fbc666535575c2d76b4`

认证网络原始训练方式为**多用户 supervised contrastive learning**。本次复现只是加载历史权重进行测试，没有重新训练。

### 数据

- `meiy/1/acc2.npy`：bo，150 samples
- `meiy/2/acc.npy` ～ `meiy/9/acc.npy`：其余 8 用户，每人约 297–300 samples
- 总样本数：2541
- stratified split，seed = 42
- train / val / test = 1524 / 508 / 509
- bo validation 中的 30 个样本用于 enrollment

## 3. 目前最需要注意的三个实验问题

### 3.1 阈值存在 test-set leakage

原脚本通过遍历 **test scores** 得到阈值并计算 EER。这样得到的 2.81% 可以用于**复现历史结果**，但不应该作为最终部署协议下的独立测试性能。

后续正式实验应改为：

1. 在 validation set 上确定阈值；
2. 冻结 threshold；
3. 在 test set 上只进行一次最终评估；
4. 报告 FAR、FRR、accuracy / balanced accuracy，并同时保留 ROC / DET 等曲线。

因此论文中最好区分：

- **Historical reproduced EER**：复现原脚本；
- **Validation-selected operating point**：validation 定阈值、test 冻结评估。

## 4. Generator 训练数据与认证测试集的隔离尚未验证

当前 loader 的 train / val / test 划分并没有证明：

- CycleGAN generator 的训练数据与 authentication test 完全不重叠；
- 数据来自独立 session；
- 测试用户 / 测试语音与生成模型训练阶段严格隔离。

所以当前结果不能直接支持：

> “模型具有跨 session 泛化能力”  
> “模型能够泛化到真正未知用户”  
> “开放集认证性能已经得到验证”

这些结论需要重新构造实验协议后才能验证。

更严格的实验应优先采用 **session-disjoint / enrollment-disjoint** 的划分方式。如果未来有多次采集数据，最好让 enrollment、validation 和 test 来自不同采集 session。

## 5. user4 是当前最重要的 hard impostor

479 个 impostor test samples 中一共有 11 个 false accepts，而且：

> **11 个 false accepts 全部来自 user4。**

user4 的 false accept rate：

```text
11 / 59 = 18.64%
```

其他 7 个 impostor 在这次测试中均为 0 false accepts。

这说明当前系统的主要错误不是均匀分布在所有 impostor 上，而是集中在一个特定用户上。因此 user4 应该作为后续分析的重点，而不是只继续优化整体 EER。

建议比较 bo 与 user4 在以下三个特征空间中的距离分布：

- shallow feature；
- bottleneck feature；
- cycle-difference / reconstruction residual feature。

需要回答的问题是：

> **到底是哪一类特征把 user4 拉向了 bo？**

如果 cycle residual 对 bo/user4 的区分并不明显，而 shallow/bottleneck 才承担主要判别作用，那么“cycle reconstruction error 是最具身份判别性的特征”这一论断就需要收敛。

## 6. 必须做 feature ablation

当前模型同时使用：

```text
shallow + bottleneck + cycle residual
```

因此目前的 2.81% EER 不能证明 cycle residual 本身有效。

至少应完成：

| Experiment | Features |
|---|---|
| A | shallow only |
| B | bottleneck only |
| C | cycle residual only |
| D | shallow + bottleneck |
| E | shallow + cycle |
| F | bottleneck + cycle |
| G | shallow + bottleneck + cycle |

重点不是只比较最终 accuracy，而是同时观察：

- EER；
- FAR；
- FRR；
- TAR / TPR at fixed FAR；
- user4 hard-impostor FAR；
- genuine / impostor score distributions。

只有当加入 cycle residual 后，在相同数据划分和训练协议下稳定改善验证/测试性能，才能更有力地支持论文中的核心假设：

> 个体特异的跨模态传播关系能够通过循环重构残差提供额外的身份信息。

## 7. 下一步应轮换全部目标用户

当前只验证了：

```text
target = bo
impostor = other 8 users
```

下一阶段应做 9 次 one-vs-rest：

```text
target=user1, impostor=user2...user9
target=user2, impostor=user1,user3...user9
...
target=user9, impostor=user1...user8
```

每个 target 都必须独立完成：

```text
train / enrollment
        ↓
validation
        ↓
threshold selection
        ↓
freeze threshold
        ↓
independent test
```

最后报告：

- 每个用户的 FAR / FRR / EER；
- macro-average；
- mean ± std；
- worst-case target；
- hard-impostor pair。

这样才能知道 bo 的 2.81% 是具有代表性的结果，还是一个相对容易的目标用户。

## 8. CycleGAN 与 Diffusion 目前不能直接比较

此前 Diffusion 实验与当前 CycleGAN 实验不是只替换了生成模型的 paired experiment。

当前 CycleGAN：

- target = bo；
- 使用训练好的 authentication network；
- shallow + bottleneck + cycle features；
- supervised contrastive authentication；
- 使用当前这套数据和 split。

此前 Diffusion：

- 单用户 Diffusion features；
- 直接 cosine matching；
- 汇总 9 个 target users；
- 数据文件和划分方式不同。

因此：

> **CycleGAN 2.81% vs. Diffusion 39.00% 不能解释为“CycleGAN 比 Diffusion 好 36.19 个百分点”。**

两者同时改变了生成模型、特征、认证器、目标用户和实验协议。

真正公平的比较必须固定：

```text
same users
same raw samples
same train / val / test split
same enrollment samples
same authentication backend
same threshold protocol
same evaluation metrics
```

然后只改变：

```text
CycleGAN ↔ Diffusion
```

这样才能把性能差异归因于生成模型。

## 9. 推荐的论文级实验流程

建议把正式实验固定成：

```text
Raw ACC / Audio
      │
      ├── strict data split
      │     ├── generator train
      │     ├── auth train
      │     ├── enrollment
      │     ├── validation
      │     └── independent test
      │
      ├── CycleGAN feature extraction
      │     ├── shallow
      │     ├── bottleneck
      │     └── cycle residual
      │
      ├── authentication model
      │
      ├── validation threshold selection
      │
      └── frozen test evaluation
            ├── FAR
            ├── FRR
            ├── EER
            ├── ROC / DET
            ├── per-user results
            └── hard-impostor analysis
```

实验顺序建议保持为：

**历史结果复现 → 修正阈值协议 → user4 诊断 → feature ablation → 9-target rotation → session-disjoint 验证 → CycleGAN/Diffusion 公平比较 → 小样本/开放集实验。**

这样可以避免不断在同一 test set 上选择配置，最终导致 test set 实际上变成 validation set。

## 10. 当前阶段可以写进论文的结论边界

现阶段可以较稳妥地写：

> 在历史实验配置下，我们成功复现了以 bo 为目标用户、其余 8 名用户为冒充者的单用户身份认证实验。测试集中 genuine accept rate 为 96.67%，FRR 为 3.33%，FAR 为 2.30%，按照原实验脚本在测试分数上计算得到 EER 为 2.81%。进一步分析发现，全部 11 个 false accepts 均来自同一冒充用户 user4，表明系统性能受到特定 hard-impostor pair 的显著影响。由于原实验的阈值选择使用了测试分数，且生成模型训练数据与认证测试数据之间的独立性尚未完全验证，该结果目前作为历史基线使用，后续实验将采用 validation-selected frozen threshold、目标用户轮换及 session-disjoint protocol 进行更严格评估。

暂时不要写成：

> “系统在 9 用户开放集认证中 EER 为 2.81%。”

也不要写成：

> “CycleGAN 显著优于 Diffusion（2.81% vs. 39%）。”

这两个结论目前的实验协议都还不足以支持。

## 11. 复现信息

服务器目录：

```text
/root/autodl-tmp/cycle1
```

复现命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u testb.py \
  --cyclegan_ckpt ./models_bo150y20.1/checkpoint_epoch_200.pth \
  --auth_ckpt ./checkpoints_auth-y-cyc-new/best_auth_model.pth \
  --out_dir ./reproduction_20260922 \
  --no_tsne
```

主要输出：

```text
original_summary.png
reproduced_summary.png
reproduction_20260922.log
```

服务器环境：

```text
PyTorch 2.4.1+cu121
NumPy 1.24.2
scikit-learn 1.3.2
```

---

这份记录的定位是：**保存当前已复现事实、明确结果解释边界，并固定下一阶段实验协议。** 后续新的实验结果应在严格的数据隔离和 validation-threshold protocol 下追加，而不覆盖历史复现结果。
