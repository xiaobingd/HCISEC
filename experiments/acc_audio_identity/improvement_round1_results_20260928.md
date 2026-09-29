# 认证效果提升方案：第一轮实测

本轮使用同一 `150_cyclegan_plus_auth_v1` 协议，对 user1/bo、user4、user5 训练了 source-wise normalized branch head。三路特征分别经过 `Linear → LayerNorm → GELU → L2`，再进入 fusion；训练使用 P-K 采样（9 个用户 × 4 个样本）。比较了保留 projection128 和直接使用 fusion512 两种输出。

## 结果

| 用户 | 现有 SupCon HTER | normalized + PK + fusion512 | normalized + PK + projector128 | 现有 AUC | fusion512 AUC |
|---|---:|---:|---:|---:|---:|
| user1/bo | 0.120 | **0.115** | 0.146 | 0.960 | **0.967** |
| user4 | 0.244 | **0.200** | **0.200** | 0.810 | **0.858** |
| user5 | 0.339 | 0.400 | **0.325** | 0.758 | 0.679 |

## 判断

- user1/bo：分支归一化、P-K 采样和去掉 projector 后小幅改善，HTER 从 12.0% 降至 11.5%。
- user4：改善明显，HTER 从 24.4% 降至 20.0%，AUC 从 0.810 提升至 0.858；两种 projector 设置的 HTER 相同，但 fusion512 的 AUC 略高。
- user5：去掉 projector 反而恶化；保留 projector 后 HTER 从 33.9% 降至 32.5%，但 AUC 没有提升，说明 threshold 和分数分布仍不稳定。

## 当前结论

1. F2 尺度主导确实是问题，source-wise normalization 对 user1 和 user4 有效。
2. P-K sampler 有帮助，但不能单独解决 user5 的混淆问题。
3. “一定去掉 projector”不是普适结论：user1/user4 更适合 fusion512，user5 暂时保留 projector 更稳。
4. 结果仍是单次训练种子，下一步需要 5 seeds，并补充 ArcFace/AAM 和 GE2E，才能确定 verification-oriented loss 是否优于 SupCon。


