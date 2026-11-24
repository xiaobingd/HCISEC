# 条件化 Audio 身份残差审计

本轮只读取已有16个训练词缓存，未读取最终测试 Audio，也未训练新神经网络。先控制同词内容与能量活动代理，检验 Audio 是否能增量预测 ACC 跨词身份混淆。

## 实际结果

```json
{
  "correct": {
    "mse_reduction": -0.00032844450087300724,
    "hard_mse_reduction": 0.00015721008214625464,
    "baseline_mse": 0.10902737902806899,
    "audio_mse": 0.10935582352894202,
    "baseline_spearman": 0.45202629710793535,
    "audio_spearman": 0.45063942302745963
  },
  "identity_permutation_p": 0.405,
  "null_mean": -0.000619948395333245,
  "null95": [
    -0.004072976524659413,
    0.004510232614219599
  ],
  "cache_alignment": "PASS",
  "pairs": 45,
  "validation_pair_tasks": 180,
  "independent_users": 10,
  "final_Audio_read": false,
  "limitation": "Pair tasks share identities, not 180 independent subjects; permutation acts on ten user labels consistently across all words. Audio and ACC have different temporal window origins from newest continuous experiment.",
  "hard_identity_permutation_p": 0.475
}
```

MSE reduction 正值代表加入 Audio 更好。身份置换将整个用户的 Audio 一致换给另一个用户，199次都重新拟合预测器；以用户标签为置换单位，不能把重复录制/身份对当独立受试者。

本轮整体预测未改善，困难子集的微小正值也未超过身份置换。当前证据不支持将完整 ECAPA 同词相似度作为 ACC 跨词身份监督。下一轮应预先固定局部周期性、谐波形状与能量轨迹候选，再在新的留出词上验证，不能从当前验证词筛出最有利特征。

## 方法与解释边界

12词建模、4词验证，所有45个身份对依次留出，训练预测器排除这两位用户。每词所有用户使用相同词；偶数遍构建注册原型、奇数遍构建不同词查询原型。基线包含 ACC 同词关系与两用户活动代理统计，Audio 组仅增加同词 Audio 身份相似度。固定标准化＋Ridge(alpha=10)，不调参。难样本由训练词 ACC 相似度四分位数定义，不使用验证目标筛选。所有词级关系先减去该词平均值。

这是原始 ACC 统计特征的条件预测检验，不是神经 E0 的认证实验。活动代理不等于人工 VAD，均值控制不等于完全剔除内容或活动；不能把结果称为纯生理身份残差。历史16训练词与最新连续实验词划分不同，仅在自己的缓存内划分，不能借用其结果宣布最新协议泛化。缓存编码器是否记住身份也未被这个回归对照完全排除。

下一步只在真实对应明显超过身份置换时，才考虑将有支持的关系送入新词/新身份验证；若未超过，先检查原始时频发声特征而非扩大 ECAPA 蒸馏。本轮相关/预测结果均不能直接说明认证 EER 改善。

匿名逐身份对误差见 per_pair_public.json，逐用户关联身份对误差见 per_user_public.json（同一身份对会计入两位用户，不能作为独立样本）。本轮难子集由训练词决定，其结果亦须超过身份置换，不能只挑困难子集的正值作为成功依据。每词中心化均值只用八位基础身份之间的28个关系，不包含留出身份对；验证词背景用于描述性去词均值，不参与回归拟合。后续还需要新词与新身份确认。

不发布原始音频、embedding、身份映射或逐样本分数。
