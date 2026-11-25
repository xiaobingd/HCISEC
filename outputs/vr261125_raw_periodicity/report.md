# 原始发声周期性：条件身份残差预测

三类候选预先固定：shape=周期性6维，envelope=谐波4维，combined=两者关系组合。只读取历史16训练词的160条Audio录制，12词拟合/4词验证；不是新盲测。

|候选|基线MSE|加Audio MSE|整体改善|置换p|Holm p|难子集改善|难子集Holm p|
|---|---:|---:|---:|---:|---:|---:|---:|
|shape|0.109027|0.106998|0.002030|0.070000|0.210000|0.001815|0.780000|
|envelope|0.109027|0.108538|0.000490|0.280000|0.280000|-0.001082|0.790000|
|combined|0.109027|0.107221|0.001806|0.105000|0.210000|0.000867|0.790000|

正改善为好；正改善且Holm p<0.05才作为候选证据。

## 方法

40–450Hz滤波，60ms窗/20ms步，60–400Hz归一化自相关搜索，固定0.6置信门槛。估计的是带限周期，不能当人工真值F0。周期6维、谐波4维定义见extraction_public.json。45身份对轮流留出双方，8人训练；固定Ridge10，199次用户置换重拟合，Holm3校正。控制同词ACC关系和活动代理，困难子集只由训练词确定。

## 限制

历史PCM头48000按44100解释尚缺物理采样率验证，因此绝对F0依赖此时间基准。采用全3秒窗，无Audio引导ACC裁剪；切窗可能重叠，物理同步未验证。所有验证词已被历史查看；10人、180关系任务不是180独立受试者。此处只是描述性关系预测，不是蒸馏或EER改善。

## 本轮判断

通过整体条件预测检验的候选：[]。即使通过也只支持进一步身份训练验证，不能直接证明认证收益。若为空，不进行该关系蒸馏。原始周期跨模态一致性另见mechanism_public.json。

方法背景：归一化周期分析与YIN均基于时域重复性，但本实现不是YIN：https://pubmed.ncbi.nlm.nih.gov/12002874/ 。原始信号、向量、轨迹、身份映射不公开。

## 原始周期对应的实际结果

```json
{
  "correct": {
    "joint_voiced_fraction": 0.01213576858108108,
    "pitch_within10pct": 0.31491953023053504,
    "median_abs_cents": 695.095334592426,
    "audio_voiced_fraction": 0.19913956925675674,
    "acc_voiced_fraction": 0.02055004222972973
  },
  "same_person_word_wrongrepeat": {
    "joint_voiced_fraction": 0.012156883445945945,
    "pitch_within10pct": 0.2948328267477204,
    "median_abs_cents": 761.2650659985546,
    "audio_voiced_fraction": 0.19913956925675674,
    "acc_voiced_fraction": 0.02055004222972973
  },
  "user_macro_match_advantage": 0.11822493343426692,
  "users_with_comparable_voiced_frames": 10,
  "paired_user_signflip_p": 0.078125,
  "descriptive_periodicity_identity_relation_spearman": -0.025592271840247264,
  "identity_permutation_p": 0.675,
  "scope": "frame comparison at legacy alignment, NO lag tuning;wrongrepeat retains user/word;notphysicalsync validation;voicing intersections maydiffer;sharedidentitydescriptors notalone proof ofidentitytransfer;allcohortscalers trainwordsonly inthisdescriptiveanalysis",
  "header_rate_pitchscale_sensitivity": {
    "correct": {
      "joint_voiced_fraction": 0.01213576858108108,
      "pitch_within10pct": 0.21400608960417572,
      "median_abs_cents": 754.5527368495373,
      "audio_voiced_fraction": 0.19913956925675674,
      "acc_voiced_fraction": 0.02055004222972973
    },
    "wrongrepeat": {
      "joint_voiced_fraction": 0.012156883445945945,
      "pitch_within10pct": 0.21580547112462006,
      "median_abs_cents": 845.3671792409801,
      "audio_voiced_fraction": 0.19913956925675674,
      "acc_voiced_fraction": 0.02055004222972973
    },
    "caveat": "pitch-only rescaling sensitivity,notfull header48000 re-extraction or clock realignment;notusedtoselect results"
  }
}
```

这些数值区分了发声周期对应与身份迁移：真实配对优于错遍仅说明对应，不证明EER改善。周期身份关系的相关性亦非蒸馏有效性。

## 逐用户预测误差

|用户|候选|基线MSE|加Audio MSE|
|---|---|---:|---:|
|U01|原始周期性|0.064486|0.064124|
|U01|谐波结构|0.064486|0.063890|
|U01|组合|0.064486|0.063662|
|U02|原始周期性|0.122173|0.119173|
|U02|谐波结构|0.122173|0.120546|
|U02|组合|0.122173|0.117998|
|U03|原始周期性|0.112591|0.112139|
|U03|谐波结构|0.112591|0.113250|
|U03|组合|0.112591|0.113074|
|U04|原始周期性|0.112720|0.110331|
|U04|谐波结构|0.112720|0.112273|
|U04|组合|0.112720|0.111006|
|U05|原始周期性|0.101802|0.099010|
|U05|谐波结构|0.101802|0.104704|
|U05|组合|0.101802|0.101962|
|U06|原始周期性|0.158762|0.147530|
|U06|谐波结构|0.158762|0.153779|
|U06|组合|0.158762|0.147374|
|U07|原始周期性|0.140703|0.140216|
|U07|谐波结构|0.140703|0.140872|
|U07|组合|0.140703|0.138110|
|U08|原始周期性|0.056704|0.056333|
|U08|谐波结构|0.056704|0.055697|
|U08|组合|0.056704|0.056330|
|U09|原始周期性|0.142391|0.142276|
|U09|谐波结构|0.142391|0.142995|
|U09|组合|0.142391|0.144557|
|U10|原始周期性|0.077942|0.078845|
|U10|谐波结构|0.077942|0.077373|
|U10|组合|0.077942|0.078136|


## 证据判断

原始周期性将预测MSE从0.109027降至0.106998，相对降低约1.86%；这是本轮值得后续确认的正向信号。其用户身份置换p=0.07，Holm校正后0.21，未达到本轮预设标准。周期特征包含置信度和可检测比例，因此当前改善也可能来自检测可靠性，而不是F0身份差异。必须在后续训练词内部先拆分F0统计与可靠性统计，再在新留出数据确认；不能继续使用本轮验证词选择方案。

ACC仅2.06%的帧通过固定周期条件，双模态共同有效约1.21%。低覆盖限制了当前检测器的判断能力，不能将本轮结果解释为原始ACC不存在发声周期信息。
