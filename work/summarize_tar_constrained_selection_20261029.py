import json
from pathlib import Path
root=Path('outputs/mei_tar_constrained_selection_20261029');data=json.loads((root/'results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
lines=['# 本人接受率约束下的选路与校准','',
'复用U4/U5全部冻结模型，无新训练、无G/F或教师更新。每身份认证80条、校准20条，最终本人47/50条及全部八个攻击来源不变。','',
'目标：开发阶段优先达到本人接受率90%，再降低FAR。完整772维基线保持不变，原始统计权重和个人教师权重均从0/.25/.5/1选择，三个教师为上轮常规、温和、较强。去重后共40种组合。只用认证训练OOF分数按词分五折，阈值由其余四折本人90%经验分位确定，汇总留折接受结果。先按本人接受率不足90%的差距，再FAR、HTER、AUC及较小权重排序。','',
'最终阈值只用独立校准集20条本人：选择仍能接受至少90%的最高观测分数。攻击校准分数仅作诊断，不用于改阈值或路线。全部选择保存后才读最终分数。','',
'|目标|方法|本人接受率|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
names={'baseline_original':'完整基线原阈值','control_original':'常规教师原阈值','baseline_tar90':'完整基线仅改90%阈值','control_tar90':'常规教师仅改90%阈值','selected_tar90':'训练OOF选路＋90%阈值'}
for u,a in data.items():
 for k,name in names.items():
  r=a['routes'][k];lines.append('|U%s|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(u,name,100*r['tar'],100*r['far'],r['auc'],100*r['hter']))
lines+=['','## 选中组合','']
for u,a in data.items():
 c=a['chosen'];r=a['routes']['selected_tar90'];lines.append('- U%s：教师%s，原始统计权重%.2f，教师权重%.2f；训练OOF TAR %.2f%%、FAR %.2f%%；校准TAR %.2f%%、FAR %.2f%%。'%(u,c['teacher_key'],c['raw_weight'],c['teacher_weight'],100*c['quality']['tar'],100*c['quality']['far'],100*r['calibration_tar'],100*r['calibration_far']))
lines+=['','## 结果解释','',
'相对常规教师原规则，U4最终TAR74.47%不变，FAR8.12%降至7.36%，AUC0.9327至0.9346，HTER16.83%至16.45%；幅度较小。U5最终TAR66%至86%，FAR15.60%至26.85%，AUC0.8599至0.8588，HTER24.80%至20.43%。后者改善操作点的总体错误代价，但没有提高排序判别能力，安全性下降。',
'校准TAR90%未推广到最终TAR：U4为74.47%、U5为86%。20条校准本人不足以保证跨词覆盖。本轮不据此继续降低阈值。','',
'## 边界','',
'这是操作点与选路试验，不是改善编码器或教师的训练。固定评分模型仅改变阈值时AUC应不变；本人接受率提高若伴随FAR升高，属于安全性/可用性权衡，不能称身份判别能力提高。90%是校准经验目标，不保证最终或总体本人接受率。',
'历史最终集已观察，已知用户闭集；内容依赖10/20周期及尾部缺行映射。认证训练OOF分数复用先前嵌套模型，其源模型跨元验证折存在重叠；本次40候选选择不是全新完整嵌套的无偏估计。补充分支模型参数仍按旧AUC规则选择。','',
'不根据最终集改目标、权重、用户或阈值。需要新词/新会话确认。']
Path('docs/experiments/2026-10-29-tar-constrained-selection-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[:30]))

