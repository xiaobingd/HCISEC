import json
from pathlib import Path
root=Path('outputs/mei_sameword_tar_constraint_20261029');data=json.loads((root/'results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
lines=['# U4/U5内容一致：本人接受率约束选路','',
'本人和所有八个攻击用户使用同一15词，同一重复编号。认证训练重复0–4共75条/身份，校准重复5/6共30条，最终重复7/8共30条；最终每词2条本人和16条攻击，总计30本人/240攻击。第9次重复统一不用，避免尾部缺行造成词频不平衡。','',
'认证词为five/six/seven/eight/nine/Next/Back/Off/Grab/Down/Up/Zoom/Record/Play/Reset。上游模型仅使用其余15词，全部认证录制与上游150条互斥。词义仍以词表顺序及10/20周期、跨用户相位一致为前提，未逐录音独立转写。','',
'复用完整772维基线和冻结个人教师，不训练G/F或教师。新增原始ACC统计与三个教师ACC向量的同词RBF：按重复分5外折/3内折，全部训练/验证折均含相同词表。0/.25/.5/1权重的40种分数融合只用认证训练OOF选择，90%本人接受优先，其次FAR。校准采用30条本人中仍接受至少90%的最高观测阈值。','',
'|目标|方法|TAR|FAR|总体AUC|逐词平均AUC|HTER|','|---|---|---:|---:|---:|---:|---:|']
names={'baseline_original':'完整772维原阈值','control_original':'常规教师分数融合原阈值','baseline_tar90':'完整772维＋90%阈值','control_tar90':'常规教师分数融合＋90%阈值','selected_tar90':'训练OOF选路＋90%阈值'}
for u,a in data.items():
 old=a['old_sameword_reference']['train_selected_teacher'];lines.append('|U%s|旧同词900维教师拼接|%.2f%%|%.2f%%|%.4f|%.4f|%.2f%%|'%(u,100*old['tar'],100*old['far'],old['auc'],old['same_word_mean_auc'],100*old['hter']))
 for key,name in names.items():
  r=a['routes'][key];lines.append('|U%s|%s|%.2f%%|%.2f%%|%.4f|%.4f|%.2f%%|'%(u,name,100*r['tar'],100*r['far'],r['auc'],r['same_word_mean_auc'],100*r['hter']))
lines+=['','## 训练选择与校准','']
for u,a in data.items():
 c=a['chosen'];r=a['routes']['selected_tar90'];lines.append('- U%s：%s教师，原始统计权重%.2f、教师权重%.2f；训练OOF TAR %.2f%% / FAR %.2f%%；校准 TAR %.2f%% / FAR %.2f%%。'%(u,c['teacher_key'],c['raw_weight'],c['teacher_weight'],100*c['quality']['tar'],100*c['quality']['far'],100*r['calibration_tar'],100*r['calibration_far']))
lines+=['','## 结果解释','',
'40候选在训练OOF中都未达到90%本人接受，规则按最小不足量回退。U4选常规教师权重.25、不选原始统计；U5不选教师，仅原始统计权重.5。两者训练OOF本人接受均89.33%。',
'U4最终TAR93.33%、FAR25.00%、AUC0.9131、HTER15.83%；旧同词900维教师为83.33%、10.83%、0.9518、13.75%。新规则提高本人接受但未优于旧教师的综合错误率及安全性。',
'U5最终TAR83.33%、FAR47.92%、AUC0.7517、HTER32.29%；旧同词900维教师为73.33%、40.00%、0.7404、33.33%。本人接受及HTER略改善，但约半数攻击被接受，不能称实用认证。',
'不能因为新同词结果比历史跨词结果差就归因内容一致有害，两套词集、录制次数和样本预算不同。','',
'## 边界','',
'新教师分数融合与旧900维拼接结构不同，不把差异全部归因于阈值。应通过相同评分模型原阈值与90%阈值的对照区分。历史最终录制已被反复观察，属于探索性闭集，不是新盲测。',
'训练OOF评分复用嵌套分支模型，元验证不是新完整嵌套，40候选搜索仍可能过拟合。个人教师设置此前在跨词实验中探索，存在历史选择影响。本轮最终成绩未用于改路线、权重、阈值或删除来源。',
'与跨内容实验使用的80条训练/20条校准/47至50条最终不同，此处75/30/30且词集15个，不能直接相减得出纯内容因素的因果增益。90%只是校准经验目标，不保证总体或最终TAR。']
Path('docs/experiments/2026-10-29-sameword-tar-constraint-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[:30]))

