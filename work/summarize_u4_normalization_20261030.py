import json,statistics
from pathlib import Path
root=Path('outputs/mei_u4_normalization_20261030');a=json.loads((root/'verified_results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
previous=json.loads(Path('outputs/mei_audio_guided_acc_kd_20261030/verified_results_public.json').read_text());modes=['global','instance','gated','dual'];names={'legacy':'旧输入CE参考','global':'全局标准化','instance':'逐样本标准化','gated':'门控标准化','dual':'归一化＋保存频谱统计'};summary={}
lines=['# U4保存频谱归一化对照结果','','九身份已知跨词实验；每人180条编码器训练，20条独立注册，30验证，20校准；U4最终47条，攻击394条，八个攻击来源全部保留。所有人的前端规则一致。三seed42/43/44。','', '旧输入CE参考复用上一轮；新四组保持相同新增参数数量/初始化/400步、同一后端和阈值规则。新编码器仍为九身份CE监督，并非重训个人CycleGAN或Audio教师。旧参考没有本轮恒零统计分支；编码器和CE头的初始化顺序相同，minibatch相同，仍作为历史参考单列。','', '## K20三seed均值±样本标准差','','|后端|方法|本人接受TAR|攻击FAR|HTER|AUC|','|---|---|---:|---:|---:|---:|']
for backend,prefix in [('余弦','cosine_K20_U4'),('RBF','rbf_K20_U4')]:
 summary[backend]={}
 for mode in ['legacy']+modes:
  rows=[v['routes']['B0_CE' if mode=='legacy' else mode][prefix] for v in (previous if mode=='legacy' else a).values()];out={k:dict(mean=statistics.mean(r[k] for r in rows),std=statistics.stdev(r[k] for r in rows)) for k in ['tar','far','hter','auc']};summary[backend][mode]=out
  cells=[('%.4f ± %.4f' if k=='auc' else '%.2f%% ± %.2f')%((1 if k=='auc' else 100)*out[k]['mean'],(1 if k=='auc' else 100)*out[k]['std']) for k in ['tar','far','hter','auc']];lines.append('|%s|%s|%s|'%(backend,names[mode],'|'.join(cells)))
lines+=['','## 各seed门控与检查点','','|seed|方法|step|全局分支门控权重|','|---|---|---:|---:|']
for seed,v in a.items():
 for mode in modes:
  m=v['metadata'][mode]['selected'];lines.append('|%s|%s|%s|%s|'%(seed,mode,m['step'],'%.4f'%m['gate'] if mode=='gated' else '—'))
lines+=['','## 八个攻击来源FAR：三seed平均','','|后端|攻击来源|全局|逐样本|门控|双分支|','|---|---|---:|---:|---:|---:|']
for backend,prefix in [('余弦','cosine_K20_U4'),('RBF','rbf_K20_U4')]:
 for user in [1,2,3,5,6,7,8,9]:
  cells=[]
  for mode in modes:
   rr=[v['routes'][mode][prefix]['attack_by_user'][str(user)] for v in a.values()];cells.append('%.2f%%'%(100*statistics.mean(r['accepted']/r['n'] for r in rr)))
  lines.append('|%s|U%d|%s|'%(backend,user,'|'.join(cells)))
lines+=['','## 边界','','旧数组已经在[-1,1]，历史前端归一化统计未完全追溯。本轮global只表示对保存频谱做训练集标量zscore；dual的mean/std/RMS/range是保存频谱统计，不是真实物理能量。不能用本轮证明原始幅度价值或完成了原始ACC重处理。','', '模型选择按独立验证九身份宏HTER；后端和阈值仅使用注册/验证/校准。最终未按U4成绩挑检查点、改门控或删攻击人。所有逐seed及逐来源接受数保留公开JSON。历史同会话跨词推断仍为探索性。']
lines+=['','## 结论','','四种新处理在U4两个K20后端都未超过旧输入CE参考，暂不替换旧前端。新四组中全局标准化最好，但不能因最终结果而称部署选择。逐样本标准化FAR下降伴随本人接受下降，不能只报告FAR改善。','', '逐样本处理相对全局处理的HTER变差，提示保存频谱中的样本均值/尺度差异可能包含可用信息，也可能含录制条件；缺少跨会话与真实原始幅度对照，不能直接认定是生理身份信息。门控权重仅0.4986至0.5055，短预算没有学出明显归一化偏好。统计支路也未自动恢复丢失信息；不能据此否定原始能量，因为其输入只是历史归一化频谱。','', '本轮是CE身份编码器的输入消融，不涉及CycleGAN重训，不是完整772维baseline或上一轮Audio MKD的重跑。若要验证图中的真实绝对尺度双分支，需要九用户未经逐样本缩放的原始信号及其与旧行号的对应。']
(root/'summary_public.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');Path('docs/experiments/2026-10-30-u4-normalization-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[:23]))
