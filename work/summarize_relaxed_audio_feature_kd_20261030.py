import json,statistics
from pathlib import Path
root=Path('outputs/mei_relaxed_audio_feature_kd_20261030');data=json.loads((root/'verified_results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
fallback=json.loads(json.dumps(data));nonzero=json.loads((root/'nonzero_verified_results_public.json').read_text());assert json.loads((root/'nonzero_verification_public.json').read_text())['status']=='PASS'
for seed,a in data.items():
 for dst,src in [('E2_AudioMKD','E2_AudioMKD_nonzero'),('E3_SelfMKD','E3_SelfMKD_nonzero'),('E2_Shuffled','E2_nonzero_Shuffled')]:a['routes'][dst]=nonzero[seed]['routes'][src]
MODES=['E0_CE','E1_LogitKL','E2_AudioMKD','E3_SelfMKD','E2_Shuffled'];summary={}
def average(a,mode,prefix):
 rr=[r for k,r in a['routes'][mode].items() if k.startswith(prefix)];assert len(rr)==9;return {k:statistics.mean(r[k] for r in rr) for k in ['tar','far','auc','hter','diagnostic_eer','lowfar_calibrated_test_tar','lowfar_calibrated_test_far']}
lines=['# Audio→ACC松弛特征蒸馏：协议A受控实验','',
'完全复用上一轮180条/人训练、20条独立注册、30条验证、20条校准、47至50条最终；九身份已知、全部八个攻击来源、三个seed42/43/44。最终只输入ACC。','',
'E0复用ACC+CE；E1逐项复用原Audio Logit-KL验证选中的模型；E2为冻结真实Audio教师→ACC特征MKD；E3为冻结ACC教师→ACC的同结构自蒸馏。E2/E3各六组非零候选：lambda .1/.5/1 × margin cos30/cos45，另允许lambda0退回E0。每候选同一Student初始化和批次序列、400步，架构/后端不变。','',
'损失采用AAAI2023 Eq4：mean max(m−cos(zS,FT),0)。FT=normalize(.6*zT+.4*MLP(zT))，三层128维MLP可训练，教师固定。与原论文相同的是松弛公式和残差投影思想；模态、网络、数据与身份损失已改造，没有QAW，不称完整VGSR复现。','',
'额外E2_Shuffled：只在训练集合中，同身份不同词做一一置换，无原始配对、无跨用户。固定E2所选lambda/m、初始模型/投影、批次和训练步数，仅验证选检查点；不为打乱组再搜索超参。','',
'## K20三seed宏平均 ± 样本标准差','', '|后端|方法|TAR|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
for backend,prefix in [('cosine','cosine_K20_'),('RBF','rbf_K20_')]:
 summary[backend]={}
 for mode in MODES:
  rr=[average(a,mode,prefix) for a in data.values()];m={k:dict(mean=statistics.mean(r[k] for r in rr),std=statistics.stdev(r[k] for r in rr)) for k in rr[0]};summary[backend][mode]=m;cells=[]
  for k in ['tar','far','auc','hter']:cells.append('%.4f ± %.4f'%(m[k]['mean'],m[k]['std']) if k=='auc' else '%.2f%% ± %.2f'%(100*m[k]['mean'],100*m[k]['std']))
  lines.append('|%s|%s|%s|'%(backend,mode,'|'.join(cells)))
lines+=['','## 允许退回CE时的验证选择（下表；上表为实际非零MKD）','', '|seed|方法|选中源|lambda|margin|','|---|---|---|---:|---|']
for s,a in data.items():
 for mode,c in a['chosen'].items():lines.append('|%s|%s|%s|%s|%s|'%(s,mode,c['key'],c.get('lambda_mkd','旧组'),c.get('margin','—')))
lines+=['','严格非零参数：'+json.dumps({s:a['chosen'] for s,a in nonzero.items()},ensure_ascii=False),'','允许退回CE时K20 HTER：'+json.dumps({b:{m:round(100*statistics.mean(average(a,m,p)['hter'] for a in fallback.values()),2) for m in MODES} for b,p in [('cosine','cosine_K20_'),('RBF','rbf_K20_')]},ensure_ascii=False)]
lines+=['','## 少样本余弦注册','', '|K|方法|TAR|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
for k in [3,5,10,20]:
 for mode in MODES:
  rr=[average(a,mode,'cosine_K%d_'%k) for a in data.values()];m={key:statistics.mean(r[key] for r in rr) for key in ['tar','far','auc','hter']};lines.append('|%d|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(k,mode,100*m['tar'],100*m['far'],m['auc'],100*m['hter']))
for backend,prefix in [('cosine','cosine_K20_U'),('RBF','rbf_K20_U')]:
 lines+=['','## 九用户%s K20 HTER：三seed均值'%backend,'', '|用户|E0|E1|AudioMKD|SelfMKD|打乱AudioMKD|','|---|---:|---:|---:|---:|---:|']
 for u in range(1,10):lines.append('|U%d|%s|'%(u,'|'.join('%.2f%%'%(100*statistics.mean(a['routes'][m][prefix+str(u)]['hter'] for a in data.values())) for m in MODES)))
lines+=['','## 解释与核验','',
'打乱组若不退步，只能说明收益对具体录制/内容配对不敏感，不自动否定身份级蒸馏；若退步也不能单独证明物理同步机制，因为本对照同时打乱词语。比较AudioMKD与SelfMKD才能检查跨模态特有增量。',
'投影可能改变原教师的分类读出，即使余弦损失降低也不等于保留身份。投影原/后验证top1和分量范数保存在完整结果中，不用该诊断选模型。',
'独立重算E0/E1、所有验证选择、打乱一一映射、Student-only推理、注册Scaler、校准阈值、最终AUC及八个来源接受数；现场检查梯度只到Student和投影、不回冻结教师。','',
'这是历史单会话、已知九身份跨词协议A，非新用户开放集。词语分组依赖10/20周期和尾部缺行假设。三个seed不是独立采集。没有按最终结果挑人、重设margin或更换阈值。',
'论文来源：https://ojs.aaai.org/index.php/AAAI/article/view/26525/26297 。']
Path('docs/experiments/2026-10-30-relaxed-audio-feature-kd-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'summary_public.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('\n'.join(lines[:32]))
