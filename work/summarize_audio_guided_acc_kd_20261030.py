import json,statistics
from pathlib import Path
root=Path('outputs/mei_audio_guided_acc_kd_20261030');data=json.loads((root/'verified_results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS';MODES=['B0_CE','B1_AAM','SelfKD','S1_AudioKD','S2_FusionKD']
lines=['# Audio辅助ACC身份表征：协议A首轮','',
'真实配对Audio只在教师训练及提供训练软标签时使用，最终所有Student只输入ACC。不经过CycleGAN。九用户已知身份跨词，不是未见用户注册。','',
'固定词组：编码器训练0–17（180条/人）；独立注册18/19（20条）；验证20–22（30条）；校准23/24（20条）；最终25–29（47至50条）。所有身份预处理相同，尾部缺行不填充，不删攻击者。训练批每次9身份×4词×2重复。模型选择原型只来自训练样本，不使用独立注册录制；认证后端可使用注册录制与开发词选择参数。','',
'所有Student相同三卷积CNN、128维L2表示、400步、AdamW3e-4；B0为CE，B1为AAM(m=.2,s=16)。Audio教师同尺寸CNN；双模态教师固定已训ACC/Audio编码器，只训练拼接MLP与九类头。KD仅为CE+lambda*T²*KL，lambda0/.1/.5/1，T2/4，包含lambda0回退。ACC自蒸馏使用相同网格；不使用SupCon或强制嵌入相等。三个seed42/43/44完整报告。','',
'## K20认证：三seed宏平均 ± 样本标准差','', '|后端|方法|TAR|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
summary={}
def seed_metrics(a,mode,prefix):
 rows=[r for k,r in a['routes'][mode].items() if k.startswith(prefix)];assert len(rows)==9;return {k:statistics.mean(r[k] for r in rows) for k in ['tar','far','auc','hter','diagnostic_eer','lowfar_calibrated_test_tar','lowfar_calibrated_test_far']}
for backend,prefix in [('cosine','cosine_K20_'),('RBF','rbf_K20_')]:
 summary[backend]={}
 for mode in MODES:
  seedrows=[seed_metrics(data[str(s)],mode,prefix) for s in [42,43,44]];stats={k:dict(mean=statistics.mean(r[k] for r in seedrows),std=statistics.stdev(r[k] for r in seedrows)) for k in seedrows[0]};summary[backend][mode]=stats
  cells=[]
  for k in ['tar','far','auc','hter']:
   m=stats[k];cells.append('%.4f ± %.4f'%(m['mean'],m['std']) if k=='auc' else '%.2f%% ± %.2f'%(100*m['mean'],100*m['std']))
  lines.append('|%s|%s|%s|'%(backend,mode,'|'.join(cells)))
lines+=['','## 余弦少样本注册','', '|注册K|方法|TAR|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
for k in [3,5,10,20]:
 for mode in MODES:
  rr=[seed_metrics(data[str(s)],mode,'cosine_K%d_'%k) for s in [42,43,44]];m={key:statistics.mean(r[key] for r in rr) for key in ['tar','far','auc','hter']};lines.append('|%d|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(k,mode,100*m['tar'],100*m['far'],m['auc'],100*m['hter']))
lines+=['','## 蒸馏选择与教师资格','', '|seed|方法|训练验证选中候选|','|---|---|---|']
for s,a in data.items():
 for mode,key in a['chosen'].items():lines.append('|%s|%s|%s|'%(s,mode,key))
lines+=['','教师开发集认证HTER（以训练原型、验证跨词评分；不是最终Audio诊断）：','', '|seed|ACC教师|Audio教师|融合教师|融合优于两者？|','|---|---:|---:|---:|---|']
for s,a in data.items():
 q=a['teacher_validation_auth'];lines.append('|%s|%.2f%%|%.2f%%|%.2f%%|%s|'%(s,100*q['acc']['hter'],100*q['audio']['hter'],100*q['fusion']['hter'],a['fusion_beats_both_validation_auth_hter']))
for backend,prefix in [('cosine','cosine_K20_U'),('RBF','rbf_K20_U')]:
 lines+=['','## 各用户%s K20 HTER：三seed均值'%backend,'', '|用户|B0CE|B1AAM|ACC自蒸馏|Audio蒸馏|融合蒸馏|','|---|---:|---:|---:|---:|---:|']
 for u in range(1,10):lines.append('|U%d|%s|'%(u,'|'.join('%.2f%%'%(100*statistics.mean(a['routes'][mode][prefix+str(u)]['hter'] for a in data.values())) for mode in MODES)))
lines+=['','## 低FAR操作点与EER说明','',
'完整JSON同时保留由校准集FAR≤5%阈值得到的最终TAR和实际最终FAR；不会用最终ROC重新选部署阈值。diagnostic_eer为最终ROC描述性指标，不参与选模/校准。','',
'## 首轮结论','',
'Audio和融合教师确实具备更好的开发集身份能力：三seed验证认证HTER平均ACC30.69%、Audio16.13%、融合12.21%，融合每个seed均优于两种单模态。但ACC Student的蒸馏收益没有稳定推广到最终词。',
'K20余弦HTER：B0 CE32.78%、AudioKD33.45%、FusionKD32.74%；RBF为33.10%、33.00%、32.80%。融合KD的均值改善很小，并且不是三个seed都超过B0；不能认定有效。AAM与ACC自蒸馏在本轮也未稳定提高基线。',
'这支持“强教师不保证跨模态迁移成功”的诊断，不能据此断言Audio身份信息完全不可由ACC学习，也不能证明LUPI范式无效。当前只是小CNN、400步、九已知身份logit-KL的首轮实验，不能代替更充分训练和独立新会话。',
'协议B尚未执行；首轮尚未满足稳定超越B0的条件，不以已知身份成绩冒充未见身份增益。','',
'## 范围与下一步','',
'这是协议A：教师和Student都见过九身份的训练词，不能称开放集新用户注册。若要协议B，必须按6开发/3未见身份重新训练所有教师、Student、归一化、选择和校准，不能只改变最终查询。',
'三seed衡量优化随机性，最终录制是同一批历史单会话数据，不是三个独立人群。词义仍依赖10/20周期、相位与尾部缺行假设。KD网格比CE有更多候选，ACC自蒸馏提供部分控制；教师额外训练预算也需区分。',
'本轮注册/训练数量与旧CycleGAN实验不同，不能直接相减归因新范式。LUPI与蒸馏是已有学习范式，不直接等同论文创新。若S1/S2未稳定超过B0与ACC自蒸馏，不继续简单加大蒸馏权重。',
'参考：ISCA2021 https://www.isca-archive.org/interspeech_2021/zhang21m_interspeech.html ；AAAI2023 https://ojs.aaai.org/index.php/AAAI/article/view/26525 。二者研究音视频/人脸，并非直接验证ACC。']
Path('docs/experiments/2026-10-30-audio-guided-acc-kd-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'summary_public.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('\n'.join(lines[:30]))

