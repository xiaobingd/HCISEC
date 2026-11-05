import json,statistics as st
from pathlib import Path
root=Path('outputs/mei_identity_relation_kd_20261030');results=json.loads((root/'results_public.json').read_text());verification=json.loads((root/'verification_public.json').read_text());assert verification['status']=='PASS'
seeds=['42','43','44'];methods=list(results['42']['routes']);summary={'K20':{},'fewshot':{},'per_user':{},'geometry':{},'teacher_validation':{s:results[s]['teacher_validation'] for s in seeds}}
names={'E0_ACC_CE':'ACC CE','E0_ACC_AAM':'ACC AAM','E1_concat_relation':'简单融合关系蒸馏','E2_interaction_relation':'交互教师关系蒸馏','E3_full':'交互教师+跨词稳定性','E4_ACC_stability':'仅ACC跨词稳定性','E5_full_shuffled':'完整方法·打乱配对','E6_self_relation':'ACC自关系蒸馏+稳定性'}
def avg(s,key,prefix):
 rr=[v for k,v in results[s]['routes'][key].items() if k.startswith(prefix)];assert len(rr)==9
 return {m:st.mean(v[m] for v in rr) for m in ['tar','far','hter','auc']}
lines=['# Audio+ACC 交互教师与身份关系蒸馏：三种子实验','',
'日期：2026-10-30。复用现有配对九用户频谱与 Audio 辅助 ACC 蒸馏代码；全部8个攻击来源保留。最终部署只输入 ACC。','',
'## 协议与隔离','',
'本轮为已知九身份的严格留词协议：训练词组0–17，每人180条；注册18–19，每人20条；验证20–22，每人30条；校准23–24，每人20条；最终测试25–29，每人47–50条。不同用户在各阶段使用相同内容词集合；训练、注册、验证、校准和测试之间无录制或词组交集。所有人的频谱预处理保持一致，缺失尾部不填充。词标签沿用历史10/20周期与缺尾映射假设，尚不等于逐条原始语义核验。','',
'这是已知身份、未见词实验，不能称未见身份开放集。最终测试为此前反复观察过的历史单会话数据，本轮是探索性比较，仍需独立新录制确认。','',
'## 改动','',
'Student保持原CNN16/32/64、GroupNorm、GELU、全局均值/标准差池化、128维L2编码和9类CE头。复用CE与AAM(m=.2,s=16)基线。每个新模型训练400步，batch72=9人×4词×2重复，AdamW lr3e-4/wd.001；同种子Student初始化与批序相同。三个种子42/43/44。','',
'教师复用各seed的真实ACC/Audio身份编码器并冻结。两侧增加128→128投影；简单融合把投影后向量拼接；交互融合先计算逐元素乘积r=a⊙m，再拼接[a+r,m+r]。两者使用相同256→128→128融合头与9类CE头，参数数量一致。交互不是完整Transformer注意力；它是本轮的最小乘法交互实现。','',
'关系蒸馏比较Student与冻结教师的批内余弦相似度矩阵，分别在“同人异词”和“异人同词”两类关系上取均方误差并等权平均。跨词稳定性由同人异词的1−cos吸引项和异人同词的max(0,cos−.2)排斥项等权构成。两项固定权重均为.1；未用最终测试挑权重。完整方法的教师与Student均加入稳定性。','',
'教师/Student每25步仅用验证词宏平均HTER选checkpoint，继而按最大FAR/FRR、AUC、CE、较早步数打破并列。注册、认证后端和阈值沿用旧协议：K3/5/10/20余弦；K20 RBF仅在注册数据拟合，有限参数网格由验证词选择；联合HTER阈值与低FAR阈值均来自校准集。所有路线锁定后才读最终分数。','',
'教师的额外预训练与融合头训练属于额外资源；Student尺寸与每个模型400步相同，不代表整个系统总训练成本与ACC-only相同。全部教师关系目标仅计算训练样本。','',
'## K20认证：九用户宏平均，再对三seed求均值±样本标准差','',
'|后端|方法|TAR|FAR|HTER|AUC|','|---|---|---:|---:|---:|---:|']
for backend,prefix in [('cosine','cosine_K20_'),('RBF','rbf_K20_')]:
 summary['K20'][backend]={}
 for key in methods:
  rr=[avg(s,key,prefix) for s in seeds];stats={m:{'mean':st.mean(v[m] for v in rr),'std':st.stdev(v[m] for v in rr),'seeds':[v[m] for v in rr]} for m in rr[0]};summary['K20'][backend][key]=stats
  cells=['%.2f%% ± %.2f'%(100*stats[m]['mean'],100*stats[m]['std']) for m in ['tar','far','hter']]+['%.4f ± %.4f'%(stats['auc']['mean'],stats['auc']['std'])]
  lines.append('|%s|%s|%s|'%(backend,names[key],'|'.join(cells)))
lines+=['','## 成对增益：HTER越低越好','', '|后端|对照|HTER变化（百分点）|3个seed中改善数|','|---|---|---:|---:|']
for b in ['cosine','RBF']:
 for key,ref in [('E3_full','E0_ACC_CE'),('E3_full','E1_concat_relation'),('E3_full','E4_ACC_stability'),('E3_full','E6_self_relation'),('E5_full_shuffled','E3_full')]:
  a=summary['K20'][b][key]['hter'];c=summary['K20'][b][ref]['hter'];lines.append('|%s|%s − %s|%+.2f|%d/3|'%(b,names[key],names[ref],100*(a['mean']-c['mean']),sum(x<y for x,y in zip(a['seeds'],c['seeds']))))
lines+=['','## 各用户 K20 HTER：三seed平均','', '|后端|用户|ACC CE|简单融合关系|完整方法|仅稳定性|打乱配对|ACC自关系|','|---|---|---:|---:|---:|---:|---:|---:|']
subset=['E0_ACC_CE','E1_concat_relation','E3_full','E4_ACC_stability','E5_full_shuffled','E6_self_relation']
for b,prefix in [('cosine','cosine_K20_U'),('RBF','rbf_K20_U')]:
 summary['per_user'][b]={}
 for u in range(1,10):
  values={key:{m:st.mean(results[s]['routes'][key][prefix+str(u)][m] for s in seeds) for m in ['tar','far','hter','auc']} for key in methods};summary['per_user'][b][str(u)]=values
  lines.append('|%s|U%d|%s|'%(b,u,'|'.join('%.2f%%'%(100*values[key]['hter']) for key in subset)))
lines+=['','## 少样本注册余弦HTER','', '|K|ACC CE|简单融合关系|完整方法|仅稳定性|打乱配对|ACC自关系|','|---|---:|---:|---:|---:|---:|---:|']
for k in [3,5,10,20]:
 values={key:st.mean(avg(s,key,'cosine_K%d_'%k)['hter'] for s in seeds) for key in methods};summary['fewshot'][str(k)]=values;lines.append('|%d|%s|'%(k,'|'.join('%.2f%%'%(100*values[key]) for key in subset)))
lines+=['','## 教师开发词身份能力','', '|seed|简单融合HTER|交互教师HTER|交互+稳定性HTER|','|---|---:|---:|---:|']
for s in seeds:lines.append('|%s|%s|'%(s,'|'.join('%.2f%%'%(100*results[s]['teacher_validation'][key]['hter']) for key in ['T_concat','T_interaction','T_interaction_stable'])))
lines+=['','## 最终身份关系诊断','', '|方法|同人异词相似度|异人同词相似度|相似度差|pair AUC|','|---|---:|---:|---:|---:|']
for key in methods:
 values={m:st.mean(results[s]['final_identity_geometry'][key][m] for s in seeds) for m in ['same_identity_crossword_mean','different_identity_sameword_mean','margin','pair_auc']};summary['geometry'][key]=values;lines.append('|%s|%s|'%(names[key],'|'.join('%.4f'%values[m] for m in values)))
lines+=['','上述pair共享录制，互相依赖，不能当作独立认证试次或独立置信区间。打乱配对是在训练身份内跨词进行双射，保留用户身份，破坏具体录制/内容配对；若打乱后不降低性能，只说明精确配对的额外价值尚未证实，不自动否定身份级蒸馏。','',
'## 本轮实际结论','',
'完整方法的K20余弦HTER为32.65%，比ACC CE的32.78%仅低0.13个百分点；相对仅ACC跨词稳定性32.68%只低约0.02个百分点。RBF完整方法33.84%，差于ACC CE33.10%，且三seed均差于仅稳定性32.51%。完整方法相对ACC CE虽有2/3 seed改善，但seed42显著变差，均值与方差不能忽略。没有稳定超过两种后端的ACC-only基线。','',
'打乱配对后HTER均值增加约0.92（余弦）/0.44（RBF）个百分点，但各自仅2/3 seed呈该方向，且原始完整方法本身没有稳定增益，不能把这个差值直接解释为已证明的跨模态身份耦合收益。最终pair AUC：ACC CE0.6994、完整方法0.6937、仅稳定性0.7065；全套方法也没有让此项身份关系指标改善。','',
'目前较有价值的信号是ACC自身的跨词稳定性约束；RBF HTER降到32.51%，但仅0.59个百分点，仍需新会话验证。Audio+ACC教师较强不等于这些关系全部可以从ACC预测。本轮证据更支持先在训练折内评估哪些教师关系可由ACC稳定预测，再只迁移这些关系。它尚不能证明固定权重的关系矩阵蒸馏适合作为创新点2。','',
'## 复核','',
'已独立重载所有Student、注册向量和RBF后端，重放最终分数、校准阈值、AUC、TAR、FAR及全部8个攻击源接受数；检查词/录制隔离、400步预算、相同Student参数量、相同教师参数量、训练折内shuffle双射、关系梯度传到Student。CE/AAM对照与旧结果逐项一致。verification_public.json为PASS。','',
'## 结论判断规则','',
'需要同时看TAR与FAR，以宏平均HTER为主，报告全部用户和seed。完整方法需稳定超过ACC CE、简单融合、仅稳定性和自蒸馏，才有证据说明新增跨模态关系监督带来增益。均值微幅改善或只在部分seed改善属于探索信号，不能据此确立论文创新点。当前没有跨佩戴会话或未见身份验证。']
root.mkdir(exist_ok=True);(root/'summary_public.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');Path('docs/experiments/2026-10-30-identity-relation-kd-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[lines.index('## K20认证：九用户宏平均，再对三seed求均值±样本标准差'):lines.index('## 各用户 K20 HTER：三seed平均')]))

