import json
from pathlib import Path
root=Path('outputs/mei_posthoc_subset_20261029');d=json.loads((root/'verified_summary_public.json').read_text())
assert d['status']=='PASS' and d['test_informed_subset']
names={'base':'完整772维','identity_acc':'身份教师ACC'};methods={'rbf':'RBF','relative':'背景相对评分','cosine':'仅本人余弦'}
lines=['# 测试知情筛选的限定攻击用户群结果','',
'**本表使用旧测试结果挑选并移除困难攻击身份，是事后限定用户群探索，不是独立测试性能或九人全体泛化成绩。** 每个目标分别移除0、1、2人，认证训练与校准也一并移除；本人80条认证训练及本人测试保持不变。','',
'背景相对评分已加入：本人中心余弦减去lambda乘以最相似保留背景中心余弦。背景原型不包含已移除身份。lambda只在保留的训练内容折中选择。纯余弦和RBF使用相同的限定用户群，阈值都由对应校准数据确定。','',
'|特征|方法|移除人数|本人接受率|限定群FAR|限定群AUC|恢复全部攻击FAR|','|---|---|---:|---:|---:|---:|---:|']
for feature,label in names.items():
 for method,title in methods.items():
  for drop in [0,1,2]:
   r=d['macro']['%s_drop%d_%s'%(feature,drop,method)];a=r['restricted'];b=r['all_attackers_diagnostic']
   lines.append('|%s|%s|%d|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(label,title,drop,a['tar']*100,a['far']*100,a['auc'],b['far']*100))
lines+=['','## 每目标移除名单','', '|目标|先移除1人|再移除第2人|','|---|---|---|']
for u in map(str,range(1,10)):
 rank=d['results'][u]['ranking'];lines.append('|U%s|U%d 旧测试FAR %.2f%%|U%d 旧测试FAR %.2f%%|'%(u,rank[0]['user'],rank[0]['prior_test_far']*100,rank[1]['user'],rank[1]['prior_test_far']*100))
lines+=['','## 移除两人后的逐目标结果','', '每格为本人接受率 / 限定群FAR / AUC。','', '|目标|完整RBF|教师RBF|教师纯余弦|教师背景相对|','|---|---|---|---|---|']
def fmt(r):return '%.2f / %.2f / %.4f'%(r['tar']*100,r['far']*100,r['auc'])
for u in map(str,range(1,10)):
 a=d['results'][u]['routes'];lines.append('|U%s|%s|%s|%s|%s|'%(u,*[fmt(a[k]) for k in ['base_drop2_rbf','identity_acc_drop2_rbf','identity_acc_drop2_cosine','identity_acc_drop2_relative']]))
lines+=['','## 背景惩罚系数','', '|目标|移除两人后 教师ACC的λ|','|---|---:|']
for u in map(str,range(1,10)):
 lines.append('|U%s|%s|'%(u,d['results'][u]['routes']['identity_acc_drop2_relative']['lam']))
lines+=['','λ=0表示训练内验证没有支持额外背景惩罚，实际回退为纯本人余弦；不能把这些用户的成绩归因于背景校正。',
'','## 解释边界','',
'限定群FAR的下降同时包含排除困难攻击者与重训的影响，不可直接写成方法泛化提升。全八人诊断列保留了被排除者的攻击成绩，用于显示范围缩小带来的差别。各目标的保留群不同，也不是一个统一七人数据集。',
'','难区分不等同于采集质量差。本轮没有测量信噪比并据此清洗；筛选依据明确是旧测试误接受率。上游教师仍使用全部九人的上游数据，并未重训生成器、教师或Diffusion。',
'','全部路线已经复核：旧测试排名、标准化拟合样本数、被移除背景中心、校准阈值、最终模型分数、限定群与全部攻击的AUC和逐来源接受数。报告中的较好结果仅用于限定群体描述；需使用新的未参与筛选数据做确认。']
Path('docs/experiments/2026-10-29-posthoc-subset-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
compact={k:d[k] for k in ['status','test_informed_subset','targets','routes_verified','macro','verification']}
compact['results']={}
keep=['tar','far','auc','hter','genuine_n','impostor_n','genuine_accepted','attack_accepted','removed','feature','method','lam']
for u,a in d['results'].items():
 compact['results'][u]={'ranking':a['ranking'],'routes':{}}
 for k,r in a['routes'].items():
  row={n:r[n] for n in keep if n in r};row['all_attackers_diagnostic']={n:r['all_attackers_diagnostic'][n] for n in keep if n in r['all_attackers_diagnostic']};compact['results'][u]['routes'][k]=row
(root/'summary_public.json').write_text(json.dumps(compact,separators=(',',':')),encoding='utf-8')
print('\n'.join(lines))

