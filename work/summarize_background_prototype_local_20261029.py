import json
from pathlib import Path
root=Path('outputs/mei_background_prototype_20261029');d=json.loads((root/'verified_summary_public.json').read_text())
assert d['status']=='PASS' and d['targets']==9
names={'base':'完整772维','identity_acc':'身份教师ACC','selected_fusion':'此前自主多选组合'}
lines=['# 背景用户原型相对余弦认证结果','',
'图示公式：本人中心余弦相似度减去lambda乘以最相似背景身份中心的余弦相似度。其余八人均为背景，只用每身份80条认证训练建立中心。它是简化背景相对分数，不是正式AS-Norm。','',
'上游、预处理、校准和最终测试不变。λ在0/0.25/0.5/1/1.5/2中按认证训练内五个内容折平均AUC选择；同时提供固定λ=1对照。测试前锁定所有设置。','',
'|特征表示|分数方式|宏TAR|宏FAR|宏AUC|宏HTER|','|---|---|---:|---:|---:|---:|']
for name,label in names.items():
 for mode,title in [('cosine','本人余弦'),('relative_fixed1','背景相对 λ=1'),('relative','训练折选λ')]:
  r=d['macro'][name+'__'+mode];lines.append('|%s|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(label,title,r['tar']*100,r['far']*100,r['auc'],r['hter']*100))
for mode,title in [('auto_cosine','训练折自动选单分支余弦'),('auto_relative','训练折自动选单分支及λ')]:
 r=d['macro'][mode];lines.append('|自动单分支|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(title,r['tar']*100,r['far']*100,r['auc'],r['hter']*100))
lines+=['','## 逐用户配对比较','', '每格 TAR% / FAR% / AUC。','', '|用户|772余弦|772相对|教师ACC余弦|教师ACC相对|','|---|---|---|---|---|']
def fmt(r):return '%.2f / %.2f / %.4f'%(r['tar']*100,r['far']*100,r['auc'])
for u in map(str,range(1,10)):
 a=d['results'][u]['routes'];lines.append('|U%s|%s|%s|%s|%s|'%(u,*[fmt(a[k]) for k in ['base__cosine','base__relative','identity_acc__cosine','identity_acc__relative']]))
lines+=['','|用户|自主组合余弦|自主组合相对|λ 772 / 教师 / 组合|','|---|---|---|---|']
for u in map(str,range(1,10)):
 a=d['results'][u]['routes'];lines.append('|U%s|%s|%s|%s|'%(u,fmt(a['selected_fusion__cosine']),fmt(a['selected_fusion__relative']),' / '.join(str(a[k+'__relative']['lam']) for k in names)))
lines+=['','## 阶段判断','',
'背景校正在同一多分支表示上同时改善本人接受率与FAR：训练折选λ后，宏TAR增加2.53个百分点、FAR降低1.64个百分点，AUC增加0.0279，HTER降低2.09个百分点。固定λ=1也有正向平均效果，不只依赖权重调参。',
'','U4多分支表示从本人接受29/47、攻击误接受43/394，改善为35/47、39/394，即TAR61.70%升到74.47%，FAR10.91%降到9.90%。教师ACC单分支也改善，但并非所有用户都有收益，U1、U2等存在接受率和FAR的权衡。',
'','结论应限定为背景相对评分对当前表示有局部及平均增益，尚不能称稳定认证解决方案；FAR仍约28%，也未验证未见背景身份。固定λ=1在当前最终测试更好不能据此事后替代预先定义的训练折选λ路线。',
'','## 与原RBF的关系','',
'80条完整RBF宏TAR68.32%、FAR29.68%、AUC0.7474、HTER30.68%；原自主多选RBF为67.90%、27.49%、0.7715、29.80%。这些是不同分类器，不能把相对RBF的全部差异归因于背景校正；须看上述同表示余弦配对。',
'','所有351条诊断路线已从训练数据重新建立身份中心，并重算校准阈值、最终分数、AUC和八个攻击来源计数。每种表示的惩罚系数选择也从训练OOF分数复核。',
'','标准化仅在训练行拟合。各分支先标准化并L2，中心为该身份归一化向量均值再L2；自主组合先按此前权重平方根缩放各归一化分支、拼接并L2。这是特征级融合，不等于此前RBF分数级融合。',
'','局限：已知八个背景身份，闭集；历史测试已观察，探索性重复评估；单种子；10/20周期及尾部缺行假设；旧多选权重已使用全80条训练，因此相应CV是条件验证。不能称开放集、未见背景泛化或正式AS-Norm。']
Path('docs/experiments/2026-10-29-background-prototype-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
compact={k:d[k] for k in ('status','targets','diagnostic_routes_verified','macro','verification')}
compact['results']={u:{'routes':{name:{k:v for k,v in r.items() if k!='attack_by_user'} for name,r in a['routes'].items()},'prior_fusion_weights':a['prior_fusion_weights']} for u,a in d['results'].items()}
(root/'summary_public.json').write_text(json.dumps(compact,indent=2),encoding='utf-8')
print('\n'.join(lines))

