import json
from pathlib import Path
root=Path('outputs/mei_personalized_teacher_20261029');data=json.loads((root/'results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
lines=['# U4/U5个人教师参数调整：单seed受控实验','',
'保留完整772维基线，G/F不训练，个人教师只用上游每身份120条训练、30条验证。认证采用每身份80条训练、20条校准、47至50条最终测试，保留全部八个攻击身份。','',
'三组教师使用相同架构、相同初始化seed3800+目标编号、相同词语均衡采样。常规与U4个性化设置最多600步；U5个性化连续四次验证不改善则早停。上游按ACC/真实Audio的逐词平均AUC、总体AUC、BCE依次选择教师。','',
'U4个性化仅增加半难负类权重：基于训练批负类得分50%至90%分位选择，权重1.5或2，负类平均权重归一为1；全部负类仍保留。U5温和/较强设置分别采用学习率3e-4/1e-4、weight decay .005/.01、跨词SupCon权重.2/.3，常规为5e-4/.003及普通SupCon .1。所有设置保留BCE、九身份CE及真实跨模态配对一致性。','',
'认证只补充新教师ACC向量；原始统计和教师可不选。复用相同80条基线与原始统计OOF、分类器；新教师也采用训练内5外折3内折RBF，稳定性前向选择和权重0/.25/.5/1。温和与较强之间按训练OOF选定，不看最终分数。','',
'|目标|路线|本人接受率|FAR|AUC|HTER|补充权重|','|---|---|---:|---:|---:|---:|---|']
for u,a in data.items():
 for key,name in [('baseline_rbf','完整772维80条基线'),('auto_without_diffusion','历史80条教师自动补充')]:
  r=a['baseline80_reference'][key];lines.append('|U%s|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|%s|'%(u,name,100*r['tar'],100*r['far'],r['auc'],100*r['hter'],json.dumps(r['weights'])))
 for key,name in [('control','新训常规对照'),('personal_mild','温和个人参数'),('personal_strong','较强个人参数')]:
  r=a['routes'][key];lines.append('|U%s|%s%s|%.2f%%|%.2f%%|%.4f|%.2f%%|%s|'%(u,name,'（OOF选中）' if key==a['personalized_choice'] else '',100*r['tar'],100*r['far'],r['auc'],100*r['hter'],json.dumps(r['weights'])))
lines+=['','## 教师训练记录','', '|目标|设置|实际步数|选中步数|上游逐词AUC|','|---|---|---:|---:|---:|']
for u,a in data.items():
 for key,m in a['teacher_metadata'].items():
  best=next(h for h in m['history'] if h['step']==m['selected_step']);lines.append('|U%s|%s|%d|%d|%.4f|'%(u,key,m['actual_steps'],m['selected_step'],best['word_auc']))
lines+=['','## 初步结论','',
'U4训练OOF选中温和半难负类权重：相对本轮常规对照，AUC0.9327升至0.9382、FAR8.12%降至7.61%，但本人接受率74.47%降至72.34%，HTER16.83%升至17.64%；不是综合改善。',
'U5训练OOF选中较强个人参数：本人接受率仍66%，FAR15.60%升至18.41%、AUC0.8599降至0.8562、HTER24.80%升至26.21%。温和设置虽本人接受70%，FAR也升至19.44%，且不是OOF选中配置。不能根据最终结果改选温和。',
'当前证据不支持以这组个人参数替换常规教师。训练折增益没有稳定推广到最终词；U4更偏向安全性与本人拒绝的权衡，U5仍需排查跨词稳定性和模型选择的错配。下一轮应预先约束开发集本人拒绝并独立评估，而不是继续按最终集试权重。','',
'## 范围与限制','',
'这是单seed机制试验，不是稳定收益或普遍适配的证据。历史最终集已观察，内容仍依赖周期10/20与尾部缺行映射，已知用户闭集。没有按最终成绩删样本、攻击来源或重设阈值。',
'新常规对照与两个个性化设置是本轮受控比较。历史教师自动补充还搜索过多种教师及Audio分支，不能将其与新路线的差异完全归因训练参数。U5同时改变多个训练项，不能单独归因跨词损失或学习率。',
'只有ACC教师向量参与认证补充，即使提高，也不证明经过生成Audio的有效身份增强。未重训生成器。训练OOF既选补充分支又选参数配置，仍有选择过拟合；需要独立新内容/会话验证。']
Path('docs/experiments/2026-10-29-personalized-teacher-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[:26]))

