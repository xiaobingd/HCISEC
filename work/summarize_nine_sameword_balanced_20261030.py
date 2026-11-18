import json,statistics
from pathlib import Path
root=Path('outputs/mei_nine_sameword_balanced_20261030');data=json.loads((root/'verified_results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS' and len(data)==9
lines=['# 九用户同内容认证：同时兼顾FAR与FRR','',
'九个目标均保留其余八个攻击身份。统一15词、相同重复编号：训练0–4共75本人/600负类，校准5/6共30/240，测试7/8共30/240；每词2本人/16攻击。所有认证录制与上游120训练＋30验证互斥。','',
'没有90%本人接受约束。RBF参数、分数融合、校准阈值均以HTER=(FAR+FRR)/2最小为首要目标，相同时先减小max(FAR,FRR)，再比较AUC/简单组合。阈值由独立校准集本人和攻击者共同确定，最终不再调节。','',
'使用原完整772维、纯ACC统计320维和三类既有冻结ACC教师64维。分支C1/10/100、gamma scale/.1，按重复5外折/3内折。融合保留完整基线，原始统计和一类教师可选，权重0/.25/.5/1，共40候选。九人使用同一候选池，不混用仅U4/U5拥有的新参数教师。','',
'## 九用户最终结果','',
'|用户|基线TAR|基线FAR|基线AUC|基线HTER|融合TAR|融合FAR|融合AUC|融合HTER|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for u,a in data.items():
 b=a['routes']['baseline772'];s=a['routes']['balanced_selected'];lines.append('|U%s|%.2f%%|%.2f%%|%.4f|%.2f%%|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(u,100*b['tar'],100*b['far'],b['auc'],100*b['hter'],100*s['tar'],100*s['far'],s['auc'],100*s['hter']))
macro={}
lines+=['','## 九用户宏平均','', '|方法|TAR|FAR|AUC|逐同词AUC|HTER|','|---|---:|---:|---:|---:|---:|']
for key,name in [('baseline772','完整772维'),('raw_acc_only','纯ACC统计'),('balanced_selected','平衡目标自主补充')]:
 m={k:statistics.mean(a['routes'][key][k] for a in data.values()) for k in ['tar','far','auc','same_word_mean_auc','hter']};macro[key]=m;lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.4f|%.2f%%|'%(name,100*m['tar'],100*m['far'],m['auc'],m['same_word_mean_auc'],100*m['hter']))
wins=sum(a['routes']['balanced_selected']['hter']<a['routes']['baseline772']['hter']-1e-12 for a in data.values());losses=sum(a['routes']['balanced_selected']['hter']>a['routes']['baseline772']['hter']+1e-12 for a in data.values())
lines+=['','相对本轮完整772维基线，自主补充最终HTER改善%d人、持平%d人、变差%d人。没有按这些最终结果回退路线。'%(wins,9-wins-losses,losses),'',
'## 训练折选择的补充','', '|用户|教师|原始统计权重|教师权重|训练OOF TAR|训练OOF FAR|训练OOF HTER|','|---|---|---:|---:|---:|---:|---:|']
for u,a in data.items():
 c=a['chosen'];q=c['quality'];lines.append('|U%s|%s|%.2f|%.2f|%.2f%%|%.2f%%|%.2f%%|'%(u,c['teacher'],c['raw_weight'],c['teacher_weight'],100*q['tar'],100*q['far'],100*q['hter']))
lines+=['','## 全部路线','', '|用户|路线|TAR|FAR|AUC|逐同词AUC|HTER|','|---|---|---:|---:|---:|---:|---:|']
for u,a in data.items():
 for key,r in a['routes'].items():lines.append('|U%s|%s|%.2f%%|%.2f%%|%.4f|%.4f|%.2f%%|'%(u,key,100*r['tar'],100*r['far'],r['auc'],r['same_word_mean_auc'],100*r['hter']))
lines+=['','## 结论','',
'本轮宏HTER从28.75%降到28.10%，仅约0.65个百分点；宏AUC0.7753至0.7958。FAR下降同时本人接受率下降，4人改善、5人变差，不能认定稳定优于完整基线。U1是当前有限测试中的零错误，不代表总体零风险；U2/U3/U5/U6/U7/U8/U9仍有明显身份重叠或拒绝问题。',
'当前不按最终结果替换各自较好版本。若要提高泛化，应另用未观察的新录制/会话检验选路，不能把训练折平衡目标当作最终两类错误必然同时下降的保证。','',
'## 核验与范围','',
'独立重新拟合225个外折模型重建OOF，重放每用户40组合选择，暴力搜索校准阈值复核平衡目标，重算全部最终分数、接受数量、AUC、每词2/16与逐攻击来源计数。全部预定用户和路线保留，不按最终成绩回退或删人。',
'G/F和教师不训练，更新的是认证RBF参数、分数组合与阈值。三类教师均为每目标单独训练的旧严格模型，认证推理只使用ACC，教师ACC增益不是生成Audio身份贡献的证明。',
'词语映射以用户词表和Audio10/20周期及跨用户相位假设为前提，未逐条语义转写。同会话历史最终数据已观察，属于探索性已知用户闭集；源特征缓存已存在，不是新盲测。嵌套分支模型的OOF被复用于阈值和融合选择，元选择不是完整独立嵌套，40候选可能选择过拟合。',
'与此前跨内容实验的词集、样本预算不同，不能直接相减归因内容。当前统一更换平衡目标及候选教师池，不能将历史U4/U5单独试验的差异全归因某一个参数。']
Path('docs/experiments/2026-10-30-nine-sameword-balanced-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'macro_public.json').write_text(json.dumps(macro,indent=2),encoding='utf-8');print('\n'.join(lines[:33]))

