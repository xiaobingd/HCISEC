import json
from pathlib import Path
root=Path('outputs/mei_auth80_20261029')
d=json.loads((root/'verified_summary_public.json').read_text())
ref=json.loads(Path('outputs/mei_strict_autosupplement_20261029/summary_public.json').read_text())
assert d['status']=='PASS' and len(d['results'])==9
lines=['# 九用户跨内容认证训练量50与80对照','',
'上游生成器和教师冻结不变，每用户认证训练本人50增至80、负类400增至640。原认证选模30条并入训练；5外折与3内折按内容分组验证。校准20条、最终47–50条本人及全部八个攻击来源保持不变。最终测试此前已观察，本次为探索性重复评估。','',
'|路线|训练本人条数|宏TAR|宏FAR|宏AUC|宏HTER|','|---|---:|---:|---:|---:|---:|']
for route,name in [('baseline_rbf','完整772维RBF'),('auto_without_diffusion','教师可选补充')]:
 for budget,label in [('fit50_reference',50),('routes',80)]:
  r=d['macro'][route][budget]
  lines.append('|%s|%d|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(name,label,100*r['tar'],100*r['far'],r['auc'],100*r['hter']))
lines+=['','每格为 TAR% / FAR% / AUC。','', '|用户|RBF50|RBF80|教师50|教师80|','|---|---|---|---|---|']
def fmt(r):return '%.2f / %.2f / %.4f'%(r['tar']*100,r['far']*100,r['auc'])
for u in map(str,range(1,10)):
 a=d['results'][u]
 assert a['fit_positive']==80 and a['fit_negative']==640
 for k,r in a['fit50_reference'].items():
  for metric in ['tar','far','auc','hter']:assert abs(r[metric]-ref['results'][u]['routes'][k][metric])<1e-12
 lines.append('|U%s|%s|%s|%s|%s|'%(u,fmt(a['fit50_reference']['baseline_rbf']),fmt(a['routes']['baseline_rbf']),fmt(a['fit50_reference']['auto_without_diffusion']),fmt(a['routes']['auto_without_diffusion'])))
lines+=['','## 结果解释','',
'增加认证训练量后，完整RBF的宏本人接受率提高1.11个百分点、FAR降低0.26个百分点，AUC略降，HTER改善0.68个百分点，整体增益有限。教师系统FAR降低0.55个百分点，但本人接受率下降2.27个百分点，HTER恶化0.86个百分点。因此不能认定80条方案整体优于50条。',
'','U1的80条教师方案本人接受48/50，误接受0/391；这是有限测试中的零误接受，不代表总体FAR为零。U4的RBF本人接受从34/47提高至38/47，但教师补充后只有35/47；安全性与可用性存在权衡。U3明显退步，增加数量没有解决表征及内容迁移问题。',
'','所有18条路线已由保存的分数独立重算AUC、接受数、HTER及逐来源攻击计数。未根据最终结果删除用户、攻击者或查询，也未回退选择较好版本。',
'','局限：单种子；同会话历史数据；内容映射依赖分段周期及尾部缺行假设。当前改动同时增加30条录制和3个训练内容组，不能区分纯录制数量与内容多样性的贡献。教师路线还允许原始ACC分支，平均增益不能全部归给教师。',
'','启动协议见2026-10-29-auth80-expansion.md。没有重训CycleGAN、教师或Diffusion，也没有重跑SupCon，不能将本表当作SupCon增样本后的结果。']
Path('docs/experiments/2026-10-29-auth80-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))

