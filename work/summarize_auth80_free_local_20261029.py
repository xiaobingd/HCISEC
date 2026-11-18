import json
from pathlib import Path
d=json.loads(Path('outputs/mei_auth80_free_selection_20261029/verified_summary_public.json').read_text())
assert d['status']=='PASS' and d['targets']==9
lines=['# 九用户80条认证自主选择特征结果','',
'完整772维基线固定，原始ACC、六个教师分支和四个Diffusion分支共11个候选，可不选或多选。取消分支家族配额。贪心前向按训练折AUC验证逐步加入，权重可为0.25、0.5、1，直到无合格增益。每次需要至少4/5外折提高且平均增益超过max(0.0025,配对标准误)。这不是全局最优搜索或显著性检验。','',
'各用户80条本人、640条负类认证训练；内容分组5外折/3内折。上游冻结、校准20条/身份、最终47至50条本人及八个攻击来源均不变。认证输入只有ACC。','',
'|路线|平均本人接受率|平均FAR|平均AUC|平均HTER|','|---|---:|---:|---:|---:|']
names={'baseline_rbf':'完整RBF','teacher_limited':'限额教师补充','diffusion_limited':'限额教师与Diffusion','free_selection':'自主多选'}
for k,r in d['macro'].items():lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(names[k],r['tar']*100,r['far']*100,r['auc'],r['hter']*100))
lines+=['','每格为 TAR% / FAR% / AUC。','', '|目标|限额教师加Diffusion|自主选择|补充分支数|','|---|---|---|---:|']
def fmt(r):return '%.2f / %.2f / %.4f'%(r['tar']*100,r['far']*100,r['auc'])
for u in map(str,range(1,10)):
 a=d['results'][u];lines.append('|U%s|%s|%s|%d|'%(u,fmt(a['fixed_quota_reference']),fmt(a['route']),a['selected_count']))
lines+=['','## 各用户训练内选择的组合','', '|用户|补充分支及权重|','|---|---|']
for u in map(str,range(1,10)):
 ww=d['results'][u]['weights'];desc='；'.join('%s × %s'%(k,w) for k,w in ww.items() if k!='base') or '无补充'
 lines.append('|U%s|%s|'%(u,desc))
lines+=['','各组合均由训练折锁定后执行最终测试，没有按最终表现回退到其他路线。独立复核重放训练折选择，并由保存的分支模型与最终特征重算融合分数、AUC、全部接受数及逐来源计数。',
'','原始ACC和教师ACC分支主要从输入提取信息，其入选不证明经过Audio的跨模态增强。Diffusion模型仍为外部ESMB0001 EMA初始化后的个人50轮模型，没有在这轮重新训练。',
'','局限：单种子；历史最终测试已被观察，属于探索性重复评估而非新盲测；内容标签仍依赖10/20分段周期及尾部缺行假设；闭集八个攻击身份均已知。扩大候选搜索空间可能增加选择过拟合，不能预先保证比限额融合更好。']
Path('docs/experiments/2026-10-29-auth80-free-selection-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))

