import json
from pathlib import Path
p=Path('outputs/mei_auth80_diffusion_20261029/verified_summary_public.json');d=json.loads(p.read_text())
before=json.loads(Path('outputs/mei_auth80_20261029/verified_summary_public.json').read_text())
assert d['status']=='PASS' and len(d['results'])==9
lines=['# 80条认证训练下Diffusion补充结果','',
'九个用户的上游模型和教师认证系统固定；只在80条本人及640条负类训练内，按内容分组嵌套交叉验证选择最多一个Diffusion分支。校准、最终测试及八个攻击身份全部不变。不重训生成器、教师或Diffusion。本轮不能称从零重训Diffusion。','',
'|路线|宏本人接受率|宏FAR|宏AUC|宏HTER|','|---|---:|---:|---:|---:|']
for name,r in [('80条教师补充',d['reference_macro']),('再允许Diffusion',d['macro'])]:
 lines.append('|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(name,r['tar']*100,r['far']*100,r['auc'],r['hter']*100))
lines+=['','每格 TAR% / FAR% / AUC。','', '|目标|80条教师|允许Diffusion|新增分支|','|---|---|---|---|']
def fmt(r):return '%.2f / %.2f / %.4f'%(r['tar']*100,r['far']*100,r['auc'])
for u in map(str,range(1,10)):
 a=d['results'][u]
 for route,record in a['reference'].items():
  assert all(before['results'][u]['routes'][route][k]==v for k,v in record.items())
 selected=a['selected_diffusion']
 desc='不加入' if not selected else '%s × %s'%(selected[0],a['route']['weights'][selected[0]])
 lines.append('|U%s|%s|%s|%s|'%(u,fmt(a['reference']['auto_without_diffusion']),fmt(a['route']),desc))
lines+=['','## 结果边界','',
'表格完整保留没有选中以及选中后退步的用户。分支与权重仅由训练折选择，不依据最终结果回退。保存分数已独立重算AUC、接受计数、HTER和八个来源标签；无新增分支时逐项核验最终分数与前一轮相同。',
'','单种子、同会话历史数据，最终测试已在之前实验中观察。因此是探索性重复评估，不是新的盲测。内容映射依赖前100条10项周期、后200条20项周期和尾部缺行假设。',
'','Diffusion来自外部ESMB0001 EMA初始化再个人50轮微调的冻结模型。潜变量1536维、引导特征492维、循环特征168维及联合2196维分别读出；认证推理只输入ACC，不使用真实测试Audio。']
Path('docs/experiments/2026-10-29-auth80-diffusion-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))

