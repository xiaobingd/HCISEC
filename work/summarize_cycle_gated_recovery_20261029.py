import json,statistics
from pathlib import Path
root=Path('outputs/mei_cycle_gated_recovery_20261029')
data=json.loads((root/'results_public.json').read_text())
assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
data={u:dict(routes=rows) for u,rows in data.items()}
lines=['# 冻结个人G/F的门控身份恢复：第二阶段','',
'U1/U4/U5，三个随机种子42/43/44。复用严格跨词划分，认证训练50、验证30、校准20条/身份；最终本人47–50条，全部八个攻击来源。生成器不训练。','',
'原始和循环640维使用同一个冻结个人G编码器读取，采用同一份仅在原始认证训练特征拟合的标准化。四条件均输出64维，最终使用相同RBF搜索范围与独立校准。','',
'原始参考MLP先用认证训练数据训练100步并冻结，用于初始化及身份一致性目标；四条件再训练400步。BCE（正类权重8）+0.1跨词SupCon；身份一致性条件额外加0.1余弦损失。检查点、RBF和阈值全部在最终测试前锁定。','',
'门控公式：z=L2Normalize(rawMLP(a)+sigmoid(g([a,b,b-a]))×delta([a,b,b-a]))。允许保留原始路径，但门值不能直接解释为循环信息贡献率。','',
'## 三seed均值 ± 样本标准差','',
'|用户|条件|本人接受率|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
summary={}
names={'raw':'原始路径，无恢复','concat':'简单拼接','gate':'门控恢复','gate_identity':'门控＋身份一致性'}
for u,entry in data.items():
 summary[u]={}
 for mode in names:
  rows=[entry['routes'][mode+'_seed%d'%s] for s in [42,43,44]]
  for row in rows:assert len(row['attack_by_user'])==8
  stats={k:dict(mean=statistics.mean(r[k] for r in rows),std=statistics.stdev(r[k] for r in rows)) for k in ['tar','far','auc','hter']};summary[u][mode]=stats
  values=[]
  for k in ['tar','far','auc','hter']:
   m=stats[k];values.append(('%.4f ± %.4f'%(m['mean'],m['std'])) if k=='auc' else ('%.2f%% ± %.2f'%(100*m['mean'],100*m['std'])))
  lines.append('|U%s|%s|%s|'%(u,names[mode],'|'.join(values)))
lines+=['','## 解释边界','',
'该实验比较整体恢复与认证流程。原始、拼接、门控的结构与参数量不同，不能将差异全部归因于门控本身。冻结原始参考约束是否有用，应优先比较门控与门控＋身份一致性。',
'历史最终集已被观察，属于探索性已知用户闭集验证；不是新盲测或开放集证明。词标签仍依赖10/20周期和尾部缺行映射。身份一致性使用的参考不是独立评分器。',
'没有按最终成绩挑种子、删用户或重新设阈值。生成器不因认证样本更新。上一步772维RBF必须作为外部基线保留，不应只与较弱神经原始路径比较。','',
'## 逐seed结果','', '|用户|条件|seed|TAR|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|---:|']
for u,entry in data.items():
 for key,r in entry['routes'].items():lines.append('|U%s|%s|%d|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(u,names[r['mode']],r['seed'],100*r['tar'],100*r['far'],r['auc'],100*r['hter']))
path=Path('docs/experiments/2026-10-29-cycle-gated-recovery-results.md');path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'summary_public.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('\n'.join(lines[:25]))

