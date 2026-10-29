import json
from pathlib import Path
root=Path('outputs/mei_optional_cycle_score_20261029');data=json.loads((root/'results_public.json').read_text())
lines=['# 完整772维基线加可关闭循环分数补充','',
'复用原始个人G/F和已锁定的二分类RBF：完整772维主分数不变，循环640维独立打分；两路各用认证训练分数均值、标准差做标准化。融合为s772+alpha*sCycle。','',
'alpha候选0/0.125/0.25/0.5/1/2，只在验证词按逐词平均AUC、总体AUC、较小权重选择；随后在独立校准集确定阈值。保留全部八个攻击来源，不修改测试样本。','',
'|用户|方法|循环权重|本人接受率|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|---:|']
for u,a in data.items():
 assert a['baseline_replay_verified'] and a['score_count_verified']
 for key,name in [('baseline772','完整772维'),('optional_cycle','自主选择循环补充')]:
  r=a['routes'][key];assert len(r['attack_by_user'])==8
  lines.append('|U%s|%s|%.3f|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(u,name,r['alpha'],100*r['tar'],100*r['far'],r['auc'],100*r['hter']))
lines+=['','## 验证集选择过程','', '|用户|alpha|验证逐词AUC|验证总体AUC|','|---|---:|---:|---:|']
for u,a in data.items():
 selected=min(a['candidates'],key=lambda r:(-r['validation_sameword_auc'],-r['validation_auc'],r['alpha']))
 assert selected==a['selection']
 for r in a['candidates']:lines.append('|U%s|%.3f|%.4f|%.4f|'%(u,r['alpha'],r['validation_sameword_auc'],r['validation_auc']))
lines+=['','## 范围与限制','',
'未训练新生成器、教师、适配器或分类器。该实验仅检验现有循环编码的独立分数是否有补充价值。基线AUC、TAR、FAR与上一阶段逐一回放一致；分数AUC与阈值接受数量重新计算。',
'标准化只使用认证训练集统计，不是正式AS-Norm。分类器参数与融合权重复用同一验证集选择，可能过拟合验证词。零权重提供候选回退，但不保证验证选择的非零权重在最终集改善。',
'历史最终集已观察，本轮仍是探索性已知用户闭集验证；不是开放集或独立盲测。词语映射仍依赖10/20周期与尾部缺行假设。没有按最终结果改alpha、删攻击用户或改阈值。']
path=Path('docs/experiments/2026-10-29-optional-cycle-score-results.md');path.write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[:18]))

