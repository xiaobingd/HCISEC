import json,statistics
from pathlib import Path
root=Path('outputs/mei_independent_acc_auth_20261029');data=json.loads((root/'results_public.json').read_text());base=json.loads(Path('outputs/mei_frozen_cycle_identity_compare_20261029/results_public.json').read_text())
assert json.loads((root/'verification_public.json').read_text())['status']=='PASS'
lines=['# 独立纯ACC身份认证对照','',
'U1/U4/U5，纯ACC三种对照：6400维频谱直接RBF、320维频谱均值标准差RBF、随机初始化小型CNN128维RBF（三seed42/43/44）。不使用个人G/F编码器、生成输出、Audio或预训练教师权重。','',
'所有路线使用同一存储ACC频谱、同一严格跨词划分：认证训练50、验证30、校准20条/身份，最终本人47至50条及全部八个攻击来源。直接频谱保留全部80×80输入；统计特征为沿两个轴分别计算均值、标准差。RBF标准化仅在训练拟合。','',
'CNN为三层卷积16/32/64＋GroupNorm＋GELU，时频全局均值标准差后映射128维L2表征。从零训练400步，均衡36样本/批，BCE正类权重8＋0.1跨词SupCon；仅认证训练数据更新参数。验证词选检查点，同一RBF网格C1/10/100、gamma scale/0.1选参，独立校准选阈值。每目标全部路线锁定后才评分最终数据。','',
'|用户|方法|本人接受率|FAR|AUC|HTER|','|---|---|---:|---:|---:|---:|']
summary={}
def values(rows):
 out={k:dict(mean=statistics.mean(r[k] for r in rows),std=statistics.stdev(r[k] for r in rows) if len(rows)>1 else 0.) for k in ['tar','far','auc','hter']};return out
for u,a in data.items():
 assert a['no_generator_or_audio'] and a['score_counts_verified'];rr=a['routes'];summary[u]={}
 groups=[('baseline772','完整772维基线',[base[u]['base772']]),('direct_flat6400','纯ACC频谱直接RBF',[rr['direct_flat6400']]),('direct_stats320','纯ACC统计RBF',[rr['direct_stats320']]),('cnn128','独立纯ACC CNN＋RBF',[rr['cnn128_seed%d'%s] for s in [42,43,44]])]
 for key,name,rows in groups:
  stats=values(rows);summary[u][key]=stats;v=[]
  for k in ['tar','far','auc','hter']:
   m=stats[k];text=('%.4f'%m['mean']) if k=='auc' else ('%.2f%%'%(100*m['mean']))
   if len(rows)>1:text+=(' ± %.4f'%m['std']) if k=='auc' else (' ± %.2f'%(100*m['std']))
   v.append(text)
  lines.append('|U%s|%s|%s|'%(u,name,'|'.join(v)))
lines+=['','## 结果解释','',
'U1独立ACC CNN三seed平均AUC0.9980、HTER2.68%，优于完整基线0.9909、5.43%；U4纯ACC统计RBF的AUC0.8818、HTER21.09%，优于完整基线0.8559、24.77%；U5完整基线AUC0.8475、HTER21.00%，优于本轮三种纯ACC路线。',
'因此不能笼统声称当前基线优于纯ACC。此处列出每位用户的差异用于分析，未根据最终成绩选择并部署各自最优架构；若要建立自主选择系统，仍需在训练/验证内选择并在独立数据检验。','',
'## 研究边界','',
'公平之处是最终录制、词语角色、全部攻击来源、RBF网格及校准规则相同。训练资源仍不完全匹配：772维基线的生成器此前使用150条上游配对录制，独立CNN仅使用50条认证训练录制；尚未补做等量上游纯ACC预训练。三seed衡量优化随机性，不代表独立会话。',
'这是对存储的ACC频谱的纯ACC对照，没有重新处理原始时域三轴。小型CNN不代表纯ACC方法上限。历史最终集已观察，属于探索性已知用户闭集结果；词语语义仍有10/20周期与尾部缺行假设。',
'最终成绩不用于挑选架构、seed、用户或阈值。报告全部预定路线。不能据此直接归因Audio增加了新的身份信息；即使基线更好，也可能来自训练数据量、表示结构和辅助任务差异。','',
'## 独立CNN逐seed结果','', '|用户|seed|TAR|FAR|AUC|HTER|','|---|---:|---:|---:|---:|---:|']
for u,a in data.items():
 for seed in [42,43,44]:
  r=a['routes']['cnn128_seed%d'%seed];lines.append('|U%s|%d|%.2f%%|%.2f%%|%.4f|%.2f%%|'%(u,seed,100*r['tar'],100*r['far'],r['auc'],100*r['hter']))
Path('docs/experiments/2026-10-29-independent-acc-auth-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'summary_public.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('\n'.join(lines[:25]))

