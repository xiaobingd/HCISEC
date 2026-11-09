"""Reports all users and predeclared methods, with paired scientific figures."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path('outputs');docs=Path('docs/experiments');allresults={}
for number in [1,2]:
 root=base/f'abcs43_ecapa_round{number}_20261110';r=json.loads((root/'results_public.json').read_text());p=json.loads((root/'per_user_public.json').read_text());allresults[number]=(r,p)
 methods=list(r['results']);lines=[f'# ABCS43 ECAPA探索第{number}轮：逐用户完整结果','','2026-11-10。23 Base / 8 Cal / 12未见Test，K=12、seed42。使用全部43位可用参与者，沿用固定划分和句子隔离。通道0为振动、通道1为Audio仍是未确认假设。查询不使用Audio。','', '[两轮预定方案](2026-11-10-abcs43-ecapa-two-round-protocol.md)。shape40固定保留；student为192维；supplement是二者等权拼接232维。各个人学生100步，相同初始化与批次规则；current复用上一实验结果。','', '## 汇总：Cal-EER（百分比）','','| 方法 | TAR | FAR | HTER | pooled AUC | 最差用户FAR |','|---|---:|---:|---:|---:|---:|']
 for m in methods:
  q=r['results'][m]['Cal_EER'];lines.append(f"| {m} | {100*q['tar']:.2f} | {100*q['far']:.2f} | {100*q['hter']:.2f} | {q['auc']:.4f} | {100*q['worst_target_far']:.2f} |")
 if number==1:
  lines+=['','## 解释','','冻结教师融合HTER=4.55%，现有全主体微调=4.67%，无Audio个人适配=5.19%。这次30步微调没有超过冻结教师。不能只凭超过原始形状6.23%就断言Audio收益：无Audio适配本身也有改善。','','同身份录制错配融合HTER=4.58%，异身份错配=4.89%，都与正确目标接近。当前证据不能证明精确逐录制跨模态对应是主要收益来源；结果可能主要依赖身份级关系、正则化和融合。异身份随机错配也未大幅崩溃，进一步削弱强因果解释；需要更多种子和新的身份划分。']
 else:
  lock=json.loads((root/'selection_public.json').read_text());lines+=['','## Cal选择（Test训练前锁定）','',f"选择：**{lock['selected_method']}**。规则：Cal01–04定阈值、Cal05–08以HTER/AUC排序。以下不是Test选择结果。",'', '| 方法 | 内层Cal验证TAR% | FAR% | HTER% | AUC |','|---|---:|---:|---:|---:|']
  for v in lock['choices']:lines.append(f"| {v['method']} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} |")
  chosen=r['results'][lock['selected_method']]['Cal_EER'];lines+=['',f"锁定方法在最终Test：TAR={100*chosen['tar']:.2f}%，FAR={100*chosen['far']:.2f}%，HTER={100*chosen['hter']:.2f}%。全部预定候选均报告，不能看到Test后改选。投影与注意力候选还同时改变了步数和学习率，不能单独归因某一层。"]
 lines+=['','## 配对差异：HTER百分点（身份bootstrap）','','| 比较 | 均值差 | 95%区间 | 改善人数/12 |','|---|---:|---|---:|']
 for name,v in r['paired'].items():
  lo,hi=v['identity_bootstrap95'];lines.append(f"| {name} | {100*v['hter_delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {v['improved_targets']} |")
 for role,title in [('Test','全部12位未见用户'),('Cal','全部8位校准用户（用于校准，不是独立Test）')]:
  lines+=['',f'## {title}：Cal-EER','','| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |','|---|---|---:|---:|---:|---:|---:|']
  for u in sorted(set(v['user'] for v in p if v['role']==role)):
   for m in methods:
    v=next(v for v in p if v['user']==u and v['method']==m and v['point']=='Cal_EER');lines.append(f"| {u} | {m} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} | {v['genuine_n']}/{v['impostor_n']} |")
 lines+=['','## Cal-FAR5工作点汇总','','| 方法 | TAR% | FAR% | HTER% |','|---|---:|---:|---:|']
 for m in methods:
  q=r['results'][m]['Cal_FAR5'];lines.append(f"| {m} | {100*q['tar']:.2f} | {100*q['far']:.2f} | {100*q['hter']:.2f} |")
 lines+=['','## 限制与可复现性','','这是同一已探索数据划分上的单种子多方法探索，不是独立确认。多项比较的区间未做多重比较校正；不能选择Test最好者作为正式最终方案。Cal-FAR5不保证未知Test FAR≤5%。每位24本人、264攻击，总288本人/3168攻击比较；同一录制参与多目标比较，不能按独立录制计算置信区间。bootstrap单位是目标身份。','','重要的训练限制：个人适配使用本人12条注册句子，而Base回放使用另一组4句；个人分类头的目标身份与背景身份没有同句困难负样本。虽然最终跨句查询和跨用户同句比较保持隔离，这仍可能让个人适配学习到内容差异；不能称适配阶段已完全消除文本混杂。此问题两轮都相同，未引入Test泄漏，但可能限制教师微调收益，下一步应补充同注册句子的Base背景录制后再做独立对照。','','Base23位仅用于通用模型训练，不伪报未见用户结果。全部20个注册目标均报告，未移除难用户或难攻击。完整双工作点逐用户数据见同目录的per_user_public.csv。教师、学生权重、阈值、样本分数、源身份映射不发布。检查记录见verification_public.json。']
 (docs/f'2026-11-10-abcs43-ecapa-round{number}-results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 with (root/'per_user_public.csv').open('w',newline='',encoding='utf8') as f:
  fields=['user','role','method','point','tar','far','frr','hter','auc','eer','genuine_n','impostor_n'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in p:w.writerow({k:(round(row[k],8) if isinstance(row[k],float) else row[k]) for k in fields})

fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,number in zip(axes,[1,2]):
 r,p=allresults[number];names=['shape40','no_audio_supplement','frozen_supplement','current_supplement']+(['same_identity_shuffle_supplement','wrong_identity_supplement'] if number==1 else ['projection_supplement','attentive_supplement']);values=[r['results'][n]['Cal_EER']['hter']*100 for n in names];labels=[n.replace('_supplement',' + shape').replace('_',' ') for n in names];ax.barh(labels,values,color=['#777777','#5c91ad','#2d817e','#c7933c','#9564a7','#a26662']);ax.invert_yaxis();ax.set_xlabel('HTER (%) at calibration EER threshold');ax.set_title(f'Round {number}: unseen targets (n=12)');ax.set_xlim(0,max(values)*1.22)
 for i,v in enumerate(values):ax.text(v+.05,i,f'{v:.2f}',va='center',fontsize=9)
fig.suptitle('ABCS43 exploratory study; fixed split / one seed / channel mapping provisional',fontsize=10);out=base/'abcs43_ecapa_round2_20261110';fig.savefig(out/'two_round_hter.png',dpi=180);fig.savefig(out/'two_round_hter.pdf');print('REPORTS_COMPLETE')
