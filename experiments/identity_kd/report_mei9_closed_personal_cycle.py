"""Full nine-user closed-set report, nominal same-word versus cross-word."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('outputs/mei9_closed_personal_cycle_20261110');r=json.loads((root/'results_public.json').read_text());per=json.loads((root/'per_user_public.json').read_text());lock=json.loads((root/'selection_public.json').read_text());gan=json.loads((root/'gan_complete_public.json').read_text());choice=lock['selected'];methods=list(r['results']);lines=['# mei九人闭集：个人CycleGAN特征补充实验','','2026-11-10。真正闭集：九位用户均参与共享身份训练，九个独立本人/非本人二分类RBF头均使用全部八位已知负类；不把开放模型all_eight_mixed测试改名为闭集。全部九人报告，不剔除难用户或攻击来源。','', '[完整预定方案](2026-11-10-mei9-closed-personal-cycle-protocol.md)。本报告的编号是raw9索引，不保证对应历史trainb U编号。输入为同一头显采集的三轴OpenVR tracking vVelocity，不是原始硬件加速度。词序未逐条转写核验，因此同词/跨词为名义条件。','', '## 数据与模型','', '- 共享ACC编码器使用九人fit身份监督700步；Audio ECAPA使用九人fit回放＋注册共144条Audio微调100步，再冻结并给ACC学生蒸馏100步。这里Audio教师是九人共享闭集教师，个人CycleGAN才是逐注册者微调。','- 原始形状120维保持不变，与学生192维构成baseline312；保留原始720维统计、学生单支路。',f"- 共享G/F预训练{gan['steps']}次更新，batch16、约{gan['equivalent_epochs']:.2f}等效轮，按36条develop录制选择第{gan['selected_step']}步。G为3→1、F为1→3，LSGAN＋配对L1＋循环L1，因通道不同不加恒等映射损失。",'- 九个个人G/F各从共享模型重置，仅用本人的12条注册配对微调100步，保存25/100步，判别器冻结；其余用户注册负类由当前目标的G/F处理。','- 每人12注册（4词×3遍），用于二分类头训练；CalA、CalB各4条（每词下一遍），仅用于阈值与分支选择。最终阈值用CalA+B的8条本人/64条负类选择。','- 名义同词查询每人16条、128条攻击；跨词每人24条、192条攻击。所有样本索引互斥；同词与跨词共用同一阈值。查询Audio不参与模型前向。缓存归档可能整体解压，不能声称从未物理读取查询Audio字节。','- 后端固定balanced RBF SVM，C=10、gamma=scale；只在108条注册特征上拟合，不在拼接后重做StandardScaler。所有方法使用相同后端和数据。','', '## 校准选择','',f"九目标平均CalB选择：step={choice['step']}，branch={choice['branch']}，alpha={choice['alpha']}；CalA选阈值。CalB HTER={100*choice['hter']:.2f}%，AUC={choice['auc']:.4f}。配置在提取最终个人G/F查询特征前锁定。",'', '## 宏平均结果（Cal-EER，百分比）','','| 方法 | 条件 | TAR | FAR | HTER | 用户平均AUC |','|---|---|---:|---:|---:|---:|']
for method in methods:
 for cond in ['sameword','crossword']:
  v=r['results'][method][cond]['Cal_EER'];lines.append(f"| {method} | {'名义同词' if cond=='sameword' else '名义跨词'} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} |")
lines+=['','## 自主选择相对baseline312的配对差异','','| 条件 | HTER差百分点 | 身份bootstrap95%区间 | 改善人数/9 |','|---|---:|---|---:|']
for cond in ['sameword','crossword']:
 v=r['paired_vs_baseline312']['selected_cycle'][cond];lo,hi=v['identity_bootstrap95'];lines.append(f"| {cond} | {100*v['hter_delta']:+.2f} | [{100*lo:+.2f},{100*hi:+.2f}] | {v['improved_users']} |")
if choice['step']==0:lines+=['','Cal选择不增加CycleGAN分支，自主选择结果等于baseline312；九个个人G/F都确实完成了25/100步训练，固定补充结果仍完整报告。']
lines+=['','## 逐用户完整结果（Cal-EER）','','| 用户 | 条件 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |','|---|---|---|---:|---:|---:|---:|---:|']
for u in sorted({v['user'] for v in per}):
 for cond in ['sameword','crossword']:
  for method in methods:
   v=next(v for v in per if v['user']==u and v['method']==method and v['condition']==cond and v['point']=='Cal_EER');lines.append(f"| {u} | {cond} | {method} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} | {v['genuine_n']}/{v['impostor_n']} |")
lines+=['','## Cal-FAR5宏平均','','| 方法 | 条件 | TAR% | FAR% | HTER% |','|---|---|---:|---:|---:|']
for method in methods:
 for cond in ['sameword','crossword']:
  v=r['results'][method][cond]['Cal_FAR5'];lines.append(f"| {method} | {cond} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} |")
lines+=['','## 解读边界','','本轮是已见身份闭集，并不证明未见用户注册泛化。与此前开放集比较还同时改变了身份训练、后端和查询数量，不能把差异全部归因闭集。只有本轮方法之间是固定数据/固定后端的配对比较。','','名义跨词仍取决于推断词序，未独立完成逐录制语义核验。数据来自同一头显、缺少重戴/跨天/跨设备条件标记；不能排除佩戴、设备或录制条件身份线索。单种子、多方法探索，bootstrap按用户身份，未经多重比较校正。最终Cal-FAR5不保证Test FAR≤5%。没有加入随机未训练GAN对照，因此即使个别分支改善，也不能单凭本轮证明跨模态生成是唯一增益来源。','','[全部九人双工作点CSV](../../outputs/mei9_closed_personal_cycle_20261110/per_user_public.csv)，[独立检查记录](../../outputs/mei9_closed_personal_cycle_20261110/verification_public.json)。原始信号、源姓名、个人权重、阈值及样本分数保持私有。']
Path('docs/experiments/2026-11-10-mei9-closed-personal-cycle-results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
with (root/'per_user_public.csv').open('w',newline='',encoding='utf8') as f:
 fields=['user','method','condition','point','tar','far','frr','hter','auc','eer','genuine_n','impostor_n'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for v in per:w.writerow({k:round(v[k],8) if isinstance(v[k],float) else v[k] for k in fields})
fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
for ax,cond in zip(axes,['sameword','crossword']):
 for method,label in [('raw120','Raw shape120'),('baseline312','Audio-assisted baseline312'),('selected_cycle','Cal-selected CycleGAN supplement')]:
  vv=[next(v['hter'] for v in per if v['user']==f'Mei9-U{u:02}' and v['method']==method and v['condition']==cond and v['point']=='Cal_EER')*100 for u in range(1,10)];ax.plot(range(1,10),vv,'o-',label=label)
 ax.set_title('Nominal '+cond);ax.set_xticks(range(1,10));ax.set_xlabel('Anonymous raw9 user');ax.set_ylabel('HTER (%)');ax.legend(fontsize=8)
fig.suptitle('Mei9 closed set: all nine training identities / all eight known attackers / one seed');fig.savefig(root/'mei9_closed_comparison.png',dpi=180);fig.savefig(root/'mei9_closed_comparison.pdf');print('CLOSED_REPORT_COMPLETE')
