"""All-user report, compact public CSV and scientific plots for personal CycleGAN."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

shared=Path('outputs/vibravox_cycle_supplement_20261110');root=Path('outputs/vibravox_personal_cycle_supplement_20261110');r=json.loads((root/'results_public.json').read_text());users=json.loads((root/'per_user_public.json').read_text());lock=json.loads((root/'selection_public.json').read_text());train=json.loads((shared/'gan_training_public.json').read_text());chosen=lock['selected'];methods=list(r['results']);lines=['# VibraVox：注册者单人CycleGAN微调与特征补充','','2026-11-10。59人：35 Base预训练、12 Cal选择与校准、12未见Test注册。每个目标12条注册配对，查询仅振动；所有11位攻击者保留。完整公开每位Test与Cal结果。','', '[技术方案与用户补充要求](2026-11-10-vibravox-cycle-supplement-protocol.md)。这是紧凑共享初始化＋单人微调新增分支，不是旧完整772维重现。VibraVox是单轴额头加速度，不是本地头显速度。','', '## 训练路线','', f"- 共享G/F只用35位Base fit，{train['steps']}次更新、batch16，约{train['equivalent_epochs']:.2f}个训练集等效轮；按70条Base-develop录制选择第{train['selected_step']}步。不是200轮。",'- G/F为width32、4残差块，PatchGAN；LSGAN＋10配对L1＋10cycleL1＋identity mapping L1。identity mapping不是用户身份损失。','- 每位Cal/Test从同一共享G/F独立重置，仅用自己的12条注册振动–Audio，微调100步、batch6、LR2e-5，保存25/100步；不使用任何其他身份样本参与个人更新。判别器冻结为Base共享训练最终步，G/F初始化为Base-dev选择步。','- 沿用Base-fit的输入归一化和特征StandardScaler，个人注册/查询不重新拟合尺度。其他用户攻击样本经过注册者的同一个模型。','- 原始shape40＋已有个人Audio教师V4学生192=baseline232保持不变。固定个人补充分支权重0.5；自主选择可不补充。匹配shared/random分支与个人选择使用同分支、同权重。','- Cal前6位定阈值、后6位选择25/100步及分支/权重；锁定后才训练Test个人G/F。最终全部12位Cal分别定Cal-EER与Cal-FAR5阈值。注册Audio共288条，查询Audio不参与生成器微调或认证。','', '## 校准选择','',f"锁定：step={chosen['step']}，branch={chosen['branch']}，alpha={chosen['alpha']}。Cal开发HTER={100*chosen['hter']:.2f}%，AUC={chosen['auc']:.4f}。不能看到Test后更换该选择。",'', '## 全部Test汇总（Cal-EER）','','| 方法 | TAR% | FAR% | HTER% | pooled AUC | 最差目标FAR% |','|---|---:|---:|---:|---:|---:|']
for m in methods:
 q=r['results'][m]['Cal_EER'];lines.append(f"| {m} | {100*q['tar']:.2f} | {100*q['far']:.2f} | {100*q['hter']:.2f} | {q['auc']:.4f} | {100*q['worst_target_far']:.2f} |")
baseline=r['results']['baseline232']['Cal_EER'];selected=r['results']['personal_selected']['Cal_EER'];delta=100*(selected['hter']-baseline['hter']);lo,hi=r['paired_vs_baseline232']['personal_selected']['identity_bootstrap95'];lines+=['','## 主要观察','',f"个人自主选择相对现baseline232，HTER变化{delta:+.2f}个百分点，身份bootstrap 95%区间[{100*lo:+.2f},{100*hi:+.2f}]个百分点。其结果是TAR={100*selected['tar']:.2f}%、FAR={100*selected['far']:.2f}%、HTER={100*selected['hter']:.2f}%。",'']
if chosen['step']==0:lines+=['Cal没有选择任何个人CycleGAN补充分支，所以自主选择结果回到原baseline232。这不是个人模型没训练：24人均已训练25/100步，固定分支结果完整列出。此时matched_shared_branch与matched_random_branch也退化为原基线，不能把它们的相同分数解释成随机网络与学习网络等效；共享模型的非零随机对照另见[共享实验逐用户CSV](../../outputs/vibravox_cycle_supplement_20261110/per_user_public.csv)。']
elif hi>=0:lines+=['该差异尚不足以确认稳定改善；不能按Test选择其他效果更好的固定分支替换主结果。']
else:lines+=['该单种子探索出现改善迹象，但还需要新身份划分/多种子及随机网络对照，不能直接确认为跨模态生成带来的身份增益。']
lines+=['','特征名：input_encoder256为G对原始振动的编码；cycle_encoder256为G对F(G(振动))的编码；cycle_statistics240为循环频谱统计。input_only100与cycle_only100仅使用个人模型编码，无原始/学生补充。shared_selected是未个人微调的共享模型自主选择参考。','','## 全部12位Test逐用户（Cal-EER）','','| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |','|---|---|---:|---:|---:|---:|---:|']
for role,title in [('Test',None),('Cal','全部12位Cal逐用户（仅校准/开发，不是独立Test）')]:
 if title:lines+=['','## '+title,'','| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |','|---|---|---:|---:|---:|---:|---:|']
 for u in sorted({v['user'] for v in users if v['role']==role}):
  for m in methods:
   q=next(v for v in users if v['user']==u and v['method']==m and v['point']=='Cal_EER');lines.append(f"| {u} | {m} | {100*q['tar']:.2f} | {100*q['far']:.2f} | {100*q['hter']:.2f} | {q['auc']:.4f} | {q['genuine_n']}/{q['impostor_n']} |")
lines+=['','## Cal-FAR5工作点','','| 方法 | TAR% | FAR% | HTER% |','|---|---:|---:|---:|']
for m in methods:
 q=r['results'][m]['Cal_FAR5'];lines.append(f"| {m} | {100*q['tar']:.2f} | {100*q['far']:.2f} | {100*q['hter']:.2f} |")
lines+=['','## 限制与复核','','每位24本人、264攻击，共288本人/3168攻击比较。攻击录制重复用于不同目标比较，bootstrap按目标身份而非录制。已有VibraVox语料多轮探索，本轮不是封存独立确认。不同用户句子不完全相同，不称同内容认证；全部句子角色通过文本哈希隔离。个人注册允许Audio，最终查询仅振动。生成误差降低不等于身份保留，个人风格可能让攻击者也趋向目标；必须看随机与共享模型对照及认证指标。','','[完整逐用户双工作点CSV](../../outputs/vibravox_personal_cycle_supplement_20261110/per_user_public.csv)，[独立检查](../../outputs/vibravox_personal_cycle_supplement_20261110/verification_public.json)。原始数据、源身份映射、模型权重、阈值和样本分数均保持私有。']
Path('docs/experiments/2026-11-10-vibravox-personal-cycle-supplement-results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
with (root/'per_user_public.csv').open('w',newline='',encoding='utf8') as f:
 fields=['user','role','method','point','tar','far','frr','hter','auc','eer','genuine_n','impostor_n'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for q in users:writer.writerow({k:round(q[k],8) if isinstance(q[k],float) else q[k] for k in fields})
names=['shape40','baseline232','shared_selected','personal25_cycle_encoder256','personal100_cycle_encoder256','personal_selected','matched_shared_branch','matched_random_branch'];fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);values=[100*r['results'][n]['Cal_EER']['hter'] for n in names];axes[0].barh(names,values,color='#438b98');axes[0].invert_yaxis();axes[0].set_xlim(0,max(values)*1.2);axes[0].set_xlabel('HTER (%) at Cal-EER threshold');axes[0].set_title('VibraVox, 12 unseen target identities')
for i,v in enumerate(values):axes[0].text(v+.03,i,f'{v:.2f}',va='center',fontsize=8)
q=[v for v in users if v['role']=='Test' and v['point']=='Cal_EER'];a=[next(v['hter'] for v in q if v['user']==f'Vib59-Test-{i:02}' and v['method']=='baseline232')*100 for i in range(1,13)];b=[next(v['hter'] for v in q if v['user']==f'Vib59-Test-{i:02}' and v['method']=='personal_selected')*100 for i in range(1,13)];axes[1].plot(range(1,13),a,'o-',label='baseline232');axes[1].plot(range(1,13),b,'s--',label='Cal-selected: no added CycleGAN' if chosen['step']==0 else 'personal CycleGAN supplement');axes[1].set_xticks(range(1,13));axes[1].set_xlabel('Anonymous Test user');axes[1].set_ylabel('HTER (%)');axes[1].legend(fontsize=8);fig.suptitle('Registration-only G/F adaptation; query vibration only; single-seed exploratory study',fontsize=10);fig.savefig(root/'authentication_comparison.png',dpi=180);fig.savefig(root/'authentication_comparison.pdf');print('REPORT_COMPLETE')

