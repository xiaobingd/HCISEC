"""Complete anonymous per-user ABCS43 open-set CycleGAN report."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('outputs/abcs43_open_personal_cycle_20261110');r=json.loads((root/'results_public.json').read_text());per=json.loads((root/'per_user_public.json').read_text());lock=json.loads((root/'selection_public.json').read_text());gan=json.loads((root/'gan_complete_public.json').read_text());c=lock['selected'];methods=list(r['results']);lines=['# ABCS43开放集：个人CycleGAN补充结果','','2026-11-10。全部43人固定23 Base / 8 Cal / 12 Test，K=12、seed42。共享训练未见Cal/Test身份，注册阶段各自只用本人12条配对微调；每个目标保留全部11位未见攻击者。','', '**通道0为振动、通道1为Audio仍是工作假设，物理归属尚未独立确认。结果不能直接表述为已验证的BC/AC认证。**','', '[预定方案](2026-11-10-abcs43-open-personal-cycle-protocol.md)。原始shape40＋既有个人ECAPA/V4学生192的232维基线保持不变。这里是紧凑CycleGAN新增分支，不是历史完整772维重现。','', '## 技术路线与隔离','',f"- 从相同2445条WAV提取统一40×32对数功率谱。共享G/F只用23位Base的1380 fit配对，{gan['steps']}更新、batch16，约{gan['equivalent_epochs']:.2f}等效轮；345 develop选择第{gan['selected_step']}步及对应判别器，不是200轮。",'- 全段去直流、16kHz、4096 Hann/1024 hop/16384 FFT、50–7600Hz几何40频带；时间插值32格。所有人使用同一Base-fit全局模态均值/标准差和3SD截断，不拟合个人尺度。Audio指频谱，非可听波形。','- G/F width32、4残差块、PatchGAN；LSGAN＋10配对L1＋10循环L1＋1恒等映射L1。恒等映射不是身份监督。','- 每位Cal/Test独立重置共享G/F，仅用本人12条注册配对微调100步、batch6、LR2e-5，保存25/100步，固定Base判别器。每个目标模型同时处理本人及全部攻击者。','- 输入G编码256、循环G编码256、生成Audio的F编码256、循环统计240、生成统计80；Scaler仅拟合共享G/F的Base-fit输出后固定。原始统计和已有学生不重训。','- Cal01–04定阈值，Cal05–08选择31个配置之一（两步数×五分支×三权重＋不补充），攻击查询限制在各自4人子群。锁定后才训练Test个人G/F。最终用全部8位Cal定Cal-EER/Cal-FAR5阈值。','- 注册12文本、Cal查询24文本、Test查询24文本在对应群体内真实一致，且与Base fit/develop角色全局互斥。查询Audio不计算特征或前向；读取立体声WAV进行CRC和通道0解码会涉及整文件，不能称未物理读取查询Audio字节。','', '## Cal选择','',f"step={c['step']}，branch={c['branch']}，alpha={c['alpha']}；Cal开发HTER={100*c['hter']:.2f}%，AUC={c['auc']:.4f}。该配置在Test个人模型训练前锁定。",'', '## 最终Test汇总（Cal-EER）','','| 方法 | TAR% | FAR% | HTER% | pooled AUC | 最差目标FAR% |','|---|---:|---:|---:|---:|---:|']
for method in methods:
 v=r['results'][method]['Cal_EER'];lines.append(f"| {method} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} | {100*v['worst_target_far']:.2f} |")
lines+=['','## 主结果与对照解释','']
if c['step']==0:lines+=['Cal选择不增加CycleGAN分支，所以selected_cycle_supplement回到current_supplement；并非个人模型未训练，全部20位注册者均完成25/100步。此时matched_shared/matched_random也退化为原基线，不能用其相同分数判断学习网络和随机网络等价；应看预定固定非零的shared/random input/cycle对照。']
else:lines += [f"Cal锁定的是个人{c['step']}步的{c['branch']}分支，权重{c['alpha']}；不能根据Test换成其他固定分支。匹配shared/random使用相同分支、权重，固定input/cycle另用0.5权重。"]
primary=r['paired']['selected_cycle_supplement_minus_current_supplement'];lo,hi=primary['identity_bootstrap95'];lines += ['',f"相对原232维基线，主结果HTER变化{100*primary['hter_delta']:+.2f}个百分点，身份bootstrap95%区间[{100*lo:+.2f},{100*hi:+.2f}]，{primary['improved_targets']}/12位改善。"]
lines+=['','## 全部配对HTER差（百分点）','','| 比较 | 均值差 | 身份bootstrap95%区间 | 改善人数/12 |','|---|---:|---|---:|']
for name,v in r['paired'].items():
 if name.endswith('_minus_current_supplement'):
  lo,hi=v['identity_bootstrap95'];lines.append(f"| {name} | {100*v['hter_delta']:+.2f} | [{100*lo:+.2f},{100*hi:+.2f}] | {v['improved_targets']} |")
for role,title in [('Test','全部12位未见目标'),('Cal','全部8位Cal诊断（用于选择与校准，不是独立Test）')]:
 lines+=['','## '+title+'：Cal-EER','','| 用户 | 方法 | TAR% | FAR% | HTER% | AUC | 本人/攻击 |','|---|---|---:|---:|---:|---:|---:|']
 for u in sorted({v['user'] for v in per if v['role']==role}):
  for method in methods:
   v=next(v for v in per if v['user']==u and v['method']==method and v['point']=='Cal_EER');lines.append(f"| {u} | {method} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} | {v['auc']:.4f} | {v['genuine_n']}/{v['impostor_n']} |")
lines+=['','## Cal-FAR5工作点汇总','','| 方法 | TAR% | FAR% | HTER% |','|---|---:|---:|---:|']
for method in methods:
 v=r['results'][method]['Cal_FAR5'];lines.append(f"| {method} | {100*v['tar']:.2f} | {100*v['far']:.2f} | {100*v['hter']:.2f} |")
lines+=['','## 边界与完整性','','每位Test 24本人/264攻击，共288本人/3168攻击比较，攻击录制用于多个目标，不是独立3168条录制；bootstrap按目标身份，不按录制。仍是单种子、已探索语料，不是封存独立确认，多个候选未做多重比较校正。Cal-FAR5不保证Test FAR≤5%。23位Base不伪报未见用户认证结果，全部Test/Cal均报告。','','原教师的Base回放与个人注册内容不同，此基线限制仍存在，本轮未重新训练该教师。生成或循环误差下降不等于身份保留，个人风格也可能让攻击输入趋向注册者。应同时看原始形状、232维、个人模型、共享模型和随机特征，并验证新的身份划分和更多种子。','','[全部逐用户双工作点CSV](../../outputs/abcs43_open_personal_cycle_20261110/per_user_public.csv)，[检查记录](../../outputs/abcs43_open_personal_cycle_20261110/verification_public.json)。原始信号、私有频谱包、模型权重、姓名/文本映射、阈值、样本分数不发布。']
if (root/'verification_public.json').exists():
 audit=json.loads((root/'verification_public.json').read_text());lines+=['',f"独立复核{audit['personal_checkpoints']}个个人G/F检查点、{audit['workpoint_checks']}个工作点通过；原四个参考结果复现，查询Audio特征/前向为0，所有查询接受/拒绝决定在GPU重算后不变，最大特征/分数差{audit['max_feature_or_score_error']:.3g}。"]
Path('docs/experiments/2026-11-10-abcs43-open-personal-cycle-results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
with (root/'per_user_public.csv').open('w',newline='',encoding='utf8') as f:
 fields=['user','role','method','point','tar','far','frr','hter','auc','eer','genuine_n','impostor_n'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for v in per:w.writerow({k:round(v[k],8) if isinstance(v[k],float) else v[k] for k in fields})
names=['shape40','current_supplement','personal25_cycle_supplement','personal100_cycle_supplement','selected_cycle_supplement','shared_cycle_supplement','random_cycle_supplement'];fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);values=[100*r['results'][m]['Cal_EER']['hter'] for m in names];axes[0].barh(names,values,color='#398592');axes[0].invert_yaxis();axes[0].set_xlim(0,max(values)*1.2);axes[0].set_xlabel('HTER (%) at Cal-EER threshold');axes[0].set_title('ABCS43 open set: 12 unseen targets')
for i,v in enumerate(values):axes[0].text(v+.03,i,f'{v:.2f}',va='center',fontsize=8)
for method,label in [('current_supplement','Current232'),('selected_cycle_supplement','Cal-selected: no added CycleGAN' if c['step']==0 else 'Personal CycleGAN Cal-selected')]:
 y=[next(v['hter'] for v in per if v['user']==f'ABCS43-Test-{u:02}' and v['method']==method and v['point']=='Cal_EER')*100 for u in range(1,13)];axes[1].plot(range(1,13),y,'o-',label=label)
axes[1].set_xticks(range(1,13));axes[1].set_xlabel('Anonymous Test user');axes[1].set_ylabel('HTER (%)');axes[1].legend(fontsize=8);fig.suptitle('Base23 / Cal8 / Test12; own12 paired registration fine-tuning; physical channel mapping provisional',fontsize=10);fig.savefig(root/'abcs43_open_cycle.png',dpi=180);fig.savefig(root/'abcs43_open_cycle.pdf');print('ABCS_OPEN_REPORT_COMPLETE')
