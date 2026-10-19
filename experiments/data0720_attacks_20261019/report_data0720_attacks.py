"""Produce aggregate external-attacker report and source-level plot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent/'outputs/data0720_attacks_20261019'
s=json.loads((root/'summary.json').read_text());old=json.loads((root.parent/'mei_baseline_auth_20261019/summary.json').read_text());audit=json.loads((root/'source_audit.json').read_text())
routes=['fusion','raw_ACC','cycle_ACC'];users=sorted(s['source_counts']);agg={}
for route in routes:
    rr=[r for r in s['results'] if r['route']==route]
    agg[route]=dict(TAR=float(np.mean([1-r['genuine']['FRR'] for r in rr])),new_sources={u:dict(n=s['source_counts'][u],mean_FAR=float(np.mean([r['attack_sources'][u]['FAR'] for r in rr])),accepted_by_seed={str(r['seed']):r['attack_sources'][u]['accepted'] for r in rr}) for u in users},new_micro_FAR=float(np.mean([r['new_attack']['micro_FAR'] for r in rr])),new_macro_FAR=float(np.mean([r['new_attack']['macro_FAR'] for r in rr])),combined_micro_FAR=float(np.mean([r['combined_with_ke']['micro_FAR'] for r in rr])),combined_macro_FAR=float(np.mean([r['combined_with_ke']['macro_FAR'] for r in rr])),combined_AUC=float(np.mean([r['combined_with_ke']['AUC'] for r in rr])))
s['aggregate']=agg;s['source_audit_aggregate']={k:audit[k] for k in ['labelled_files','unique_user_keys','unique_content','counts','unlabelled_files','unlabelled_matched','content_duplicates_removed','selection']};s['source_audit_aggregate']['empty_files']=len(audit['invalid_files']);s['source_audit_aggregate']['empty_unique_keys']=len({(r['user'],r['key']) for r in audit['invalid_files']})
(root/'summary.json').write_text(json.dumps(s,indent=2))
lines=['# Data0720：冻结 mei 基线后的新增攻击验证','', '本轮仅增加攻击验证，不重新训练 CycleGAN、认证头、注册模板或阈值。原模型训练只见过 mei 和 ke；Data0720 的三位身份未参与训练与校准。本人仍为原先30条 mei，原有 ke 攻击118条，新增899条攻击，合计1017条攻击。两次 head seed 复用同一批数据，样本数不翻倍。','', '## 结果','', '| 特征路线 | 本人接受率 | fanweibo FAR | fanweiqi FAR | fanweiqisis FAR | 新增899条 FAR | 含ke的1017条 FAR | 四来源宏平均 FAR |','|---|---:|---:|---:|---:|---:|---:|---:|']
names=dict(fusion='CycleGAN 多源融合',raw_ACC='输入ACC（未循环）',cycle_ACC='循环重建ACC')
for route in routes:
    a=agg[route];vals=[a['TAR']]+[a['new_sources'][u]['mean_FAR'] for u in users]+[a['new_micro_FAR'],a['combined_micro_FAR'],a['combined_macro_FAR']];lines.append('|'+names[route]+'|'+'|'.join(f'{v:.2%}' for v in vals)+'|')
lines+=['','百分比为认证头 seed42、43 的均值。微平均按样本加权；宏平均先计算每位攻击者的FAR，再等权平均，避免新增用户样本更多掩盖 ke 的结果。FAR为攻击被接受比例，越低越好。','', '多源融合头对新增攻击的FAR为33.98%，此前对训练已见身份ke的FAR为0.85%。原低FAR未推广到新增身份。多源融合在此次固定阈值下比输入ACC与循环ACC控制拒绝更多攻击，但仍误接受约三分之一新增攻击且本人误拒23.33%，尚不支持可靠身份认证。分数分布跨用户变化、仅ke负类训练造成的边界覆盖不足、采集批次差异都是待区分的解释，本实验不能确定单一原因；也不能从此次操作点差异直接证明CycleGAN身份信息因果增益。','', '## 每次训练的实际接受数','', '| 路线 | seed | 本人 /30 | fanweibo /300 | fanweiqi /300 | fanweiqisis /299 |','|---|---:|---:|---:|---:|---:|']
for r in s['results']:lines.append('|'+names[r['route']]+'|'+str(r['seed'])+'|'+str(r['genuine']['accepted'])+'|'+'|'.join(str(r['attack_sources'][u]['accepted']) for u in users)+'|')
lines+=['','## 数据核验与适用范围','', '- 1298个有效的带用户名ACC文件按用户和采集键去重后899条；另有同一空录制在两个任务目录中的副本，无法输入模型。汇总acc目录300条均有带用户名的原文件，不重复加入。没有根据认证分数删除攻击样本。','- 三个新增用户都通过原有同一251×201前端，使用原先冻结的归一化、频带和时间处理；未做逐用户参数调整。899条均通过原2秒长度要求。','- 检查原始ACC文件与原实验各角色无重复、实际模型输入与原训练/注册/校准/最终验证无重复、新增实际输入之间无重复；验证前后锁文件及模型SHA不变。','- 新增三位身份是头训练未见的攻击者，但这是固定mei注册者、已知采集条件下的一次扩展验证，不能当作多注册者、跨日重复采集的身份泛化证明。Data0720与Data0723的采集批次差异可能影响结果。','- 不增加本人测试样本；原30条本人属于此前已经检查过的实验划分，不是重新独立采集。新增攻击结果不能消除原有本人误拒问题。','- 输入ACC指同样预处理后的模型输入，并非未经处理的时域原始信号。两次head seed只有一个生成模型，不是两次独立GAN训练。','', '## 复现','', '先执行 pack_data0720_attacks.py 打包带用户标签ACC并审计重复，再将私有压缩包传到GPU服务器，运行 evaluate_data0720_attacks.py。它读取原 mei_baseline_auth_20261019 的锁定模型，输出每来源计数、Wilson95区间与私有逐样本分数。最后运行 report_data0720_attacks.py。代码适配当前实验目录依赖，模型/模板/数据不公开。','', '关联原基线：[mei_baseline_auth_20261019](../mei_baseline_auth_20261019/README.md)。']
(root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(4);w=.24
for j,route in enumerate(routes):
    vals=[old['aggregate'][route]['FAR']]+[agg[route]['new_sources'][u]['mean_FAR'] for u in users];b=ax.bar(x+(j-1)*w,100*np.array(vals),w,label=route)
    ax.bar_label(b,fmt='%.2f',fontsize=8,padding=3)
ax.set_xticks(x,['ke (seen identity, n=118)']+[u+' (n='+str(s['source_counts'][u])+')' for u in users]);ax.set_ylabel('Attack acceptance FAR (%)');ax.set_title('Frozen mei authentication: mean of head seeds 42/43');ax.legend();ax.margins(y=.2);fig.tight_layout();fig.savefig(root/'attack_sources.png',dpi=160);plt.close(fig)
print(json.dumps(agg))
