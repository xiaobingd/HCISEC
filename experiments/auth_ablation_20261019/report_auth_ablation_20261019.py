"""Local report of predeclared matched-budget ablations; no public publication."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent/'outputs/auth_ablation_20261019';s=json.loads((root/'summary.json').read_text());conditions=['old_ke_only','binary_long','binary_short','identity_short','identity_episode'];names=dict(old_ke_only='原仅ke训练',binary_long='多负类二值标签：6000更新',binary_short='多负类二值标签：1500更新',identity_short='逐身份标签：1500更新',identity_episode='逐身份标签+模拟注册选模：1500更新');routes=['fusion','raw_ACC','cycle_ACC'];rn=dict(fusion='CycleGAN多源融合',raw_ACC='输入ACC（未循环）',cycle_ACC='循环重建ACC')
lines=['# 多负类认证头：预算、标签与注册式选模消融','', '完成12条新的优化轨迹、18个锁定认证头：三种特征路线×两个head seeds×二值/逐身份训练；逐身份轨迹按两种选模方法保存头，不重复训练。原生成模型冻结，公共预处理、全部角色划分和认证512维接口不变。', '', '## 固定方案','', '- binary_short：原二值SupCon与原二值intra-inter选模，最大25epochs×60updates=1500。与上一轮binary_long相比，最大更新6000→1500，余弦学习率日程同时压缩到25epochs，因此是训练预算方案对比，不能归为步数单因素。','- identity_short：与binary_short同初始化、同批次构造、同优化器和学习率日程，只把SupCon标签改为mei及四攻击者的5个身份。本人占约一半训练项，其他四位合占一半；未更改采样来混入额外因素。选模仍使用原二值指标。','- identity_episode：完全复用identity_short训练轨迹，只改checkpoint选择。验证集10本人固定拆成5模拟注册、5查询，10组互补拆分共20次episode；以查询对120攻击的平均AUC选模，平局时先低方差再高本人类内相似度。真正注册10条、校准130条均未参与选模。','- 全部18个checkpoint选择锁定后才读取实际注册和校准特征；沿用原阈值规则。所有模型、模板、阈值锁定后才读最终集。没有根据最终结果重训、改阈值或删样本。', '', '## 最终集上的结果','', '所有条件使用相同30条本人、477条攻击（ke118、fanweibo120、fanweiqi120、fanweiqisis119）。两次head seed结果取均值，样本数不翻倍。', '', '| 特征 | 条件 | 本人接受率↑ | FAR↓ | HTER↓ | AUC↑ |','|---|---|---:|---:|---:|---:|']
for route in routes:
    for c in conditions:
        a=s['aggregate'][c][route];lines.append('|'+rn[route]+'|'+names[c]+'|'+ '|'.join(f'{a[k]:.2%}' for k in ['TAR','FAR','HTER'])+f"|{a['AUC']:.4f}|")
lines+=['','## 多源融合：逐来源攻击接受率','', '| 条件 | ke | fanweibo | fanweiqi | fanweiqisis |','|---|---:|---:|---:|---:|']
users=['chengkeyi','fanweibo','fanweiqi','fanweiqisis']
for c in conditions:
    a=s['aggregate'][c]['fusion'];lines.append('|'+names[c]+'|'+'|'.join(f"{a['sources'][u]['FAR']:.2%}" for u in users)+'|')
lines+=['','## 新多源头：每个随机种子的计数与模板稳定性','', '| 条件 | seed | 选中更新数 | 本人接受/30 | 攻击接受/477 | FAR | AUC | 注册模板与本人训练中心cos |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in s['results']:
    if r['condition'] not in conditions[2:] or r['route']!='fusion':continue
    fp=sum(v['accepted'] for v in r['attack_sources'].values());lines.append('|'+names[r['condition']]+'|'+str(r['seed'])+'|'+str(r['best_epoch']*60)+'|'+str(r['genuine']['accepted'])+'|'+str(fp)+f"|{r['FAR']:.2%}|{r['AUC']:.4f}|{r['enrollment_to_training_center']:.4f}|")
lines+=['','## 结果解释与限制','']
for a,b,label in [('binary_long','binary_short','缩短训练预算'),('binary_short','identity_short','修改训练身份标签'),('identity_short','identity_episode','改为模拟注册选模')]:
    aa=s['aggregate'][a]['fusion'];bb=s['aggregate'][b]['fusion'];lines.append(f"- {label}：多源本人接受率{aa['TAR']:.2%}→{bb['TAR']:.2%}，FAR{aa['FAR']:.2%}→{bb['FAR']:.2%}，HTER{aa['HTER']:.2%}→{bb['HTER']:.2%}，AUC{aa['AUC']:.4f}→{bb['AUC']:.4f}。")
lines+=['','不能只看某个FAR降低就判定认证修复，还要同时检查本人误拒、AUC及两个种子差异。所有攻击者身份均已见于训练；最终样本此前也反复用于诊断，属于探索性录制留出，不是新采集盲测、未见身份开集或跨日验证。两个头初始化不等于多个独立生成模型。','', '此次模拟注册仅改选模，未给训练增加新正类、正类损失或增强。不能把结果解释为已经测试了所有身份保留训练方法。逐身份训练与原二值选模的语义存在不一致，因此identity_short用于过渡消融，不应默认是推荐最终方法。','', '脚本：train_auth_ablation_20261019.py、evaluate_auth_ablation_20261019.py、report_auth_ablation_20261019.py。完整模型、锁文件与协议保存服务器 /root/imu_cyclegan_trial_20261016/auth_ablation_20261019，另备份 /autodl-fs/data/auth_ablation_20261019。详细结果仅本地/私有保存，未尝试公开此前审批拒绝范围内的内容。']
(root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(4);labels=['Binary 6000','Binary 1500','Identity 1500','Identity + episodes'];cc=conditions[1:]
for ax,metric,title in [(axs[0],'FAR','Attack acceptance FAR (%)'),(axs[1],'TAR','Genuine acceptance TAR (%)')]:
    for j,route in enumerate(routes):
        vals=[100*s['aggregate'][c][route][metric] for c in cc];bars=ax.bar(x+(j-1)*.25,vals,.25,label=route);ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
    ax.set_xticks(x,labels,rotation=15);ax.set_title(title);ax.set_ylim(0,115);ax.legend(fontsize=8)
fig.suptitle('Same 30 genuine /477 attacks; mean of seeds42/43');fig.tight_layout();fig.savefig(root/'ablation_comparison.png',dpi=160);plt.close(fig)
print(json.dumps({c:s['aggregate'][c]['fusion'] for c in conditions},indent=2))
