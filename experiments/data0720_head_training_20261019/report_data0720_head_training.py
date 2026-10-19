"""Render the added-negative training result on matched final recordings."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent/'outputs/data0720_head_training_20261019'
s=json.loads((root/'summary.json').read_text());routes=['fusion','raw_ACC','cycle_ACC'];names=dict(fusion='CycleGAN多源融合',raw_ACC='输入ACC（未循环）',cycle_ACC='循环重建ACC');conditions=['ke_only','expanded_negatives'];cn=dict(ke_only='仅ke负类训练',expanded_negatives='加入Data0720负类训练')
lines=['# Data0720新增用户参与认证头训练','', '保持同一个mei专属CycleGAN不变，只重训原trainb核心SupCon认证头。对比三种特征路线，各跑head seeds42/43；没有重新训练生成模型。新增用户作为负类，与ke共同训练，mei为唯一正类。','', '## 数据划分','', '| 用户 | 头训练 | 头验证 | 注册 | 阈值校准 | 最终测试 |','|---|---:|---:|---:|---:|---:|']
users=['wangjingmei','chengkeyi','fanweibo','fanweiqi','fanweiqisis'];roles=['head_train','head_validation','enrollment','calibration','final_test']
for u in users:lines.append('|'+u+'|'+'|'.join(str(s['counts'][r][u]) for r in roles)+'|')
lines+=['','mei另有180条只用于生成模型训练、30条只用于生成检查，与上表的认证数据隔离。各来源去重后按固定随机种子划分，没有按分数删除样本。','', '## 在完全相同最终样本上的比较','', '最终集：30条本人、477条攻击（ke118、fanweibo120、fanweiqi120、fanweiqisis119）。下表是两个认证头种子的均值，样本量不翻倍。旧头用原先锁定分数在这批最终子集上重新统计，不能把其FAR与上一轮全部899条新攻击的统计直接当作同一分母。','', '| 特征 | 训练设置 | 本人接受率↑ | 全部攻击FAR↓ | 攻击来源宏平均FAR↓ | HTER↓ | AUC↑ |','|---|---|---:|---:|---:|---:|---:|']
for route in routes:
    for condition in conditions:
        a=s['aggregate'][condition][route];lines.append('|'+names[route]+'|'+cn[condition]+'|'+ '|'.join([f"{a[k]:.2%}" for k in ['TAR','FAR','macro_FAR','HTER']])+f"|{a['AUC']:.4f}|")
lines+=['','## 加入新负类后，各攻击来源接受率','', '| 特征 | ke /118 | fanweibo /120 | fanweiqi /120 | fanweiqisis /119 |','|---|---:|---:|---:|---:|']
for route in routes:
    a=s['aggregate']['expanded_negatives'][route];lines.append('|'+names[route]+'|'+'|'.join(f"{a['sources'][u]['FAR']:.2%}" for u in users[1:])+'|')
lines+=['','## 每个认证头的实际计数','', '| 特征 | seed | 本人接受 /30 | ke接受 /118 | fanweibo接受 /120 | fanweiqi接受 /120 | fanweiqisis接受 /119 |','|---|---:|---:|---:|---:|---:|---:|']
for r in s['results']:
    if r['condition']!='expanded_negatives':continue
    lines.append('|'+names[r['route']]+'|'+str(r['seed'])+'|'+str(r['genuine']['accepted'])+'|'+'|'.join(str(r['attack_sources'][u]['accepted']) for u in users[1:])+'|')
f=s['aggregate']['expanded_negatives']['fusion'];b=s['aggregate']['ke_only']['fusion']
fusion_runs=[r for r in s['results'] if r['condition']=='expanded_negatives' and r['route']=='fusion']
lines+=['','## 结论与边界','',f"在相同最终录制上，多源融合的FAR从{b['FAR']:.2%}变为{f['FAR']:.2%}，本人接受率从{b['TAR']:.2%}变为{f['TAR']:.2%}。应同时考察误接受与本人误拒，不能只报告攻击接受率。",'',f"多源融合AUC从{b['AUC']:.4f}降至{f['AUC']:.4f}，不只是某一个阈值下的变化；新增训练seed42/43的FAR分别为{fusion_runs[0]['FAR']:.2%}/{fusion_runs[1]['FAR']:.2%}，训练不稳定。直接增加负类不足以修复这条冻结编码器+原头训练路径。输入ACC对照误接受减少，但本人接受仍只有65%；循环ACC没有改善综合错误率。本轮不支持可靠认证，也不能证明生成编码器已保留足够可泛化身份信息。",'', '- 这是四位负类身份都参与训练的录制留出验证，不是训练未见身份的开集结果；这些用户下一次作为完全未知攻击者的效果仍需另行验证。','- 先固定划分，完成头选择、模板和校准阈值锁定，再导出最终集并评估；原有mei/ke各组数组保持逐元素一致。原始ACC与实际模型输入均检查各角色不重复，所有用户同一前端与冻结归一化。','- 全部899条Data0720来源此前已被旧头诊断打分。因此当前留出集不属于从未接触的独立确认数据；划分不依据这些分数，新头选模与阈值不使用最终分数。','- 复用原SupCon算法、温度0.07、Adam学习率0.001、100epochs、512维认证向量。负类480条、正类30条；原类别平衡规则每类重复到960个缓存项，每轮60次更新，原ke-only每轮15次。重复不增加独立录制量。本比较同时改变负类覆盖、更新次数、验证和校准数据，不能归因为某一因素。','- 输入ACC是同样预处理后的251通道输入的均值/标准差特征；循环ACC是重建ACC同类统计。多源特征使用当前ResNet层及循环残差，原trainb核心类复用，接口适配有明确记录。不同路线阈值分别来自同一校准规则，不是相同FAR操作点的因果对比。','- 单一mei注册者，两次头初始化、一个生成模型；日期/设备/任务差异仍可能影响结果，不能据此证明跨日稳定身份特征。','', '## 文件与复现','', 'prepare_data0720_head_training.py（先development后final）、run_data0720_head_training.py（执行原训练循环，仅更新协议描述）、evaluate_data0720_head_training.py（检查锁定与同源对比）、report_data0720_head_training.py（生成报告和图）。原模型依赖 experiments/mei_baseline_auth_20261019 的实现。详细分数、模板及模型包保存在私有实验目录。']
(root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(3);w=.35
for j,condition in enumerate(conditions):
    values=[s['aggregate'][condition][route]['FAR']*100 for route in routes];bars=axs[0].bar(x+(j-.5)*w,values,w,label=condition);axs[0].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
axs[0].set_xticks(x,routes);axs[0].set_ylabel('All attacks FAR (%)');axs[0].set_title('Same 477 attack recordings');axs[0].legend();axs[0].margins(y=.25)
x=np.arange(4)
for j,condition in enumerate(conditions):
    values=[s['aggregate'][condition]['fusion']['sources'][u]['FAR']*100 for u in users[1:]];bars=axs[1].bar(x+(j-.5)*w,values,w,label=condition);axs[1].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
axs[1].set_xticks(x,['ke','fanweibo','fanweiqi','fanweiqisis'],rotation=15);axs[1].set_ylabel('FAR (%)');axs[1].set_title('Fusion: per attacker');axs[1].legend();axs[1].margins(y=.25);fig.tight_layout();fig.savefig(root/'training_comparison.png',dpi=160);plt.close(fig)
print(json.dumps(s['aggregate'],indent=2))
