import json
from pathlib import Path
root=Path('outputs/mei_formal_asnorm_20261029');d=json.loads((root/'verified_summary_public.json').read_text())
assert d['status']=='PASS' and d['targets']==9
prior=json.loads(Path('outputs/mei_posthoc_subset_20261029/verified_summary_public.json').read_text())
names={'base':'完整772维','identity_acc':'身份教师ACC'}
lines=['# 正式AS-Norm1认证实验结果','',
'实现Matejka等2017公式7：注册端、测试端独立选Top-K背景身份，分别减均值除样本标准差，再取平均。不是减最大背景分数，也不是AS-Norm2或TAS-Norm。标准差ddof=1，下限1e-4。','',
'每目标认证训练本人80条，上游及输入预处理不变。注册向量、cohort中心及Scaler均只由保留的认证训练行建立；cohort排除目标。背景数量为8/7/6。','',
'**删1/2人名单由旧测试FAR挑选，属于测试知情限定用户群探索，不是独立测试性能。不删人结果也使用了已观察的历史测试。**','',
'|表示|删人数|评分|本人接受率|限定群FAR|AUC|HTER|全部攻击FAR|','|---|---:|---|---:|---:|---:|---:|---:|']
for feature,label in names.items():
 for drop in [0,1,2]:
  for mode,title in [('cosine','纯本人余弦'),('relative','旧背景相对'),('asnorm_fixed4','AS-Norm1 K=4'),('asnorm_selected','AS-Norm1训练选K'),('snorm','S-Norm全背景')]:
   key='%s_drop%d_%s'%(feature,drop,mode)
   if mode in ['cosine','relative']:
    a=prior['macro'][key]['restricted'];b=prior['macro'][key]['all_attackers_diagnostic']
   else:a=d['macro'][key]['restricted'];b=d['macro'][key]['all_attackers_diagnostic']
   lines.append('|%s|%d|%s|%.2f%%|%.2f%%|%.4f|%.2f%%|%.2f%%|'%(label,drop,title,a['tar']*100,a['far']*100,a['auc'],a['hter']*100,b['far']*100))
lines+=['','## 教师ACC的逐用户正式AS-Norm结果','', '每格为本人接受率 / 限定群FAR / AUC。固定K=4为预设对照，选K只使用训练内容折。','', '|目标|全部8人背景 K=4|删2人后 K=4|删2人后训练选K|所选K|','|---|---|---|---|---:|']
def fmt(r):return '%.2f / %.2f / %.4f'%(r['tar']*100,r['far']*100,r['auc'])
for u in map(str,range(1,10)):
 a=d['results'][u]['routes'];selected=a['identity_acc_drop2_asnorm_selected']
 lines.append('|U%s|%s|%s|%s|%d|'%(u,fmt(a['identity_acc_drop0_asnorm_fixed4']),fmt(a['identity_acc_drop2_asnorm_fixed4']),fmt(selected),selected['k']))
lines+=['','## 阶段结论','',
'正式AS-Norm1未在当前小背景集上超过旧的最大背景惩罚评分。教师ACC删两人时，训练选K的TAR74.27%、限定群FAR27.71%、AUC0.7953，旧相对评分为76.10%、25.74%、0.8084，三项均退步。固定K=4为75.21%、27.37%、0.7932，也没有整体改善。',
'','不删人时正式AS-Norm1训练选K提高本人接受率，但FAR也提高，属于权衡。不能仅根据接受率提高就宣称认证增强。S-Norm全背景没有被适应性Top-K稳定超过。背景数量不足、这些表示未必适合语音领域评分分布，均是待验证解释，尚不能认定为已证实原因。',
'','正式版本的相似度和Top-K统计使用float64。初始float32版本在小cohort标准差下未通过严格独立分数重算，因此未作为正式结果报告；升级精度后重新选择训练K、重算校准及最终分数，并通过独立核验。',
'','## 复核及解释','',
'固定K=4与训练选K均在最终评分前锁定，S-Norm使用全部cohort。没有依据最终结果在路线之间回退。独立实现通过cohort索引和样本矩重算全部分数，并核验中心身份、标准化拟合数量、校准阈值、AUC和逐来源接受数。',
'','背景仅6至8个身份，样本标准差估计不稳定；不能将大规模语音验证论文的效果直接迁移到本实验。cohort中包含已知攻击身份，不代表开放集泛化。各目标删人名单不同，限定群不是统一七人数据集。',
'','参考：[Matejka等2017](https://www.fit.vut.cz/research/group/speech/public/publi/2017/matejka_interspeech2017_IS170803.pdf)；[asv-subtools实现 cross_select=false](https://github.com/Snowdar/asv-subtools/blob/master/score/ScoreNormalization.py)。']
Path('docs/experiments/2026-10-29-formal-asnorm-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
compact={k:d[k] for k in ['status','targets','routes_verified','macro','verification']}
keep=['tar','far','auc','hter','genuine_n','impostor_n','genuine_accepted','attack_accepted','feature','removed','k','variant']
compact['results']={}
for u,a in d['results'].items():
 compact['results'][u]={'routes':{}}
 for name,r in a['routes'].items():
  row={k:r[k] for k in keep if k in r};row['all_attackers_diagnostic']={k:r['all_attackers_diagnostic'][k] for k in keep if k in r['all_attackers_diagnostic']};compact['results'][u]['routes'][name]=row
(root/'summary_public.json').write_text(json.dumps(compact,separators=(',',':')),encoding='utf-8')
print('\n'.join(lines))

