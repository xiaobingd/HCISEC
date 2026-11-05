import json
from pathlib import Path
root=Path('outputs/mei_u4_sameword_raw_increment_20261030');r=json.loads((root/'results_public.json').read_text(encoding='utf-8-sig'));v=json.loads((root/'verification_public.json').read_text(encoding='utf-8-sig'));assert v['status']=='PASS'
names={'baseline772':'完整772维baseline','raw_only':'ACC统计单独','baseline_raw_oof':'baseline＋ACC（训练折选权重）','baseline_teacher':'baseline＋ACC教师','baseline_teacher_raw_oof':'baseline＋ACC教师＋ACC（训练折选权重）','baseline_raw_fixed1':'baseline＋ACC固定权重1','baseline_teacher_raw_fixed1':'baseline＋教师＋ACC固定权重1'}
lines=['# U4同内容认证：输入ACC补充复核','','同内容协议：九身份使用相同15词；每人认证训练75条（5词）、校准30条（2词）、测试30条（2词），其余八身份均作攻击者，每用户240条测试负样本。复用原772维完整CycleGAN循环特征及冻结ACC身份教师，没有重训生成器或分支RBF。','', 'ACC支路是输入频谱的320维统计：80频点沿时间的均值/标准差与80时间点沿频率的均值/标准差。绕过生成器，和完整772维分数相加。不是未经归一化时域信号或物理能量。','', '分支分数依照既有同内容实验，以其训练Scaler与训练分数均值/标准差标准化。新增路线各自在同一校准集按HTER=(FAR+FRR)/2选阈值；ACC附加权重只在训练OOF五折中从0/.25/.5/1选择。最终集仅评估。']
lines+=['','## U4最终测试','', '|方法|ACC权重|本人接受TAR|攻击FAR|HTER|AUC|','|---|---:|---:|---:|---:|---:|']
for key in names:
 if key not in r['routes']:continue
 a=r['routes'][key];weight=a['weights'].get('raw',0);lines.append('|%s|%.2f|%.2f%%|%.2f%%|%.2f%%|%.4f|'%(names[key],weight,100*a['tar'],100*a['far'],100*a['hter'],a['auc']))
lines+=['','## 结果解读','','完整baseline加ACC后HTER由19.58%到19.38%，仅改善0.21个百分点；本人接受率从80.00%降至76.67%，FAR从19.17%降至15.42%。因此综合错误略好，但本人拒绝增多。训练折对“baseline＋ACC”选择权重1。','', '加入教师后再加ACC，HTER由16.67%升至21.25%，本人接受率由86.67%降到70.00%。训练折仍选了ACC权重1，说明此处OOF选择未能泛化至最终内容组。完整teacher＋ACC路线与既有平衡融合结果逐指标精确复现。不能宣称ACC补充稳定提升；需更多未观察内容/录制验证。','', 'ACC统计独立支路单独效果弱：HTER32.08%，本人接受率46.67%。其意义来自与baseline结合的互补性，而非独立认证能力。','', '全部八个攻击来源保留，每来源30条；没有根据测试结果删除攻击者或更换阈值。已重放原baseline及历史所选组合以校验数据和分数。此为历史最终数据上的探索性复核，不能当成新盲测。词组语义与尾部缺失沿用既有映射假设。','', '核验状态：'+json.dumps(v,ensure_ascii=False)]
Path('docs/experiments/2026-10-30-u4-sameword-raw-increment-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[:24]))
