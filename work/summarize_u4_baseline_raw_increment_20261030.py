import json
from pathlib import Path
root=Path('outputs/mei_u4_baseline_raw_increment_20261030');r=json.loads((root/'results_public.json').read_text());assert json.loads((root/'verification_public.json').read_text())['status']=='PASS';lock=json.loads((root/'locked_before_final.json').read_text());names={'baseline772':'完整772维baseline','raw_only':'输入ACC320单独','baseline_teacher':'baseline＋身份保持ACC教师','baseline_raw_fixed1':'baseline＋ACC固定权重1','baseline_raw_optional':'baseline＋ACC可选','baseline_teacher_raw_optional':'baseline＋教师＋ACC可选'}
lines=['# U4完整baseline加输入ACC：跨词复核','','复用既有个人CycleGAN、个人教师、完整772维和已拟合二分类RBF；本轮没有重训生成器/教师/认证器。仅训练OOF选择可选原始输入ACC分数、独立校准阈值并复算最终。是一次冻结配置的历史结果复核，不是新三seed实验。','', '## 统一平衡校准后的结果','','|方法|本人接受TAR|FAR|HTER|AUC|','|---|---:|---:|---:|---:|']
for key,name in names.items():
 a=r['routes'][key];lines.append('|%s|%.2f%%|%.2f%%|%.2f%%|%.4f|'%(name,100*a['tar'],100*a['far'],100*a['hter'],a['auc']))
lines+=['','## 结论','','基于同一772维baseline加入输入ACC，HTER从23.57%降至20.74%，改善2.83个百分点；固定身份保持ACC教师后，再补充输入ACC，从17.29%降至15.36%，改善1.92个百分点。两处原始ACC可选权重都由训练OOF选为1。','', '这支持绕过生成器保留输入直接信息，而不是把输入ACC整体替换成循环表示。注意baseline+教师再加ACC时，TAR由76.60%升至82.98%，FAR也由11.17%升至13.71%；不是所有指标都改善。不能按最终测试换阈值来消除此权衡。','', '这些是历史已知九用户、U4目标跨词探索性结果，不能推及所有用户或未见攻击人，也不能单凭融合增益证明Audio提供了新增身份信息。','', '旧保存频谱已归一化；ACC320是时间/频率均值和标准差，不是真实原始时域幅度。之前预处理实验仅为CE编码器输入消融，不能与本表的772维RBF数值混比。这里所有路线统一独立校准最小HTER，旧脚本的接受率优先阈值不同，本表baseline也不可直接与旧阈值报告比较。','', '## 选择与数据','','认证拟合50本人/400负类，五个训练词的nested外5/内3折；可选权重0/.25/.5/1，保留原forward_select的4折改善和配对标准误条件。固定教师锚点来自历史训练OOF非Diffusion选择去掉raw，没有按最终重新选教师。校准20本人/160负类，最终47本人/394攻击；全部八来源保留。词序周期和缺失尾部假设仍未由逐条转写独立确认。','', '## 各来源接受数','','|来源|baseline|baseline＋ACC|baseline＋教师|baseline＋教师＋ACC|','|---|---:|---:|---:|---:|']
for u in [1,2,3,5,6,7,8,9]:
 cells=[]
 for key in ['baseline772','baseline_raw_optional','baseline_teacher','baseline_teacher_raw_optional']:
  c=r['routes'][key]['attack_by_user'][str(u)];cells.append('%d/%d'%(c['accepted'],c['n']))
 lines.append('|U%d|%s|'%(u,'|'.join(cells)))
lines+=['','六路线已独立复算：完整772维与ACC320缓存逐行一致、450条训练Scaler、训练OOF权重、平衡校准阈值、最终分数/AUC及八来源计数全部PASS。']
Path('docs/experiments/2026-10-30-u4-baseline-raw-increment-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines[:17]))
