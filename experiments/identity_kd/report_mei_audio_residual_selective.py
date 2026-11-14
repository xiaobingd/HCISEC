import json
from pathlib import Path
import numpy as np
p=Path('outputs/mei_audio_residual_selective_20261114')
r=json.loads((p/'results_public.json').read_text());rows=json.loads((p/'per_user_public.json').read_text());sel=json.loads((p/'selection_public.json').read_text())
lines=['# Mei9：原始ACC保留＋选择性Audio关系补充','','本轮复用归档冻结Audio教师输出，未重训ECAPA或CycleGAN。教师已用训练/注册Audio微调；仅使用144条合法训练/注册目标，新补充分支仅输入ACC。所有用户已知，八个负类来源全部保留。词标签按采集顺序推断，非逐条转写核验。','','原始分支为720维L2统计，补充分支738→128→64，400步，CE＋跨词同人拉近＋同词异人推远。均匀/选择性关系监督权重0.1；选择性权重由训练Audio跨词一致性、Audio与原始ACC的异人混淆差异构成。它是简化余弦关系监督，非完整RKD或已验证的可靠性估计。三个种子42/43/44相同初始化及预算。','','注册每人12条；CalA/B各4条，在0/.25/.5/1中选择全局权重（允许0）；最终CalA+B设每人阈值。同词16条、跨词24条查询；仅Cal选择，不根据查询改选。','','|方法|条件|TAR%|FAR%|HTER%|AUC|','|---|---|---:|---:|---:|---:|']
for v in r['summary']:lines.append('|'+v['method']+'|'+v['condition']+'|'+ '|'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter'])+f'|{v["auc"]:.4f}|')
lines+=['','## 校准选择','','|种子|方法|补充权重|','|---|---|---:|']
for v in sel:lines.append(f'|{v["seed"]}|{v["method"]}|{v["alpha"]}|')
lines+=['','## 逐用户（3种子均值）','','每格为TAR/FAR/HTER百分比；每个用户保留全部八个攻击来源。','','|用户|条件|原始ACC|纯身份补充|均匀Audio|选择性Audio|同人打乱|','|---|---|---|---|---|---|---|']
for u in [f'U{i}' for i in range(1,10)]:
 for c in ['sameword','crossword']:
  vals=[]
  for m in ['raw','identity','uniform','selective','shuffle_same_user']:
   rr=[v for v in rows if v['user']==u and v['condition']==c and v['method']==m];assert len(rr)==3
   vals.append('/'.join(f'{100*np.mean([v[k] for v in rr]):.2f}' for k in ['tar','far','hter']))
  lines.append('|'+u+'|'+c+'|'+'|'.join(vals)+'|')
lines+=['','本轮为已探索小数据闭集实验。若选择权重为0，只能得出校准未支持加入该分支，不能据此证明任何Audio信息无用；同人打乱对照在权重为0时也不能用于判断配对价值。没有未见用户、重新佩戴或跨天验证。完整逐用户各种子结果见per_user_public.json。']
(p/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(r,indent=2))
