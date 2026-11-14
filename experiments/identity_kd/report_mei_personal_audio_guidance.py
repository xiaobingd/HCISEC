import json
from pathlib import Path
p=Path('outputs/mei_personal_audio_guidance_20261114');r=json.loads((p/'results_public.json').read_text());rows=json.loads((p/'per_user_public.json').read_text());sel=json.loads((p/'selection_public.json').read_text())
lines=['# Mei9：用户特异Audio关系指导ACC','','本轮是第一阶段个人化关系监督，不包含F0/包络/共享频谱等教师目标类型选择。复用已冻结归档ECAPA和前一轮纯身份分支seed42初始化；不重训CycleGAN。每人独立训练200步、相同MLP和身份损失；纯身份、均匀Audio和个人Audio三个分支采用相同个人随机起点和训练预算。','','个人权重仅从144条合法训练/注册样本计算：Audio同人跨词一致性、原始ACC相似而Audio不相似的训练关系，以及当前本人参与的关系。所有九人的身份监督仍保留；八个攻击来源均参与RBF训练及测试。个人化权重为固定启发式，不是已经证明最优的监督选择。','','原始720维统计保留；每个用户独立用CalA设阈值、CalB选择0/.25/.5/1补充权重，再用CalA+B设最终阈值。注册12/人、校准8/人、同词16/人、跨词24/人。阈值与旧基线一致，ROC不丢中间点，并列优先较低FAR。','','|方法|条件|TAR%|FAR%|HTER%|AUC|','|---|---|---:|---:|---:|---:|']
for v in r['summary']:lines.append(f'|{v["method"]}|{v["condition"]}|'+ '|'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter'])+f'|{v["auc"]:.4f}|')
lines+=['','## 全部用户','','每格TAR/FAR/HTER%，不根据测试删用户。','','|用户|条件|原始ACC|纯身份补充|均匀Audio|个人Audio|','|---|---|---|---|---|---|']
for u in range(1,10):
 for c in ['sameword','crossword']:
  values=[]
  for m in ['raw','identity','uniform','personal']:
   v=next(v for v in rows if v['user']==f'U{u}' and v['condition']==c and v['method']==m);values.append('/'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter']))
  lines.append(f'|U{u}|{c}|'+ '|'.join(values)+'|')
lines+=['','## 校准选择','','|用户|纯身份权重|均匀Audio权重|个人Audio权重|','|---|---:|---:|---:|']
for u in range(1,10):lines.append(f'|U{u}|'+ '|'.join(str(next(v['alpha'] for v in sel if v['user']==f'U{u}' and v['method']==m)) for m in ['identity','uniform','personal'])+'|')
lines+=['','## 边界','','单次个人微调种子，已探索的九人闭集；初始分支来自归档seed42。词语标签按采集顺序推断、未逐条核验。没有最终查询数据参与权重筛选，但校准每人只有8条，个人权重选择可能不稳定。原始信号为OpenVR tracking速度，不是原始硬件加速度。不能从本轮推出未见用户或跨佩戴泛化。完整逐用户结果和选择记录同目录JSON。']
(p/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(r,indent=2))
