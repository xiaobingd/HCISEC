import json
from pathlib import Path
p=Path('outputs/mei_personal_physical_auxiliary_20261114');r=json.loads((p/'results_public.json').read_text());rows=json.loads((p/'per_user_public.json').read_text());selection=json.loads((p/'selection_public.json').read_text());targets=json.loads((p/'teacher_targets_public.json').read_text())
lines=['# Mei9个人物理Audio辅助监督','','每人独立微调ACC738→128→64补充分支200步，同一身份初始化（归档identity42）、相同个人随机起点、相同CE/跨词身份/锚定损失。保留原始720维统计。比较无Audio、三类物理Audio目标全部加入、按训练内部逐词留出诊断选目标。辅助头仅用本人16条合法训练/注册Audio；目标在本人训练记录上标准化，逐类MSE等权平均，辅助权重0.1。预测头训练后丢弃，认证只需ACC。','','周期性是自动自相关候选，非人工核验F0；低频谱10–300Hz为现有前端匹配候选，尚无波形相干性证明；包络是相对时长28点log能量。选择来自上一轮8词训练内部诊断，不使用最终查询。无候选的用户使用纯身份训练。','','每人注册12、CalA/B各4，同词16、跨词24查询；全部八个负类保留。每人只用CalA设阈值、CalB选0/.25/.5/1补充权重，再用CalA+B设最终阈值。阈值与旧基线相同。单种子，所有超参数固定。','','|方法|条件|TAR%|FAR%|HTER%|AUC|','|---|---|---:|---:|---:|---:|']
for v in r['summary']:lines.append(f'|{v["method"]}|{v["condition"]}|'+ '|'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter'])+f'|{v["auc"]:.4f}|')
lines+=['','## 每用户结果','','每格为TAR/FAR/HTER%，全部八个攻击来源。','','|用户|条件|原始ACC|纯身份|全部物理目标|个人选择目标|','|---|---|---|---|---|---|']
for u in range(1,10):
 for c in ['sameword','crossword']:
  cells=[]
  for m in ['raw','identity','all_physical','selected_physical']:
   v=next(v for v in rows if v['user']==f'U{u}' and v['condition']==c and v['method']==m);cells.append('/'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter']))
  lines.append(f'|U{u}|{c}|'+ '|'.join(cells)+'|')
lines+=['','## 目标与校准选择','','|用户|候选目标|个人目标分支权重|','|---|---|---:|']
for u in range(1,10):lines.append(f'|U{u}|'+(','.join(targets[str(u)]) or 'none')+'|'+str(next(v['alpha'] for v in selection if v['user']==f'U{u}' and v['method']=='selected_physical'))+'|')
lines+=['','边界：闭集已知九人、单次训练种子、归档初始化、已探索语料。逐词诊断每人只有16条/8名义词，不是大样本可靠性证据。词标签按采集顺序推断，未逐条转写；输入为OpenVR tracking速度频谱。校准每人8条，选择可能不稳定。本轮没有重训ECAPA/CycleGAN，没有证明未见用户、跨天或重新佩戴泛化。']
(p/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(r,indent=2))
