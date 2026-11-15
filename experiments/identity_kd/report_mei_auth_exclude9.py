import json
from pathlib import Path
p=Path('outputs/mei8_closed_auth_20261115');r=json.loads((p/'results_public.json').read_text());rows=json.loads((p/'per_user_public.json').read_text());lines=['# Mei限定八人：认证阶段移除U9','','按用户选项2：U9从认证头训练、校准与最终测试全部移除，不评估U9本人。只保留U1–U8，每人七个攻击来源。共享ACC/Audio教师、CycleGAN和特征标准化仍复用九人训练模型，因此不是上游训练排除U9，也不是未见用户实验。','','RBF C10/gamma=scale/balanced在96条注册数据上重新拟合；每人12正类、84负类。CalA/CalB各32条，CalA设阈值、CalB重新选择19个补充配置；最终Cal64设阈值。同词每人16本人/112攻击，跨词24本人/168攻击。查询未参与选择。','','校准选择：`'+json.dumps(r['selected'])+'`。','','|方法|条件|校准规则|TAR%|FAR%|HTER%|AUC|','|---|---|---|---:|---:|---:|---:|']
for v in r['summary']:lines.append(f'|{v["method"]}|{v["condition"]}|{v["point"]}|'+ '|'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter'])+f'|{v["auc"]:.4f}|')
lines+=['','## 全部用户','','|用户|方法|条件|校准规则|TAR%|FAR%|HTER%|AUC|','|---|---|---|---|---:|---:|---:|---:|']
for v in rows:lines.append(f'|{v["user"]}|{v["method"]}|{v["condition"]}|{v["point"]}|'+ '|'.join(f'{100*v[k]:.2f}' for k in ['tar','far','hter'])+f'|{v["auc"]:.4f}|')
lines+=['','移除U9是观察既往结果后进行的限定用户群分析，不能把整体改善归因于算法提升；目标和攻击人群同时改变。词语标签仍为采集顺序推断，未逐条转写核验，单种子、同设备同采集条件。verification_public为认证角色索引/数量检查；未重新运行上游模型重提取，复用已通过的九人CPU模型审核。Cal-FAR5不保证查询FAR≤5%。']
(p/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(r['selected'])
