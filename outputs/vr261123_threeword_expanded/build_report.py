"""Validate source-count expansion and publish aggregate factorial results."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
rows=json.loads((p/'results_public.json').read_text())
locks=json.loads((p/'thresholds_public.json').read_text())
protocol=json.loads((p/'protocol_public.json').read_text())
verify=json.loads((p/'verification_public.json').read_text())
assert len(rows)==780 and len(locks)==300
keys=['seed','user','registration','calibration','grouping','rule','query']
assert len({tuple(r[k] for k in keys) for r in rows})==780
for r in rows:
    n={'legacy_first4':7,'additional_last4':6,'expanded_all8':13}[r['query']]
    assert r['genuine_trials']==n and r['attack_trials']==9*n
    assert abs(r['far']-r['false_accepts']/(9*n))<1e-12
    assert abs(r['frr']-r['false_rejects']/n)<1e-12
    assert abs(r['hter']-(r['far']+r['frr'])/2)<1e-12
for v in locks:
    assert np.isfinite(v['threshold'])
    if v['rule']=='strict5':assert v['development']['far']<=.05+1e-12
metrics=['far','frr','hter','eer','auc']
configs=list(dict.fromkeys(tuple(r[k] for k in keys[2:]) for r in rows))
def get(reg,cal,query='expanded_all8',rule='quantile_gap',user=None,seed=None,group='nonoverlap'):
    return [v for v in rows if v['registration']==reg and v['calibration']==cal and v['query']==query
        and v['rule']==rule and v['grouping']==group and (user is None or v['user']==user)
        and (seed is None or v['seed']==seed)]
def avg(part):return {k:float(np.mean([v[k] for v in part])) for k in metrics}
macro=[];peruser=[]
for reg,cal,group,rule,query in configs:
    cfg=dict(registration=reg,calibration_available=20 if group=='legacy' else cal,
             calibration_used=20 if group=='legacy' else {20:18,40:39}[cal],grouping=group,rule=rule,query=query)
    macro.append(dict(**cfg,**avg(get(reg,cal,query,rule,group=group))))
    for us in range(1,11):peruser.append(dict(user=us,**cfg,**avg(get(reg,cal,query,rule,us,group=group))))
assert len(macro)==26 and len(peruser)==260
oldpath=p.parent/'threeword_calibration/results_public.json'
old=json.loads(oldpath.read_text())
errs=[]
for r in rows:
    if r['grouping']=='legacy':
        v=next(v for v in old if v['seed']==r['seed'] and v['user']==r['user'] and v['normalization']=='raw' and v['rule']==r['rule'])
        errs += [abs(r[k]-v[k]) for k in metrics]
assert max(errs)==0
verify.update(local_count_rate_checks=True,local_old_baseline_replay_max_error=max(errs))
(p/'verification_public.json').write_text(json.dumps(verify,indent=2))
(p/'summary_public.json').write_text(json.dumps(dict(protocol=protocol,macro=macro,per_user=peruser,
    verification=verify),ensure_ascii=False,separators=(',',':')),encoding='utf-8')
for seed in (42,43,44):
    part=[v for v in rows if v['seed']==seed]
    assert len(part)==260
    (p/f'results_seed{seed}_public.json').write_text(json.dumps(part,separators=(',',':')),encoding='utf-8')
lines=['# 3词联合分位数校准：扩大注册、校准与查询录制数量', '',
 '2026-11-23。沿用同一批10位用户、40个词、每词8遍，共3200条原始硬件ACC缓存；扩大认证阶段的录制使用量，未增加身份数量、词汇覆盖、会话或编码器训练量。', '',
 '## 主要结果：固定扩大查询集的2×2对照', '',
 '所有行均使用同一组每人13次3词查询（39条不同录制），全部9位攻击者保留。规则固定为上一轮开发集选定的分位数间隙，未重新搜索参数。', '',
 '|注册录制/人|实际校准录制/人|FAR%|FRR%|HTER%|EER%|', '|---:|---:|---:|---:|---:|---:|']
for nr,nc in [(24,20),(24,40),(32,20),(32,40)]:
    v=avg(get(nr,nc));lines.append(
        f'|{nr}|{ {20:18,40:39}[nc]}|'+'|'.join(f'{100*v[k]:.2f}' for k in metrics[:4])+'|')
lines += ['', '24条注册时，实际校准录制18→39，HTER从4.22%降到3.32%，FRR从6.92%降到5.13%，FAR保持1.51%。'
 '三个种子均改善。32条注册时，扩大校准也将HTER从5.46%降到4.46%。'
 '但在相同校准量下，32条注册没有超过24条注册；因此这轮支持扩大校准，而不支持同词注册重复次数越多越好。', '',
 '这四组完整呈现，不依据查询结果为用户挑选注册数量或阈值。24＋39是本轮观察较好的配置，尚未通过新开发选择与新数据确认定版。', '',
 '## 实验方法与数据处理', '',
 '冻结CE种子42/43/44的762→256→192 ACC编码器、原训练标准化和RBF-SVM C10 balanced。'
 '统计输入含频谱均值、标准差、四段均值、幅度与原始统计。无Audio、重训、质量门控或重录策略。', '',
 '注册词缓存ID20–23：每词6遍或8遍，对应每人24或32条，同时重新拟合合法注册RBF正负类。'
 '校准词8、9、18、19、24，使用每词前4遍或全部8遍，分别有20或40条可用录制。'
 '查询词25–29，和训练、注册、校准词集合不重叠。所有阈值写盘后才计算查询指标。', '',
 '每阶段按上一轮固定seed1123词序排列，再以连续3条组成不同词挑战。主分组不循环补齐：'
 '20条可用录制实际使用18条组成6组；40条实际使用39条组成13组。剩下1/2条由固定索引决定，'
 '不是按质量或表现剔除；组间没有重复source_id。这里比较的是不同缓存录制条目的数量，不能把重组挑战算成新增原始数据。', '',
 '分位数阈值固定为τ=(Q10(本人校准3词平均分)+Q95(攻击者校准3词平均分))/2。'
 '保留旧strict5规则对照（校准FAR≤5%下最小FRR，再最小FAR、最高阈值）。'
 '分位数规则没有逐用户FAR≤5%的数学保证；所有查询FAR/FRR均保留实测结果。', '',
 '## 三种查询评估', '',
 '|查询|源录制可用/实际使用，每人|3词挑战/人|攻击挑战/目标|是否源录制复用|',
 '|---|---|---:|---:|---|',
 '|旧前4遍原分组|20 / 20|7|63|首条一次复用，兼容复算|',
 '|新增后4遍不复用分组|20 / 18|6|54|否|',
 '|全部8遍不复用分组，主评估|40 / 39|13|117|否|', '',
 '新增后4遍是新增纳入主评估的录制，其中部分可能曾作历史质量门控的备用录制，不能称为全新盲测。'
 '全部8遍的分组与旧7组不同，EER变化不能直接归因于校准。数据集同一会话，不能验证摘戴、跨天或换设备泛化。', '',
 '## 固定分位数规则的全部查询条件', '',
 '|注册|校准实际使用|查询|FAR%|FRR%|HTER%|EER%|', '|---:|---:|---|---:|---:|---:|---:|']
ql={'legacy_first4':'旧前4遍','additional_last4':'新增后4遍','expanded_all8':'全部8遍'}
for nr,nc in [(24,20),(24,40),(32,20),(32,40)]:
    for q in ql:
        v=avg(get(nr,nc,q));lines.append(f'|{nr}|{ {20:18,40:39}[nc]}|{ql[q]}|'+
            '|'.join(f'{100*v[k]:.2f}' for k in metrics[:4])+'|')
lines += ['', '24条注册＋39条校准在新增后4遍上的FAR0.93%、FRR5.56%、HTER3.24%。'
 '扩大后的全部查询EER为1.89%，说明原先0.24%的经验EER未在更大且不同分组的查询中原样保持，'
 '但分位数校准的低错误水平没有仅限于旧7组。', '',
 '## 扩大查询的逐用户结果', '',
 '以下为分位数规则，三个种子平均。每目标每种子13次本人及117次攻击；用户和攻击者不作筛选。', '',
 '|匿名用户|24注册18校准 HTER%|24注册39校准 FAR%|FRR%|HTER%|32注册39校准 HTER%|',
 '|---|---:|---:|---:|---:|---:|']
for us in range(1,11):
    a=avg(get(24,20,user=us));b=avg(get(24,40,user=us));c=avg(get(32,40,user=us))
    lines.append(f'|U{us}|'+ '|'.join(f'{100*v:.2f}' for v in [a['hter'],b['far'],b['frr'],b['hter'],c['hter']])+'|')
lines += ['', '24注册下扩大校准后，U2的FRR从30.77%降到20.51%；U5的FAR仍为8.26%。'
 '32注册降低了U5的FAR至5.13%，但增加了部分其他用户的错误。'
 '这说明统一增加注册重复数不能同时解决所有人的误拒和误接受，不据此对查询用户进行事后选参。', '',
 '## 主评估种子结果', '', '|种子|注册|校准实际使用|FAR%|FRR%|HTER%|', '|---:|---:|---:|---:|---:|---:|']
for seed in (42,43,44):
    for nr,nc in [(24,20),(24,40),(32,20),(32,40)]:
        v=avg(get(nr,nc,seed=seed));lines.append(f'|{seed}|{nr}|{ {20:18,40:39}[nc]}|'+
            '|'.join(f'{100*v[k]:.2f}' for k in metrics[:3])+'|')
lines += ['', '## strict5规则对照：相同扩大查询', '',
 '|注册|校准实际使用|FAR%|FRR%|HTER%|', '|---:|---:|---:|---:|---:|']
for nr,nc in [(24,20),(24,40),(32,20),(32,40)]:
    v=avg(get(nr,nc,rule='strict5'));lines.append(f'|{nr}|{ {20:18,40:39}[nc]}|'+
        '|'.join(f'{100*v[k]:.2f}' for k in metrics[:3])+'|')
lines += ['', '## 核验和结果边界', '',
 '3200条sample_id唯一，频谱＋幅度按字节SHA256检查无完全重复行；这是缓存特征核验，'
 '并非重新检查原始波形的完整去重或切片时间区间重叠。训练、注册、校准和查询按词隔离；主分组三条不同词且source_id不复用。'
 '780个查询工作点、300个锁定阈值的计数与比率通过核验。旧7组、24条注册、两种阈值规则逐用户／种子复算最大误差0。', '',
 '三种查询集合互有源记录重叠，三个种子也复用同一物理数据，不将它们相加当作独立受试者，'
 '不提供依赖独立录制假设的虚假置信区间。主2×2比较使用完全相同的39条查询源录制，'
 '因此能分别观察注册和校准样本量变化；不把与旧7组的绝对差异归因为方法本身。', '',
 '发布完整26个条件的260条匿名逐用户均值和780条种子级用户汇总，不发布原始波形、embedding、实名映射、逐样本分数。'
 '源代码依赖同目录calibrate_threeword.py的cutoff与measure函数，版本和上一轮一致。历史模型、创新点一V1和旧结果未改动。', '',
 '## 下一步判断', '',
 '优先补充校准词与独立录制，而非盲目增加同词注册次数。'
 '在开发身份任务里检验逐用户FAR约束和注册词覆盖，再用新采集或新会话确认；'
 '不依据本轮U2/U5查询结果直接调整个人阈值。', '']
(p/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(main=[v for v in macro if v['query']=='expanded_all8' and v['rule']=='quantile_gap'],
                      verification=verify)))

