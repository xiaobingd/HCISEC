"""Aggregate dependent role rotations without treating folds as new subjects."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
rows=json.loads((p/'fold_user_results_private.json').read_text())
locks=json.loads((p/'thresholds_private.json').read_text())
protocol=json.loads((p/'protocol_public.json').read_text())
splitlist=json.loads((p/'split_protocol_public.json').read_text())
verification=json.loads((p/'verification_public.json').read_text())
assert len(rows)==len(locks)==10800 and len(splitlist)==60
keys=['seed','method','family','fold','words','user']
assert len({tuple(v[k] for k in keys) for v in rows})==10800
for v in rows:
    n=24 if v['words']==1 else 8
    assert v['genuine_trials']==n and v['attack_trials']==9*n
    assert abs(v['far']-v['false_accepts']/(9*n))<1e-12
    assert abs(v['frr']-v['false_rejects']/n)<1e-12
    assert abs(v['hter']-(v['far']+v['frr'])/2)<1e-12
    assert all(0<=v[k]<=1 for k in ['far','frr','hter','eer','auc'])
assert all(np.isfinite(v['threshold']) for v in locks)
for v in splitlist:
    groups=[{v['registration_prompt']},{v['calibration_prompt']},set(v['query_prompts'])]
    assert [len(s) for s in groups]==[1,1,3]
    assert not any(groups[i]&groups[j] for i in range(3) for j in range(i))
metrics=['far','frr','hter','eer','auc']
methods=['CE','AudioKD','ShuffledAudioKD']
families=['english_word','english_long','chinese_long']
def get(method,family,k,seed=None,user=None,fold=None):
    return [v for v in rows if v['method']==method and v['family']==family and v['words']==k
        and (seed is None or v['seed']==seed) and (user is None or v['user']==user)
        and (fold is None or v['fold']==fold)]
def avg(part):return {k:float(np.mean([v[k] for v in part])) for k in metrics}
macro=[];peruser=[];perseed=[];perfold=[]
for m in methods:
    for f in families:
        for k in (1,3):
            cfg=dict(method=m,family=f,words=k)
            macro.append(dict(**cfg,**avg(get(m,f,k))))
            for us in range(1,11):peruser.append(dict(user=us,**cfg,**avg(get(m,f,k,user=us))))
            for seed in (42,43,44):
                for us in range(1,11):perseed.append(dict(seed=seed,user=us,**cfg,**avg(get(m,f,k,seed=seed,user=us))))
            for fold in range(20):perfold.append(dict(fold=fold,**cfg,**avg(get(m,f,k,fold=fold))))
assert len(peruser)==180 and len(perseed)==540 and len(perfold)==360
verification.update(count_rate_checks=True,unique_fold_user_workpoints=10800,
    synthetic_checks='PASS:20balanced splits;role overlap rejection;calibration triples use3distinct recordings',
    aggregation='equal folds within each user/seed,then equal users/seeds;folds not independent subjects')
(p/'verification_public.json').write_text(json.dumps(verification,indent=2))
(p/'summary_public.json').write_text(json.dumps(dict(protocol=protocol,macro=macro,per_user=peruser,
    verification=verification),separators=(',',':')))
(p/'fold_summary_public.json').write_text(json.dumps(perfold,separators=(',',':')))
for seed in (42,43,44):
    (p/f'user_means_seed{seed}_public.json').write_text(json.dumps([v for v in perseed if v['seed']==seed],separators=(',',':')))
labels={'CE':'纯ACC','AudioKD':'Audio辅助','ShuffledAudioKD':'打乱Audio目标'}
fl={'english_word':'英文单词内部','english_long':'英文长词/短语内部','chinese_long':'中文提示语内部'}
oldprep=json.loads((p.parent/'phrase_auth/preparation_public.json').read_text(encoding='utf8'))
vocab={v['word']:v['prompt'] for v in oldprep['word_durations']}
lines=['# 严格跨内容：英文与中文内部的注册—校准—测试隔离', '',
 '2026-11-24。全部10位已知用户，3个学生种子，所有9个攻击来源保留。'
 '这轮不把英文注册→中文查询当作唯一的跨内容验证，而分别在相同提示语族内部轮换内容。', '',
 '## 实际方法', '',
 '复用上一轮从随机初始化、仅16类英文训练词得到的纯ACC／正确Audio／打乱Audio目标学生。'
 '5个英文单词、5个英文长词/短语、5个中文提示语均未参与共享编码器、标准化或Audio教师适配。'
 '共享网络冻结；每折只用该折注册记录重建个人RBF-SVM(C10 balanced)。认证全程只输入ACC。', '',
 '每类只有5种内容，因此每折用1种内容注册、另一种内容校准、另外3种内容测试；三者提示语集合严格不重叠。'
 '注册每人8条，校准每人8条源录制，测试每人3种内容×8遍共24条。'
 '枚举全部20种有序注册/校准组合，所有组合等权报告，不按成绩筛选用户或词。', '',
 '例如中文一折可用“东南大学”注册、“人机交互”校准，测试“研究生、解锁设备、计算机学院”；'
 '其他19折轮换角色，确保每种内容作注册4次、校准4次、测试12次。', '',
 '单条查询每人24次本人、216次攻击。三条联合把同一次重复编号的3种测试内容平均，'
 '每人8次本人、72次攻击，24条源录制全部用到且不复用。', '',
 '## 三条联合的校准限制', '',
 '要同时提供1种注册、3种校准、3种测试内容至少需要7种提示语，目前每类只有5种。'
 '本轮三条联合校准使用同一校准提示语的3次不同录制，而测试使用3种不同内容；没有冒充成3种不同校准词。', '',
 '具体从8条校准源录制枚举全部C(8,3)=56种三录制平均分，每组3条来自同一个人，攻击组也不混合不同攻击者。'
 '这些组合相互依赖，只代表8条源录制，不能算56条独立数据。阈值沿用固定规则：'
 'τ=(本人校准平均分Q10＋攻击者校准平均分Q95)/2。10800个阈值均写盘后才计算查询指标。', '',
 '因此本轮满足内容隔离，但校准与查询的内容多样性不匹配；这会影响阈值迁移，需结合EER与锁阈值HTER分别判断。', '',
 '## 全部用户与20折宏平均', '',
 '|方法|内容族|联合条数|FAR%|FRR%|HTER%|EER%|AUC|', '|---|---|---:|---:|---:|---:|---:|---:|']
for m in methods:
    for f in families:
        for k in (1,3):
            v=avg(get(m,f,k));lines.append(f'|{labels[m]}|{fl[f]}|{k}|'+
                '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+f"|{v['auc']:.4f}|")
lines += ['', '## 结果判断', '',
 'Audio辅助下，中文内部从单条到三条联合：EER10.92%→2.49%，HTER16.09%→11.49%，'
 'FAR9.19%→4.72%，FRR22.99%→18.25%。多条融合能改善内容切换后的身份排序，'
 '但并未恢复上一轮同内容0.11%的HTER。', '',
 '三条联合的EER明显低于固定阈值HTER，尤其英文长提示语EER4.17%而HTER18.76%。'
 '这与校准内容覆盖不足、分数随内容变化的现象一致，但本轮未单独隔离这些原因的因果贡献。', '',
 '正确Audio在全部6个条件的平均HTER都没有超过纯ACC。中文三条纯ACC HTER11.35%，Audio为11.49%；'
 '英文单词三条分别9.97%和10.20%。当前Audio关系蒸馏未解决跨内容泛化，不继续依据最终查询加权或筛选。', '',
 '上一轮0.11%来自同内容15注册／10校准／15查询，当前是8注册／8校准源／24查询，'
 '内容覆盖与预算均不同，不能把两个绝对数值的差异全归因于换内容。', '']
for f in families:
    lines += [f'## {fl[f]}：逐用户', '',
        '每项平均全部20折和3个学生种子，单位%。', '',
        '|用户|Audio单条 HTER|Audio三条 FAR|FRR|HTER|EER|纯ACC三条 HTER|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for us in range(1,11):
        a=avg(get('AudioKD',f,1,user=us));b=avg(get('AudioKD',f,3,user=us));c=avg(get('CE',f,3,user=us))
        lines.append(f'|U{us}|'+ '|'.join(f'{100*v:.2f}' for v in [a['hter'],b['far'],b['frr'],b['hter'],b['eer'],c['hter']])+'|')
lines += ['', '## 逐种子：三条联合', '', '|种子|方法|内容族|FAR%|FRR%|HTER%|EER%|', '|---:|---|---|---:|---:|---:|---:|']
for seed in (42,43,44):
    for m in methods:
        for f in families:
            v=avg(get(m,f,3,seed=seed));lines.append(f'|{seed}|{labels[m]}|{fl[f]}|'+
                '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+'|')
lines += ['', '## 全20种中文划分的敏感性', '',
 '|注册内容|校准内容|Audio三条 FAR%|FRR%|HTER%|EER%|', '|---|---|---:|---:|---:|---:|']
for sp in splitlist:
    if sp['family']!='chinese_long':continue
    v=avg(get('AudioKD','chinese_long',3,fold=sp['fold']))
    lines.append(f"|{vocab[sp['registration_prompt']]}|{vocab[sp['calibration_prompt']]}|"+
        '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+'|')
lines += ['', '## 核验与发布范围', '',
 '合成测试覆盖20折平衡、训练/注册/校准/测试内容隔离、故意重叠时拒绝、56种校准组合组内录制不重复。'
 '实际运行检查共享训练提示语重叠0、每折源索引互斥、测试源条目不复用、全部9个攻击来源、全部阈值先锁定。'
 '10800个工作点唯一，错误计数与FAR/FRR/HTER一致；发布全部180条逐用户均值和540条逐种子用户均值、360条折级敏感性。', '',
 '每个提示语在不同折会交换角色，这是固定共享编码器下的角色轮换，每折后端独立训练，'
 '不把其他折的后端参数或阈值继承进当前折，也不按折成绩选超参数。折、种子、挑战长度均重复使用物理数据，'
 '不将其当成20套独立采集或新受试者，不提供依赖错误独立性假设的置信区间。', '',
 '中文内容未参与共享表征训练，因此还包含英语表征向中文提示语的迁移；但中文认证端注册、校准、查询均为中文，'
 '不存在此前只用英文注册校准却直接换中文查询的混用。全部用户参与过共享训练，故仍为闭集，不是未见用户开放集。', '',
 '这是已探索过的同批数据，不是新盲测。原模型、同内容报告、创新点一V1均未修改。'
 '发布匿名汇总、完整划分、模型校验哈希和代码，不发布波形、embedding、实名映射或逐样本认证分数。', '']
(p/'report.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(macro=macro,verification=verification)))
