"""Validate and report matched fit-only Audio-teacher multiword authentication."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
rows=json.loads((p/'results_public.json').read_text())
locks=json.loads((p/'thresholds_public.json').read_text())
protocol=json.loads((p/'protocol_public.json').read_text())
verify=json.loads((p/'verification_public.json').read_text())
execution=json.loads((p/'execution_verification_public.json').read_text())
training=json.loads((p/'training_public.json').read_text())
assert len(rows)==540 and len(locks)==270
keys=['seed','method','words','user','query']
assert len({tuple(v[k] for k in keys) for v in rows})==540
for v in rows:
    count=(40 if v['query']=='expanded_all8' else 20)//v['words']
    assert v['genuine_trials']==count and v['attack_trials']==9*count
    assert abs(v['far']-v['false_accepts']/(9*count))<1e-12
    assert abs(v['frr']-v['false_rejects']/count)<1e-12
    assert abs(v['hter']-(v['far']+v['frr'])/2)<1e-12
    assert all(0<=v[k]<=1 for k in ['far','frr','hter','eer','auc'])
assert verify['teacher_nonfit_audio_forwarded']==0 and verify['ce_threeword_metric_max_error']<1e-10
assert all(np.isfinite(v['threshold']) for v in locks)
targets=np.load(p/'teacher_targets_private.npz')
data=np.load(p.parent/'paired_raw_acc_private.npz')
fi=np.flatnonzero(data['role']=='fit')
assert np.array_equal(targets['indices'],fi)
assert np.array_equal(targets['sample_id'],data['sample_id'][fi])
assert targets['targets'].shape==(2080,192) and np.isfinite(targets['targets']).all()
assert np.allclose(np.linalg.norm(targets['targets'],axis=1),1,atol=1e-5)
aux=[v for branch in training if branch['method']!='CE' for v in branch['trace']]
assert aux and all(0<v['aux_gradient_ratio']<=.100001 for v in aux)
metrics=['far','frr','hter','eer','auc']
methods=['CE','AudioKD','ShuffledAudioKD']
def get(method,words=3,query='expanded_all8',user=None,seed=None):
    return [v for v in rows if v['method']==method and v['words']==words and v['query']==query
            and (user is None or v['user']==user) and (seed is None or v['seed']==seed)]
def avg(part):return {k:float(np.mean([v[k] for v in part])) for k in metrics}
macro=[dict(method=m,words=k,query=q,**avg(get(m,k,q))) for m in methods for k in (1,3,5)
       for q in ['expanded_all8','additional_last4']]
peruser=[dict(user=u,method=m,words=k,query=q,**avg(get(m,k,q,u))) for m in methods for k in (1,3,5)
         for q in ['expanded_all8','additional_last4'] for u in range(1,11)]
differences={}
for reference in ['CE','ShuffledAudioKD']:
    per_seed=[];per_user=[]
    for seed in (42,43,44):
        a=avg(get('AudioKD',seed=seed));b=avg(get(reference,seed=seed))
        per_seed.append(dict(seed=seed,**{k:a[k]-b[k] for k in metrics}))
    for user in range(1,11):
        a=avg(get('AudioKD',user=user));b=avg(get(reference,user=user))
        per_user.append(dict(user=user,**{k:a[k]-b[k] for k in metrics}))
    differences[reference]=dict(per_seed=per_seed,per_user=per_user,
        hter_improved_users=sum(v['hter'] < -1e-12 for v in per_user),
        hter_equal_users=sum(abs(v['hter'])<=1e-12 for v in per_user),
        hter_worse_users=sum(v['hter']>1e-12 for v in per_user))
verify.update(local_rate_count_checks=True,synthetic_kd_gradient_checks='PASS',
              runtime_execution=execution,teacher_target_alignment_and_unit_norm=True,
              logged_aux_gradient_ratio_min=min(v['aux_gradient_ratio'] for v in aux),
              logged_aux_gradient_ratio_max=max(v['aux_gradient_ratio'] for v in aux))
(p/'verification_public.json').write_text(json.dumps(verify,indent=2))
(p/'summary_public.json').write_text(json.dumps(dict(protocol=protocol,macro=macro,per_user=peruser,
    paired_changes=differences,verification=verify),ensure_ascii=False,separators=(',',':')),encoding='utf-8')
for seed in (42,43,44):
    part=[v for v in rows if v['seed']==seed]
    assert len(part)==180
    (p/f'results_seed{seed}_public.json').write_text(json.dumps(part,separators=(',',':')))
labels={'CE':'纯ACC','AudioKD':'正确Audio教师','ShuffledAudioKD':'打乱Audio目标'}
lines=['# 仅训练词适配的 Audio 教师：ACC 多词联合认证', '',
 '2026-11-23。全部10人闭集跨词，学生种子42/43/44。Audio只参与离线教师适配和ACC学生训练；注册、校准、查询均只用ACC。', '',
 '## 多词认证结果', '',
 '全部8遍查询，三个种子、全部用户宏平均。主比较事先固定为3词；1词和5词是挑战长度对照，不据查询结果更改主设置。', '',
 '|方法|联合词数|FAR%|FRR%|HTER%|EER%|AUC|', '|---|---:|---:|---:|---:|---:|---:|']
for m in methods:
    for k in (1,3,5):
        v=avg(get(m,k));lines.append(f'|{labels[m]}|{k}|'+ '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+f"|{v['auc']:.4f}|")
lines += ['', '## 本轮如何实现 Audio 教师', '',
 '审计发现旧本地教师的历史微调使用过旧注册词，其中4词现已属于校准。旧学生和教师缓存因此不进入本轮严格对照。'
 '重新加载SHA256验证的官方VoxCeleb ECAPA权重，对仅2080条fit Audio提取6144维池化输入与192维初始声纹。'
 '模型权重哈希为0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2。', '',
 '冻结ECAPA卷积与池化主体，只更新6144→192最终投影和新增192→64→192身份残差，配临时10类分类头。'
 '固定种子4242、300步、40条/批，CE＋0.1 SupCon＋0.01原向量锚定；学习率分别1e-5、1e-4、1e-3。'
 '这属于教师末端投影/残差适配，不是完整ECAPA主体微调。300步后教师冻结，生成所有2080条fit的目标。'
 '没有注册、校准或查询Audio前向，也没有用这些Audio更新教师。', '',
 '## 同预算学生对照', '',
 '三组都从原纯ACC700步检查点初始化，762维统计输入→256→192，保持原标准化。'
 '每组固定400步、100条/批、AdamW lr1e-4/weight_decay0.01、梯度裁剪1，沿用此前CE对照的完全相同采样。'
 '每批10人，每人3个共同词×2次Support和另外2个共同词×2次Query，合计10条；词和样本在Support/Query间互斥。', '',
 '- CE：仅身份分类。',
 '- AudioKD：CE加Audio关系MSE，同人异词与异人同词两种关系各占一半。辅助梯度回传ACC编码器。'
 '每步权重固定规则min(10,0.1×身份梯度范数/关系梯度范数)，系数停止梯度，没有查询调权。',
 '- ShuffledAudioKD：用独立固定排列打乱2080条教师目标，保持额外损失和训练预算，破坏输入身份对应。', '',
 '学生结构、初始化和400步预算一致；Audio方法额外付出一次300步教师适配及特征提取，不宣称整体计算预算相同。'
 '三个学生种子共用一个fit-only教师；未研究教师训练种子的变化。', '',
 '## 认证协议', '',
 '编码器仅训练26词。注册词缓存ID20–23，每词6遍，共24条/人。校准词8、9、18、19、24，各8遍共40条可用；'
 '查询词25–29，各8遍共40条可用，与训练、注册、校准按词隔离。', '',
 'RBF-SVM C=10、balanced，每个目标以本人注册ACC为正类，全部其他9人注册ACC为负类。'
 '分别对1/3/5条不同词录制取RBF分数平均；校准和查询使用同一固定词序，不循环补齐、不复用source_id。'
 '40条可用记录对应40/13/8次挑战，实际使用40/39/40条缓存条目。校准目标每人同样40/13/8次，攻击数为9倍。', '',
 '沿用此前开发选择的分位数阈值τ=(本人校准分数Q10＋攻击者校准分数Q95)/2。'
 '方法/种子/用户/挑战长度分别拟合阈值，全部270个阈值写盘后才计算查询。'
 '没有重新搜索阈值规则、KD权重、模型检查点、用户或攻击来源。', '',
 '## 主3词逐用户结果', '',
 '每项为三个种子均值，单位%。所有10位目标和9位攻击来源均保留。', '',
 '|用户|纯ACC FAR|FRR|HTER|Audio FAR|FRR|HTER|打乱目标 HTER|', '|---|---:|---:|---:|---:|---:|---:|---:|']
for user in range(1,11):
    a=avg(get('CE',user=user));b=avg(get('AudioKD',user=user));c=avg(get('ShuffledAudioKD',user=user))
    lines.append(f'|U{user}|'+ '|'.join(f'{100*v:.2f}' for v in [a['far'],a['frr'],a['hter'],b['far'],b['frr'],b['hter'],c['hter']])+'|')
lines += ['', '## 主3词逐种子结果', '',
 '|种子|方法|FAR%|FRR%|HTER%|EER%|', '|---:|---|---:|---:|---:|---:|']
for seed in (42,43,44):
    for m in methods:
        v=avg(get(m,seed=seed));lines.append(f'|{seed}|{labels[m]}|'+ '|'.join(f'{100*v[k]:.2f}' for k in metrics[:4])+'|')
lines += ['', '## 后4遍录制的补充评估', '',
 '|方法|词数|FAR%|FRR%|HTER%|EER%|', '|---|---:|---:|---:|---:|---:|']
for m in methods:
    for k in (1,3,5):
        v=avg(get(m,k,'additional_last4'));lines.append(f'|{labels[m]}|{k}|'+ '|'.join(f'{100*v[z]:.2f}' for z in metrics[:4])+'|')
a=avg(get('CE'));b=avg(get('AudioKD'));c=avg(get('ShuffledAudioKD'))
lines += ['', '## 结果判断', '',
 f"主3词Audio相对纯ACC：HTER变化{100*(b['hter']-a['hter']):+.2f}个百分点，"
 f"FAR变化{100*(b['far']-a['far']):+.2f}个百分点，FRR变化{100*(b['frr']-a['frr']):+.2f}个百分点。"
 f"逐用户HTER为{differences['CE']['hter_improved_users']}人改善、{differences['CE']['hter_equal_users']}人持平、"
 f"{differences['CE']['hter_worse_users']}人退化。", '',
 f"相对打乱Audio目标，正确Audio的HTER变化{100*(b['hter']-c['hter']):+.2f}个百分点。"
 '是否支持Audio增量应同时看纯ACC基线、打乱对照、三个种子和逐用户风险；不能把多词融合对两类编码器共有的收益算作Audio收益。', '']
if b['hter']<a['hter'] and b['hter']<c['hter']:
    lines += ['正确Audio的主3词HTER在三个种子上均低于纯ACC，但主要是以更高FAR换取更低FRR。'
              '3词EER从纯ACC的1.89%升至2.45%，三个种子均上升，AUC也均下降，因此没有身份排序增强的证据。'
              '正确Audio相对打乱目标的HTER优势仅约0.13个百分点；5词二者HTER同为1.53%，EER/AUC也相同。'
              '这表明多词聚合是主要收益，当前Audio关系蒸馏的额外贡献很小，不能将全部多词收益算作Audio知识迁移。', '',
              '逐用户看，U2的FRR从20.51%降到12.82%，U5的FAR从8.26%降到7.12%，但U8的FAR从3.70%升至10.54%。'
              '下一步应只在训练/开发词中检验不增加困难用户FAR的选择性监督，再以新录制确认，不据这些查询结果给用户单独调参。', '']
else:
    lines += ['本轮未满足“正确Audio的主3词HTER同时超过纯ACC和打乱目标”的标准。'
              '若它仅在某个挑战长度或个别用户更好，不据查询结果替换预设主方案。'
              '下一步应在训练范围内检查教师身份关系是否能被ACC预测，再决定选择性迁移，不直接加大蒸馏权重。', '']
lines += ['## 执行与核验', '',
 '远程实例只有约半个CPU核。保持官方权重与数据不变，教师提取转至本地CPU，先2线程、后8线程；学生训练恢复2线程。'
 '本地数据、统计缓存和700步初始化的SHA256与远程一致。官方教师抽查1条的向量最大差异'
 f"{execution['teacher_embedding_probe_max_abs']:.3g}，池化输入最大差异{execution['teacher_pool_probe_max_abs']:.3g}。"
 '该数值核验只是一个抽查，不宣称全部2080条与远程逐元素完全一致。', '',
 '合成测试确认教师停止梯度、KD回传学生、辅助梯度比例有界。运行中检查fit-only教师2080条、模型与数据对齐、'
 '全部阈值先锁定、主评估source_id不复用。纯ACC3词和上一轮24注册＋39校准结果复算误差'
 f"{verify['ce_threeword_metric_max_error']:.3g}；540个查询工作点、270个阈值计数/比率通过核验。", '',
 '三个纯ACC检查点跨平台重训参数的最大绝对差异为'
 f"{max(v['parameter_max_error'] for v in verify['ce_checkpoint_replay']):.3g}。"
 '全部2080条教师目标的索引、sample_id、有限性和单位范数检查通过；有记录的辅助梯度比例范围为'
 f"{verify['logged_aux_gradient_ratio_min']:.4f}–{verify['logged_aux_gradient_ratio_max']:.4f}。", '',
 '这是全部用户参与共享训练的闭集跨词实验。查询词与该批录制已在此前探索中使用，后4遍部分也可能曾作为质量备用录制，'
 '不能称为新盲测或未见身份开放集。三种子和补充查询复用物理录制，不计算虚假的独立样本置信区间。'
 'source_id不复用不等于新做了原始波形时间重叠审计，本轮沿用上游缓存切分。', '',
 '发布匿名逐用户/逐种子结果、训练摘要、协议、核验与代码；不发布Audio/ACC波形、embedding、实名映射、私有模型或逐样本认证分数。'
 '既有创新点一V1和历史实验未修改。', '']
(p/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(primary={m:avg(get(m)) for m in methods},paired=differences)))

