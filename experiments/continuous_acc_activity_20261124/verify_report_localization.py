"""Audit selector isolation and raw feature replay; render all anonymous results."""
import json,itertools
import numpy as np
from explore_phoneme_localization import BASE,OUT,GROUPS,GRID,ARMS,macro_auc,choose
from audit_local_identity_residual import holm

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
summary,users,rows,sel,comparisons=[read(n) for n in ['summary_public.json','per_user_public.json','per_word_user_public.json','selection_public.json','comparison_public.json']]
c=np.load(BASE/'phoneme/features_private.npz');p=np.load(BASE/'phoneme/localization_private.npz')
u,w=c['user'],c['word'];assert len(sel)==90 and len(rows)==630 and len(users)==210 and len(summary)==21
for v in sel:
    assert v['query_word'] not in v['selector_words'] and v['target'] not in v['selector_users']
    for arm in ['fixed','audio']:assert GRID[choose(v['candidate_auc'][arm])]==v[arm+'_offset']
for v in summary:
    rr=[a for a in users if a['group']==v['group'] and a['arm']==v['arm']]
    assert len(rr)==10
    for key in ['auc','margin']:assert np.isclose(v[key],np.mean([a[key] for a in rr]),atol=1e-12)
primary=[v for v in comparisons if v['control']=='fixed_support_selected']
assert np.allclose(holm([v['user_signflip_p'] for v in primary]),[v['holm_primary_p'] for v in primary])
assert np.allclose(p['audio'][2],c['shared_phone'],atol=1e-9) and np.allclose(p['audio'][1],c['offset_minus100'],atol=1e-9)
# Perturb all outer-query-word AND target-user features: internal scores must be invariant.
qword,target=0,1;cohort=list(range(2,11));groups=[[wo for wo in words if wo!=qword] for words in GROUPS.values()];groups=[g for g in groups if len(g)>=2]
z=p['audio'][0].copy();changed=z.copy();changed[(w==qword)|(u==target)]+=1000
maxerr=0.
for words in groups:
  for held in words:
    support=[wo for wo in words if wo!=held]
    a=macro_auc(z,u,w,support,held,cohort,target);b=macro_auc(changed,u,w,support,held,cohort,target)
    maxerr=max(maxerr,abs(a-b));assert abs(a-b)<1e-12
verification=read('verification_public.json');verification.update(independent_aggregate_check=True,selector_offset_check=True,holm_recomputed=True,
  isolation_perturbation='all outerqueryword andtargetuser candidate features changed+1000;innerAUC unchanged',max_isolation_auc_change=maxerr)
(OUT/'verification_public.json').write_text(json.dumps(verification,indent=2))
names={'word_center':'词中心120ms','fixed_support_selected':'固定相对位置（支持词选）','correct_Audio_selected':'正确Audio边界（支持词选偏移）','wrongrepeat_Audio_sameoffset':'同人同词错遍边界','ACC_energy_peak':'ACC局部高频能量峰','Audio_nominal':'原始音素位置','Audio_minus100_fixed':'音素提前100ms（历史参照）'}
lines=['# 音素局部定位：Audio独立贡献与固定位置对照','','## 研究问题','','上一轮所有音素组提前100ms后更好。本轮检验该现象是否需要本人这一次Audio边界，还是固定位置/ACC自身能量已经可以解释。三组全部保留，未按查询结果删用户或挑偏移。','','## 严格的本轮隔离','','同一10人、9词、720片段，全部属于历史训练词。每次留出一个查询词与一位目标用户；偏移选择只用其余用户及其余词，在音素组内部继续做跨词留出。固定位置和Audio位置均5候选（-200/-100/0/+100/+200ms），使用相同选择预算。错遍Audio保持本人和词相同，交换另一遍的相对音素位置，并沿用正确Audio组选出的偏移。没有按其他人的平均信号做对齐。','','ACC能量峰使用40–450Hz三轴平方能量，在固定3秒缓存中央0.8秒内按20ms步寻找120ms窗最大值，不调参。各组都是同一120ms、同一24维原始谱/幅度统计、同一余弦原型流程。','','所有候选规则和词都已经受到历史实验启发/查看，因此即使本轮选择隔离，也仍是探索复核，不是新的盲测。正确/错遍Audio组在诊断查询中使用缓存Audio边界，不是可部署的ACC-only方案。固定位置和能量组不再使用音素边界，但3秒原始缓存本身已经依赖历史词分段，也不能直接声称连续信号端到端不使用Audio。','','## 跨词AUC（10用户宏平均）','','|组|方法|AUC↑|本人-最难冒充间隔↑|','|---|---|---:|---:|']
for v in summary:lines.append(f"|{v['group']}|{names[v['arm']]}|{v['auc']:.4f}|{v['margin']:+.4f}|")
lines+=['','## 正确Audio相对对照','','按10用户配对符号置换；主要检验为正确Audio对固定位置，三组Holm校正。其他对照是探索性诊断。','','|组|对照|AUC差|改善用户|单侧p|主要检验校正p|','|---|---|---:|---:|---:|---:|']
for v in comparisons:lines.append(f"|{v['group']}|{names[v['control']]}|{v['delta_auc']:+.4f}|{v['improved_users']}/10|{v['user_signflip_p']:.4f}|{v.get('holm_primary_p','—')}|")
lines+=['','## 逐用户AUC','','|组|用户|方法|AUC|','|---|---|---|---:|']
for v in users:lines.append(f"|{v['group']}|U{v['user']:02d}|{names[v['arm']]}|{v['auc']:.4f}|")
passed=[v['group'] for v in primary if v['delta_auc']>0 and v['holm_primary_p']<.05]
lines+=['','## 当前判断','',f'正确Audio超过固定相对位置并通过校正的组：{passed}。该判断仍需与错遍和ACC能量控制结合：仅超过固定位置不等于利用了本次Audio，超过错遍也不自动证明最终认证受益。', '', '如果固定相对位置/ACC能量已达到或超过Audio，优先研究ACC自身稳健局部聚合；若正确Audio稳定超过固定位置、错遍和能量，才值得将这种局部定位关系作为训练期监督。不得用这批查询词反复选择窗口。', '', '本轮尚未做新录制物理同步标定或人工音素真值核验，因此不能把偏移解释为固定硬件延迟。没有训练新神经网络，没有EER/HTER阈值结果，不将AUC诊断写成认证基线增益。','','## 核验','','90个选择折均排除外层查询词与目标用户；将这些排除对象特征人为加1000，内部选择AUC完全不变。旧音素位置和-100ms特征重现上一轮；重新核算宏平均、偏移选择与Holm校正。原始ACC缓存与44100时间重建清单的中心差为0，未发现缓存中心错用旧清单的问题。','','公开匿名逐用户/逐词汇总、选择候选分数与代码；不含信号、片段特征、样本边界、身份映射或逐样本认证分数。']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(verification))
