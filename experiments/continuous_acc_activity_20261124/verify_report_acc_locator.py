import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from train_acc_phoneme_locator import OUT,BASE,ARMS,unit
from audit_local_identity_residual import holm

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
summary,users,rows,comp,audit=[read(n) for n in ['summary_public.json','per_user_public.json','per_word_user_public.json','comparison_public.json','exclusion_audit_public.json']]
assert len(summary)==6 and len(users)==60 and len(rows)==240 and len(audit)==40
for v in audit:assert v['target'] not in v['fit_users'] and v['query_word'] not in v['fit_words'] and v['fit_utterances']==216
for v in summary:
    rr=[a for a in users if a['arm']==v['arm']]
    for key in ['auc','margin']:assert np.isclose(v[key],np.mean([a[key] for a in rr]),atol=1e-12)
assert np.allclose(holm([v['p'] for v in comp[:2]]),[v['holm_p'] for v in comp[:2]])
p=np.load(BASE/'phoneme/locator_parameters_private.npz');d=np.load(BASE/'phoneme/locator_inputs_private.npz');c=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz')
idx=d['indices'];u,w=c['user'][idx],c['word'][idx];x=d['features'];maxerr=0.
for arm in ['correct_Audio_trained','wrongrepeat_Audio_trained','ACC_self_energy_trained']:
    prefix='w6_u1_'+arm
    scores=((x-p[prefix+'_mean'])/p[prefix+'_scale'])@p[prefix+'_coef']+p[prefix+'_intercept'];k=np.argmax(scores,axis=1);z=x[np.arange(320),k]
    support=w!=6;query=w==6;fit=support&(u!=1);zz=StandardScaler().fit(z[fit]).transform(z)
    proto=unit(np.stack([zz[support&(u==us)].mean(0) for us in range(1,11)]));q=unit(zz[query]);auc=roc_auc_score(u[query]==1,q@proto[0])
    expected=next(v['auc'] for v in rows if v['user']==1 and v['query_word']==6 and v['arm']==arm);maxerr=max(maxerr,abs(auc-expected));assert abs(auc-expected)<1e-12
verification=read('verification_public.json');verification.update(independent_saved_weight_ACC_only_replays=3,max_auc_replay_error=maxerr,aggregate_recomputed=True,holm_recomputed=True)
(OUT/'verification_public.json').write_text(json.dumps(verification,indent=2))
names={'correct_Audio_trained':'正确Audio训练定位器','wrongrepeat_Audio_trained':'错遍Audio训练定位器','ACC_self_energy_trained':'ACC能量自监督定位器（补充控制）','ACC_energy_peak':'直接ACC能量峰','fixed_support_selected':'支持词选择固定位置','oracle_Audio_selected':'查询Audio定位（参照）'}
lines=['# Audio音素位置能否教会ACC自行定位：轻量迁移试验','','## 任务','','针对上一轮EH组（Exit/Help/Menu/Next）继续做探索。上一轮该组正确Audio边界胜过错遍，但没有稳定胜过ACC能量。本轮只用其他用户与其他词的Audio边界训练Ridge定位器，注册和查询均由冻结定位器仅根据ACC选片段。','','四个历史词均已经查看，本轮不是独立盲测确认。原始3秒缓存已依赖历史词分段，因此这里只实现局部定位阶段不使用查询Audio，不能声称整个连续采集流程完全不依赖Audio。','','## 固定方法','','10用户、4词、320片段；40个外层查询词/目标身份折。每折用其余9身份、其余3词共216片段拟合。候选中心位于3秒缓存中央0.8秒内，20ms步、固定120ms窗；每窗24维ACC频带/幅度统计。Ridge(alpha=10)预测以训练Audio音素位置为中心、sigma=100ms的软标签；输入无时间坐标、词ID、用户ID。冻结后只输入ACC候选窗特征，取最高得分位置。','','正确/错遍定位器具有同结构、同数据量、同预算；错遍保留本人和词、轮换另一遍边界。偏移来自上一轮排除外层词和目标身份的支持词选择，不用查询结果选择。第一轮读出后，另增加同结构ACC能量自监督定位器作为补充控制，未调模型权重。','','身份后端保留相同24维原始统计、归一化原型余弦。所有用户与查询保留。查询Audio边界只用于离线定位误差诊断，不进入定位推理、身份评分或模型选择。','','## 跨词身份结果','','|方法|AUC↑|本人-最难冒充间隔↑|','|---|---:|---:|']
for v in summary:lines.append(f"|{names[v['arm']]}|{v['auc']:.4f}|{v['margin']:+.4f}|")
lines+=['','## 正确Audio训练定位器相对对照','','|对照|AUC差|改善用户|单侧p|主要检验Holm p|','|---|---:|---:|---:|---:|']
for v in comp:lines.append(f"|{names[v['control']]}|{v['delta_auc']:+.4f}|{v['improved_users']}/10|{v['p']:.4f}|{v.get('holm_p','补充控制')}|")
lines+=['','## 离线定位误差','','这里只为诊断使用历史查询Audio边界，不参与任何选择或身份评分。误差相对自动音素映射位置，并非人工真值。','','|方法|平均绝对误差/ms|','|---|---:|']
diag=read('offline_localization_public.json')
for arm in ['correct_Audio_trained','wrongrepeat_Audio_trained','ACC_self_energy_trained']:
    lines.append(f"|{names[arm]}|{np.mean([v['offline_teacher_center_MAE_ms'] for v in diag if v['arm']==arm]):.1f}|")
lines+=['','## 逐用户AUC','','|用户|方法|AUC↑|','|---|---|---:|']
for v in users:lines.append(f"|U{v['user']:02d}|{names[v['arm']]}|{v['auc']:.4f}|")
passed=all(v['delta_auc']>0 and v['holm_p']<.05 for v in comp[:2])
lines+=['','## 当前结论','',f'正确Audio同时超过错遍及ACC能量并通过主要校正：{passed}。ACC可以学习训练Audio给出的局部位置模式，但如果错遍或ACC自监督同样有效，收益不能归因于本次Audio的身份特异性信息。该结果支持研究稳健局部选择，尚不支持直接声称Audio身份知识迁移成功。', '', '本轮没有神经身份编码器训练或EER/HTER锁阈值实验，不能把定位AUC当论文最终认证增益。下一步需要在未反复查看的新数据上，将正确音素监督与ACC自定位保持同结构/预算比较；当前历史四词不再用于反复调sigma/alpha。', '', '已重算全部宏平均、主要Holm校正并从保存的权重重放3个ACC-only身份结果，误差为0。原始信号、模型参数、局部特征、边界与身份映射均不公开。']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(verification))
