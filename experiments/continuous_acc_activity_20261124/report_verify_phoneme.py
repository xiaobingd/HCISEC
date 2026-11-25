import json
import numpy as np
from audit_phoneme_identity import BASE,OUT,GROUPS,ARMS,descriptor,window
from audit_local_identity_residual import holm

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
summary,users,rows,cov,comp=[read(n) for n in ['summary_public.json','per_user_public.json','per_word_user_public.json','coverage_public.json','comparison_public.json']]
c=np.load(BASE/'phoneme/features_private.npz');raw=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz')
idx=c['indices'];assert len(idx)==720 and np.all(raw['fit'][idx])
for key in ['user','word','sample_id']:assert np.array_equal(c[key],raw[key][idx])
assert len(rows)==540 and len(users)==180 and len(summary)==18 and len(cov)==30
for v in summary:
    rr=[q for q in users if q['group']==v['group'] and q['arm']==v['arm']]
    for key in ['auc','margin']:assert np.isclose(v[key],np.mean([q[key] for q in rr]),atol=1e-12)
assert sum(v['total'] for v in cov)==720 and sum(v['valid'] for v in cov)==int(c['success'].sum())
assert c['success'].all()
for arm in ARMS:assert c[arm].shape==(720,24) and np.isfinite(c[arm]).all()
primary=[v for v in comp if v['control']=='word_center' and v['group'] in ['AE','EH']]
assert np.allclose(holm([v['user_signflip_p'] for v in primary]),[v['holm_primary_p'] for v in primary])
manifest=json.loads((BASE/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
overlaps=0
for wo in sorted(set(c['word'])):
  for us in range(1,11):
    pos=np.flatnonzero((c['word']==wo)&(c['user']==us)&c['success']);bounds=c['phonebounds'][pos];bounds=bounds[np.argsort(bounds[:,0])]
    overlaps+=int(np.sum(bounds[1:,0]<bounds[:-1,1]))
rawacc=raw['acc'];maxerr=0.
for pos in [0,359,719]:
    i=int(idx[pos]);rec=manifest[i];ac=c['phonebounds'][pos].mean();lo,hi=rec['audio_start_s'],rec['audio_end_s'];il,ih=rec['imu_interval']
    center=1.5+(ac-(lo+hi)/2)*(ih-il)/(hi-lo)
    got=window(rawacc[i],center);maxerr=max(maxerr,float(np.max(abs(got-c['shared_phone'][pos]))));assert np.allclose(got,c['shared_phone'][pos],atol=1e-10)
verification=dict(status='PASS',samples=720,recordings=90,users=10,groups=3,arms=6,rows=540,identity_word_role='all selected oldfit words only;leavewordout diagnostic',
  aggregate_recomputed=True,holm_recomputed=True,within_user_word_phone_interval_overlaps=overlaps,raw_ACC_replayed_windows=3,max_feature_replay_error=maxerr,
  final_Audio_opened=False,manual_phone_validation=False,oracle_query_Audio=True,ACC_only_deployment_claim=False)
duration=c['phonebounds'][:,1]-c['phonebounds'][:,0]
verification['estimated_phone_duration_median_ms']=float(np.median(duration)*1000)
verification['estimated_phone_shorter_than_120ms_fraction']=float(np.mean(duration<.12))
(OUT/'verification_public.json').write_text(json.dumps(verification,indent=2))
labels={'shared_phone':'共享元音120ms','word_center':'词中心120ms','wrong_phone':'错音素120ms','offset_minus100':'元音位置提前100ms','offset_plus100':'元音位置延后100ms','whole_word':'完整词（更长窗口）'}
lines=['# Audio引导共享音素的跨词ACC身份诊断','','## 实际判断','','音素定位在当前设置下没有超过同长度词中心对照，不能宣称音素级Audio监督已有效。共享元音优于错音素，说明局部位置选择影响身份信息，但错音素常为辅音，因此不能完全排除发声活动/能量差异。统一提前100ms在三组中都更好，提示需要审计时间映射、音素边界和发声起始上下文；没有依据把它直接解释为固定硬件延迟或选为最终最优偏移。','','这是训练词内、Audio辅助定位的诊断：Audio也用于诊断查询片段定位，不是最终ACC-only查询实验。最终保留词的Audio未读取，也未训练神经网络。','','## 预先固定的方法','','10人，9个历史训练词，720次已有独立录制片段，90条原始词录音。/AE/：Back、Cancel、Grab；/EH/：Exit、Help、Menu、Next；/AY/：five、nine。图中time不在词表。720次自动对齐全部产生结构有效的结果；这不是音素准确率100%，尚无人工音素真值。','','PocketSphinx5.0.4自带US-English模型，已知单词文本，两遍强制对齐到音素状态。依原有Audio/IMU时间区间映射到ACC，40–120ms以上残差可能大于音素本身。本次没有人工核验物理同步。对齐方法文档：https://pocketsphinx.readthedocs.io/en/stable/pocketsphinx.html 。','','每组逐词留出查询，其余词做身份原型注册。标准化只用其余9位用户的注册词；目标本人只提供注册原型，不进入标准化拟合。ACC统一24维原始谱/幅度统计，余弦认证分数。所有120ms局部组使用相同前端；完整词对照有更长数据，只作为上下文参照。没有搜索超参数或删人。','','## 跨词AUC（10用户宏平均）','','|音素组|窗口方法|AUC↑|本人-最难冒充者间隔↑|','|---|---|---:|---:|']
for v in summary:lines.append(f"|{v['group']}|{labels[v['arm']]}|{v['auc']:.4f}|{v['margin']:+.4f}|")
lines+=['','## 配对对照','','差值为共享元音AUC减去对照；按10用户聚类作精确符号置换，AE/EH对词中心的两个主要检验做Holm校正；AY与错音素比较是探索性，不单独当成功依据。','','|音素组|对照|AUC差|改善用户|单侧p|主要检验校正p|','|---|---|---:|---:|---:|---:|']
for v in comp:lines.append(f"|{v['group']}|{labels[v['control']]}|{v['delta_auc']:+.4f}|{v['improved_users']}/10|{v['user_signflip_p']:.4f}|{v.get('holm_primary_p','—')}|")
lines+=['','## 逐用户AUC','','|组|用户|窗口|AUC↑|','|---|---|---|---:|']
for v in users:lines.append(f"|{v['group']}|U{v['user']:02d}|{labels[v['arm']]}|{v['auc']:.4f}|")
lines+=['','## 下一步与边界','',f"估计音素中位时长{np.median(duration)*1000:.1f}ms；{100*np.mean(duration<.12):.1f}%短于固定120ms窗，因此部分窗口包含相邻音素，并非纯音素隔离。",'', '先在基础身份/训练词里核验自动音素边界与两模态时间映射，评估100ms偏移现象究竟来自时延还是更有身份信息的发声起始部分；不得利用这批已看过词选择偏移后，再将其作为独立验证。待定位可靠后，才训练带时间不确定性的局部ACC聚合器；部署查询必须由ACC自己定位。','','本轮没有新的独立盲测词、跨天、重戴或设备对照。留词验证只控制词标签，不能完全控制共发音、语速、活动与前后音素。没有EER/HTER锁阈值实验，因此本报告不把诊断AUC称为完整身份认证基线提升。','','对齐时间、片段特征、模型、音频及身份映射保留私有；公开代码、词典覆盖和匿名逐用户汇总。']
(OUT/'report.md').write_text('\n'.join(lines).replace('720次已有独立录制片段','720个重复发声片段').replace('40–120ms以上残差可能大于音素本身','时间映射误差可能与音素长度同量级'),encoding='utf-8')
print(json.dumps(verification))
