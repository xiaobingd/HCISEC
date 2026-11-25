"""Reconcile public aggregates and attach mechanism results, no private export."""
import json
import numpy as np
from extract_raw_periodicity import PRIVATE,OUT,BASE,track,summarize
from audit_local_identity_residual import holm

c=np.load(PRIVATE/'features_private.npz');old=np.load(BASE/'conditional_correspondence/raw_common_dynamics/local_power_private.npz')
for key in ['sample_id','indices','user','word','repeat']:assert np.array_equal(c[key],old[key])
for key in ['audio_periodicity','acc_periodicity','audio_harmonics','acc_harmonics','audio_track','acc_track']:assert np.isfinite(c[key]).all()
raw=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');rawacc=raw['acc']
replay_error=0.
for pos in [0,639,1279]:
    got=track(rawacc[int(c['indices'][pos])],1000)
    err=float(np.max(abs(got-c['acc_track'][pos])));replay_error=max(replay_error,err)
    assert np.allclose(got,c['acc_track'][pos],atol=1e-6)
s=json.loads((OUT/'summary_public.json').read_text());rows=json.loads((OUT/'per_pair_public.json').read_text())
assert len(s)==3 and len(rows)==135
for v in s:
    rr=[r for r in rows if r['candidate']==v['candidate']]
    for key in ['baseline_mse','audio_mse']:assert np.isclose(v[key],np.mean([r[key] for r in rr]),atol=1e-12)
assert np.allclose(holm([v['permutation_p'] for v in s]),[v['holm_p'] for v in s])
m=json.loads((OUT/'mechanism_public.json').read_text())
labels={'shape':'原始周期性','envelope':'谐波结构','combined':'组合'}
p=OUT/'report.md';text=p.read_text(encoding='utf-8').split('\n\n## 原始周期对应的实际结果')[0]
text+='\n\n## 原始周期对应的实际结果\n\n```json\n'+json.dumps(m,ensure_ascii=False,indent=2)+'\n```\n'
text+='\n这些数值区分了发声周期对应与身份迁移：真实配对优于错遍仅说明对应，不证明EER改善。周期身份关系的相关性亦非蒸馏有效性。\n'
text+='\n## 逐用户预测误差\n\n|用户|候选|基线MSE|加Audio MSE|\n|---|---|---:|---:|\n'
for v in json.loads((OUT/'per_user_public.json').read_text()):text+=f"|U{v['user']:02d}|{labels[v['candidate']]}|{v['baseline_mse']:.6f}|{v['audio_mse']:.6f}|\n"
p.write_text(text,encoding='utf-8')
verification=dict(status='PASS',aligned_samples=1280,candidates=3,endpoint_pairs_per_candidate=45,all_features_finite=True,public_aggregates_recomputed=True,holm_recomputed=True,
    raw_ACC_replayed_windows=3,max_track_replay_error=replay_error,private_export=False,final_Audio_opened=False,source_audio_fit_recordings=160)
(OUT/'verification_public.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification))
