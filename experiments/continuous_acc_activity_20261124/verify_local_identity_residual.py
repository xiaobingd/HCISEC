"""Aggregate accounting, multiplicity and descriptive repeat stability."""
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from audit_local_identity_residual import BASE,OUT,descriptors,unit,holm

summary=json.loads((OUT/'summary_public.json').read_text())
rows=json.loads((OUT/'per_pair_public.json').read_text())
users=json.loads((OUT/'per_user_public.json').read_text())
assert len(rows)==135 and len(users)==30 and len(summary)==3
for s in summary:
    rr=[v for v in rows if v['candidate']==s['candidate']]
    assert len(rr)==45
    for k in ['baseline_mse','audio_mse']:assert np.isclose(np.mean([v[k] for v in rr]),s[k],atol=1e-12)
    assert np.isclose(s['mse_reduction'],s['baseline_mse']-s['audio_mse'])
assert np.allclose(holm([s['permutation_p'] for s in summary]),[s['holm_p'] for s in summary])
c=np.load(BASE/'conditional_correspondence/raw_common_dynamics/local_power_private.npz')
p=json.loads((OUT/'protocol_public.json').read_text());w,u,r=[c[k] for k in ['word','user','repeat']]
local=descriptors(c['audio'].astype(float),c['audio_activity'].astype(float));stability=[]
for k,x in local.items():
    z=StandardScaler().fit(x[np.isin(w,p['train_words'])]).transform(x)
    for wo in sorted(set(w)):
        relations=[]
        for parity in (0,1):
            pr=unit(np.stack([z[(w==wo)&(u==us)&(r%2==parity)].mean(0) for us in range(1,11)]))
            relations.append((pr@pr.T)[np.triu_indices(10,1)])
        stability.append(dict(candidate=k,word=int(wo),spearman=float(spearmanr(*relations).statistic)))
(OUT/'repeat_stability_public.json').write_text(json.dumps(stability,indent=2))
v=dict(status='PASS',candidate_groups=3,identity_pairs_per_group=45,user_rows=30,aggregate_recomputed=True,holm_recomputed=True,
       repeat_stability_scope='descriptive all-cohort train-word standardized even-vs-odd pairrelations; no predictive selection',
       final_test_audio_opened=False)
(OUT/'verification_public.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
