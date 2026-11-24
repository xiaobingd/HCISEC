import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'conditional_correspondence/enrollment_template_report';OUT.mkdir(exist_ok=True)
d=np.load(ROOT/'phrase_auth/frontend_private.npz');u,w=d['user'],d['word'];reg=np.isin(w,[20,21,22,23]);ix=np.flatnonzero(reg)
c=np.load(ROOT/'conditional_correspondence/enrollment_identity_quality/registered_ecapa_private.npz');assert np.array_equal(c['indices'],ix)
az=np.zeros((3200,192));az[ix]=c['embedding'];views=np.load(ROOT/'conditional_correspondence/stage3/views_private.npz')
records=[];aucs=[]
for name,z in [('Audio',az)]+[(f'ACC_{seed}',views[f'{seed}_A_ACC']) for seed in (42,43,44)]:
 for a in (20,21,22,23):
  for b in (20,21,22,23):
   if a==b:continue
   p=np.stack([z[reg&(u==us)&(w==a)].mean(0) for us in range(1,11)]);q=np.stack([z[reg&(u==us)&(w==b)].mean(0) for us in range(1,11)])
   p/=np.linalg.norm(p,axis=1,keepdims=True);q/=np.linalg.norm(q,axis=1,keepdims=True)
   aucs.append(dict(modality=name,enrollment_word=a,query_registration_word=b,auc=float(roc_auc_score(np.eye(10).ravel(),(p@q.T).ravel()))))
for seed in (42,43,44):
 z=views[f'{seed}_A_ACC']
 for us in range(1,11):
  for wo in (20,21,22,23):
   part=np.flatnonzero(reg&(u==us)&(w==wo));other=np.flatnonzero(reg&(u==us)&(w!=wo))
   ap=az[other].mean(0);ap/=np.linalg.norm(ap);xp=z[other].mean(0);xp/=np.linalg.norm(xp)
   aq=az[part]@ap;xq=z[part]@xp
   rho=float(spearmanr(aq,xq).statistic);assert np.isfinite(rho)
   records.append(dict(seed=seed,user=us,word=wo,quality_spearman=rho))
users=[dict(user=us,registration_audio_acc_quality_rank_rho=float(np.mean([a['quality_spearman'] for a in records if a['user']==us]))) for us in range(1,11)]
summary=dict(registered_audio_only=True,registered_source_records=320,audio_registration_crossword_auc=float(np.mean([a['auc'] for a in aucs if a['modality']=='Audio'])),
 acc_registration_crossword_auc_mean=float(np.mean([a['auc'] for a in aucs if a['modality'].startswith('ACC')])),
 quality_rank_spearman_mean=float(np.mean([a['quality_spearman'] for a in records])),per_user=users,
 note='registration-only diagnostics,not authentication queries or independent users;quality rankagreement isnot proof of incremental gain')
(OUT/'registration_diagnostic_public.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
