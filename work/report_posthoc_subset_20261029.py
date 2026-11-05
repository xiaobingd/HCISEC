"""Independent verification of explicitly test-informed restricted-cohort results."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_posthoc_subset_20261029');PREV=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def check(g,n,t,r):
 assert len(g)==r['genuine_n'] and len(n)==r['impostor_n']
 assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
 assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
 r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
rows={}
for u in range(1,10):
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text());locked=json.loads((folder/'locked_before_queries.json').read_text())
 assert locked['selection_test_informed'] is True
 previous=json.loads((PREV/('U%d'%u)/'results_public.json').read_text());attacks=previous['reference']['baseline_rbf']['attack_by_user']
 ranking=sorted([dict(user=int(v),prior_test_far=b['accepted']/b['n'],prior_test_accepted=b['accepted'],prior_test_n=b['n']) for v,b in attacks.items()],key=lambda x:(-x['prior_test_far'],x['user']))
 assert ranking==a['ranking']
 p=old.protocol(u);scores=np.load(str(folder/'scores_private.npz'));models=old.sel.joblib.load(str(folder/'models_private.joblib'))
 dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(PREV/('U%d'%u)/'final_features_private.npz'))
 assert np.array_equal(scores['attack_labels'],p.labels['final_impostor'])
 for key,r in a['routes'].items():
  spec=locked['routes'][key];removed=[v['user'] for v in ranking[:r['drop']]];assert removed==r['removed'] and u not in removed
  fit_n=80*(9-r['drop']);model=models[key]
  scaler=model[0] if r['method']=='relative' else model[0][0]
  assert int(scaler.n_samples_seen_)==fit_n
  if r['method']=='relative':assert sorted((model[1]+1).tolist())==[v for v in range(1,10) if v not in removed]
  def values(x):
   if r['method']=='rbf':return model[1].decision_function(scaler.transform(x)/np.sqrt(x.shape[1]))
   z=norm(scaler.transform(x));sim=z@model[2].T;uid=model[1]
   return sim[:,uid==u-1].ravel()-r['lam']*sim[:,uid!=u-1].max(1)
  calmask=~np.isin(p.labels['impostor_calibration'],np.asarray(removed)-1)
  g=values(dev['target_dev_pool__'+r['feature']]);n=values(dev['impostor_calibration__'+r['feature']][calmask])
  t=spec['threshold'];assert abs(old.base.original.choose_threshold(g,n)-t)<1e-12
  for role in ('final_genuine','final_impostor'):assert np.allclose(values(final[role+'__'+r['feature']]),scores[key+'__'+role],atol=1e-12,rtol=1e-12)
  g=scores[key+'__final_genuine'];n=scores[key+'__final_impostor'];keep=~np.isin(scores['attack_labels'],np.asarray(removed)-1)
  check(g,n[keep],t,r);check(g,n,t,r['all_attackers_diagnostic'])
  assert len(r['attack_by_user'])==8-r['drop']
  for v,b in r['attack_by_user'].items():
   assert int(v) not in removed;s=n[scores['attack_labels']==int(v)-1];assert len(s)==b['n'] and int((s>=t).sum())==b['accepted']
  diag=r['all_attackers_diagnostic']
  for name,mask in [('retained_attackers',keep),('removed_attackers',~keep)]:
   b=diag[name];s=n[mask];assert b['n']==len(s) and b['accepted']==int((s>=t).sum())
 rows[str(u)]=a
macro={key:{scope:{k:float(np.mean([(a['routes'][key] if scope=='restricted' else a['routes'][key]['all_attackers_diagnostic'])[k] for a in rows.values()])) for k in ['tar','far','auc','hter']} for scope in ['restricted','all_attackers_diagnostic']} for key in rows['1']['routes']}
out=dict(status='PASS',test_informed_subset=True,targets=9,routes_verified=sum(len(a['routes']) for a in rows.values()),results=rows,macro=macro,verification='Replayed prior FINAL-FAR identity ranking; refit scaler sample counts; excluded center identities; calibration and model scores; restricted/all-source AUC, accepted counts and per-source labels')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

