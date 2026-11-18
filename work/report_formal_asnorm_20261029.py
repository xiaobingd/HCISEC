"""Independent AS-Norm1 score computation via cohort indices and sample moments."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0,'/root')
import run_posthoc_subset_20261029 as previous
old=previous.old;ROOT=Path('/autodl-fs/data/mei_formal_asnorm_20261029_f64');PRIOR=previous.ROOT
def refscore(model,x,u,k):
 s,ids,c=model;z=previous.norm(s.transform(x)).astype(np.float64);e=c[ids==u-1][0].astype(np.float64);cohort=c[ids!=u-1].astype(np.float64)
 raw=z@e;es=cohort@e;ts=z@cohort.T
 indices=np.argpartition(es,-k)[-k:];et=es[indices]
 ti=np.argpartition(ts,-k,axis=1)[:,-k:];tt=np.take_along_axis(ts,ti,axis=1)
 mu_e=et.sum()/k;mu_t=tt.sum(1)/k
 sd_e=np.sqrt(((et-mu_e)**2).sum()/(k-1));sd_t=np.sqrt(((tt-mu_t[:,None])**2).sum(1)/(k-1))
 result=((raw-mu_e)/max(sd_e,1e-4)+(raw-mu_t)/np.maximum(sd_t,1e-4))/2
 return result
def check(g,n,t,r):
 assert r['genuine_n']==len(g) and r['impostor_n']==len(n)
 assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
 assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
 r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
rows={}
for u in range(1,10):
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_queries.json').read_text())
 models=old.sel.joblib.load(str(PRIOR/('U%d'%u)/'models_private.joblib'));p=old.protocol(u)
 dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(previous.PRIOR/('U%d'%u)/'final_features_private.npz'))
 saved=np.load(str(folder/'scores_private.npz'));assert np.array_equal(saved['attack_labels'],p.labels['final_impostor'])
 oof=np.load(str(folder/'oof_private.npz'));labels=np.concatenate([p.labels[k] for k in ['auth_train','model_validation']])
 words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ['auth_train','model_validation'] for v in old.USERS])
 for key,r in a['routes'].items():
  spec=lock['routes'][key];model=models['%s_drop%d_relative'%(r['feature'],r['drop'])];assert len(model[1])-1==8-r['drop']
  assert sorted((model[1]+1).tolist())==[v for v in range(1,10) if v not in r['removed']]
  assert int(model[0].n_samples_seen_)==80*(9-r['drop'])
  k=r['k'];assert 2<=k<=8-r['drop']
  if r['variant']=='asnorm_fixed4':assert k==4
  if r['variant']=='asnorm_selected':
   assert k==min(spec['cv'],key=lambda b:(-b['mean_auc'],-b['k']))['k']
   keeptrain=~np.isin(labels,np.asarray(r['removed'])-1);y=labels[keeptrain]==u-1;ww=words[keeptrain]
   folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,ww))
   for candidate in spec['cv']:
    ss=oof['%s_drop%d_K%d'%(r['feature'],r['drop'],candidate['k'])]
    assert abs(np.mean([roc_auc_score(y[va],ss[va]) for _,va in folds])-candidate['mean_auc'])<1e-12
  if r['variant']=='snorm':assert k==8-r['drop']
  t=r['threshold'];calmask=~np.isin(p.labels['impostor_calibration'],np.asarray(r['removed'])-1)
  cg=refscore(model,dev['target_dev_pool__'+r['feature']],u,k);cn=refscore(model,dev['impostor_calibration__'+r['feature']][calmask],u,k)
  assert abs(old.base.original.choose_threshold(cg,cn)-t)<1e-8
  for role in ['final_genuine','final_impostor']:assert np.allclose(refscore(model,final[role+'__'+r['feature']],u,k),saved[key+'__'+role],atol=1e-8,rtol=1e-10)
  g=saved[key+'__final_genuine'];n=saved[key+'__final_impostor'];keep=~np.isin(saved['attack_labels'],np.asarray(r['removed'])-1)
  check(g,n[keep],t,r);check(g,n,t,r['all_attackers_diagnostic'])
  assert len(r['attack_by_user'])==8-r['drop']
  for v,b in r['attack_by_user'].items():
   assert int(v) not in r['removed'];s=n[saved['attack_labels']==int(v)-1];assert len(s)==b['n'] and int((s>=t).sum())==b['accepted']
 rows[str(u)]=a
macro={key:{scope:{m:float(np.mean([(a['routes'][key] if scope=='restricted' else a['routes'][key]['all_attackers_diagnostic'])[m] for a in rows.values()])) for m in ['tar','far','auc','hter']} for scope in ['restricted','all_attackers_diagnostic']} for key in rows['1']['routes']}
out=dict(status='PASS',targets=9,routes_verified=sum(len(a['routes']) for a in rows.values()),results=rows,macro=macro,verification='Independent cohort-index and sample-moment AS-Norm1 scores; fit counts/background IDs; K locks; calibration thresholds; all-source and restricted AUC/counts')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

