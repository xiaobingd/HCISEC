"""Replay train-only feature selection and independently verify final scores."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
FIXED=Path('/autodl-fs/data/mei_auth80_diffusion_20261029')
rows={}
for u in range(1,10):
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text())
 lock=json.loads((folder/'locked_before_queries.json').read_text());p=old.protocol(u)
 oo=dict(np.load(str(folder/'oof_private.npz')))
 y=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')])==u-1
 words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
 folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 ww,meta=old.sel.forward_select(oo,y,folds,sorted(k for k in oo if k!='base'),{'base':1.},11)
 assert ww==lock['weights']==a['weights']==a['route']['weights']
 assert len(ww)-1==a['selected_count']
 data=np.load(str(folder/'scores_private.npz'));features=np.load(str(folder/'final_features_private.npz'))
 models=old.sel.joblib.load(str(FIXED/('U%d'%u)/'models_private.joblib'))
 for role in ('final_genuine','final_impostor'):
  score=0
  for k,w in ww.items():
   m=models[k];score+=w*(old.sel.rbf.score(m['model'],[features[role+'__'+k]])-m['mean'])/m['std']
  assert np.allclose(score,data[role],atol=1e-12,rtol=1e-12)
 g=data['final_genuine'];n=data['final_impostor'];t=lock['threshold'];r=a['route']
 assert np.array_equal(data['attack_labels'],p.labels['final_impostor'])
 assert r['genuine_n']==len(g)==len(p.labels['final_genuine']) and r['impostor_n']==len(n)==len(p.labels['final_impostor'])
 assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
 assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
 assert len(r['attack_by_user'])==8
 for v,b in r['attack_by_user'].items():
  s=n[data['attack_labels']==int(v)-1];assert len(s)==b['n'] and int((s>=t).sum())==b['accepted']
 r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum());rows[str(u)]=a
macro={name:{k:float(np.mean([fn(a)[k] for a in rows.values()])) for k in ['tar','far','auc','hter']} for name,fn in [('baseline_rbf',lambda a:a['reference']['baseline_rbf']),('teacher_limited',lambda a:a['reference']['auto_without_diffusion']),('diffusion_limited',lambda a:a['fixed_quota_reference']),('free_selection',lambda a:a['route'])]}
out=dict(status='PASS',targets=9,results=rows,macro=macro,verification='Replayed train-fold greedy selection and final score fusion from fixed models/private features; recomputed pairwise AUC, all counts, HTER and eight-source labels')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

