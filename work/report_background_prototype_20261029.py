"""Replay prototype scores, training-only lambda choices and final metrics."""
import sys,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_background_prototype_20261029');PREV=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
rows={}
for u in range(1,10):
 folder=ROOT/('U%d'%u);record=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_queries.json').read_text())
 p=old.protocol(u);data=np.load(str(folder/'scores_private.npz'));oof=np.load(str(folder/'oof_private.npz'))
 dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));final=np.load(str(PREV/('U%d'%u)/'final_features_private.npz'))
 labels=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')]);y=labels==u-1
 words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 pairs={}
 for name,rep in lock['representations'].items():
  candidates=rep['candidates']
  for r in candidates:
   s=oof[name+'__target']-r['lam']*oof[name+'__background'];values=[float(roc_auc_score(y[va],s[va])) for _,va in folds]
   assert np.allclose(values,r['fold_aucs'],rtol=0,atol=1e-12)
  best=min(candidates,key=lambda r:(-r['mean_auc'],r['lam']));assert best['lam']==rep['lambda_selected']
  scalers=[];blocks=[]
  for branch,w in rep['descriptor']:
   x=np.concatenate([dev[k+'__'+branch] for k in ('auth_train','model_validation')]);scaler=StandardScaler().fit(x)
   scalers.append((branch,w,scaler));blocks.append(np.sqrt(w)*norm(scaler.transform(x)))
  z=norm(np.concatenate(blocks,axis=1));centers=norm(np.asarray([z[labels==v].mean(axis=0) for v in range(9)]))
  pairs[name]={}
  for role in ('final_genuine','final_impostor','target_dev_pool','impostor_calibration'):
   source=final if role.startswith('final_') else dev
   query=norm(np.concatenate([np.sqrt(w)*norm(scaler.transform(source[role+'__'+branch])) for branch,w,scaler in scalers],axis=1))
   similarity=query@centers.T;pairs[name][role]=(similarity[:,u-1],similarity[:,np.arange(9)!=u-1].max(axis=1))
 for route,r in record['diagnostic_routes'].items():
  spec=lock['routes'][route];name=spec['feature'];lam=spec['lam'];t=spec['threshold']
  scores={role:aa-lam*bb for role,(aa,bb) in pairs[name].items()}
  assert abs(old.base.original.choose_threshold(scores['target_dev_pool'],scores['impostor_calibration'])-t)<1e-12
  for role in ('final_genuine','final_impostor'):assert np.allclose(scores[role],data[route+'__'+role],atol=1e-12,rtol=1e-12)
  g=scores['final_genuine'];n=scores['final_impostor']
  assert len(g)==r['genuine_n'] and len(n)==r['impostor_n'] and np.array_equal(data['attack_labels'],p.labels['final_impostor'])
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
  assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
  assert len(r['attack_by_user'])==8
  for v,b in r['attack_by_user'].items():
   a=n[data['attack_labels']==int(v)-1];assert len(a)==b['n'] and int((a>=t).sum())==b['accepted']
  r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
 for name in record['routes']:
  key=lock['auto_cosine']+'__cosine' if name=='auto_cosine' else lock['auto_relative']+'__relative' if name=='auto_relative' else name
  record['routes'][name]=record['diagnostic_routes'][key]
 rows[str(u)]=record
macro={route:{k:float(np.mean([a['routes'][route][k] for a in rows.values()])) for k in ('tar','far','auc','hter')} for route in rows['1']['routes']}
out=dict(status='PASS',targets=9,diagnostic_routes_verified=351,results=rows,macro=macro,verification='Rebuilt all training-only class centers and calibration/final scores; replayed train-fold lambda ranking, calibration thresholds, AUC and eight-source counts')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro)))

