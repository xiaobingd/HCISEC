"""Add pure target-cosine controls to the frozen restricted-cohort prototype models."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_posthoc_subset_20261029 as trial
old=trial.old;ROOT=trial.ROOT;rows={}
for u in old.ORDER:
 folder=ROOT/('U%d'%u);p=old.protocol(u)
 result=json.loads((folder/'results_public.json').read_text());lock=json.loads((folder/'locked_before_queries.json').read_text())
 models=old.sel.joblib.load(str(folder/'models_private.joblib'));dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'))
 added={}
 for key,spec in list(lock['routes'].items()):
  if not key.endswith('_relative'):continue
  new=key.replace('_relative','_cosine');removed=spec['removed'];keep=~np.isin(p.labels['impostor_calibration'],np.asarray(removed)-1)
  g,_=trial.proto_pair(models[key],dev['target_dev_pool__'+spec['feature']],u)
  n,_=trial.proto_pair(models[key],dev['impostor_calibration__'+spec['feature']][keep],u)
  z=dict(spec,lam=0.,variant='pure_target_cosine_control',threshold=float(old.base.original.choose_threshold(g,n)))
  z.pop('candidates',None);added[new]=z;models[new]=models[key]
 lock['routes'].update(added);lock['cosine_controls_locked_before_new_score_calculation']=True
 trial.save(folder/'locked_before_queries.json',lock);old.sel.joblib.dump(models,str(folder/'models_private.joblib'))
 final=np.load(str(trial.PRIOR/('U%d'%u)/'final_features_private.npz'));scores=dict(np.load(str(folder/'scores_private.npz')))
 for key,spec in added.items():
  ss={}
  for role in ('final_genuine','final_impostor'):
   s,_=trial.proto_pair(models[key],final[role+'__'+spec['feature']],u);ss[role]=s;scores[key+'__'+role]=s
  full=old.sel.evaluate_route(ss,p,u,spec['threshold']);full.update(spec)
  removed=np.isin(p.labels['final_impostor'],np.asarray(spec['removed'])-1)
  for name,mask in [('removed_attackers',removed),('retained_attackers',~removed)]:
   s=ss['final_impostor'][mask];full[name]=dict(n=len(s),accepted=int((s>=spec['threshold']).sum()),far=float((s>=spec['threshold']).mean()) if len(s) else None)
  import types
  protocol=types.SimpleNamespace(labels={'final_impostor':p.labels['final_impostor'][~removed]})
  restricted=old.sel.evaluate_route({'final_genuine':ss['final_genuine'],'final_impostor':ss['final_impostor'][~removed]},protocol,u,spec['threshold'])
  restricted.update(spec);restricted['test_informed_subset']=True;restricted['all_attackers_diagnostic']=full;result['routes'][key]=restricted
 np.savez_compressed(str(folder/'scores_private.npz'),**scores);trial.save(folder/'results_public.json',result);rows[str(u)]=result
trial.save(ROOT/'results_public.json',rows);print('COSINE_CONTROLS_ADDED',len(rows)*6)

