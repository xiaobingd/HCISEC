"""Replay frozen branch predictions, calibration and all six U4 routes."""
import sys,json
import numpy as np
sys.path.insert(0,'/root')
import run_u4_baseline_raw_increment_20261030 as run
s=run.s;b=run.b;p,source,dev,final,models,oof,anchor=run.prepare();lock=json.loads((run.ROOT/'locked_before_final.json').read_text());r=json.loads((run.ROOT/'results_public.json').read_text());saved=np.load(str(run.ROOT/'scores_private.npz'));y=p.labels['auth_train']==3;folds=list(s.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,p.words))
for key,start in [('baseline_raw_optional',{'base':1.}),('baseline_teacher_raw_optional',anchor)]:
 ww,meta=s.sel.forward_select(oof,y,folds,['raw'],start,1);assert ww==lock['weights'][key] and meta==lock['selection'][key]
for key,row in r['routes'].items():
 weights=lock['weights'][key]
 def score(src,role):
  branches=[]
  for name,weight in weights.items():
   model=models[name];scaler,svc=model['model'];x=src[role+'__'+name];assert scaler[0].n_samples_seen_==450
   values=s.sel.rbf.score(model['model'],[x]);branches.append(weight*(values-model['mean'])/model['std'])
  return sum(branches)
 g=score(final,'final_genuine');n=score(final,'final_impostor');cg=score(dev,'target_dev_pool');cn=score(dev,'impostor_calibration');t=b.balanced_threshold(cg,cn);assert t==lock['thresholds'][key]
 assert np.allclose(g,saved[key+'__genuine']) and np.allclose(n,saved[key+'__impostor'])
 assert len(g)==47 and len(n)==394 and abs((g>=t).mean()-row['tar'])<1e-12 and abs((n>=t).mean()-row['far'])<1e-12
 auc=((g[:,None]>n)+.5*(g[:,None]==n)).mean();assert abs(auc-row['auc'])<1e-12
 assert abs((1-row['tar']+row['far'])/2-row['hter'])<1e-12 and len(row['attack_by_user'])==8
 for user,counts in row['attack_by_user'].items():
  z=n[p.labels['final_impostor']==int(user)-1];assert len(z)==counts['n'] and int((z>=t).sum())==counts['accepted']
run.s.save(run.ROOT/'verification_public.json',dict(status='PASS',routes=6,checks=['base772 and raw320 cache versus protocol inputs','train-only 450sample branchscalers','trainingOOF-only raw weight replay','calibration jointHTER thresholds','recomputed frozen branch scores counts AUC all8attackers']))
print('PASS',flush=True)
