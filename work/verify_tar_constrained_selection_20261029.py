"""Replay constrained OOF choice, calibration quantile, and final score counts."""
import sys,json
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_tar_constrained_selection_20261029 as run
prior=run.prior;old=run.old
for u in [4,5]:
 folder=run.ROOT/('U%d'%u);src=prior.ROOT/('U%d'%u);auth=prior.AUTH80/('U%d'%u);p=old.protocol(u)
 lock=json.loads((folder/'locked_before_final.json').read_text());result=json.loads((folder/'results_public.json').read_text());saved=np.load(str(folder/'scores_private.npz'));bo=np.load(str(auth/'oof_private.npz'));to=np.load(str(src/'oof_private.npz'))
 y=np.concatenate([p.labels[r] for r in ['auth_train','model_validation']])==u-1;words=np.concatenate([prior.first.words(p,r) for r in ['auth_train','model_validation']]);folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 for c in lock['candidates']:
  s=bo['base']+c['raw_weight']*bo['raw']+(c['teacher_weight']*to[c['teacher_key']] if c['teacher_weight'] else 0.)
  q=run.oof_quality(s,y,words,folds)
  for k in ['tar','far','auc','hter','shortfall']:assert abs(q[k]-c['quality'][k])<1e-12
 assert min(lock['candidates'],key=run.rank)==lock['chosen']==result['chosen']
 cached=old.sel.joblib.load(str(auth/'models_private.joblib'));basefeatures=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'));basefinal=np.load(str(auth/'scores_private.npz'));teachers={};models={}
 def standardized(m,x):return (old.sel.rbf.score(m['model'],[x])-m['mean'])/m['std']
 def score(spec,role):
  s=basefinal['baseline_rbf__'+role].copy() if role.startswith('final_') else standardized(cached['base'],basefeatures[role+'__base'])
  if spec['raw_weight']:
   x=p.arrays[role];raw=np.concatenate([x.mean(2),x.std(2),x.mean(1),x.std(1)],1);s+=spec['raw_weight']*standardized(cached['raw'],raw)
  if spec['teacher_weight']:
   key=spec['teacher_key']
   if key not in teachers:
    ck=torch.load(str(src/(key+'_teacher_private.pt')),map_location=prior.D,weights_only=False);t=old.base.Teacher().to(prior.D);t.load_state_dict(ck['state']);t.eval();teachers[key]=t;models[key]=old.sel.joblib.load(str(src/(key+'_rbf_private.joblib')))
   s+=spec['teacher_weight']*standardized(models[key],prior.embed(teachers[key],p.arrays[role]))
  return s
 for key,spec in lock['routes'].items():
  if key.endswith('tar90'):
   g=score(spec,'target_dev_pool');n=score(spec,'impostor_calibration');t=float(np.sort(g)[len(g)-int(np.ceil(.9*len(g)-1e-9))]);assert t==spec['threshold'];assert (g>=t).mean()>=.9
   assert abs((g>=t).mean()-spec['calibration_tar'])<1e-12 and abs((n>=t).mean()-spec['calibration_far'])<1e-12
  for role in ['final_genuine','final_impostor']:assert np.allclose(score(spec,role),saved[key+'__'+role],atol=1e-7,rtol=1e-6)
  r=result['routes'][key];g=saved[key+'__final_genuine'];n=saved[key+'__final_impostor'];t=spec['threshold']
  assert len(g)==r['genuine_n'] and len(n)==r['impostor_n'] and abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12 and len(r['attack_by_user'])==8
  for v,a in r['attack_by_user'].items():
   z=n[saved['attack_labels']==int(v)-1];assert len(z)==a['n'] and int((z>=t).sum())==a['accepted']
 print('VERIFIED',u,flush=True)
old.save(run.ROOT/'verification_public.json',dict(status='PASS',targets=[4,5],routes=10,checks=['40 OOF candidates pertarget replay','training-only chosen fusion replay','independent quantile90 formula','calibration TAR/FAR replay','independent saved models final inference','AUC/counts all8sources']))

