"""Independent replay of same-word role balance, frozen model inference and TAR calibration."""
import sys,json
import numpy as np
import torch
sys.path.insert(0,'/root')
import run_sameword_tar_constraint_20261029 as run
old=run.old;prior=run.prior
M=run.same.manifest()
for u in [4,5]:
 folder=run.ROOT/('U%d'%u);data=run.dataset(u,M);lock=json.loads((folder/'locked_before_final.json').read_text());result=json.loads((folder/'results_public.json').read_text());saved=np.load(str(folder/'scores_private.npz'));oof=np.load(str(folder/'oof_private.npz'));models=old.sel.joblib.load(str(folder/'models_private.joblib'))
 y=data['auth_train']['labels']==u-1;reps=data['auth_train']['repeats'];folds=list(old.sel.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,reps));teachers={}
 for tr,va in folds:assert not set(reps[tr])&set(reps[va]) and set(data['auth_train']['words'][tr])==set(data['auth_train']['words'][va])==set(run.same.GROUPS)
 for c in lock['candidates']:
  s=oof['base']+c['raw_weight']*oof['raw']+(c['teacher_weight']*oof[c['teacher_key']] if c['teacher_weight'] else 0.)
  q=run.rule.oof_quality(s,y,reps,folds)
  for k in ['tar','far','auc','hter','shortfall']:assert abs(q[k]-c['quality'][k])<1e-12
 assert min(lock['candidates'],key=run.rule.rank)==lock['chosen']==result['chosen']
 for key,m in models.items():
  sc=m['model'][0][0];assert int(sc.n_samples_seen_)==675
  if key=='base':train=data['auth_train']['base']
  elif key=='raw':train=run.stats(data['auth_train']['a'])
  else:
   ck=torch.load(str(prior.ROOT/('U%d'%u)/(key+'_teacher_private.pt')),map_location=prior.D,weights_only=False);t=old.base.Teacher().to(prior.D);t.load_state_dict(ck['state']);t.eval();teachers[key]=t;train=prior.embed(t,data['auth_train']['a'])
  assert np.allclose(sc.mean_,train.mean(0),atol=2e-5,rtol=2e-5)
 def score(spec,role):
  d=data[role];s=run.standardized(models['base'],d['base'])
  if spec['raw_weight']:s+=spec['raw_weight']*run.standardized(models['raw'],run.stats(d['a']))
  if spec['teacher_weight']:s+=spec['teacher_weight']*run.standardized(models[spec['teacher_key']],prior.embed(teachers[spec['teacher_key']],d['a']))
  return s
 pos=data['test']['labels']==u-1;assert np.array_equal(saved['labels'],data['test']['labels']) and np.array_equal(saved['words'],data['test']['words'])
 for key,spec in lock['routes'].items():
  s=score(spec,'calibration');mask=data['calibration']['labels']==u-1;g=s[mask];n=s[~mask]
  t=float(np.sort(g)[len(g)-int(np.ceil(.9*len(g)-1e-9))]) if key.endswith('tar90') else float(old.base.original.choose_threshold(g,n));assert abs(t-spec['threshold'])<1e-12
  if key.endswith('tar90'):assert (g>=t).mean()>=.9
  prediction=score(spec,'test');assert np.allclose(prediction,saved[key],atol=1e-7,rtol=1e-6);g=prediction[pos];n=prediction[~pos];r=result['routes'][key]
  assert len(g)==30 and len(n)==240 and abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12 and len(r['attack_by_user'])==8
  for v,a in r['attack_by_user'].items():
   ss=prediction[data['test']['labels']==int(v)-1];assert len(ss)==a['n']==30 and int((ss>=t).sum())==a['accepted']
  for w in run.same.GROUPS:
   mm=data['test']['words']==w;wp=pos[mm];ss=prediction[mm];assert wp.sum()==2 and (~wp).sum()==16;pr=r['per_word'][run.same.WORDS[w]]
   assert abs(((ss[wp,None]>ss[~wp])+.5*(ss[wp,None]==ss[~wp])).mean()-pr['auc'])<1e-12
 print('VERIFIED',u,flush=True)
old.save(run.ROOT/'verification_public.json',dict(status='PASS',targets=[4,5],routes=10,checks=['same15words allidentities repeated roles balanced','auth rows disjoint from upstream','repeat-fold vocabulary match','all scaler675train rows only','40OOFfusion candidates and selected route','independent quantilecal30','saved models final inference','AUC and perword2genuine16attacks','all8sources30each']))

