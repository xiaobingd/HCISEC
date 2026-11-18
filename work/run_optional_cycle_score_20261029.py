"""Keep original772 RBF intact; optional standardized cycle640 score supplement."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root')
import run_frozen_cycle_identity_compare_20261029 as first
old=first.old;trial=first.trial
ROOT=Path('/autodl-fs/data/mei_optional_cycle_score_20261029')
ALPHAS=[0.,.125,.25,.5,1.,2.]

def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);source=first.ROOT/('U%d'%u)
 p=old.protocol(u);dev=np.load(str(source/'development_private.npz'));scores={};normalizers={}
 source_lock=json.loads((source/'locked_before_final.json').read_text())
 for branch in ['base772','cycle640']:
  model=old.sel.joblib.load(str(source/(branch+'_private.joblib')))
  raw={r:old.sel.rbf.score(model,[dev[r+'__'+branch]]) for r in first.ROLES[:4]}
  mu=float(raw['auth_train'].mean());sd=float(raw['auth_train'].std()+1e-6)
  normalizers[branch]=dict(mean=mu,std=sd,source_C=source_lock['routes'][branch]['C'],source_gamma=source_lock['routes'][branch]['gamma'])
  scores[branch]={r:(s-mu)/sd for r,s in raw.items()}
 y=p.labels['model_validation']==u-1;w=first.words(p,'model_validation');candidates=[]
 for alpha in ALPHAS:
  s=scores['base772']['model_validation']+alpha*scores['cycle640']['model_validation'];q=trial.quality(s,y,w)
  candidates.append(dict(alpha=alpha,validation_sameword_auc=q[0],validation_auc=q[1]))
 selected=min(candidates,key=lambda r:(-r['validation_sameword_auc'],-r['validation_auc'],r['alpha']))
 routes={'baseline772':0.,'optional_cycle':selected['alpha']};locks={}
 for name,alpha in routes.items():
  g=scores['base772']['target_dev_pool']+alpha*scores['cycle640']['target_dev_pool'];n=scores['base772']['impostor_calibration']+alpha*scores['cycle640']['impostor_calibration']
  locks[name]=dict(alpha=alpha,threshold=float(old.base.original.choose_threshold(g,n)))
 old.save(folder/'locked_before_final.json',dict(target=u,routes=locks,normalizers=normalizers,candidates=candidates,selected=selected,final_seen=False))
 # Locked selection before final-score cache access. No re-training or final-based alpha changes.
 cache=np.load(str(source/'scores_private.npz'));results={};dump={}
 for name,spec in locks.items():
  ss={}
  for role in first.ROLES[4:]:
   b=(cache['base772__'+role]-normalizers['base772']['mean'])/normalizers['base772']['std']
   c=(cache['cycle640__'+role]-normalizers['cycle640']['mean'])/normalizers['cycle640']['std']
   ss[role]=b+spec['alpha']*c;dump[name+'__'+role]=ss[role]
  r=old.sel.evaluate_route(ss,p,u,spec['threshold']);r.update(spec)
  g=ss['final_genuine'];n=ss['final_impostor'];t=spec['threshold']
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12 and len(r['attack_by_user'])==8
  wg=first.words(p,'final_genuine');wn=first.words(p,'final_impostor')
  r['per_word']={str(word):old.sel.exp.metrics(g[wg==word],n[wn==word],t) for word in sorted(np.unique(wg))}
  r['same_word_mean_auc']=float(np.mean([v['auc'] for v in r['per_word'].values()]));results[name]=r
 reference=json.loads((source/'results_public.json').read_text())['routes']['base772']
 for k in ['tar','far','auc']:assert abs(results['baseline772'][k]-reference[k])<1e-12
 if selected['alpha']==0:
  for k in ['tar','far','auc']:assert results['baseline772'][k]==results['optional_cycle'][k]
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 out=dict(target=u,routes=results,selection=selected,candidates=candidates,baseline_replay_verified=True,score_count_verified=True)
 old.save(folder/'results_public.json',out);print('COMPLETE',u,json.dumps(out),flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);old.records()
 old.save(ROOT/'protocol_public.json',dict(targets=[1,4,5],alpha_grid=ALPHAS,base='unchanged stage1 full772 RBF',supplement='independent stage1 cycle640 RBF',normalization='each branch decision score mean/std from auth_train only',selection='heldout validation per-word meanAUC then pooledAUC then smaller alpha;0 allowed',calibration='independent calibration set after alpha lock',training='no generator,teacher,adapter or RBF retraining;reuse locked stage1 classifiers',limitations=['historical final set reused;exploratory known-user closed set','same validation set reused for branch RBF hyperparameters and fusion alpha','validation choice does not guarantee final non-degradation','content10/20period and missing-tail mapping','not formal AS-Norm;training score normalization only']))
 results={str(u):target(u) for u in [1,4,5]};old.save(ROOT/'results_public.json',results);old.save(ROOT/'COMPLETE.json',dict(status='PASS',targets=[1,4,5]))
if __name__=='__main__':main()

