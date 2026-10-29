"""Train-fold background-relative cosine prototype authentication; no new upstream training."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_background_prototype_20261029')
PREV=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
LAMBDAS=[0.,.25,.5,1.,1.5,2.]
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def fit(x,ids):
 scaler=StandardScaler().fit(x);z=unit(scaler.transform(x));centers=unit(np.stack([z[ids==v].mean(0) for v in range(9)]));return scaler,centers
def pair(model,x,u):
 scaler,centers=model;sim=unit(scaler.transform(x))@centers.T
 return sim[:,u-1],np.max(np.delete(sim,u-1,axis=1),axis=1)
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);p=old.protocol(u)
 dev=dict(np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz')))
 names=sorted(k.split('__',1)[1] for k in dev if k.startswith('auth_train__'))
 labels=np.concatenate([p.labels[k] for k in ('auth_train','model_validation')]);y=labels==u-1
 words=np.concatenate([np.asarray(old.M['sources'][str(v)]['content_groups'])[old.M['sources'][str(v)]['roles'][role]] for role in ('auth_train','model_validation') for v in old.USERS])
 ids=sum([p.sample_ids[k] for k in ('auth_train','model_validation')],[])
 blocked={str(v)+':'+str(i) for v in old.USERS for k in ('gan_fit','gan_val','calibration','test') for i in old.M['sources'][str(v)]['roles'][k]}
 assert len(ids)==len(set(ids))==720 and not set(ids)&blocked and y.sum()==80
 weights=json.loads((PREV/('U%d'%u)/'locked_before_queries.json').read_text())['weights']
 # Prior multi-branch weights are frozen, not selected using this experiment's final scores.
 descriptors={k:[(k,1.)] for k in names};descriptors['selected_fusion']=list(weights.items())
 def matrix(desc,role):return np.concatenate([np.sqrt(w)*dev[role+'__'+k] for k,w in desc],axis=1)
 # Fit each block's scaler and unit normalization to avoid dimension/scale domination in fusion.
 def prepare(desc,train,query):
  tr=[];qq=[]
  for k,w in desc:
   s=StandardScaler().fit(train[k]);tr.append(np.sqrt(w)*unit(s.transform(train[k])));qq.append(np.sqrt(w)*unit(s.transform(query[k])))
  z=unit(np.concatenate(tr,1));q=unit(np.concatenate(qq,1));centers=unit(np.stack([z[train['labels']==v].mean(0) for v in range(9)]));sim=q@centers.T
  return sim[:,u-1],np.max(np.delete(sim,u-1,axis=1),axis=1)
 train={k:np.concatenate([dev[role+'__'+k] for role in ('auth_train','model_validation')]) for k in names};train['labels']=labels
 folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
 locks={};oofsave={}
 for name,desc in descriptors.items():
  target_score=np.zeros(len(y));background=np.zeros(len(y));aucs=[]
  for tr,va in folds:
   assert not set(words[tr])&set(words[va])
   a,b=prepare(desc,{k:v[tr] for k,v in train.items()},{k:v[va] for k,v in train.items()});target_score[va]=a;background[va]=b
  candidates=[]
  for lam in LAMBDAS:
   score=target_score-lam*background;values=[float(roc_auc_score(y[va],score[va])) for _,va in folds]
   candidates.append(dict(lam=lam,mean_auc=float(np.mean(values)),fold_aucs=values))
  best=min(candidates,key=lambda r:(-r['mean_auc'],r['lam']))
  oofsave[name+'__target']=target_score;oofsave[name+'__background']=background
  locks[name]=dict(descriptor=desc,lambda_selected=best['lam'],candidates=candidates)
 # Choose a single representation from the twelve individual branches using TRAIN CV only.
 auto_cos=min(names,key=lambda n:(-locks[n]['candidates'][0]['mean_auc'],n))
 auto_relative=min(names,key=lambda n:(-max(r['mean_auc'] for r in locks[n]['candidates']),n))
 routes={}
 for name in descriptors:
  for mode,lam in [('cosine',0.),('relative',locks[name]['lambda_selected']),('relative_fixed1',1.)]:
   desc=descriptors[name];cal={}
   for role in ('target_dev_pool','impostor_calibration'):
    a,b=prepare(desc,train,{k:dev[role+'__'+k] for k in names});cal[role]=a-lam*b
   threshold=float(old.base.original.choose_threshold(cal['target_dev_pool'],cal['impostor_calibration']))
   routes[name+'__'+mode]=dict(feature=name,lam=lam,threshold=threshold)
 save(folder/'locked_before_queries.json',dict(target=u,representations=locks,routes=routes,auto_cosine=auto_cos,auto_relative=auto_relative,final_seen=False,background_users=[v for v in range(1,10) if v!=u],fit_per_identity=80))
 np.savez_compressed(str(folder/'oof_private.npz'),**oofsave)
 # Only access existing final feature cache after all new choices and thresholds lock.
 final=dict(np.load(str(PREV/('U%d'%u)/'final_features_private.npz')));results={};dump={}
 for route,locked in routes.items():
  ss={}
  for role in ('final_genuine','final_impostor'):
   a,b=prepare(descriptors[locked['feature']],train,{k:final[role+'__'+k] for k in names});ss[role]=a-locked['lam']*b;dump[route+'__'+role]=ss[role]
  r=old.sel.evaluate_route(ss,p,u,locked['threshold']);r.update(locked);results[route]=r
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 main={k:results[k] for name in ('base','identity_acc','selected_fusion') for k in [name+'__cosine',name+'__relative',name+'__relative_fixed1']}
 main['auto_cosine']=results[auto_cos+'__cosine'];main['auto_relative']=results[auto_relative+'__relative']
 ref=json.loads((PREV/('U%d'%u)/'results_public.json').read_text())
 save(folder/'results_public.json',dict(target=u,routes=main,diagnostic_routes=results,rbf_reference=ref['reference']['baseline_rbf'],free_selection_reference=ref['route'],prior_fusion_weights=weights))
 print('TARGET_DONE',u,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(__import__('os').getpid()))
 save(ROOT/'protocol_public.json',dict(method='Target cosine minus lambda maximum cosine to eight background identity prototypes; NOT AS-Norm',fit_per_identity=80,fit_negative=640,lambda_grid=LAMBDAS,selection='5 content-group folds TRAIN mean AUC, ties smaller lambda; optional train-only single-branch selector',preprocessing='Per branch StandardScaler fitted on training rows; L2 each block; weighted concatenation and L2; class mean normalized',fixed_representations=['base772','identity_acc64','previous train-selected weighted feature fusion'],upstream_frozen=True,all_eight_attacks=True,limitations=['Exploratory repeated final evaluation','Known negative identities not open set','Prior fusion weights chosen on full auth80 so fusion CV is conditional, not unbiased outer validation','Candidate selection CV is not an unbiased accuracy estimate','Content mapping10/20 and tail-missing assumptions'],source_sha256=old.sha(Path(__file__))))
 rows={}
 for u in old.ORDER:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));target(u)
  rows[str(u)]=json.loads((ROOT/('U%d'%u)/'results_public.json').read_text());save(ROOT/'results_public.json',rows)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':main()

