"""Exploratory training-only repeated word-OOF error trimming; fixed full tests."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json, hashlib
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import joblib

ROOT=Path('/autodl-fs/data/mei_error_trim_20261028')
CACHE=Path('/autodl-fs/data/pira_stage1_scaleonly_20261028')
SOURCE=json.loads(Path('/autodl-fs/data/complete_baseline_teacher_20261028/source_manifest.json').read_text())
TARGETS=(1,2,4,5,7); UNKNOWN=(3,8); SEEDS=(42,43,44)
NAMES=['bo']+['user%d'%v for v in range(2,10)]

def save(p,obj):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(v,role):
 a=np.asarray(SOURCE[NAMES[v-1]]['auth_indices'],dtype=int);w=a//10
 assert not set(a)&set(SOURCE[NAMES[v-1]]['cyclegan_indices'])
 return a[w<15] if role=='fit' else a[(w>=15)&(w<20)] if role=='cal' else a[w>=20]
def fit(blocks,y,c,g):
 scalers=[StandardScaler().fit(x) for x in blocks]
 z=np.concatenate([s.transform(x)/np.sqrt(x.shape[1]) for s,x in zip(scalers,blocks)],1)
 return scalers,SVC(C=c,gamma=g,kernel='rbf',class_weight='balanced').fit(z,y)
def score(model,blocks):
 ss,m=model;return m.decision_function(np.concatenate([s.transform(x)/np.sqrt(x.shape[1]) for s,x in zip(ss,blocks)],1))
def cut(pos,neg):
 v=np.unique(np.r_[pos,neg]);ts=np.r_[v[0]-1e-6,(v[:-1]+v[1:])/2,v[-1]+1e-6]
 return float(min(ts,key=lambda t:(abs(np.mean(neg>=t)-np.mean(pos<t)),(np.mean(neg>=t)+np.mean(pos<t))/2)))
def metrics(y,s,t):
 p=y.astype(bool);a=int(np.sum(s[p]>=t));b=int(np.sum(s[~p]>=t))
 return dict(genuine_count=int(p.sum()),genuine_accepted=a,attack_count=int((~p).sum()),attack_accepted=b,TAR=a/int(p.sum()),FAR=b/int((~p).sum()),AUC=float(roc_auc_score(y,s)),threshold=float(t))
def blocks(cache,users,role,mode):
 ids=np.concatenate([np.full(len(rows(v,role)),v) for v in users]);idx=np.concatenate([rows(v,role) for v in users])
 keys=['base'] if mode=='baseline772' else ['base','teacher']
 xs=[np.concatenate([cache['U%d_%s'%(v,k)][rows(v,role)] for v in users]) for k in keys]
 return xs,ids,idx
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 protocol=dict(targets=list(TARGETS),unknown=list(UNKNOWN),roles=dict(fit_words=[0,14],cal_words=[15,19],test_words=[20,29]),features=['baseline772','baseline772_teacher64'],screening_seeds=list(SEEDS),error_boundary=0,minimum_error_votes=2,per_identity_removal_cap=.1,parameter_selection='5 word-grouped folds seed42 mean AUC on complete authentication fit; identical chosen C/gamma for paired filtered/unfiltered arms',filter_selection='fixed rule; neither calibration nor final chooses removals or filtering arm',caveats=['Historical test recordings reused: exploratory, not blind','Upstream150 random rows saw test word categories: authentication-stage content holdout only','OOF removes training difficult samples, not proven sensor noise','U4/U7 missing rows assumed at end; word mapping provisional'])
 save(ROOT/'protocol.json',protocol);result={};scores={};removals={}
 for u in TARGETS:
  folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True)
  source=CACHE/('U%d'%u)/'feature_cache.npz';cache=dict(np.load(str(source)))
  known=[v for v in range(1,10) if v not in UNKNOWN]
  target_result={};remove_private={}
  for mode in protocol['features']:
   x,uid,idx=blocks(cache,known,'fit',mode);y=uid==u;words=idx//10
   folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(x[0],y,words))
   candidates=[]
   for c in (1.,10.,100.):
    for g in ('scale',.1):
     auc=[]
     for tr,va in folds:
      assert not set(words[tr])&set(words[va])
      m=fit([a[tr] for a in x],y[tr],c,g);auc.append(float(roc_auc_score(y[va],score(m,[a[va] for a in x]))))
     candidates.append(dict(C=c,gamma=g,mean_fold_auc=float(np.mean(auc))))
   config=max(candidates,key=lambda r:(r['mean_fold_auc'],-r['C'],r['gamma']=='scale'))
   votes=np.zeros(len(y),dtype=int);margins=[]
   for seed in SEEDS:
    oof=np.full(len(y),np.nan)
    for tr,va in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(x[0],y,words):
     assert not set(words[tr])&set(words[va])
     m=fit([a[tr] for a in x],y[tr],config['C'],config['gamma']);oof[va]=score(m,[a[va] for a in x])
    assert np.isfinite(oof).all();votes+=(oof>=0)!=y;margins.append(np.where(y,oof,-oof))
   margin=np.mean(margins,axis=0);keep=np.ones(len(y),dtype=bool);removed_counts={}
   for v in known:
    loc=np.flatnonzero(uid==v);bad=loc[votes[loc]>=2];order=np.lexsort((idx[bad],margin[bad],-votes[bad]));drop=bad[order[:int(np.floor(.1*len(loc)))]]
    keep[drop]=False;removed_counts[str(v)]=dict(total=len(loc),candidates=len(bad),removed=len(drop))
   assert all(np.any(keep&(uid==v)) for v in known)
   remove_private[mode]=[dict(user=int(uid[i]),row=int(idx[i]),votes=int(votes[i]),mean_signed_margin=float(margin[i])) for i in np.flatnonzero(~keep)]
   xc,uc,ic=blocks(cache,known,'cal',mode);yc=uc==u
   locks={};models={}
   for arm,mask in [('unfiltered',np.ones(len(y),bool)),('trimmed',keep)]:
    m=fit([a[mask] for a in x],y[mask],config['C'],config['gamma']);s=score(m,xc);t=cut(s[yc],s[~yc]);models[arm]=m
    locks[arm]=dict(C=config['C'],gamma=config['gamma'],threshold=t,fit_rows=int(mask.sum()),fit_positive=int(np.sum(y&mask)),fit_negative=int(np.sum(~y&mask)),calibration=metrics(yc,s,t))
   save(folder/(mode+'_locked.json'),dict(config=config,candidates=candidates,arms=locks,removed_by_identity=removed_counts,feature_cache_sha256=digest(source)))
   joblib.dump(models,str(folder/(mode+'_models.joblib')))
   # No calibration/final row influences the OOF votes or drop lists. Both arms locked before queries.
   xt,ut,it=blocks(cache,list(range(1,10)),'test',mode);yt=ut==u;unknown=np.isin(ut,UNKNOWN);seen=(ut!=u)&~unknown
   final={}
   for arm in models:
    s=score(models[arm],xt);t=locks[arm]['threshold'];key='U%d_%s_%s'%(u,mode,arm)
    scores[key+'_scores']=s;scores[key+'_users']=ut;scores[key+'_rows']=it
    final[arm]=dict(all=metrics(yt,s,t),unknown=metrics(yt[yt|unknown],s[yt|unknown],t),known=metrics(yt[yt|seen],s[yt|seen],t),by_attacker={str(v):dict(count=int(np.sum(ut==v)),accepted=int(np.sum(s[ut==v]>=t)),FAR=float(np.mean(s[ut==v]>=t))) for v in range(1,10) if v!=u},fit=locks[arm])
   target_result[mode]=dict(routes=final,removed_by_identity=removed_counts,config=config)
   print('MODE_DONE',u,mode,final['unfiltered']['unknown'],final['trimmed']['unknown'],flush=True)
  result[str(u)]=target_result;removals[str(u)]=remove_private
  save(ROOT/'results_public.json',result);save(ROOT/'removed_training_private.json',removals);np.savez_compressed(str(ROOT/'scores_private.npz'),**scores)
 save(ROOT/'COMPLETE.json',dict(targets=len(result),routes=20))
 print('ERROR_TRIM_COMPLETE',flush=True)
if __name__=='__main__':main()
