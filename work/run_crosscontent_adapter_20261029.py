"""Frozen772 -> Linear128 GELU Linear64 L2; cross-word and enrollment-query ablations."""
import sys,json,os,copy,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_strict_autosupplement_20261029 as old
ROOT=Path('/autodl-fs/data/mei_crosscontent_adapter_20261029')
FINAL=Path('/autodl-fs/data/mei_auth80_free_selection_20261029')
SEEDS=[42,43,44];MODES=['ordinary_supcon','crossword_only','episode_only','combined'];STEPS=400
DEVICE=old.D
class Adapter(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(772,128),nn.GELU(),nn.Linear(128,64))
 def forward(self,x):return F.normalize(self.net(x),dim=1)
def save(p,x):old.save(p,x)
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def contrast(z,labels,words,crossword):
 same=labels[:,None]==labels[None,:];sameword=words[:,None]==words[None,:]
 eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
 pos=same&~eye
 if crossword:pos=pos&~sameword
 valid=~eye if not crossword else (pos|~same)
 logits=z@z.T/.07
 if crossword:logits=logits+((~same)&sameword).float()*np.log(2.)
 logp=logits-torch.logsumexp(logits.masked_fill(~valid,-1e9),dim=1,keepdim=True)
 assert torch.all(pos.sum(1)>0)
 return -((logp*pos).sum(1)/pos.sum(1)).mean()
def episode(z,labels_np,words_np,rng):
 sw,qw=rng.choice(np.unique(words_np),2,replace=False)
 support=[];queries=[];targets=[]
 for uid in range(9):
  support.append(rng.choice(np.flatnonzero((labels_np==uid)&(words_np==sw)),2,replace=False))
  queries.extend(rng.choice(np.flatnonzero((labels_np==uid)&(words_np==qw)),2,replace=False).tolist());targets.extend([uid]*2)
 assert sw!=qw and not set(np.concatenate(support))&set(queries)
 centers=F.normalize(torch.stack([z[ix].mean(0) for ix in support]),dim=1)
 return F.cross_entropy(z[queries]@centers.T/.1,torch.as_tensor(targets,device=z.device))
def centers(z,labels):
 if isinstance(z,np.ndarray):
  c=np.stack([z[labels==v].mean(0) for v in range(9)]);return c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-12)
 return F.normalize(torch.stack([z[torch.as_tensor(labels==v,device=z.device)].mean(0) for v in range(9)]),dim=1)
def quality(score,y,words):
 values=[roc_auc_score(y[words==w],score[words==w]) for w in sorted(np.unique(words))]
 return float(np.mean(values)),float(roc_auc_score(y,score))
def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True);p=old.protocol(u)
 dev=np.load(str(old.ROOT/('U%d'%u)/'development_features_private.npz'))
 x=dev['auth_train__base'];v=dev['model_validation__base'];labels=p.labels['auth_train'];val_labels=p.labels['model_validation'];y=labels==u-1
 def word_ids(role):
  return np.asarray([old.M['sources'][s.split(':')[0]]['content_groups'][int(s.split(':')[1])] for s in p.sample_ids[role]])
 words=word_ids('auth_train');valwords=word_ids('model_validation')
 assert len(x)==450 and y.sum()==50 and len(v)==270 and (val_labels==u-1).sum()==30
 stagegroups=[set(word_ids(role)) for role in ['auth_train','model_validation','target_dev_pool','final_genuine']]
 for i in range(len(stagegroups)):
  for j in range(i):assert not stagegroups[i]&stagegroups[j]
 all_auth=set(sum([p.sample_ids[k] for k in ['auth_train','model_validation','target_dev_pool','impostor_calibration','final_genuine','final_impostor']],[]))
 upstream={str(vv)+':'+str(i) for vv in old.USERS for role in ['gan_fit','gan_val'] for i in old.M['sources'][str(vv)]['roles'][role]}
 assert not all_auth&upstream
 scaler=StandardScaler().fit(x);xx=tensor(scaler.transform(x));vv=tensor(scaler.transform(v));tl=torch.as_tensor(labels,device=DEVICE);tw=torch.as_tensor(words,device=DEVICE)
 old.sel.joblib.dump(scaler,str(folder/'scaler_private.joblib'));locks={};models={};history={}
 # Matched RBF uses the same held-out validation content as adapter checkpoint selection.
 best=None
 for c in [1.,10.,100.]:
  for gamma in ['scale',.1]:
   fitted=old.sel.rbf.fit([x],y,c,gamma);s=old.sel.rbf.score(fitted,[v]);q=quality(s,val_labels==u-1,valwords)
   rank=(-q[0],-q[1],c,0 if gamma==.1 else 1)
   if best is None or rank<best[0]:best=(rank,fitted,c,gamma,q)
 models['rbf_matched']=best[1];locks['rbf_matched']=dict(C=best[2],gamma=best[3],validation_sameword_auc=best[4][0],validation_auc=best[4][1])
 old.sel.joblib.dump(best[1],str(folder/'rbf_private.joblib'))
 for mode in MODES:
  for ss in SEEDS:
   seed(ss);net=Adapter().to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001);rng=np.random.RandomState(ss)
   best=None;hist=[]
   for step in range(1,STEPS+1):
    net.train();z=net(xx)
    if mode=='ordinary_supcon':loss=contrast(z,tl,tw,False)
    elif mode=='crossword_only':loss=contrast(z,tl,tw,True)
    elif mode=='episode_only':loss=episode(z,labels,words,rng)
    else:loss=contrast(z,tl,tw,True)+episode(z,labels,words,rng)
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    if step%25==0:
     net.eval()
     with torch.no_grad():
      cc=centers(net(xx),labels);zs=net(vv);scores=(zs@cc[u-1]).cpu().numpy();q=quality(scores,val_labels==u-1,valwords)
      ce=float(F.cross_entropy(zs@cc.T/.1,torch.as_tensor(val_labels,device=DEVICE)))
     hist.append(dict(step=step,training_loss=float(loss.detach()),validation_sameword_auc=q[0],validation_auc=q[1],validation_multiclass_ce=ce))
     rank=(-q[0],-q[1],ce,step)
     if best is None or rank<best[0]:best=(rank,copy.deepcopy(net.state_dict()),step,q)
   net.load_state_dict(best[1]);net.eval()
   for par in net.parameters():par.requires_grad_(False)
   key=mode+'_seed%d'%ss;models[key]=net;history[key]=hist
   locks[key]=dict(mode=mode,seed=ss,selected_step=best[2],validation_sameword_auc=best[3][0],validation_auc=best[3][1])
   torch.save(dict(state=best[1],metadata=locks[key]),str(folder/(key+'_private.pt')))
   print('ADAPTER_DONE',u,key,best[2],best[3],flush=True)
 def score(key,raw):
  if key=='rbf_matched':return old.sel.rbf.score(models[key],[raw])
  with torch.no_grad():
   if key=='raw772_cosine':
    zz=F.normalize(tensor(scaler.transform(raw)),dim=1);cc=centers(F.normalize(xx,dim=1),labels)
   else:zz=models[key](tensor(scaler.transform(raw)));cc=centers(models[key](xx),labels)
   return (zz@cc[u-1]).cpu().numpy()
 locks['raw772_cosine']=dict(feature=772,training='none')
 for key in locks:
  g=score(key,dev['target_dev_pool__base']);n=score(key,dev['impostor_calibration__base'])
  locks[key]['threshold']=float(old.base.original.choose_threshold(g,n))
 save(folder/'locked_before_final.json',dict(target=u,routes=locks,history=history,final_scores_seen=False))
 # Models, checkpoints and calibration thresholds are all locked before final cache access.
 final=np.load(str(FINAL/('U%d'%u)/'final_features_private.npz'));results={};dump={}
 for key,spec in locks.items():
  scores={role:score(key,final[role+'__base']) for role in ['final_genuine','final_impostor']}
  result=old.sel.evaluate_route(scores,p,u,spec['threshold']);result.update(spec)
  wg=word_ids('final_genuine');wn=word_ids('final_impostor');result['per_word']={}
  for w in sorted(np.unique(wg)):
   result['per_word'][str(w)]=old.sel.exp.metrics(scores['final_genuine'][wg==w],scores['final_impostor'][wn==w],spec['threshold'])
  result['same_word_mean_auc']=float(np.mean([r['auc'] for r in result['per_word'].values()]));results[key]=result
  for role,s in scores.items():dump[key+'__'+role]=s
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 save(folder/'results_public.json',dict(target=u,routes=results));print('TARGET_COMPLETE',u,flush=True)
def main():
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'QUEUE_LOCK').open('x') as f:f.write(str(os.getpid()))
 old.records()
 save(ROOT/'protocol_public.json',dict(targets=[1,4,5],seeds=SEEDS,modes=MODES,architecture='772->128 GELU->64 L2',steps=STEPS,optimizer='AdamW lr3e-4 wd1e-3 clip1',contrast='temperature.07;crossword positives sameID differentword;sameID sameword excluded;differentID sameword negatives weighted2',episode='9-way prototype CE temperature.1;one supportword and different queryword;2samples/identity on each side',combined_weights=[1,1],fit_per_user=50,validation_per_user=30,calibration_per_user=20,all_eight_attackers=True,upstream_frozen=True,checkpoint_selection='heldout validation content per-word meanAUC,then pooledAUC,then multiclassCE;no final',groups=old.M['groups'],limitations=['Single-session historical data and already observed final set','Three seeds do not create new users','Closed-set known identities,not unseen-user registration','Training prototypes also serve final enrollment;not separate enrollment recording pool','Content semantics still conditional on10/20row mapping and missing-tail assumption','Matched RBF uses validation content;not same hyperparameter selection as historical baseline'],source_sha256=old.sha(Path(__file__))))
 rows={}
 for u in [1,4,5]:
  save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));target(u)
  rows[str(u)]=json.loads((ROOT/('U%d'%u)/'results_public.json').read_text());save(ROOT/'results_public.json',rows)
 save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':main()

