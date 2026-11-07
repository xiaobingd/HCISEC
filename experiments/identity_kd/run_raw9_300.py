"""Full raw nine-user leave-one-user-out ACC authentication with frozen ECAPA KD."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
import argparse, copy, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root')
import run_pilot as p
from run_ecapa_pilot import teacher_load, EXPECTED, sha

NAMES=['bo','bro','jin','ke','mei','qi','qis','xin','yue']
BRANCHES=['acc_vel_300','acc_deriv_300']
BETAS=[0.0,0.1,0.5]
KS=[1,5,10,20]

class Student(nn.Module):
 def __init__(self):
  super().__init__(); self.body=p.resnet18(weights=None); self.body.fc=nn.Linear(512,192)
 def forward(self,x): return F.normalize(self.body(x),dim=1)

@torch.no_grad()
def encode(net,x):
 net.eval(); out=[]
 for i in range(0,len(x),64): out.append(net(torch.as_tensor(x[i:i+64],device=p.DEVICE)).cpu().numpy())
 return np.concatenate(out)

def norm(a): return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def pair_auc(z,y,w):
 sim=z@z.T; upper=np.triu(np.ones(sim.shape,bool),1)
 pos=upper&(y[:,None]==y[None,:])&(w[:,None]!=w[None,:])
 neg=upper&(y[:,None]!=y[None,:])&(w[:,None]==w[None,:])
 if not pos.any() or not neg.any(): return 0.5
 return float(roc_auc_score(np.r_[np.ones(pos.sum()),np.zeros(neg.sum())],np.r_[sim[pos],sim[neg]]))

def load_data(npz,branch):
 a=np.load(npz,allow_pickle=False)
 return {k:a[k] for k in a.files}

def scale_branch(d,branch,fit_users):
 ix=np.isin(d['user'],fit_users)&(d['role']=='fit'); x=d[branch].astype('float32')
 lo,hi=np.percentile(x[ix],[1,99]); y=np.clip(2*(x-lo)/(hi-lo)-1,-1,1)
 return y.astype('float32'),dict(low=float(lo),high=float(hi),fit_ids=d['sample_id'][ix].tolist())

@torch.no_grad()
def audio_targets(model,d,fit_users,x):
 ix=np.flatnonzero(np.isin(d['user'],fit_users)&(d['role']=='fit'))
 feats=np.zeros((len(d['user']),192),dtype='float32')
 for s in range(0,len(ix),16):
  jj=ix[s:s+16]; w=torch.as_tensor(d['wave'][jj].astype('float32'),device=p.DEVICE)
  z=model.encode_batch(w).squeeze(1); feats[jj]=F.normalize(z,dim=-1).cpu().numpy()
 return feats,ix

def train(x,d,fit_users,tf,beta,out,steps=200):
 p.seed(42);net=Student().to(p.DEVICE);head=nn.Linear(192,len(fit_users)).to(p.DEVICE)
 params=list(net.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
 rng=np.random.RandomState(42);fit=np.flatnonzero(np.isin(d['user'],fit_users)&(d['role']=='fit'))
 dev=np.flatnonzero(np.isin(d['user'],fit_users)&(d['role']=='develop'))
 words=np.unique(d['word'][fit]); lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in fit_users for w in words}
 assert all(len(lookup[k])>=2 for k in lookup)
 best=(-1,None,None);hist=[]
 for step in range(1,steps+1):
  net.train(); ww=rng.choice(words,2,replace=False)
  ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in fit_users for w in ww])
  yy=torch.as_tensor([fit_users.index(d['user'][i]) for i in ix],device=p.DEVICE)
  z=net(torch.as_tensor(x[ix],device=p.DEVICE));ce=F.cross_entropy(head(z),yy);sc=p.sc_loss(z,yy)
  kd=z.new_zeros(())
  if beta:
   t=torch.as_tensor(tf[ix],device=p.DEVICE);kd=(1-(z*t).sum(1)).mean()
  loss=ce+.1*sc+beta*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
  if step%50==0:
   emb=encode(net,x[dev]);q=pair_auc(emb,d['user'][dev],d['word'][dev]);hist.append(dict(step=step,ce=float(ce.detach()),supcon=float(sc.detach()),kd=float(kd.detach()),dev_pair_auc=q))
   if q>best[0]:best=(q,copy.deepcopy(net.state_dict()),step)
 net.load_state_dict(best[1]);net.eval();out.mkdir(parents=True,exist_ok=True)
 torch.save({'state':best[1],'selected_step':best[2]},out/'student_private.pt')
 (out/'training_private.json').write_text(json.dumps(dict(beta=beta,steps=steps,fit_users=fit_users,fit_ids=d['sample_id'][fit].tolist(),dev_ids=d['sample_id'][dev].tolist(),selected_step=best[2],history=hist),indent=2))
 return net,best[2],best[0]

def scores_for(net,x,d,target,users,k,seed):
 en=np.flatnonzero((d['user']==target)&(d['role']=='enroll'));te=np.flatnonzero((d['role']=='final')&np.isin(d['user'],users))
 ze=encode(net,x[en]);zt=encode(net,x[te]);rng=np.random.RandomState(seed); words=np.unique(d['word'][en]); picks=[]
 shuffled={w:list(rng.permutation(np.flatnonzero(d['word'][en]==w))) for w in words}
 while len(picks)<k:
  for w in rng.permutation(words):
   if shuffled[w]: picks.append(int(shuffled[w].pop()))
   if len(picks)==k: break
 proto=norm(ze[picks].mean(0));return zt@proto,d['user'][te]==target,d['user'][te]

def eer_threshold(g,n):
 vals=np.unique(np.r_[g,n]); vals=np.r_[vals,np.nextafter(vals[-1],np.inf)]
 return float(min(vals,key=lambda t:(abs((g<t).mean()-(n>=t).mean()),(g<t).mean()+(n>=t).mean(),-t)))

def main(a):
 root=Path(a.out);root.mkdir(parents=True,exist_ok=True);d=load_data(a.data,BRANCHES[0]);
 model=teacher_load(root)
 summary={'scope':'full raw nine-user; outer target excluded from ACC classifier and threshold calibration','branches':{},'teacher_hashes':EXPECTED,'outer_users':NAMES,'methods':BETAS,'K':KS,'rows':int(len(d['user']))}
 for branch in BRANCHES:
  summary['branches'][branch]={}
  for target in range(1,10):
   others=[u for u in range(1,10) if u!=target];cal=others[(target-1)%len(others)];fit_users=[u for u in others if u!=cal]
   x,scale=scale_branch(d,branch,fit_users);tf=np.zeros((len(d['user']),192),dtype='float32');tt,ti=audio_targets(model,d,fit_users,x);tf[:]=tt
   fold=root/branch/NAMES[target-1];fold.mkdir(parents=True,exist_ok=True)
   (fold/'scaling_private.json').write_text(json.dumps(dict(branch=branch,target=NAMES[target-1],fit_users=[NAMES[u-1] for u in fit_users],calibration_user=NAMES[cal-1],**scale)))
   candid=[]
   for beta in BETAS:
    net,step,devauc=train(x,d,fit_users,tf,beta,fold/('beta_'+str(beta)),a.steps)
    all_scores=[];all_pos=[];all_neg=[]
    for k in [5]:
     for s in range(10):
      sc,pos,who=scores_for(net,x,d,cal,range(1,10),k,4200+s);all_scores.extend(sc.tolist());all_pos.extend(pos.tolist());
      all_neg.extend((~pos).tolist())
    ar=np.asarray(all_scores);pp=np.asarray(all_pos);tau=eer_threshold(ar[pp],ar[~pp]);tar=float((ar[pp]>=tau).mean());far=float((ar[~pp]>=tau).mean());hter=(1-tar+far)/2
    candid.append(dict(beta=beta,net=net,step=step,dev_pair_auc=devauc,threshold=tau,cal_tar=tar,cal_far=far,cal_hter=hter))
   chosen=min(candid,key=lambda z:(z['cal_hter'],-z['dev_pair_auc'],z['beta']))
   result={'fit_users':[NAMES[u-1] for u in fit_users],'calibration_user':NAMES[cal-1],'selected_beta':chosen['beta'],'selected_step':chosen['step'],'calibration':{k:chosen[k] for k in ['threshold','cal_tar','cal_far','cal_hter','dev_pair_auc']},'candidates':[{k:v for k,v in c.items() if k not in ['net']} for c in candid],'K':{}}
   score_dump={}
   for k in KS:
    reps=[]
    for s in range(10):
     sc,pos,who=scores_for(chosen['net'],x,d,target,range(1,10),k,4200+s);tau=chosen['threshold'];tar=float((sc[pos]>=tau).mean());far=float((sc[~pos]>=tau).mean());auc=float(roc_auc_score(pos.astype(int),sc));reps.append(dict(seed=4200+s,tar=tar,far=far,hter=(1-tar+far)/2,auc=auc,genuine_n=int(pos.sum()),impostor_n=int((~pos).sum())))
     score_dump['K'+str(k)+'_s'+str(s)]=dict(scores=sc.tolist(),is_genuine=pos.tolist(),user_ids=who.tolist())
    result['K'][str(k)]={m:float(np.mean([r[m] for r in reps])) for m in ['tar','far','hter','auc']};result['K'][str(k)]['repetitions']=reps
   (fold/'result_private.json').write_text(json.dumps(result,indent=2));(fold/'scores_private.json').write_text(json.dumps(score_dump))
   summary['branches'][branch][NAMES[target-1]]=result
   print('FOLD_DONE',branch,NAMES[target-1],result['selected_beta'],result['K'],flush=True)
   del candid,tf,x
 (root/'results_private.json').write_text(json.dumps(summary,indent=2));print('RAW9_COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=200);main(ap.parse_args())
