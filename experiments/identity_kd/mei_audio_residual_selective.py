"""Closed Mei9: raw statistics plus identity/uniform/selective Audio relation residual.
Only training/registration Audio targets; CalA/B selects alpha, queries evaluate.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,roc_curve

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
def threshold(s,y):
 f,t,th=roc_curve(y,s);return float(th[np.argmin(abs(f-(1-t)))])
def metric(s,y,tau):
 tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());return dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s)))
def main(a):
 torch.set_num_threads(2);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=np.load(a.input);ix={k:np.array(v) for k,v in json.loads(str(d['split'])).items()};x=d['x'];raw=d['stats'];ti=d['indices'];y=d['user'];word=d['word'];target=norm(d['targets']);n=len(ti);assert not set(ti)&set(np.r_[ix['calA'],ix['calB'],ix['sameword'],ix['crossword']]);results=[];locks=[]
 # Reliability is cross-word agreement of train-only Audio identity prototypes.
 centers=norm(np.stack([target[y[ti]==u].mean(0) for u in range(1,10)]));reliability=[]
 for j,i in enumerate(ti):
  others=(y[ti]==y[i])&(word[ti]!=word[i]);reliability.append(max(0.,float(target[j]@norm(target[others].mean(0,keepdims=True))[0])))
 rel=np.array(reliability);same=y[ti,None]==y[ti][None,:];diffword=word[ti,None]!=word[ti][None,:];sameword=~diffword;mask=(same&diffword)|(~same&sameword);mask &= ~np.eye(n,dtype=bool)
 np.savez_compressed(out/'reliability_private.npz',indices=ti,reliability=rel,mask=mask)
 for seed in [42,43,44]:
  for mode in ['identity','uniform','selective','shuffle_same_user']:
   torch.manual_seed(seed);net=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Linear(128,64));head=nn.Linear(64,9);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-3);xx=torch.tensor(x[ti]);yy=torch.tensor(y[ti]-1,dtype=torch.long);t=target.copy()
   if mode=='shuffle_same_user':
    rng=np.random.RandomState(seed+1000)
    for u in range(1,10):j=np.flatnonzero(y[ti]==u);t[j]=t[rng.permutation(j)]
   tt=torch.tensor(t);tm=tt@tt.T;weights=np.ones((n,n),dtype='float32')
   if mode=='selective':
    rr=norm(x[ti]);gap=np.maximum(0.,(rr@rr.T)-(target@target.T));weights=np.sqrt(rel[:,None]*rel[None,:])*(1+gap)
   w=torch.tensor(weights*mask,dtype=torch.float32);mt=torch.tensor(mask);positive=torch.tensor(same&diffword);negative=torch.tensor((~same)&sameword)
   for step in range(400):
    z=F.normalize(net(xx),dim=1);sim=z@z.T;loss=F.cross_entropy(head(z)*8,yy)+.1*(1-sim[positive]).mean()+.1*F.relu(sim[negative]-.2).mean()
    if mode!='identity':loss=loss+.1*((sim-tm).square()*w).sum()/w.sum().clamp_min(1)
    opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():z=F.normalize(net(torch.tensor(x)),dim=1).numpy()
   torch.save(net.state_dict(),out/f'{mode}_{seed}_private.pt');choices=[]
   for alpha in [0,.25,.5,1.]:
    v=raw if alpha==0 else np.c_[raw,alpha*z]/np.sqrt(1+alpha**2);pp=[]
    for u in range(1,10):
     model=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix['register']],y[ix['register']]==u);cs=model.decision_function(v[ix['calA']]);ss=model.decision_function(v[ix['calB']]);pp.append(metric(ss,y[ix['calB']]==u,threshold(cs,y[ix['calA']]==u))['hter'])
    choices.append((float(np.mean(pp)),alpha))
   alpha=min(choices)[1];locks.append(dict(seed=seed,method=mode,alpha=alpha,cal_choices=choices));v=raw if alpha==0 else np.c_[raw,alpha*z]/np.sqrt(1+alpha**2)
   for u in range(1,10):
    for name,vv in [('raw',raw),(mode,v)]:
     if name=='raw' and mode!='identity':continue
     model=SVC(C=10,gamma='scale',class_weight='balanced').fit(vv[ix['register']],y[ix['register']]==u);cal=np.r_[ix['calA'],ix['calB']];tau=threshold(model.decision_function(vv[cal]),y[cal]==u)
     for cond in ['sameword','crossword']:
      j=ix[cond];results.append(dict(seed=seed,method=name,user=f'U{u}',condition=cond,**metric(model.decision_function(vv[j]),y[j]==u,tau)))
   print('COMPLETE',seed,mode,'alpha',alpha,flush=True)
 (out/'selection_public.json').write_text(json.dumps(locks,indent=2));(out/'per_user_public.json').write_text(json.dumps(results,indent=2));summary=[]
 for mode in ['raw','identity','uniform','selective','shuffle_same_user']:
  for cond in ['sameword','crossword']:
   r=[v for v in results if v['method']==mode and v['condition']==cond];summary.append(dict(method=mode,condition=cond,**{k:float(np.mean([v[k] for v in r])) for k in ['tar','far','hter','auc']}))
 (out/'results_public.json').write_text(json.dumps(dict(state='complete',summary=summary,limits=['nominal word labels','closed known identities','archived explored Audio teacher','teacher training includes registration Audio','simple cosine relation matching,not full RKD','alpha chosen only CalA/B','CPU400updates three seeds']),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);main(p.parse_args())
