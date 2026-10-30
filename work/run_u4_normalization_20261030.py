"""Stored-spectrum normalization ablation; NOT recovery of physical ACC energy."""
import sys,json,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/root')
import run_audio_guided_acc_kd_20261030 as p
ROOT=Path('/autodl-fs/data/mei_u4_normalization_20261030')
MODES=['global','instance','gated','dual']

class Model(nn.Module):
 def __init__(self,mode,stats):
  super().__init__();self.mode=mode;self.encoder=p.Encoder();self.head=nn.Linear(128,9)
  self.gate=nn.Parameter(torch.tensor(0.));self.aux=nn.Linear(4,128,bias=False)
  self.register_buffer('mu',torch.tensor(stats['mean']));self.register_buffer('sd',torch.tensor(stats['std']))
  self.register_buffer('sm',torch.tensor(stats['stats_mean']));self.register_buffer('ss',torch.tensor(stats['stats_std']))
 def forward(self,x):
  mean=x.mean((2,3),keepdim=True);std=x.std((2,3),unbiased=False,keepdim=True).clamp_min(1e-6)
  g=(x-self.mu)/self.sd;i=(x-mean)/std
  a=g if self.mode=='global' else i
  if self.mode=='gated':a=torch.sigmoid(self.gate)*g+(1-torch.sigmoid(self.gate))*i
  z=self.encoder(a)
  raw=torch.cat([mean.flatten(1),std.flatten(1),x.square().mean((2,3)).sqrt(),x.amax((2,3))-x.amin((2,3))],1)
  aux=(raw-self.sm)/self.ss if self.mode=='dual' else torch.zeros_like(raw)
  z=F.normalize(z+self.aux(aux),dim=1)
  return z,self.head(z)

def fit(data,seed,mode,stats,folder):
 p.seed(seed);net=Model(mode,stats).to(p.D);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.001);rng=np.random.RandomState(seed);lookup={(u,w):np.flatnonzero((data['train']['labels']==u)&(data['train']['words']==w)) for u in range(9) for w in p.ROLES['train']};best=None;history=[]
 for step in range(1,401):
  net.train();ww=rng.choice(p.ROLES['train'],4,replace=False);ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in range(9) for w in ww]);_,logits=net(p.tensor(data['train']['acc'][ix,None]));loss=F.cross_entropy(logits,torch.as_tensor(data['train']['labels'][ix],device=p.D));opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  if step%25==0:
   q=p.validation_quality(net,data,'acc');row=dict(step=step,quality=q,loss=float(loss.detach()),gate=float(torch.sigmoid(net.gate)));history.append(row);rank=p.rank(q)+(step,)
   if best is None or rank<best[0]:best=(rank,copy.deepcopy(net.state_dict()),row)
 net.load_state_dict(best[1]);net.eval();meta=dict(mode=mode,seed=seed,history=history,selected=best[2],parameters=sum(v.numel() for v in net.parameters()));torch.save(dict(state=net.state_dict(),metadata=meta,stats=stats),str(folder/(mode+'_private.pt')));print('TRAINED',seed,mode,best[2],flush=True);return net,meta

def main():
 ROOT.mkdir(exist_ok=True);data=p.data_load();x=data['train']['acc'];mean=x.mean((1,2));std=x.std((1,2));s=np.stack([mean,std,np.sqrt((x*x).mean((1,2))),x.max((1,2))-x.min((1,2))],1)
 stats=dict(mean=float(x.mean()),std=float(x.std()),stats_mean=s.mean(0).tolist(),stats_std=np.maximum(s.std(0),1e-6).tolist(),fit_n=len(x),physical_energy=False)
 p.old.save(ROOT/'preprocess_public.json',dict(stats=stats,roles=p.ROLES,source='unchanged old stored normalized 80x80 arrays; historical frontend not inverted',scope='U4 authentication against all8 identities; encoders CE9 as preceding experiment',rule='all users same transform; scalar global statistics train-only; fourth branch uses stored-spectrum mean/std/rms/range NOT raw energy'))
 p.old.save(ROOT/'manifest_private.json',{r:dict(ids=d['ids'],words=d['words'].tolist(),labels=d['labels'].tolist()) for r,d in data.items()});allrows={}
 for seed in p.SEEDS:
  folder=ROOT/('seed%d'%seed);folder.mkdir(exist_ok=True);nets={};metas={};locks={}
  for mode in MODES:
   nets[mode],metas[mode]=fit(data,seed,mode,stats,folder)
   z={r:p.inference(nets[mode],data,r)[0] for r in ['enroll','validation','calibration']};locks[mode]=p.backend_lock(z,data,folder,mode)
  p.old.save(folder/'locked_before_final.json',dict(stats=stats,metadata=metas,routes=locks,final_seen=False))
  rows={mode:p.evaluate_final(net,data,folder,mode,locks[mode]) for mode,net in nets.items()}
  out=dict(seed=seed,routes=rows,metadata=metas);p.old.save(folder/'results_public.json',out);allrows[str(seed)]=out;p.old.save(ROOT/'results_public.json',allrows);print('SEED_DONE',seed,flush=True)
 p.old.save(ROOT/'COMPLETE.json',dict(status='complete',seeds=p.SEEDS))
if __name__=='__main__':main()
