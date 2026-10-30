"""ACC-only controls: direct spectrum/statistics and independently trained CNN."""
import sys,copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/root')
import run_frozen_cycle_identity_compare_20261029 as first
old=first.old;trial=first.trial;D=old.D
ROOT=Path('/autodl-fs/data/mei_independent_acc_auth_20261029');SEEDS=[42,43,44];STEPS=400

class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,32,3,2,1),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU())
  self.proj=nn.Linear(128,128);self.cls=nn.Linear(128,1)
 def forward(self,x):
  h=self.conv(x).flatten(2);return F.normalize(self.proj(torch.cat([h.mean(2),h.std(2,unbiased=False)],1)),dim=1)

def input_tensor(x):return old.te.tensor(x)
@torch.no_grad()
def embedding(net,x):
 return np.concatenate([net(input_tensor(x[i:i+32])).cpu().numpy() for i in range(0,len(x),32)])
def statistics(x):return np.concatenate([x.mean(2),x.std(2),x.mean(1),x.std(1)],1)

def fit_rbf(feat,p,u):
 y=p.labels['auth_train']==u-1;vy=p.labels['model_validation']==u-1;best=None
 for c in [1.,10.,100.]:
  for gamma in ['scale',.1]:
   m=old.sel.rbf.fit([feat['auth_train']],y,c,gamma);s=old.sel.rbf.score(m,[feat['model_validation']]);q=trial.quality(s,vy,first.words(p,'model_validation'));rank=(-q[0],-q[1],c,0 if gamma==.1 else 1)
   if best is None or rank<best[0]:best=(rank,m,c,gamma,q)
 m=best[1];g=old.sel.rbf.score(m,[feat['target_dev_pool']]);n=old.sel.rbf.score(m,[feat['impostor_calibration']])
 return m,dict(C=best[2],gamma=best[3],validation_sameword_auc=best[4][0],validation_auc=best[4][1],threshold=float(old.base.original.choose_threshold(g,n)))

def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);p=old.protocol(u)
 arrays={r:p.arrays[r] for r in first.ROLES[:4]};models={};nets={};locks={}
 labels=p.labels['auth_train'];words=first.words(p,'auth_train');vy=p.labels['model_validation']==u-1;vw=first.words(p,'model_validation')
 # No G/F or Audio is loaded anywhere in this experiment.
 for key,fn in [('direct_flat6400',lambda x:x.reshape(len(x),-1)),('direct_stats320',statistics)]:
  feat={r:fn(x) for r,x in arrays.items()};models[key],locks[key]=fit_rbf(feat,p,u)
 for seed in SEEDS:
  trial.seed(seed);rng=np.random.RandomState(seed);net=Encoder().to(D);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.001);best=None;history=[]
  for step in range(1,STEPS+1):
   # Balanced identities, two words and two recordings/word:36 examples.
   selected_words=rng.choice(np.unique(words),2,replace=False);ix=np.concatenate([rng.choice(np.flatnonzero((labels==v)&(words==w)),2,replace=False) for v in range(9) for w in selected_words])
   z=net(input_tensor(arrays['auth_train'][ix]));y=trial.tensor((labels[ix]==u-1).astype('float32'))
   loss=F.binary_cross_entropy_with_logits(net.cls(z).flatten(),y,pos_weight=trial.tensor(8.))+.1*trial.contrast(z,torch.as_tensor(labels[ix],device=D),torch.as_tensor(words[ix],device=D),True)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
   if step%25==0:
    net.eval();ev=embedding(net,arrays['model_validation']);s=net.cls(trial.tensor(ev)).flatten().detach().cpu().numpy();q=trial.quality(s,vy,vw);ce=float(F.binary_cross_entropy_with_logits(trial.tensor(s),trial.tensor(vy.astype('float32')),pos_weight=trial.tensor(8.)))
    history.append(dict(step=step,word_auc=q[0],auc=q[1],bce=ce));rank=(-q[0],-q[1],ce,step)
    if best is None or rank<best[0]:best=(rank,copy.deepcopy(net.state_dict()),step)
    net.train()
  net.load_state_dict(best[1]);net.eval();key='cnn128_seed%d'%seed;nets[key]=net
  feat={r:embedding(net,x) for r,x in arrays.items()};models[key],locks[key]=fit_rbf(feat,p,u);locks[key].update(seed=seed,selected_step=best[2],history=history)
  torch.save(best[1],str(folder/(key+'_private.pt')));print('CNN_LOCKED',u,seed,best[2],flush=True)
 for key,m in models.items():old.sel.joblib.dump(m,str(folder/(key+'_rbf_private.joblib')))
 old.save(folder/'locked_before_final.json',dict(routes=locks,final_seen=False))
 finals={r:p.arrays[r] for r in first.ROLES[4:]};rows={};dump={}
 for key,m in models.items():
  fn=(lambda x:x.reshape(len(x),-1)) if key=='direct_flat6400' else statistics if key=='direct_stats320' else lambda x:embedding(nets[key],x)
  scores={r:old.sel.rbf.score(m,[fn(x)]) for r,x in finals.items()};spec=locks[key];r=old.sel.evaluate_route(scores,p,u,spec['threshold']);r.update({k:v for k,v in spec.items() if k!='history'})
  g=scores['final_genuine'];n=scores['final_impostor'];t=spec['threshold']
  assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
  assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12 and len(r['attack_by_user'])==8
  for v,a in r['attack_by_user'].items():
   s=n[p.labels['final_impostor']==int(v)-1];assert len(s)==a['n'] and int((s>=t).sum())==a['accepted']
  rows[key]=r
  for role,s in scores.items():dump[key+'__'+role]=s
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 out=dict(target=u,routes=rows,no_generator_or_audio=True,score_counts_verified=True);old.save(folder/'results_public.json',out);return out

def main():
 ROOT.mkdir(exist_ok=True);old.records()
 old.save(ROOT/'protocol_public.json',dict(targets=[1,4,5],seeds=SEEDS,steps=STEPS,controls=['direct6400 spectrum RBF','direct320 mean/std across frequency and time RBF','independent CNN128 RBF'],cnn='3convs16/32/64 GroupNorm GELU; global mean/std;128L2; random initialization, no G/F weights',training='authfit only: balanced36batch9identities2words2repetitions;BCE posweight8 +0.1 crosswordSupCon; AdamW3e-4 wd.001 clip1',normalization='same stored ACC spectra all users;direct RBF scaling fitted by original RBF helper on training only;CNN receives same stored scale without per-person normalization',selection='validation wordAUC/pooledAUC/BCE checkpoint;matched RBF gridC1/10/100 gamma scale/.1; calibration threshold; all routes locked before final',limitations=['historical final reused exploratory known-user closed set','word mapping10/20period missing-tail assumptions','CNN has no upstream generative pretraining;resources differ from baseline','processed ACC spectra,not raw waveform new preprocessing','not claim best possible ACC-only model']))
 rows={}
 for u in [1,4,5]:
  old.save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));rows[str(u)]=target(u);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

