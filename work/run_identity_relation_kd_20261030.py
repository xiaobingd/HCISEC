"""Matched ACC-only students with real-paired cross-modal identity relation teachers."""
import sys,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/root')
import run_audio_guided_acc_kd_20261030 as p
from run_relaxed_audio_feature_kd_20261030 import load_previous,shuffle_map
ROOT=Path('/autodl-fs/data/mei_identity_relation_kd_20261030')
LREL=.1; LSTABLE=.1

class Teacher(nn.Module):
 def __init__(self,acc,audio,interaction=False):
  super().__init__();self.acc=copy.deepcopy(acc.encoder);self.audio=copy.deepcopy(audio.encoder)
  self.pa=nn.Linear(128,128);self.pm=nn.Linear(128,128)
  self.fuse=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,128));self.head=nn.Linear(128,9);self.interaction=interaction
  for q in list(self.acc.parameters())+list(self.audio.parameters()):q.requires_grad_(False)
 def forward(self,a,m):
  x=F.normalize(self.pa(self.acc(a)),dim=1);y=F.normalize(self.pm(self.audio(m)),dim=1)
  if self.interaction:
   cross=x*y;x,y=x+cross,y+cross
  z=F.normalize(self.fuse(torch.cat([x,y],1)),dim=1);return z,self.head(z)

def masks(labels,words):
 same=labels[:,None]==labels[None,:];sw=words[:,None]==words[None,:]
 return same&~sw,~same&sw

def stability(z,labels,words):
 pos,neg=masks(labels,words);sim=z@z.T
 return .5*((1-sim[pos]).mean()+F.relu(sim[neg]-.2).mean())

def relation(z,tz,labels,words):
 pos,neg=masks(labels,words);error=(z@z.T-tz@tz.T).square()
 return .5*(error[pos].mean()+error[neg].mean())

def train(data,s,folder,key,acc=None,audio=None,interaction=False,stable=False,target=None,mapping=None):
 p.seed(s);teacher=acc is not None
 net=Teacher(acc,audio,interaction).to(p.D) if teacher else p.Classifier().to(p.D)
 params=[q for q in net.parameters() if q.requires_grad];opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=.001)
 rng=np.random.RandomState(s);lookup={(u,w):np.flatnonzero((data['train']['labels']==u)&(data['train']['words']==w)) for u in range(9) for w in p.ROLES['train']}
 best=None;history=[];start=time.monotonic()
 for step in range(1,p.STEPS+1):
  net.train();ww=rng.choice(p.ROLES['train'],4,replace=False);ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in range(9) for w in ww]);y=torch.as_tensor(data['train']['labels'][ix],device=p.D);w=torch.as_tensor(data['train']['words'][ix],device=p.D)
  a=p.tensor(data['train']['acc'][ix,None]);z,l=net(a,p.tensor(data['train']['audio'][ix,None])) if teacher else net(a)
  ce=F.cross_entropy(l,y);ls=stability(z,y,w) if stable else ce.new_zeros(())
  lr=relation(z,target[ix if mapping is None else mapping[ix]],y,w) if target is not None else ce.new_zeros(())
  loss=ce+LSTABLE*ls+LREL*lr;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
  if step%25==0:
   domain='fusion' if teacher else 'acc';q=p.validation_quality(net,data,domain)
   history.append(dict(step=step,ce=float(ce.detach()),stability=float(ls.detach()),relation=float(lr.detach()),quality=q))
   rr=p.rank(q)+(step,)
   if best is None or rr<best[0]:best=(rr,copy.deepcopy(net.state_dict()),step,q)
 net.load_state_dict(best[1]);net.eval()
 for q in net.parameters():q.requires_grad_(False)
 meta=dict(key=key,seed=s,steps=p.STEPS,selected_step=best[2],quality=best[3],history=history,teacher=teacher,interaction=interaction,stability=stable,relation=target is not None,shuffled=mapping is not None,lambda_relation=LREL,lambda_stability=LSTABLE,parameters=sum(q.numel() for q in net.parameters()),trainable_parameters=sum(q.numel() for q in params),seconds=time.monotonic()-start)
 torch.save(dict(state=best[1],metadata=meta),str(folder/(key+'_private.pt')));print('TRAINED',s,key,best[2],round(meta['seconds'],1),flush=True)
 return net,meta

def geometry(z,labels,words):
 z=p.tensor(z);y=torch.as_tensor(labels,device=p.D);w=torch.as_tensor(words,device=p.D);pos,neg=masks(y,w);sim=(z@z.T).cpu().numpy();po=sim[pos.cpu().numpy()];ne=sim[neg.cpu().numpy()]
 return dict(same_identity_crossword_mean=float(po.mean()),different_identity_sameword_mean=float(ne.mean()),margin=float(po.mean()-ne.mean()),pair_auc=float(p.roc_auc_score(np.r_[np.ones(len(po)),np.zeros(len(ne))],np.r_[po,ne])),positive_pairs=len(po),negative_pairs=len(ne),dependent_pairs=True)

def seed_run(s,data):
 folder=ROOT/('seed%d'%s);folder.mkdir(exist_ok=True);source=p.ROOT/('seed%d'%s)
 b0,ck=load_previous(source,'B0_CE');audio,_=load_previous(source,'T1_Audio');aam,_=load_previous(source,'B1_AAM')
 # Same encoder count, same projection/fusion parameter count for all three teachers.
 teachers={};metas={};targets={}
 for name,inter,stable in [('T_concat',False,False),('T_interaction',True,False),('T_interaction_stable',True,True)]:
  net,meta=train(data,s+2000,folder,name,acc=b0,audio=audio,interaction=inter,stable=stable)
  teachers[name]=net;metas[name]=meta;targets[name]=p.tensor(p.inference(net,data,'train','fusion')[0])
 models={'E0_ACC_CE':b0,'E0_ACC_AAM':aam};metadata={}
 specs=[('E1_concat_relation','T_concat',False,False),('E2_interaction_relation','T_interaction',False,False),('E3_full','T_interaction_stable',True,False),('E4_ACC_stability',None,True,False),('E5_full_shuffled','T_interaction_stable',True,True)]
 mapping=shuffle_map(data,s);np.save(folder/'shuffle_map_private.npy',mapping)
 for key,t,stable,shuffled in specs:
  net,meta=train(data,s,folder,key,stable=stable,target=targets[t] if t else None,mapping=mapping if shuffled else None)
  models[key]=net;metadata[key]=meta
 # Self relation control matches full identity loss and targets frozen ACC instead of fusion.
 target=p.tensor(p.inference(b0,data,'train')[0]);net,meta=train(data,s,folder,'E6_self_relation',stable=True,target=target)
 models['E6_self_relation']=net;metadata['E6_self_relation']=meta
 locks={};dev={}
 for key,net in models.items():
  z={r:p.inference(net,data,r)[0] for r in ['enroll','validation','calibration']};locks[key]=p.backend_lock(z,data,folder,key);dev[key]=geometry(z['validation'],data['validation']['labels'],data['validation']['words'])
 qualifications={key:p.validation_quality(net,data,'fusion') for key,net in teachers.items()}
 p.old.save(folder/'locked_before_final.json',dict(seed=s,metadata=metadata,teachers=metas,teacher_validation=qualifications,development_geometry=dev,routes=locks,final_seen=False))
 rows={};diag={}
 for key,net in models.items():
  rows[key]=p.evaluate_final(net,data,folder,key,locks[key]);zt=p.inference(net,data,'test')[0];diag[key]=geometry(zt,data['test']['labels'],data['test']['words'])
 # Replay cached baseline ensures old and new code use the identical backend and scores.
 prior=json.loads((source/'results_public.json').read_text())
 for key,pk in [('E0_ACC_CE','B0_CE'),('E0_ACC_AAM','B1_AAM')]:
  for name,r in rows[key].items():
   for metric in ['tar','far','hter','auc']:assert abs(r[metric]-prior['routes'][pk][name][metric])<1e-12
 out=dict(seed=s,routes=rows,teacher_validation=qualifications,final_identity_geometry=diag,baseline_replayed=True)
 p.old.save(folder/'results_public.json',out);print('SEED_DONE',s,flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);data=p.data_load()
 for i,r in enumerate(p.ROLES):
  for q in list(p.ROLES)[:i]:assert not set(data[r]['ids'])&set(data[q]['ids']) and not set(data[r]['words'])&set(data[q]['words'])
 p.old.save(ROOT/'manifest_private.json',{r:dict(ids=v['ids'],labels=v['labels'].tolist(),words=v['words'].tolist()) for r,v in data.items()})
 p.old.save(ROOT/'protocol_public.json',dict(seeds=p.SEEDS,steps=p.STEPS,groups=p.ROLES,student='unchanged CNN128,CE9;cached AAM control;identical initialization and 72-record minibatches',teachers='frozen real ACC/Audio encoders, independent128 projections;concat versus elementwise cross interaction;equal parameters and400updates',loss='CE+.1*balanced relation cosine-matrix MSE+.1*crossword sameID attraction/sameword differentID hinge .2',relation_masks='same identity different word;different identity same word, equal mean weights',backend='unchanged K3/5/10/20 cosine and K20RBF validation grid,calibration threshold',selection='checkpoint every25steps by heldout validation macro HTER,worst error,AUC,CE;all weights fixed before final',controls=['CE','AAM','concat relation','interaction relation','full','ACC stability','full shuffled withinidentity differentword','selfrelation+stability'],deployment='ACC student only;no Audio/projection/fusion at enrollment or test',limitations=['known9identities;not identity-disjoint open set','historical single-session heldout words reused exploratorily','tail missing and word-order metadata assumption inherited','teacher pretraining adds resources;students have equal400updates','pair geometry observations dependent','shuffle tests exact recording/content matching;equal scores can still indicate identity-level transfer']))
 rows={}
 for s in p.SEEDS:
  p.old.save(ROOT/'status_public.json',dict(state='running',seed=s,completed=list(rows)))
  rows[str(s)]=seed_run(s,data);p.old.save(ROOT/'results_public.json',rows)
 p.old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);p.old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

