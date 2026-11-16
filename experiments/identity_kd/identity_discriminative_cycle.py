"""Stage A: matched shared encoders, no personal adaptation, all nine identities.
E0 archived baseline; E1 GAN; E2 CE+SupCon only; E3 GAN+CE;
E4 GAN+CE+SupCon; E5 frozen E1 plus trainable 256-d projection.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch,joblib
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mei9_closed_personal_cycle import Generator,Discriminator,indices
from run_mei9_shape_v4 import Student,encode
from run_rkd01_vibravox import norm,metrics,thresholds
from vibravox_cycle_supplement import pooled

def dump(path,v):path.write_text(json.dumps(v,indent=2))
def contrast(z,y,word,kind):
 sim=z@z.T/.1;eye=torch.eye(len(z),device=z.device,dtype=torch.bool);positive=(y[:,None]==y[None,:])&~eye
 if kind=='crossword':positive &= word[:,None]!=word[None,:]
 logp=sim-torch.logsumexp(sim.masked_fill(eye,-1e9),1,keepdim=True)
 return -((logp*positive).sum(1)/positive.sum(1).clamp_min(1)).mean()
def batch(d,fit,rng):
 users=rng.choice(np.arange(1,10),4,replace=False);words=sorted(set(d['word'][fit]));eligible=[w for w in words if all(np.sum((d['user'][fit]==u)&(d['word'][fit]==w))>=2 for u in users)];ww=rng.choice(eligible,2,replace=False)
 return np.concatenate([rng.choice(fit[(d['user'][fit]==u)&(d['word'][fit]==w)],2,replace=False) for u in users for w in ww])
def train(mode,seed,d,data,ix,out,a):
 folder=out/f'seed{seed}'/mode;folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);rng=np.random.RandomState(seed);device=a.device
 g=Generator(3,1).to(device);f=Generator(1,3).to(device);da=Discriminator(3).to(device);db=Discriminator(1).to(device);head=nn.Linear(256,9).to(device);projection=nn.Linear(256,256).to(device)
 if mode=='E5':
  ck=torch.load(out/f'seed{seed}'/'E1'/'checkpoint_private.pt',map_location='cpu',weights_only=False);assert ck['step']==a.steps;g.load_state_dict(ck['G']);g.eval()
  for p in g.parameters():p.requires_grad_(False)
  with torch.no_grad():projection.weight.copy_(torch.eye(256,device=device));projection.bias.zero_()
 genparams=list(g.encoder.parameters()) if mode=='E2' else list(projection.parameters()) if mode=='E5' else list(g.parameters())+list(f.parameters())
 params=genparams+([] if mode=='E1' else list(head.parameters()));opt=torch.optim.Adam(params,lr=2e-4,betas=(.5,.999));od=torch.optim.Adam(list(da.parameters())+list(db.parameters()),lr=2e-4,betas=(.5,.999));dest=folder/'checkpoint_private.pt';start=0;history=[]
 if dest.exists():
  ck=torch.load(dest,map_location=device,weights_only=False)
  for name,obj in [('G',g),('F',f),('DA',da),('DB',db),('head',head),('projection',projection),('optimizer',opt),('discriminator_optimizer',od)]:obj.load_state_dict(ck[name])
  start=ck['step'];rng.set_state(ck['rng']);history=ck['history'];assert ck['supcon']==a.supcon
 t0=time.time()
 for step in range(start+1,a.steps+1):
  j=batch(d,ix['fit'],rng);x=torch.tensor(data['a'][j],device=device);m=torch.tensor(data['b'][j],device=device);y=torch.tensor(d['user'][j]-1,device=device,dtype=torch.long);w=torch.tensor(d['word'][j],device=device)
  lr=2e-4*(1 if step<=a.steps/2 else max(0.,2*(1-step/a.steps)))
  for o in [opt,od]:
   for group in o.param_groups:group['lr']=lr
  h=g.encoder(x);z=F.normalize(pooled(h),dim=1)
  if mode=='E5':z=F.normalize(projection(z.detach()),dim=1)
  ce=F.cross_entropy(head(z),y);con=contrast(z,y,w,a.supcon);identity=ce if mode=='E3' else ce+con
  pair=cycle=adv=x.new_zeros(())
  if mode in ['E1','E3','E4']:
   for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(False)
   fb=g.decoder(h);fa=f(m);pair=F.l1_loss(fb,m)+F.l1_loss(fa,x);cycle=F.l1_loss(f(fb),x)+F.l1_loss(g(fa),m);adv=((db(fb)-1)**2).mean()+((da(fa)-1)**2).mean();loss=adv+10*pair+10*cycle
   if mode!='E1':loss=loss+.1*identity
  else:loss=identity
  if step==1 and mode in ['E2','E3','E4']:
   grads=torch.autograd.grad(identity,list(g.encoder.parameters()),retain_graph=True);gn=float(torch.sqrt(sum(q.square().sum() for q in grads)));assert gn>0;dump(folder/'identity_gradient_public.json',dict(encoder_gradient_norm=gn,identity_gradient_reaches_encoder=True))
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,5);opt.step()
  if mode in ['E1','E3','E4']:
   for p in list(da.parameters())+list(db.parameters()):p.requires_grad_(True)
   dl=.5*(((da(x)-1)**2).mean()+da(fa.detach()).square().mean()+((db(m)-1)**2).mean()+db(fb.detach()).square().mean());od.zero_grad();dl.backward();od.step()
  if step%100==0 or step==a.steps:
   history.append(dict(step=step,ce=float(ce.detach()),supcon=float(con.detach()),paired=float(pair.detach()),cycle=float(cycle.detach()),seconds=round(time.time()-t0)));print(seed,mode,history[-1],flush=True)
  if step%250==0 or step==a.steps:
   ck={name:obj.state_dict() for name,obj in [('G',g),('F',f),('DA',da),('DB',db),('head',head),('projection',projection),('optimizer',opt),('discriminator_optimizer',od)]};ck.update(step=step,rng=rng.get_state(),history=history,supcon=a.supcon,fit_indices=ix['fit']);torch.save(ck,dest);dump(folder/'training_public.json',dict(step=step,seed=seed,method=mode,history=history,device=device))
 # Generation development metrics use held recordings; fixed final-step selection for all modes.
 if mode in ['E1','E3','E4']:
  g.eval();f.eval();vals=[]
  with torch.no_grad():
   for j in np.array_split(ix['dev'],max(1,len(ix['dev'])//4)):
    x=torch.tensor(data['a'][j],device=device);m=torch.tensor(data['b'][j],device=device);fb=g(x);fa=f(m);vals.append([float(F.l1_loss(fb,m)),float(F.l1_loss(fa,x)),float(F.l1_loss(f(fb),x)),float(F.l1_loss(g(fa),m))])
  dump(folder/'development_public.json',dict(names=['audio_paired_L1','acc_paired_L1','acc_cycle_L1','audio_cycle_L1'],values=np.mean(vals,0).tolist()))
 return g,projection
@torch.no_grad()
def extract(g,p,mode,a,ii,device):
 g.eval();p.eval();vv=[]
 for j in np.array_split(ii,max(1,int(np.ceil(len(ii)/16)))):
  z=pooled(g.encoder(torch.tensor(a[j],device=device)))
  if mode=='E5':z=p(F.normalize(z,dim=1))
  vv.append(z.cpu().numpy())
 return np.concatenate(vv)
def evaluate(g,p,mode,seed,d,data,ix,out,a,base):
 folder=out/f'seed{seed}'/mode;sc=StandardScaler().fit(extract(g,p,mode,data['a'],ix['fit'],a.device));pre=np.unique(np.r_[ix['register'],ix['calA'],ix['calB']]);z=np.zeros((len(base),256),dtype='float32');z[pre]=norm(sc.transform(extract(g,p,mode,data['a'],pre,a.device)));choices=[]
 def view(alpha):return base if alpha==0 else np.c_[base,alpha*z]/np.sqrt(1+alpha**2)
 def head(v,u):return SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix['register']],d['user'][ix['register']]==u)
 for alpha in [0,.25,.5,1.]:
  v=view(alpha);rr=[]
  for u in range(1,10):
   model=head(v,u);aa=ix['calA'];bb=ix['calB'];tau=thresholds(model.decision_function(v[aa]),d['user'][aa]==u)['Cal_EER'];rr.append(metrics(model.decision_function(v[bb]),d['user'][bb]==u,tau)['hter'])
  choices.append((float(np.mean(rr)),alpha))
 alpha=min(choices)[1];dump(folder/'selection_public.json',dict(alpha=alpha,cal_choices=choices));query=np.unique(np.r_[ix['sameword'],ix['crossword']]);z[query]=norm(sc.transform(extract(g,p,mode,data['a'],query,a.device)));rows=[]
 for name,v in [('fused',view(alpha)),('encoder_only',z),('baseline_E0',base)]:
  for u in range(1,10):
   model=head(v,u);cal=np.r_[ix['calA'],ix['calB']];ts=thresholds(model.decision_function(v[cal]),d['user'][cal]==u)
   for cond in ['sameword','crossword']:
    j=ix[cond];s=model.decision_function(v[j]);y=d['user'][j]==u
    for point,tau in ts.items():rows.append(dict(user=f'U{u}',condition=cond,point=point,readout=name,seed=seed,method=mode,**metrics(s,y,tau)))
 dump(folder/'per_user_public.json',rows)
def main(a):
 torch.set_num_threads(2);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(a.frontend));ix=indices(d);data=dict(np.load(Path(a.source)/'prepared_private.npz'));net=Student();net.load_state_dict(torch.load(Path(a.source)/'student_private.pt',map_location='cpu',weights_only=False));base=np.c_[data['shape'],encode(net,data['x'])]/np.sqrt(2)
 for role in ['register','calA','calB','sameword','crossword']:assert not set(ix['fit'])&set(ix[role])
 dump(out/'protocol_public.json',dict(users=9,seeds=a.seeds,steps=a.steps,phase='A shared only;no personal updates',supcon=a.supcon,baseline='archived frozen seed42 E0 reused;generator seeds42/43/44,not three newly trained E0teachers',word_labels='nominal not transcript-verified',methods=a.methods))
 for seed in a.seeds:
  for mode in a.methods:
   g,p=train(mode,seed,d,data,ix,out,a);evaluate(g,p,mode,seed,d,data,ix,out,a,base);del g,p
 print('STAGE_A_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--frontend',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=4000);p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);p.add_argument('--methods',nargs='+',choices=['E1','E2','E3','E4','E5'],default=['E1','E2','E3','E4','E5']);p.add_argument('--supcon',choices=['standard','crossword'],default='standard');p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');main(p.parse_args())
