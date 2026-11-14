"""Personal training-only Audio relation weights; individual CalA/B fusion."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.svm import SVC
from mei_audio_residual_selective import norm,threshold,metric

torch.set_num_threads(2)
out=Path('outputs/mei_personal_audio_guidance_20261114');out.mkdir(exist_ok=True,parents=True)
d=np.load('outputs/mei_residual_input_private.npz');ix={k:np.array(v) for k,v in json.loads(str(d['split'])).items()};x=d['x'];raw=d['stats'];ti=d['indices'];y=d['user'];word=d['word'];t=norm(d['targets']);n=len(ti)
assert not set(ti)&set(np.r_[ix['calA'],ix['calB'],ix['sameword'],ix['crossword']])
same=y[ti,None]==y[ti][None,:];dw=word[ti,None]!=word[ti][None,:];pos=same&dw;neg=(~same)&(~dw);mask=pos|neg
rel=[]
for j,i in enumerate(ti):
 other=(y[ti]==y[i])&(word[ti]!=word[i]);rel.append(max(0.,float(t[j]@norm(t[other].mean(0,keepdims=True))[0])))
rel=np.array(rel);gap=np.maximum(0.,norm(x[ti])@norm(x[ti]).T-t@t.T);xx=torch.tensor(x[ti]);yy=torch.tensor(y[ti]-1,dtype=torch.long);tm=torch.tensor(t@t.T);rows=[];locks=[]
for u in range(1,10):
 owner=(y[ti]==u);ownerpair=owner[:,None]|owner[None,:]
 for mode in ['raw','identity','uniform','personal']:
  if mode=='raw':z=None
  else:
   torch.manual_seed(14000+u);net=nn.Sequential(nn.Linear(738,128),nn.GELU(),nn.Linear(128,64));net.load_state_dict(torch.load('outputs/mei_audio_residual_selective_20261114/identity_42_private.pt',map_location='cpu',weights_only=True));head=nn.Linear(64,9);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-3)
   with torch.no_grad():anchor=F.normalize(net(xx),dim=1)
   # Every owner has a different teacher-reliability/confusion weighting, with all identities retained.
   w=np.ones((n,n),dtype='float32')
   if mode=='personal':w=np.sqrt(rel[:,None]*rel[None,:])*(1+gap)*ownerpair
   w=torch.tensor(w*mask,dtype=torch.float32);pp=torch.tensor(pos);nnn=torch.tensor(neg)
   for step in range(200):
    z=F.normalize(net(xx),dim=1);s=z@z.T;loss=F.cross_entropy(head(z)*8,yy)+.1*(1-s[pp]).mean()+.1*F.relu(s[nnn]-.2).mean()+.1*(1-(z*anchor).sum(1)).mean()
    if mode!='identity':loss=loss+.1*((s-tm).square()*w).sum()/w.sum().clamp_min(1)
    opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():z=F.normalize(net(torch.tensor(x)),dim=1).numpy()
   torch.save(net.state_dict(),out/f'U{u}_{mode}_private.pt')
  def view(alpha):return raw if alpha==0 else np.c_[raw,alpha*z]/np.sqrt(1+alpha**2)
  choices=[]
  for alpha in ([0] if mode=='raw' else [0,.25,.5,1.]):
   v=view(alpha);model=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix['register']],y[ix['register']]==u);a=ix['calA'];b=ix['calB'];tau=threshold(model.decision_function(v[a]),y[a]==u);vcal=metric(model.decision_function(v[b]),y[b]==u,tau);choices.append((vcal['hter'],alpha))
  alpha=min(choices)[1];locks.append(dict(user=f'U{u}',method=mode,alpha=alpha,cal_choices=choices));v=view(alpha);model=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix['register']],y[ix['register']]==u);cal=np.r_[ix['calA'],ix['calB']];tau=threshold(model.decision_function(v[cal]),y[cal]==u)
  for cond in ['sameword','crossword']:
   j=ix[cond];rows.append(dict(user=f'U{u}',method=mode,condition=cond,**metric(model.decision_function(v[j]),y[j]==u,tau)))
  print('COMPLETE',u,mode,alpha,flush=True)
(out/'selection_public.json').write_text(json.dumps(locks,indent=2));(out/'per_user_public.json').write_text(json.dumps(rows,indent=2));summary=[]
for m in ['raw','identity','uniform','personal']:
 for c in ['sameword','crossword']:
  rr=[v for v in rows if v['method']==m and v['condition']==c];summary.append(dict(method=m,condition=c,**{k:float(np.mean([v[k] for v in rr])) for k in ['tar','far','hter','auc']}))
(out/'results_public.json').write_text(json.dumps(dict(state='complete',summary=summary,scope='known9,all8attackers,nominal words,one seed;archived teacher+shared identity42 initialization;owner-specific relation weights,not physical-target feature selection;CalA/B owner fusion;no query tuning'),indent=2));print(summary,flush=True)
