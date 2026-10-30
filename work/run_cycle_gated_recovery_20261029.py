"""Stage2 frozen G/F feature recovery; matched binary heads and RBF readout."""
import sys,copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'/root')
import run_frozen_cycle_identity_compare_20261029 as first
trial=first.trial;old=first.old;D=old.D
ROOT=Path('/autodl-fs/data/mei_cycle_gated_recovery_20261029')
SEEDS=[42,43,44];MODES=['raw','concat','gate','gate_identity'];STEPS=400

class Recovery(nn.Module):
 def __init__(self,reference,mode):
  super().__init__();self.mode=mode;self.raw=copy.deepcopy(reference)
  self.concat=nn.Sequential(nn.Linear(1280,128),nn.GELU(),nn.Linear(128,64))
  self.delta=nn.Sequential(nn.Linear(1920,128),nn.GELU(),nn.Linear(128,64))
  self.gate=nn.Sequential(nn.Linear(1920,32),nn.GELU(),nn.Linear(32,1))
  nn.init.constant_(self.gate[-1].bias,-2.)
  self.cls=nn.Linear(64,1)
 def forward(self,a,b):
  g=None
  if self.mode=='raw':h=self.raw(a)
  elif self.mode=='concat':h=self.concat(torch.cat([a,b],1))
  else:
   x=torch.cat([a,b,b-a],1);g=torch.sigmoid(self.gate(x));h=self.raw(a)+g*self.delta(x)
  return F.normalize(h,dim=1),g

def tensor(x):return trial.tensor(x)
def get(net,arrays):
 with torch.no_grad():
  z,g=net(*arrays)
 return z.cpu().numpy(),None if g is None else g.cpu().numpy()

def target(u):
 folder=ROOT/('U%d'%u);folder.mkdir(parents=True,exist_ok=True);p=old.protocol(u)
 dev=np.load(str(first.ROOT/('U%d'%u)/'development_private.npz'))
 scaler=StandardScaler().fit(dev['auth_train__raw640'])
 # One shared transform for both branches preserves their coordinate relationship.
 arrays={r:[tensor(scaler.transform(dev[r+'__'+k])) for k in ['raw640','cycle640']] for r in first.ROLES[:4]}
 labels=p.labels['auth_train'];words=first.words(p,'auth_train');y=tensor((labels==u-1).astype('float32'))
 tl=torch.as_tensor(labels,device=D);tw=torch.as_tensor(words,device=D)
 vy=p.labels['model_validation']==u-1;vw=first.words(p,'model_validation');locks={};nets={};rbfs={}
 old.sel.joblib.dump(scaler,str(folder/'scaler_private.joblib'))
 for seed in SEEDS:
  trial.seed(seed);reference=nn.Sequential(nn.Linear(640,128),nn.GELU(),nn.Linear(128,64)).to(D);cls=nn.Linear(64,1).to(D)
  opt=torch.optim.AdamW(list(reference.parameters())+list(cls.parameters()),lr=3e-4,weight_decay=.001)
  # Shared raw-only initialization and frozen reference: auth_train only, no final.
  for step in range(100):
   z=F.normalize(reference(arrays['auth_train'][0]),dim=1)
   loss=F.binary_cross_entropy_with_logits(cls(z).flatten(),y,pos_weight=tensor(8.))+.1*trial.contrast(z,tl,tw,True)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(reference.parameters())+list(cls.parameters()),1.);opt.step()
  reference.eval()
  for q in reference.parameters():q.requires_grad_(False)
  with torch.no_grad():targetz=F.normalize(reference(arrays['auth_train'][0]),dim=1)
  for mode in MODES:
   trial.seed(seed);net=Recovery(reference,mode).to(D)
   for q in net.raw.parameters():q.requires_grad_(True)
   opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.001);best=None;history=[]
   for step in range(1,STEPS+1):
    z,g=net(*arrays['auth_train'])
    identity=(1-(z*targetz).sum(1)).mean()
    loss=F.binary_cross_entropy_with_logits(net.cls(z).flatten(),y,pos_weight=tensor(8.))+.1*trial.contrast(z,tl,tw,True)
    if mode=='gate_identity':loss=loss+.1*identity
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    if step%25==0:
     zz,_=get(net,arrays['model_validation']);ss=net.cls(tensor(zz)).flatten().detach().cpu().numpy();q=trial.quality(ss,vy,vw)
     ce=float(F.binary_cross_entropy_with_logits(tensor(ss),tensor(vy.astype('float32')),pos_weight=tensor(8.)))
     history.append(dict(step=step,word_auc=q[0],auc=q[1],bce=ce,identity_loss=float(identity.detach())))
     rank=(-q[0],-q[1],ce,step)
     if best is None or rank<best[0]:best=(rank,copy.deepcopy(net.state_dict()),step)
   net.load_state_dict(best[1]);net.eval();key=mode+'_seed%d'%seed;nets[key]=net
   em={r:get(net,a)[0] for r,a in arrays.items()};rb=None
   for c in [1.,10.,100.]:
    for gamma in ['scale',.1]:
     fit=old.sel.rbf.fit([em['auth_train']],labels==u-1,c,gamma);ss=old.sel.rbf.score(fit,[em['model_validation']]);q=trial.quality(ss,vy,vw);rank=(-q[0],-q[1],c,0 if gamma==.1 else 1)
     if rb is None or rank<rb[0]:rb=(rank,fit,c,gamma,q)
   rbfs[key]=rb[1];cg=old.sel.rbf.score(rb[1],[em['target_dev_pool']]);cn=old.sel.rbf.score(rb[1],[em['impostor_calibration']])
   locks[key]=dict(mode=mode,seed=seed,selected_step=best[2],C=rb[2],gamma=rb[3],validation_sameword_auc=rb[4][0],validation_auc=rb[4][1],threshold=float(old.base.original.choose_threshold(cg,cn)),history=history)
   torch.save(net.state_dict(),str(folder/(key+'_private.pt')));old.sel.joblib.dump(rb[1],str(folder/(key+'_rbf_private.joblib')))
   print('HEAD_LOCKED',u,key,best[2],flush=True)
 old.save(folder/'locked_before_final.json',dict(routes=locks,final_seen=False))
 m=old.load_gan(u);state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 finals={r:first.extract(m,p.arrays[r]) for r in first.ROLES[4:]}
 assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in m.state_dict().items())
 finalarrays={r:[tensor(scaler.transform(a[k])) for k in ['raw640','cycle640']] for r,a in finals.items()};rows={};dump={}
 for key,net in nets.items():
  scores={};gate={}
  for r,a in finalarrays.items():
   z,g=get(net,a);scores[r]=old.sel.rbf.score(rbfs[key],[z]);dump[key+'__'+r]=scores[r]
   if g is not None:gate[r]=dict(mean=float(g.mean()),std=float(g.std()))
  spec=locks[key];result=old.sel.evaluate_route(scores,p,u,spec['threshold']);result.update({k:v for k,v in spec.items() if k!='history'});result['gates']=gate
  g=scores['final_genuine'];n=scores['final_impostor'];t=spec['threshold']
  assert abs(float(((g[:,None]>n)+.5*(g[:,None]==n)).mean())-result['auc'])<1e-12
  assert abs(float((g>=t).mean())-result['tar'])<1e-12 and abs(float((n>=t).mean())-result['far'])<1e-12 and len(result['attack_by_user'])==8
  rows[key]=result
 np.savez_compressed(str(folder/'scores_private.npz'),**dump,attack_labels=p.labels['final_impostor'])
 old.save(folder/'results_public.json',dict(target=u,routes=rows,frozen_GF_verified=True,score_replay_verified=True))
 return rows

def main():
 ROOT.mkdir(exist_ok=True);old.records()
 old.save(ROOT/'protocol_public.json',dict(targets=[1,4,5],seeds=SEEDS,modes=MODES,steps=STEPS,reference='raw640 MLP640->128 GELU->64 trained100steps authfit only then frozen; shared initialization across conditions',gate='z=normalize(rawMLP(a)+sigmoid(g([a,b,b-a]))*delta([a,b,b-a])); sigmoid bias -2; no G/F gradients',loss='binary BCE posweight8 +0.1 cross-word SupCon; gate_identity adds0.1 cosine to frozen raw reference',classifier='matched final RBF on64dims C1/10/100 gamma scale/.1; reference stage1 RBF remains external comparator',selection='checkpoint validation wordAUC/pooledAUC/BCE; RBF validation wordAUC/pooledAUC; threshold calibration only; all12routes lock before final',limitations=['exploratory historical final reused','known-user closed set,all8attacks','same raw teacher used for regularization is not independent evaluation proof','same preprocessing and original strict content split','raw/cycle640 use frozen personal G encoder; not independent physical identity features','NR-SV-RC-inspired hypothesis,not claimed paper replication']))
 rows={}
 for u in [1,4,5]:
  old.save(ROOT/'status_public.json',dict(state='running',target=u,completed=list(rows)));rows[str(u)]=target(u);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

