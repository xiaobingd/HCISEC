"""Teacher reliability and predictable identity relations, yue excluded throughout."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root');import run_raw9_300 as r
ROOT=Path('/autodl-fs/data');SRC=ROOT/'identity_kd_all9_raw_full_multisource_20261103';OUT=ROOT/'mei8_selective_teacher_residual_20261108';OUT.mkdir(exist_ok=True);DEVICE='cpu';torch.set_num_threads(1)
def status(**v):
 p=OUT/'status.tmp';p.write_text(json.dumps(v,indent=2));p.replace(OUT/'status.json')
a=np.load(SRC/'raw9_full_features_private.npz');keep=a['user']!=9;d={k:a[k][keep] for k in ['user','word','role','sample_id']};users=list(range(1,9));za=np.load(ROOT/'mei8_exclude_yue_20261108/acc_embeddings_private.npy');tm=np.load(ROOT/'mei9_vibravox_teacher_transfer_20261107/vibravox_audio_embeddings_private.npy')[keep];assert len(za)==len(d['user']);fit=np.flatnonzero(d['role']=='fit');dev=np.flatnonzero(d['role']=='develop');assert set(d['user'])==set(users)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
# Cross-word prediction is fitted only within training groups, with all users retained.
pred=np.zeros((len(fit),192),dtype='float32');cv=[]
for tr,va in GroupKFold(n_splits=4).split(za[fit],d['user'][fit],groups=d['word'][fit]):
 scaler=StandardScaler().fit(za[fit[tr]]);reg=Ridge(alpha=100).fit(scaler.transform(za[fit[tr]]),tm[fit[tr]]);pred[va]=norm(reg.predict(scaler.transform(za[fit[va]])));cv.append(dict(train_word_ids=np.unique(d['word'][fit[tr]]).tolist(),heldout_word_ids=np.unique(d['word'][fit[va]]).tolist()))
predictable=np.clip((pred*tm[fit]).sum(1),0,1);margin=np.zeros(len(fit));accmargin=np.zeros(len(fit))
for j,i in enumerate(fit):
 u=d['user'][i];w=d['word'][i];own=fit[(d['user'][fit]==u)&(d['word'][fit]!=w)];neg=[fit[(d['user'][fit]==v)&(d['word'][fit]==w)] for v in users if v!=u]
 margin[j]=tm[i]@norm(tm[own].mean(0))-max(tm[i]@norm(tm[ii].mean(0)) for ii in neg)
 accmargin[j]=za[i]@norm(za[own].mean(0))-max(za[i]@norm(za[ii].mean(0)) for ii in neg)
weights=np.zeros(len(za),dtype='float32');weights[fit]=np.clip(margin/.2,0,1)*predictable
quality={r.NAMES[u-1]:dict(audio_crossword_margin=float(margin[d['user'][fit]==u].mean()),acc_crossword_margin=float(accmargin[d['user'][fit]==u].mean()),audio_predictability_cosine=float(predictable[d['user'][fit]==u].mean()),audio_distillation_weight=float(weights[fit[d['user'][fit]==u]].mean())) for u in users}
(OUT/'teacher_diagnostics_private.json').write_text(json.dumps(dict(per_user=quality,crossword_folds=cv),indent=2));status(state='teacher_quality_audit_complete',weights_mean=float(weights[fit].mean()))
class Residual(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(192,64);self.b=nn.Linear(64,192);nn.init.zeros_(self.b.weight);nn.init.zeros_(self.b.bias)
 def forward(self,x):
  delta=self.b(F.gelu(self.a(x)));delta=.1*delta/torch.linalg.vector_norm(delta,dim=1,keepdim=True).clamp_min(1.);return F.normalize(x+delta,dim=1)
def relation(z,t,y,w):
 e=(z@z.T-t@t.T).square();eye=torch.eye(len(z),dtype=torch.bool);same=(y[:,None]==y[None,:])&~eye;diff=y[:,None]!=y[None,:];p=w[:,None]*w[None,:];return .5*((e*p*same).sum()/(p*same).sum().clamp_min(1e-8)+(e*p*diff).sum()/(p*diff).sum().clamp_min(1e-8))*w.mean()
words=np.unique(d['word'][fit]);lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in users for w in words};outputs={'raw_acc8_teacher':za};selected={};baseline_dev=r.pair_auc(za[dev],d['user'][dev],d['word'][dev])
for mode in ['residual_identity_only','residual_uniform_audio','residual_selective_audio']:
 torch.manual_seed(42);rng=np.random.RandomState(42);net=Residual();head=nn.Linear(192,8,bias=False)
 with torch.no_grad():head.weight.copy_(torch.tensor(np.stack([norm(za[fit[d['user'][fit]==u]].mean(0)) for u in users])))
 opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-4);best=baseline_dev;best_state={k:v.detach().clone() for k,v in net.state_dict().items()};chosen=0;history=[]
 for step in range(1,1001):
  ww=rng.choice(words,2,replace=False);uu=rng.permutation(users);ii=np.array([int(rng.choice(lookup[u,w])) for u in uu for w in ww]);y=torch.tensor(d['user'][ii]-1,dtype=torch.long);x=torch.tensor(za[ii]);z=net(x);ce=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y);loss=ce+.1*r.p.sc_loss(z,y)+.1*(1-(x*z).sum(1)).mean();kd=z.new_zeros(())
  if mode!='residual_identity_only':
   w=torch.ones(len(ii)) if mode=='residual_uniform_audio' else torch.tensor(weights[ii]);kd=relation(z,torch.tensor(tm[ii]),y,w);loss=loss+.1*kd
  opt.zero_grad();loss.backward();opt.step()
  if step%100==0:
   with torch.no_grad():zz=net(torch.tensor(za[dev])).numpy()
   q=r.pair_auc(zz,d['user'][dev],d['word'][dev]);history.append(dict(step=step,dev_pair_auc=q,loss=float(loss.detach()),kd=float(kd.detach())))
   if q>best:best=q;chosen=step;best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
   status(state='training_residual',mode=mode,step=step,total_steps=1000,dev_pair_auc=q)
 net.load_state_dict(best_state)
 with torch.no_grad():outputs[mode]=net(torch.tensor(za)).numpy()
 selected[mode]=dict(step=chosen,dev_pair_auc=best);torch.save(best_state,OUT/(mode+'_private.pt'));(OUT/(mode+'_history.json')).write_text(json.dumps(history,indent=2));print('TRAINED',mode,selected[mode],flush=True)
train=[];cal=[]
for u in users:
 for w in np.unique(d['word'][d['role']=='enroll']):
  ii=np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='enroll'));ii=np.random.RandomState(42+u+int(w)).permutation(ii);train.extend(ii[:5]);cal.extend(ii[5:])
train=np.array(train);cal=np.array(cal);test=np.flatnonzero(d['role']=='final');results={}
for mode,z in outputs.items():
 scaler=StandardScaler().fit(z[train]);zz=scaler.transform(z);results[mode]={}
 for u in users:
  opts=[]
  for C in [.1,1,10]:
   for gamma in ['scale',.01,.1]:
    clf=SVC(C=C,gamma=gamma,class_weight='balanced').fit(zz[train],d['user'][train]==u);sc=clf.decision_function(zz[cal]);pos=d['user'][cal]==u;tau=r.eer_threshold(sc[pos],sc[~pos]);hter=float(((sc[pos]<tau).mean()+(sc[~pos]>=tau).mean())/2);opts.append((hter,C,str(gamma),clf,tau))
  _,C,gamma,clf,tau=min(opts,key=lambda q:(q[0],q[1],q[2]));sc=clf.decision_function(zz[test]);pos=d['user'][test]==u;tar=float((sc[pos]>=tau).mean());far=float((sc[~pos]>=tau).mean());results[mode][r.NAMES[u-1]]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(pos,sc)),genuine_n=int(pos.sum()),impostor_n=int((~pos).sum()),C=C,gamma=gamma,threshold=tau)
 (OUT/'results_private.json').write_text(json.dumps(results,indent=2));print('EVALUATED',mode,flush=True)
(OUT/'protocol.json').write_text(json.dumps(dict(scope='eight registered identities;complete yue exclusion;not user-unseen authentication',teacher_ACC='fresh eight-user ACC backbone,fit/dev only',teacher_Audio='VibraVox-only fullyfinetuned Audio backbone,never locally finetuned on nine users;do not reuse yue-influenced teacher',features='ACC anchors original192Dgeometry;Audio confidence uses own different-word center vs other-user same-word centers;predictability uses fourfold training-word-heldout ridge prediction',weights='clip(Audio margin/0.2,0,1)*clip(OOF predictedAudio dot realAudio,0,1);balanced relationalKD scaled by mean confidence',student='192Didentity-initialized residual,bottleneck64,correction norm<=0.1;no dimension-replacing projection',training='three matched1000step conditions,seed42;original step0 eligible;checkpoint only on ACC development pairAUC',selection=selected,backend='same20head-train+20calibration per user,RBF;all7attack sources;final recordings untouched',limits='one seed;inferred word groups;closed registered setting;new weighting is an unproven hypothesis'),indent=2));status(state='complete',registered_users=8)
