"""VibraVox Audio teacher -> supervised projection adaptation on raw mei nine-user cohort."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,gc,ctypes
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root');import run_raw9_300 as r
SRC=Path('/autodl-fs/data/identity_kd_all9_raw_full_multisource_20261103');OUT=Path('/autodl-fs/data/mei9_vibravox_teacher_transfer_20261107');OUT.mkdir(exist_ok=True);DEVICE='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2 if DEVICE=='cuda' else 1)
def status(**v):
 p=OUT/'status.tmp';p.write_text(json.dumps(v,indent=2));p.replace(OUT/'status.json')
for n in ['model','deps']:
 if not (OUT/n).exists():(OUT/n).symlink_to(SRC/n,target_is_directory=True)
afile=np.load(SRC/'raw9_full_features_private.npz',allow_pickle=False);d={k:afile[k] for k in ['user','word','role','sample_id']};N=len(d['user']);fit=np.flatnonzero(d['role']=='fit');dev=np.flatnonzero(d['role']=='develop')
# Real paired Audio is used only for encoder fit rows, never enrollment/calibration/test.
cp=OUT/'audio_cache_progress.json';cache=OUT/'vibravox_audio_embeddings_private.npy';done=json.loads(cp.read_text())['done'] if cp.exists() else 0;tf=np.lib.format.open_memmap(cache,mode='r+' if cache.exists() else 'w+',dtype='float32',shape=(N,192))
if done<len(fit):
 model=r.teacher_load(OUT);ck=torch.load('/autodl-fs/data/vibravox_dual_teacher_20261105/audio_teacher/best_backbone_private.pt',map_location='cpu',weights_only=False);model.mods.embedding_model.load_state_dict(ck['embedding_model']);del ck;model.eval();waves=afile['wave'];batchsize=8 if DEVICE=='cuda' else 1;started=time.time()
 for start in range(done,len(fit),batchsize):
  ii=fit[start:start+batchsize]
  with torch.no_grad():z=F.normalize(model.encode_batch(torch.tensor(waves[ii].copy(),device=DEVICE)).squeeze(1),dim=1).cpu().numpy()
  tf[ii]=z;tf.flush();cp.write_text(json.dumps(dict(done=start+len(ii),total=len(fit))))
  if start%10==0:status(state='extracting_frozen_audio_teacher',done=start+len(ii),total=len(fit),device=DEVICE,elapsed_seconds=time.time()-started)
 del waves,model;gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0);torch.cuda.empty_cache()
# Reuse pure ACC local backbone trained only on fit, selected on develop.
accpath=OUT/'local_acc_embeddings_private.npy'
if accpath.exists():za=np.load(accpath,mmap_mode='r')
else:
 d['acc_vel_300']=afile['acc_vel_300'];x,_=r.scale_branch(d,'acc_vel_300',list(range(1,10)));net=r.Student().to(DEVICE);ck=torch.load('/autodl-fs/data/raw9_audioft_closed_20261103/beta_0.0/student_private.pt',map_location='cpu',weights_only=False);net.load_state_dict(ck['state']);del ck;net.eval();za=np.zeros((N,192),dtype='float32');bs=32 if DEVICE=='cuda' else 4
 with torch.no_grad():
  for start in range(0,N,bs):za[start:start+bs]=net(torch.tensor(x[start:start+bs].copy(),device=DEVICE)).cpu().numpy()
 np.save(accpath,za);del net,x;d.pop('acc_vel_300');gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0);torch.cuda.empty_cache()
class Project(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,128))
 def forward(self,x):return F.normalize(self.net(x),dim=1)
def supcon(z,y):
 eye=torch.eye(len(z),device=z.device,dtype=torch.bool);positive=(y[:,None]==y[None,:])&~eye;s=z@z.T/.1;lp=s-torch.logsumexp(s.masked_fill(eye,-1e9),1,keepdim=True);return -((lp*positive).sum(1)/positive.sum(1).clamp_min(1)).mean()
outputs={'local_acc_backbone':za};selected={};users=list(range(1,10));words=np.unique(d['word'][fit]);lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in users for w in words}
for mode in ['acc_projection','vibravox_audio_assisted_projection']:
 torch.manual_seed(42);rng=np.random.RandomState(42);am=Project().to(DEVICE);sm=Project().to(DEVICE);head=nn.Linear(128,9).to(DEVICE);opt=torch.optim.AdamW(list(am.parameters())+list(sm.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-4);best=float('inf');history=[];best_state=None
 for step in range(1,2001):
  uu=rng.choice(users,8,replace=False);ww=rng.choice(words,2,replace=False);ii=np.array([int(rng.choice(lookup[u,w])) for u in uu for w in ww]);y=torch.tensor(d['user'][ii]-1,device=DEVICE,dtype=torch.long);zs=sm(torch.tensor(za[ii].copy(),device=DEVICE));loss=.5*F.cross_entropy(head(zs),y)+.1*supcon(zs,y)
  if mode=='vibravox_audio_assisted_projection':
   zt=am(torch.tensor(tf[ii].copy(),device=DEVICE));loss=(F.cross_entropy(head(zt),y)+F.cross_entropy(head(zs),y))/2+.1*supcon(torch.cat([zt,zs]),torch.cat([y,y]))+.1*(1-(zt*zs).sum(1)).mean()
  opt.zero_grad();loss.backward();opt.step()
  if step%100==0:
   with torch.no_grad():zz=sm(torch.tensor(za[dev].copy(),device=DEVICE));vl=float(F.cross_entropy(head(zz),torch.tensor(d['user'][dev]-1,device=DEVICE,dtype=torch.long)))
   history.append(dict(step=step,dev_loss=vl));status(state='adapting_projection',mode=mode,step=step,dev_loss=vl)
   if vl<best:best=vl;best_state={k:v.detach().cpu().clone() for k,v in sm.state_dict().items()};selected[mode]=dict(step=step,dev_loss=vl)
 sm.load_state_dict(best_state);torch.save(best_state,OUT/(mode+'_private.pt'))
 with torch.no_grad():outputs[mode]=np.concatenate([sm(torch.tensor(za[i:i+128].copy(),device=DEVICE)).cpu().numpy() for i in range(0,N,128)])
 (OUT/(mode+'_history.json')).write_text(json.dumps(history,indent=2))
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
(OUT/'protocol.json').write_text(json.dumps(dict(dataset='mei9 raw multisource2697paired rows; identical processing and historical content role assignment',fit_per_identity=160,develop_per_identity=40,authentication_train_per_identity=20,authentication_calibrate_per_identity=20,final_per_identity='58-60;all8attack sources',teacher='fullybackbonefinetuned VibraVox Audio ECAPA;frozen on local data;no local enrollment/test Audio inference',student='frozen local pure ACC ResNet18 baseline body + trainable192-128-128projection;not VibraVox ACC backbone transfer',selection=selected,limits='local content mapping partly inferred;closed registered identity setting;one seed;no claim of unseen local identity;Audio projection trainable but teacherbackbone frozen'),indent=2));status(state='complete',registered_users=9)

