import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,gc,ctypes
from pathlib import Path
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root');import run_raw9_300 as r
SRC=Path('/autodl-fs/data/identity_kd_all9_raw_full_multisource_20261103');OUT=Path('/autodl-fs/data/mei8_exclude_yue_20261108');OUT.mkdir(exist_ok=True);DEVICE='cuda' if torch.cuda.is_available() else 'cpu';r.p.DEVICE=DEVICE;torch.set_num_threads(2 if DEVICE=='cuda' else 1)
def status(**v):
 p=OUT/'status.tmp';p.write_text(json.dumps(v,indent=2));p.replace(OUT/'status.json')
a=np.load(SRC/'raw9_full_features_private.npz',allow_pickle=False);keep=a['user']!=9;d={k:a[k][keep] for k in ['user','word','role','sample_id']};users=list(range(1,9));assert set(d['user'])==set(users);N=len(d['user']);path=OUT/'scaled_private.npy'
if not path.exists():
 status(state='preparing_eight_user_inputs',rows=N);d['acc_vel_300']=a['acc_vel_300'][keep];x,scale=r.scale_branch(d,'acc_vel_300',users);np.save(path,x);(OUT/'scaling_private.json').write_text(json.dumps(scale,indent=2));del x;d.pop('acc_vel_300');gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
x=np.load(path,mmap_mode='r');status(state='training_fresh_eight_user_acc_backbone',steps=200,device=DEVICE)
import inspect
r._phase_status=status
code=inspect.getsource(r.train).replace('opt.step()',"opt.step()\n  if step==1 or step%10==0: _phase_status(state='training_fresh_eight_user_acc_backbone',step=step,total_steps=steps,loss=float(loss.detach())); print('TRAIN8',step,float(loss.detach()),flush=True)")
exec(code,r.__dict__)
net,step,devauc=r.train(x,d,users,np.zeros((N,192),dtype='float32'),0.,OUT/'pure_acc_backbone',200);z=r.encode(net,x);np.save(OUT/'acc_embeddings_private.npy',z);outputs={'fresh_acc8':z};selected=dict(step=step,dev_crossword_auc=devauc);status(state='evaluating_eight_users',backbone_selected=selected)
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
(OUT/'protocol.json').write_text(json.dumps(dict(excluded_identity='yue/user9;excluded from scaling,backbone fit/develop,head fit,calibration and final queries',scope='eight registered identities;not identity-unseen open set;exploratory user population chosen after observing prior results',backbone='fresh ResNet18 random initialization;no nine-user checkpoint reuse;CE+0.1SupCon,200steps,seed42',selection=selected,head='StandardScaler+balanced RBF,SVC C/gamma/threshold from calibration only;20train+20cal/user',teacher='no Audio teacher in this first exclusion-control experiment;do not reuse locally nine-user-finetuned teacher for a strict no-yue condition',data='same source and inferred content groups,removed yue only;do not delete original raw files',counts={u:int((d['user']==i).sum()) for i,u in enumerate(r.NAMES[:8],1)},limits='word labels partly inferred;single seed;comparison to prior nine-user setting changes identity population'),indent=2));status(state='complete',registered_users=8)
