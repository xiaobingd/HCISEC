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
ROOT=Path('/autodl-fs/data');SRC=ROOT/'identity_kd_all9_raw_full_multisource_20261103';OUT=ROOT/'mei8_sameword_teacher_residual_20261108';OUT.mkdir(exist_ok=True);DEVICE='cpu';torch.set_num_threads(1)
def status(**v):
 p=OUT/'status.tmp';p.write_text(json.dumps(v,indent=2));p.replace(OUT/'status.json')
a=np.load(SRC/'raw9_full_features_private.npz');keep=a['user']!=9;d={k:a[k][keep] for k in ['user','word','role','sample_id']};users=list(range(1,9));za=np.load(ROOT/'mei8_exclude_yue_20261108/acc_embeddings_private.npy');tm=np.load(ROOT/'mei9_vibravox_teacher_transfer_20261107/vibravox_audio_embeddings_private.npy')[keep];assert len(za)==len(d['user']);fit=np.flatnonzero(d['role']=='fit');dev=np.flatnonzero(d['role']=='develop');assert set(d['user'])==set(users)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
class Residual(nn.Module):
 def __init__(self):
  super().__init__();self.a=nn.Linear(192,64);self.b=nn.Linear(64,192);nn.init.zeros_(self.b.weight);nn.init.zeros_(self.b.bias)
 def forward(self,x):
  delta=self.b(F.gelu(self.a(x)));delta=.1*delta/torch.linalg.vector_norm(delta,dim=1,keepdim=True).clamp_min(1.);return F.normalize(x+delta,dim=1)
outputs={'raw_acc8_teacher':za}
BASE=ROOT/'mei8_selective_teacher_residual_20261108'
for mode in ['residual_identity_only','residual_uniform_audio','residual_selective_audio']:
 net=Residual();net.load_state_dict(torch.load(BASE/(mode+'_private.pt'),map_location='cpu'));net.eval()
 with torch.no_grad():outputs[mode]=net(torch.tensor(za)).numpy()
train=[];cal=[];test=[]
for u in users:
 for w in range(24,30):
  ii=np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:8]
  assert len(ii)==8
  ii=np.random.RandomState(42+int(w)).permutation(ii)
  train.extend(ii[:4]);cal.extend(ii[4:6]);test.extend(ii[6:8])
train=np.array(train);cal=np.array(cal);test=np.array(test);results={}
assert not set(train)&set(test) and not set(cal)&set(test)
(OUT/'split_private.json').write_text(json.dumps({k:d['sample_id'][v].tolist() for k,v in [('train',train),('cal',cal),('test',test)]},indent=2))
for mode,z in outputs.items():
 scaler=StandardScaler().fit(z[train]);zz=scaler.transform(z);results[mode]={}
 for u in users:
  opts=[]
  for C in [.1,1,10]:
   for gamma in ['scale',.01,.1]:
    clf=SVC(C=C,gamma=gamma,class_weight='balanced').fit(zz[train],d['user'][train]==u);sc=clf.decision_function(zz[cal]);pos=d['user'][cal]==u;tau=r.eer_threshold(sc[pos],sc[~pos]);hter=float(((sc[pos]<tau).mean()+(sc[~pos]>=tau).mean())/2);opts.append((hter,C,str(gamma),clf,tau))
  _,C,gamma,clf,tau=min(opts,key=lambda q:(q[0],q[1],q[2]));sc=clf.decision_function(zz[test]);pos=d['user'][test]==u;tar=float((sc[pos]>=tau).mean());far=float((sc[~pos]>=tau).mean());results[mode][r.NAMES[u-1]]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(pos,sc)),genuine_n=int(pos.sum()),impostor_n=int((~pos).sum()),C=C,gamma=gamma,threshold=tau)
 (OUT/'results_private.json').write_text(json.dumps(results,indent=2));print('EVALUATED',mode,flush=True)
(OUT/'protocol.json').write_text(json.dumps(dict(scope='eight registered users; same six words across all identities and auth roles',split='only former final recordings; first eight common repeats per word, same seeded permutation across users:4train/2cal/2test',backbones='frozen; no new teacher/residual training',limits='one split seed; inferred word mapping; not directly sample-count-matched to previous crossword result'),indent=2));status(state='complete')
