"""Stage 1: frozen per-target features, word-held-out authentication, unseen U3/U8.
All K are prespecified ablations; K=2 is the primary multiprototype hypothesis.
No test or unknown-identity data may select any parameter.
"""
import sys,json,hashlib,copy,importlib.util,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,roc_curve
import joblib
sys.path.insert(0,'/root')
import personal_rbf_auth_20261028 as rbf
exp=rbf.exp;base=exp.base
ROOT=Path('/autodl-fs/data/pira_stage1_20261028');ROOT.mkdir(parents=True,exist_ok=True)
TARGETS=(1,2,4,5,7);UNKNOWN=(3,8);SEEDS=(42,43,44);BUDGETS=(10,20,40,60)
NAMES=['bo']+['user%d'%i for i in range(2,10)]
SOURCE=json.loads(Path('/autodl-fs/data/complete_baseline_teacher_20261028/source_manifest.json').read_text())
DEVICE=exp.D;torch.set_num_threads(2)

def js(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def sha(p):
 h=hashlib.sha256()
 with open(str(p),'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows_for(u):
 a=np.asarray(SOURCE[NAMES[u-1]]['auth_indices']);w=a//10
 out={'fit':a[w<15],'cal':a[(w>=15)&(w<20)],'test':a[w>=20]}
 assert all(not set(out[x])&set(out[y]) for x,y in [('fit','cal'),('fit','test'),('cal','test')])
 assert not set(a)&set(SOURCE[NAMES[u-1]]['cyclegan_indices'])
 return out
ROWS={u:rows_for(u) for u in range(1,10)}
DATA={u:np.load(str(base.PROJECT/'meiy'/str(u)/'acc.npy')).astype('float32')[:,:80,:80] for u in range(1,10)}
for u in DATA:assert sha(base.PROJECT/'meiy'/str(u)/'acc.npy')==SOURCE[NAMES[u-1]]['sha256']

def balanced_rows(u,seed,n):
 a=ROWS[u]['fit'];rng=np.random.RandomState(seed+1000*u);words=list(np.unique(a//10));rng.shuffle(words)
 groups=[list(rng.permutation(a[a//10==w])) for w in words];out=[]
 while len(out)<n:
  for group in groups:
   if group and len(out)<n:out.append(int(group.pop()))
  if not any(groups) and len(out)<n:raise ValueError('not enough enrollment samples')
 return np.asarray(out)

def load_features(u):
 folder=ROOT/('U%d'%u);folder.mkdir(exist_ok=True)
 old=Path('/autodl-fs/data/additional_personal_users_20261028') if u in (2,7) else Path('/autodl-fs/data/complete_baseline_teacher_20261028')
 definition=Path('/autodl-fs/data/cyclegan_forward_aligned_20260928/user%d/cyclegan/model_drop.py'%u) if u in (2,7) else base.PROJECT/'experiments/samebo200safe_user3/model_drop.py'
 meta=json.loads((old/('U%d_baseline/trainb_binary_results.json'%u)).read_text());ckpt=Path(meta['cyclegan_ckpt'])
 splitdir=Path('/autodl-fs/data/cyclegan_identity_isolated_20260928/data_split') if u in (2,7) else Path('/autodl-fs/data/auth_full_pipeline_150x150_20260924/data_split')
 train=np.load(str(splitdir/('user%d/cyclegan_acc.npy'%u)))
 original=np.load(str(base.PROJECT/'meiy'/str(u)/'acc.npy'))
 assert np.array_equal(train,original[np.asarray(SOURCE[NAMES[u-1]]['cyclegan_indices'])]),('upstream index mismatch',u)
 manifest={'target':u,'upstream_checkpoint':str(ckpt),'upstream_sha256':sha(ckpt),'architecture':str(definition),'upstream150_rows':SOURCE[NAMES[u-1]]['cyclegan_indices'],
 'rows':{str(v):{k:x.tolist() for k,x in rs.items()} for v,rs in ROWS.items()},'unknown_users':list(UNKNOWN),'word_roles':{'fit':[0,14],'cal':[15,19],'test':[20,29]},
 'scope':'Word separation applies to authentication only; frozen upstream model has seen these word categories. Historical test recordings reused.',
 'non_targets_without_compatible_frozen_model':[6,9],'source_hashes':{str(v):SOURCE[NAMES[v-1]]['sha256'] for v in DATA}}
 if (folder/'feature_cache.npz').exists():
  prior=json.loads((folder/'manifest.json').read_text());assert prior==manifest
  return dict(np.load(str(folder/'feature_cache.npz'))),manifest
 import modelb
 spec=importlib.util.spec_from_file_location('pira_arch_%d'%u,str(definition));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 modelb.GeneratorAccToAudio=m.GeneratorAccToAudio;modelb.GeneratorAudioToAcc=m.GeneratorAudioToAcc
 model,_=modelb.create_multi_source_model(str(ckpt),freeze_generators=True,proj_dim=128,device=DEVICE);model.eval()
 teacher=base.Teacher().to(DEVICE);ts=torch.load(str(old/('U%d_teacher.pt'%u)),map_location=DEVICE,weights_only=False)
 source_ids=ts['metadata']['source_indices'];assert not any(NAMES[v-1] in source_ids for v in UNKNOWN)
 for name,part in source_ids.items():assert not set(part['fit']+part['selection'])&set(SOURCE[name]['auth_indices'])
 teacher.load_state_dict(ts['state']);teacher.eval()
 cache={}
 with torch.no_grad():
  for v in DATA:
   acc=DATA[v];a=[];t=[]
   for i in range(0,len(acc),16):
    x=exp.tensor(acc[i:i+16]);a.append(base.fusion_inputs(model,x).cpu().numpy());t.append(teacher.acc(x).cpu().numpy())
   cache['U%d_base'%v]=np.concatenate(a);cache['U%d_teacher'%v]=np.concatenate(t);cache['U%d_raw'%v]=rbf.profile(acc)
   assert cache['U%d_base'%v].shape==(len(acc),772)
 np.savez_compressed(str(folder/'feature_cache.npz'),**cache);js(folder/'manifest.json',manifest)
 del model,teacher;torch.cuda.empty_cache()
 return cache,manifest

class Transform:
 def fit(self,blocks):self.scalers=[StandardScaler().fit(b) for b in blocks];return self
 def __call__(self,blocks):return np.concatenate([s.transform(b)/np.sqrt(b.shape[1]) for s,b in zip(self.scalers,blocks)],axis=1)

def prototype(z,k,method,local,seed):
 z=norm(z)
 if k==1: centers=norm(z.mean(0,keepdims=True));labels=np.zeros(len(z),dtype=int)
 else:
  km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(z);centers=norm(km.cluster_centers_)
  if method=='medoids':
   d=np.clip(1-z@z.T,0,2)
   for _ in range(20):
    labels=np.argmin(1-z@centers.T,1);new=[]
    for j in range(k):
     ix=np.flatnonzero(labels==j)
     if not len(ix):new.append(centers[j]);continue
     new.append(z[ix[np.argmin(d[np.ix_(ix,ix)].sum(1))]])
    new=np.stack(new)
    if np.allclose(new,centers):break
    centers=new
  labels=np.argmin(1-z@centers.T,1)
 d=np.clip(1-z@centers.T,0,2);own=d[np.arange(len(z)),labels];global_s=max(float(np.median(own)),1e-4)
 scales=[]
 for j in range(k):
  v=own[labels==j];median=float(np.median(v)) if len(v) else global_s
  scales.append(max((len(v)*median+5*global_s)/(len(v)+5),global_s*.1,1e-4) if local else 1.)
 return {'centers':centers,'scales':np.asarray(scales),'counts':np.bincount(labels,minlength=k)}
def proto_score(model,x):return -np.min(np.clip(1-norm(x)@model['centers'].T,0,2)/model['scales'][None,:],axis=1)

def positive_threshold(s,alpha=.1):
 # Lower-score rejection: finite-sample split-conformal target FRR calibration.
 k=int(np.floor((len(s)+1)*alpha));return float(np.sort(s)[k-1]) if k>=1 else -float('inf')
def metrics(g,n,t):
 a=exp.metrics(g,n,t);a['tar']=1-a['frr'];y=np.r_[np.ones(len(g)),np.zeros(len(n))];ss=np.r_[g,n]
 fpr,tpr,_=roc_curve(y,ss);a['tar_at_far5_roc']=float(tpr[fpr<=.05].max())
 a['tar_at_far5_status']='Test ROC descriptive only; not the deployed threshold or a calibrated guarantee.'
 a['genuine_n']=len(g);a['impostor_n']=len(n);return a

def svdd_fit(x,seed):
 base.seed(seed);tx=torch.tensor(x,dtype=torch.float32,device=DEVICE)
 encoder=nn.Sequential(nn.Linear(x.shape[1],32,bias=False),nn.LeakyReLU(.1),nn.Linear(32,16,bias=False)).to(DEVICE)
 decoder=nn.Sequential(nn.Linear(16,32,bias=False),nn.LeakyReLU(.1),nn.Linear(32,x.shape[1],bias=False)).to(DEVICE)
 opt=torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=.001,weight_decay=1e-4)
 for _ in range(20):
  loss=(decoder(encoder(tx))-tx).square().mean();opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad():
  center=encoder(tx).mean(0);small=center.abs()<.1;center[small]=torch.where(center[small]<0,-torch.ones_like(center[small])*.1,torch.ones_like(center[small])*.1)
 opt=torch.optim.Adam(encoder.parameters(),lr=.001,weight_decay=1e-4)
 for _ in range(50):
  loss=(encoder(tx)-center).square().sum(1).mean();opt.zero_grad();loss.backward();opt.step()
 encoder.eval();return encoder,center
@torch.no_grad()
def svdd_score(pair,x):return -(pair[0](torch.tensor(x,dtype=torch.float32,device=DEVICE))-pair[1]).square().sum(1).cpu().numpy()

def rbf_cv(blocks,y,words):
 folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words));rows=[]
 for c in (1.,10.,100.):
  for gamma in ('scale',.1):
   auc=[]
   for tr,va in folds:
    assert not set(words[tr])&set(words[va]);assert len(np.unique(y[tr]))==2 and len(np.unique(y[va]))==2
    m=rbf.fit([x[tr] for x in blocks],y[tr],c,gamma);auc.append(float(roc_auc_score(y[va],rbf.score(m,[x[va] for x in blocks]))))
   rows.append({'C':c,'gamma':gamma,'cv_auc':float(np.mean(auc))})
 return min(rows,key=lambda r:(-r['cv_auc'],r['C'],str(r['gamma'])))

def main():
 config={'targets':list(TARGETS),'unknown':list(UNKNOWN),'seeds':list(SEEDS),'registration_sizes':list(BUDGETS),'K':[1,2,3,4],
 'primary_multi_K':2,'K_selection':'All K are prespecified ablations, no choosing best K on final scores. Phase 1 screens the multiprototype hypothesis before any K optimization.',
 'features':['raw_profile320','cycle772','cycle772_teacher64'],'frr_calibration':.1,'teacher':'Frozen original personal teacher; ACC branch only, no final real Audio input',
 'cv':'Known-user RBF only, 5 grouped folds inside registration-word training data','next_stage':'Do not auto-start adapters/reliability/new generators; assess stage1 first',
 'P1_scope':'Unknown U3/U8 absent from generator target training and teacher development. Known users used only by RBF fitting/selection; prototype/SVDD boundary and preprocessing use target enrollment only.',
 'sampling':'Word-balanced round-robin enrollment, nested10/20/40/60 for each seed; all final attacks retained',
 'limitations':['No independent new recordings','Authentication-only word holdout, not upstream word holdout','Tail missing row//10 word mapping assumption','No claim of novel method or reproduced PGBL/PAF','No existing compatible frozen G for U6/U9 in this audited cohort; U3/U8 are reserved unknown identities']}
 js(ROOT/'locked_protocol.json',config)
 for u in TARGETS:
  folder=ROOT/('U%d'%u)
  if (folder/'COMPLETE').exists():continue
  print('EXTRACT_START',u,flush=True);cache,manifest=load_features(u);print('EXTRACT_DONE',u,flush=True)
  known=[v for v in range(1,10) if v!=u and v not in UNKNOWN];results=[];scores={}
  def blocks(v,ix,feature):
   if feature=='raw_profile320':return [cache['U%d_raw'%v][ix]]
   if feature=='cycle772':return [cache['U%d_base'%v][ix]]
   return [cache['U%d_base'%v][ix],cache['U%d_teacher'%v][ix]]
  test_ids=np.concatenate([ROWS[v]['test'] for v in range(1,10) if v!=u]);test_uid=np.concatenate([np.full(len(ROWS[v]['test']),v) for v in range(1,10) if v!=u])
  query_users=[v for v in range(1,10) if v!=u]
  for seed in SEEDS:
   enroll={v:balanced_rows(v,seed,60) for v in range(1,10) if v not in UNKNOWN}
   for n in BUDGETS:
    ix=enroll[u][:n]
    for feature in config['features']:
     xb=blocks(u,ix,feature);transform=Transform().fit(xb);x=transform(xb)
     cal=transform(blocks(u,ROWS[u]['cal'],feature));g=transform(blocks(u,ROWS[u]['test'],feature))
     negblocks=[np.concatenate([blocks(v,ROWS[v]['test'],feature)[j] for v in query_users]) for j in range(len(xb))]
     neg=transform(negblocks);locked_models={}
     for method in ('kmeans','medoids'):
      for k in (1,2,3,4):
       if k==1 and method=='medoids':continue
       for local in (False,True):
        name='%s_K%d_%s'%(method,k,'local' if local else 'cosine')
        model=prototype(x,k,method,local,seed);locked_models[name]=(proto_score,model,positive_threshold(proto_score(model,cal)))
     svdd=svdd_fit(x,seed);locked_models['deep_svdd']=(svdd_score,svdd,positive_threshold(svdd_score(svdd,cal)))
     # Known negatives used only for supervised SVM; unknown identities absent.
     train_blocks=[np.concatenate([xb[j]]+[blocks(v,enroll[v][:n],feature)[j] for v in known]) for j in range(len(xb))]
     labels=np.r_[np.ones(n,dtype=bool),np.zeros(n*len(known),dtype=bool)]
     words=np.concatenate([ix//10]+[enroll[v][:n]//10 for v in known])
     c=rbf_cv(train_blocks,labels,words);svm=rbf.fit(train_blocks,labels,c['C'],c['gamma']);svm_t=positive_threshold(rbf.score(svm,blocks(u,ROWS[u]['cal'],feature)))
     lock={'target':u,'seed':seed,'n':n,'feature':feature,'enrollment_rows':ix.tolist(),'known_negative_users':known,'rbf_config':c,'thresholds':{k:float(v[2]) for k,v in locked_models.items()},'rbf_threshold':svm_t}
     tag='seed%d_n%d_%s'%(seed,n,feature);js(folder/(tag+'_locked.json'),lock)
     # All model parameters and thresholds are locked before final scoring.
     for name,(fn,model,t) in list(locked_models.items())+[('rbf', (None,None,svm_t))]:
      if name=='rbf':sg=rbf.score(svm,blocks(u,ROWS[u]['test'],feature));sn=rbf.score(svm,negblocks)
      else:sg=fn(model,g);sn=fn(model,neg)
      row={'target':u,'seed':seed,'n_enroll':n,'feature':feature,'method':name,'threshold':t,'known':metrics(sg,sn[np.isin(test_uid,known)],t),'unknown':metrics(sg,sn[np.isin(test_uid,UNKNOWN)],t),
       'attacks':{str(v):{'n':int(np.sum(test_uid==v)),'accepted':int(np.sum(sn[test_uid==v]>=t))} for v in query_users}}
      results.append(row);scores[tag+'_'+name+'_g']=sg;scores[tag+'_'+name+'_n']=sn
     joblib.dump({'transform':transform,'prototypes':{name:v[1] for name,v in locked_models.items() if name!='deep_svdd'},'rbf':svm,'svdd_state':{k:v.detach().cpu() for k,v in svdd[0].state_dict().items()},'svdd_center':svdd[1].detach().cpu(), 'thresholds':lock['thresholds'],'rbf_threshold':svm_t},str(folder/(tag+'_models.joblib')))
     js(folder/'results.json',results);print('READOUT_DONE',u,seed,n,feature,flush=True)
  np.savez_compressed(str(folder/'scores_private.npz'),negative_uid=test_uid,negative_rows=test_ids,**scores);(folder/'COMPLETE').write_text('complete\n');print('TARGET_DONE',u,flush=True)
 print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
