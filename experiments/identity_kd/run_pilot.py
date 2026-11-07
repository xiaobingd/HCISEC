"""Seed42 bounded public-teacher pilot; gate before any local KD training.

Historical local labels are inferred, so results never certify word-disjointness.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,copy,hashlib,io,json,random,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from scipy import signal,ndimage
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score,roc_curve
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)

def seed(s=42):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def save(path,x):Path(path).write_text(json.dumps(x,indent=2))
class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.body=resnet18(weights=None);self.body.conv1=nn.Conv2d(1,64,7,2,3,bias=False);self.body.fc=nn.Linear(512,128)
 def forward(self,x):return F.normalize(self.body(x),dim=1)
def sc_loss(z,y):
 sim=z@z.T/.1;diag=torch.eye(len(y),device=y.device,dtype=torch.bool);pos=(y[:,None]==y[None,:])&~diag;valid=pos.any(1)
 logp=sim-torch.logsumexp(sim.masked_fill(diag,-1e9),1,keepdim=True)
 return -(logp.masked_fill(~pos,0).sum(1)/pos.sum(1).clamp_min(1))[valid].mean()
@torch.no_grad()
def encode(net,x):
 net.eval();return np.concatenate([net(torch.as_tensor(x[i:i+64,None],device=DEVICE)).cpu().numpy() for i in range(0,len(x),64)])
def pair_diag(z,y,w):
 sim=z@z.T;upper=np.triu(np.ones(sim.shape,bool),1);pos=upper&(y[:,None]==y[None,:])&(w[:,None]!=w[None,:]);neg=upper&(y[:,None]!=y[None,:])&(w[:,None]==w[None,:]);a=sim[pos];b=sim[neg]
 assert len(a) and len(b)
 return dict(auc=float(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[a,b])),same_user_different_word_mean=float(a.mean()),different_user_same_word_mean=float(b.mean()),positive_pairs=len(a),hard_negative_pairs=len(b))
def spectrogram(w,fs):
 if w.dtype.kind=='i':w=w.astype('float64')/np.iinfo(w.dtype).max
 else:w=w.astype('float64')
 assert fs==16000 and w.ndim==1;w=w-w.mean();w=np.pad(w,(0,max(0,16000-len(w))))[:16000]
 _,_,z=signal.stft(w,fs=fs,nperseg=512,noverlap=352,boundary=None,padded=False)
 hz=np.linspace(0,8000,257);edges=700*(10**(np.linspace(0,2595*np.log10(1+8000/700),82)/2595)-1)
 filters=np.maximum(0,np.minimum((hz[None]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),(edges[2:,None]-hz[None])/(edges[2:,None]-edges[1:-1,None])))
 filters*=2/(edges[2:]-edges[:-2])[:,None];db=10*np.log10(filters@(abs(z)**2)+1e-10)
 return ndimage.zoom(db,(80/db.shape[0],80/db.shape[1]),order=1).astype('float32')
def public_data(root):
 m=json.loads((root/'manifest.json').read_text());out={}
 with zipfile.ZipFile(root/'mini_speech_commands.zip') as archive:
  for role,users in m['speaker_roles'].items():
   x=[];y=[];w=[]
   for u,who in enumerate(users):
    for item in m['records'][who]:
     fs,a=wavfile.read(io.BytesIO(archive.read(item['member'])));x.append(spectrogram(a,fs));y.append(u);w.append(item['word'])
   out[role]=dict(x=np.stack(x),y=np.array(y),w=np.array(w))
 lo,hi=np.percentile(out['train']['x'],[1,99])
 for role in out:out[role]['x']=np.clip(2*(out[role]['x']-lo)/(hi-lo)-1,-1,1).astype('float32')
 return out,dict(lo=float(lo),hi=float(hi),manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
def teacher_train(public,root):
 folder=root/'public_teacher';folder.mkdir(exist_ok=True);seed();net=Encoder().to(DEVICE);head=nn.Linear(128,64).to(DEVICE);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=3e-4,weight_decay=1e-4);rng=np.random.RandomState(42);x=public['train']['x'];y=public['train']['y'];best=None;hist=[]
 for step in range(1,401):
  net.train();users=rng.choice(64,8,replace=False);ix=np.concatenate([rng.choice(np.flatnonzero(y==u),4,replace=False) for u in users]);xx=torch.as_tensor(x[ix,None],device=DEVICE);yy=torch.as_tensor(y[ix],device=DEVICE);xx=torch.roll(xx,int(rng.randint(-3,4)),-1)+.005*torch.randn_like(xx);z=net(xx);ce=F.cross_entropy(head(z),yy);sc=sc_loss(z,yy);loss=ce+.1*sc;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.);opt.step()
  if step%50==0:
   q=pair_diag(encode(net,public['validation']['x']),public['validation']['y'],public['validation']['w']);hist.append(dict(step=step,ce=float(ce.detach()),supcon=float(sc.detach()),diagnostic=q));print('PUBLIC_TEACHER',step,q,flush=True)
   if best is None or q['auc']>best[0]:best=(q['auc'],copy.deepcopy(net.state_dict()),step)
   save(root/'status.json',dict(state='public_teacher_training',step=step,steps=400))
 net.load_state_dict(best[1]);net.eval()
 for p in net.parameters():p.requires_grad_(False)
 torch.save(dict(state=best[1],selected_step=best[2]),folder/'teacher_private.pt');test=pair_diag(encode(net,public['test']['x']),public['test']['y'],public['test']['w']);save(folder/'diagnostics.json',dict(history=hist,selected_step=best[2],test=test,source='GSC official mini;64train16dev16test speakers disjoint',steps=400,seed=42,architecture='ResNet18 mono128;CE64+.1SupCon;not word classifier'))
 return net,test
def local_data(data,manifest):
 rows=json.loads(manifest.read_text())['records'];out={k:[] for k in ['acc','audio','user','word','role','id']}
 for u in range(1,10):
  acc=np.load(data/('acc.%d.npy'%u)).astype('float32')[:,:80,:80];audio=np.load(data/('audio.%d.npy'%u)).astype('float32')[:,:80,:80]
  for r in rows:
   if r['subject_id']!=u:continue
   out['acc'].append(acc[r['row']]);out['audio'].append(audio[r['row']]);out['user'].append(u);out['word'].append(r['word_id']);out['role'].append(r['role']);out['id'].append(r['sample_id'])
 return {k:np.asarray(v) for k,v in out.items()}
def standardize(data,fit):
 out=dict(data);stats={}
 for domain in ['acc','audio']:
  mean=float(data[domain][fit].mean());std=float(data[domain][fit].std());assert std>1e-8;out[domain]=((data[domain]-mean)/std).astype('float32');stats[domain]=dict(mean=mean,std=std)
 return out,stats
def threshold(g,n):
 vals=np.unique(np.r_[g,n]);vals=np.r_[vals,np.nextafter(vals[-1],np.inf)];return float(min(vals,key=lambda t:(round(((g<t).mean()+(n>=t).mean())/2,12),round(max((g<t).mean(),(n>=t).mean()),12),-t)))
def trial(net,data,target,imp,k=5,threshold_value=None):
 en=np.flatnonzero((data['user']==target)&(data['role']=='enroll'));pos=np.flatnonzero((data['user']==target)&(data['role']=='final'));neg=np.flatnonzero((data['user']==imp)&(data['role']=='final'));assert not set(en)&(set(pos)|set(neg));ze=encode(net,data['acc'][en]);zg=encode(net,data['acc'][pos]);zn=encode(net,data['acc'][neg]);rows=[];raw=[]
 for ss in range(10):
  # Balanced over enrolment words before taking repeated examples.
  rng=np.random.RandomState(4200+ss);chosen=[];words=rng.permutation(np.unique(data['word'][en]))
  while len(chosen)<k:
   for w in words:
    available=[j for j in np.flatnonzero(data['word'][en]==w) if j not in chosen]
    if available:chosen.append(int(rng.choice(available)))
    if len(chosen)==k:break
  proto=norm(ze[chosen].mean(0));g=zg@proto;n=zn@proto;raw.append((g,n));tau=threshold(g,n) if threshold_value is None else threshold_value
  auc=float(roc_auc_score(np.r_[np.ones(len(g)),np.zeros(len(n))],np.r_[g,n]));tar=float((g>=tau).mean());far=float((n>=tau).mean());rows.append(dict(registration_seed=4200+ss,k=k,tar=tar,far=far,hter=(1-tar+far)/2,auc=auc,genuine_n=len(g),impostor_n=len(n),threshold=tau))
 # Only internal-development calls may use labels to choose the threshold.
 t=threshold(np.concatenate([g for g,n in raw]),np.concatenate([n for g,n in raw])) if threshold_value is None else threshold_value
 for row,(g,n) in zip(rows,raw):
  row['threshold']=t;row['tar']=float((g>=t).mean());row['far']=float((n>=t).mean());row['hter']=(1-row['tar']+row['far'])/2
 return {m:float(np.mean([x[m] for x in rows])) for m in ['tar','far','hter','auc']},t,rows,raw
def train_student(data,fitusers,teacher_state,teacher_features,key,beta,root,steps=200):
 folder=root/key;folder.mkdir(exist_ok=True,parents=True);seed();net=Encoder().to(DEVICE)
 if key.startswith(('KD1','KD3','NEG')):net.load_state_dict(teacher_state)
 head=nn.Linear(128,len(fitusers)).to(DEVICE);opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-4);rng=np.random.RandomState(42);fit=np.flatnonzero(np.isin(data['user'],fitusers)&(data['role']=='fit'));dev=np.flatnonzero(np.isin(data['user'],fitusers)&(data['role']=='develop'));mapping={u:i for i,u in enumerate(fitusers)};lookup={(u,w):fit[(data['user'][fit]==u)&(data['word'][fit]==w)] for u in fitusers for w in range(12)};assert all(len(x)>=2 for x in lookup.values());best=None;history=[]
 for step in range(1,steps+1):
  net.train();words=rng.choice(12,2,replace=False);ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in fitusers for w in words]);y=torch.as_tensor([mapping[u] for u in data['user'][ix]],device=DEVICE);z=net(torch.as_tensor(data['acc'][ix,None],device=DEVICE));ce=F.cross_entropy(head(z),y);sc=sc_loss(z,y);kd=z.new_zeros(())
  if beta:
   ti=ix.copy()
   if key.startswith('NEG1'):
    for j,i in enumerate(ix):
     u=data['user'][i];other=fitusers[(fitusers.index(u)+1)%len(fitusers)];ti[j]=rng.choice(lookup[other,data['word'][i]])
   target=torch.as_tensor(teacher_features[ti],device=DEVICE);kd=(1-(z*target).sum(1)).mean()
  loss=ce+.1*sc+beta*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.);opt.step()
  if step%50==0:
   q=pair_diag(encode(net,data['acc'][dev]),data['user'][dev],data['word'][dev]);history.append(dict(step=step,ce=float(ce.detach()),supcon=float(sc.detach()),kd=float(kd.detach()),develop=q))
   if best is None or q['auc']>best[0]:best=(q['auc'],copy.deepcopy(net.state_dict()),step)
 net.load_state_dict(best[1]);net.eval();torch.save(dict(state=best[1],selected_step=best[2]),folder/'student_private.pt');save(folder/'training.json',dict(steps=steps,seed=42,beta=beta,fit_users=fitusers,fit_ids=data['id'][fit].tolist(),checkpoint_develop_ids=data['id'][dev].tolist(),history=history,selected_step=best[2]));print('STUDENT_DONE',key,beta,best[2],flush=True);return net
def main(a):
 root=Path(a.out);root.mkdir(exist_ok=True);audit=json.loads((root/'audit_public.json').read_text());assert audit['pilot_allowed'];assert audit['status']=='PASS' or a.allow_exploratory,'formal P0 failed;refusing without explicit exploratory mode'
 save(root/'status.json',dict(state='public_features',formal_status=audit['status']));public,front=public_data(Path(a.public));save(root/'public_frontend.json',front);teacher,pubdiag=teacher_train(public,root)
 raw=local_data(Path(a.data),root/'manifest_private.json');fit=np.isin(raw['user'],[2,3,5,6,7])&(raw['role']=='fit');data,stats=standardize(raw,fit);save(root/'inner_normalization_private.json',stats);ix=np.flatnonzero(np.isin(data['user'],[2,3,5,6,7])&(data['role']=='develop'));localdiag=pair_diag(encode(teacher,data['audio'][ix]),data['user'][ix],data['word'][ix]);gate=dict(public=pubdiag,local_base_audio=localdiag,minimum_auc=.60,pass_gate=pubdiag['auc']>=.60 and localdiag['auc']>=.60,scope='pilot minimum diagnostic,not statistical proof')
 save(root/'teacher_gate.json',gate)
 if not gate['pass_gate']:
  save(root/'status.json',dict(state='stopped_teacher_gate',formal_status=audit['status'],reason='Teacher identity diagnostic below preregistered pilot AUC0.60;no student KD or outer final evaluated'));print('STOP_TEACHER_GATE',gate,flush=True);return
 save(root/'status.json',dict(state='inner_training',formal_status=audit['status']));state=copy.deepcopy(teacher.state_dict());seed(2042);random_teacher=Encoder().to(DEVICE).eval()
 def fit_targets(net,d,users):
  ix=np.flatnonzero(np.isin(d['user'],users)&(d['role']=='fit'));features=np.zeros((len(d['user']),128),dtype='float32');features[ix]=encode(net,d['audio'][ix]);return features
 tf=fit_targets(teacher,data,[2,3,5,6,7]);rf=fit_targets(random_teacher,data,[2,3,5,6,7]);dev={};selected={}
 for key in ['KD0','KD1','KD2','KD3','NEG1','NEG2']:
  candidates=[]
  for beta in ([0.] if key in ['KD0','KD1'] else [.1,.5]):
   name=key+'_B'+str(beta);net=train_student(data,[2,3,5,6,7],state,rf if key=='NEG2' else tf,name,beta,root/'inner',steps=a.steps);q,t,rows,_=trial(net,data,8,9,5);candidates.append((q['hter'],-q['auc'],beta,t,q,name));dev[name]=dict(quality=q,threshold=t,registration_trials=rows)
  chosen=min(candidates);selected[key]=dict(beta=chosen[2],threshold=chosen[3],quality=chosen[4],name=chosen[5])
 save(root/'locked_outer_protocol.json',dict(seed=42,steps=a.steps,selected=selected,inner_development=dev,outer_target=1,outer_impostor=4,base=[2,3,5,6,7,8,9],K=5,registration_seeds=list(range(4200,4210)),outer_final_seen=False,formal_status=audit['status']))
 base=[2,3,5,6,7,8,9];data,stats=standardize(raw,np.isin(raw['user'],base)&(raw['role']=='fit'));save(root/'outer_normalization_private.json',stats);tf=fit_targets(teacher,data,base);rf=fit_targets(random_teacher,data,base);final={};dump={}
 for key,cfg in selected.items():
  net=train_student(data,base,state,rf if key=='NEG2' else tf,key+'_outer',cfg['beta'],root/'outer',steps=a.steps);q,t,rows,scores=trial(net,data,1,4,5,cfg['threshold']);final[key]=dict(quality=q,trials=rows,beta=cfg['beta'])
  for j,(g,n) in enumerate(scores):dump[key+'_g_'+str(j)]=g;dump[key+'_n_'+str(j)]=n
 np.savez_compressed(root/'final_scores_private.npz',**dump);save(root/'results.json',dict(seed=42,scope='exploratory single directed-pair pipecheck;not formalword-disjoint;not9userestimate',target='U1',impostor='U4',formal_status=audit['status'],results=final));save(root/'status.json',dict(state='complete_exploratory',formal_status=audit['status']));print('PILOT_COMPLETE',final,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--data',required=True);p.add_argument('--public',required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--allow-exploratory',action='store_true');a=p.parse_args()
 try:main(a)
 except Exception as e:
  save(Path(a.out)/'status.json',dict(state='failed',error=repr(e)));raise
