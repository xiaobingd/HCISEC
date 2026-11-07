"""Matched last-ResNet-block finetuning, with/without relational Audio KD."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,gc,ctypes,copy
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/root');import run_raw9_300 as r
SRC=Path('/autodl-fs/data/identity_kd_all9_raw_full_multisource_20261103');OUT=Path('/autodl-fs/data/mei9_acc_backbone_finetune_20261107');OUT.mkdir(exist_ok=True);DEVICE='cuda' if torch.cuda.is_available() else 'cpu';r.p.DEVICE=DEVICE;torch.set_num_threads(2 if DEVICE=='cuda' else 1)
def status(**v):
 p=OUT/'status.tmp';p.write_text(json.dumps(v,indent=2));p.replace(OUT/'status.json')
a=np.load(SRC/'raw9_full_features_private.npz',allow_pickle=False);d={k:a[k] for k in ['user','word','role','sample_id']};N=len(d['user']);fit=np.flatnonzero(d['role']=='fit');dev=np.flatnonzero(d['role']=='develop');users=list(range(1,10));words=np.unique(d['word'][fit]);tf=np.load('/autodl-fs/data/mei9_local_finetuned_teacher_transfer_20261107/vibravox_audio_embeddings_private.npy',mmap_mode='r');assert (np.linalg.norm(tf[fit],axis=1)>.9).all()
checkpoint=torch.load('/autodl-fs/data/raw9_audioft_closed_20261103/beta_0.0/student_private.pt',map_location='cpu',weights_only=False);initial_state=checkpoint['state'];net=r.Student().to(DEVICE);net.load_state_dict(initial_state);net.eval()
fp=OUT/'layer3_features_private.npy';progress=OUT/'feature_progress.json';done=json.loads(progress.read_text())['done'] if progress.exists() else 0;features=np.lib.format.open_memmap(fp,mode='r+' if fp.exists() else 'w+',dtype='float32',shape=(N,256,5,5))
if done<N:
 d['acc_vel_300']=a['acc_vel_300'];x,_=r.scale_branch(d,'acc_vel_300',users);bs=32 if DEVICE=='cuda' else 4
 with torch.no_grad():
  for i in range(done,N,bs):
   xx=torch.tensor(x[i:i+bs].copy(),device=DEVICE);b=net.body;y=b.maxpool(b.relu(b.bn1(b.conv1(xx))));y=b.layer3(b.layer2(b.layer1(y)));assert y.shape[1:]==(256,5,5);features[i:i+len(y)]=y.cpu().numpy();features.flush();progress.write_text(json.dumps(dict(done=i+len(y),total=N)))
   if i%40==0:status(state='caching_frozen_early_blocks',done=i+len(y),total=N,device=DEVICE)
 del x;d.pop('acc_vel_300');gc.collect();ctypes.CDLL('libc.so.6').malloc_trim(0)
original=np.load('/autodl-fs/data/mei9_vibravox_teacher_transfer_20261107/local_acc_embeddings_private.npy',mmap_mode='r');outputs={'frozen_local_acc':original};selected={};lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in users for w in words}
def encode_indices(ii):
 result=[];net.eval()
 with torch.no_grad():
  for i in range(0,len(ii),32):
   ff=torch.tensor(features[ii[i:i+32]].copy(),device=DEVICE);h=net.body.layer4(ff);result.append(F.normalize(net.body.fc(torch.flatten(net.body.avgpool(h),1)),dim=1).cpu().numpy())
 return np.concatenate(result)
for mode in ['acc_finetune','audio_guided_acc_finetune']:
 torch.manual_seed(42);rng=np.random.RandomState(42);net.load_state_dict(initial_state);net.eval()
 for name,p in net.named_parameters():p.requires_grad_(name.startswith('body.layer4.') or name.startswith('body.fc.'))
 head=nn.Linear(192,9,bias=False).to(DEVICE)
 with torch.no_grad():head.weight.copy_(torch.tensor(np.stack([r.norm(original[fit[d['user'][fit]==u]].mean(0)) for u in users]),device=DEVICE))
 p4=[p for name,p in net.named_parameters() if name.startswith('body.layer4.')];pf=list(net.body.fc.parameters());opt=torch.optim.AdamW([{'params':p4,'lr':1e-5},{'params':pf,'lr':5e-5},{'params':head.parameters(),'lr':1e-3}],weight_decay=1e-4);watch=net.body.layer4[0].conv1.weight;start_weight=watch.detach().clone();best=-1;best_state=None;history=[]
 for step in range(1,501):
  ww=rng.choice(words,2,replace=False);uu=rng.choice(users,8,replace=False);ii=np.array([int(rng.choice(lookup[u,w])) for u in uu for w in ww]);yy=torch.tensor(d['user'][ii]-1,device=DEVICE,dtype=torch.long);f=torch.tensor(features[ii].copy(),device=DEVICE);h=net.body.layer4(f);z=F.normalize(net.body.fc(torch.flatten(net.body.avgpool(h),1)),dim=1);ce=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),yy);preserve=(1-(z*torch.tensor(original[ii].copy(),device=DEVICE)).sum(1)).mean();loss=ce+.1*r.p.sc_loss(z,yy)+.1*preserve;kd=z.new_zeros(())
  if mode=='audio_guided_acc_finetune':
   t=torch.tensor(tf[ii].copy(),device=DEVICE);error=(z@z.T-t@t.T).square();eye=torch.eye(len(ii),device=DEVICE,dtype=torch.bool);same=(yy[:,None]==yy[None,:])&~eye;different=(yy[:,None]!=yy[None,:]);kd=.5*(error[same].mean()+error[different].mean());loss=loss+.1*kd
  opt.zero_grad();loss.backward();assert watch.grad is not None;nn.utils.clip_grad_norm_(p4+pf+list(head.parameters()),1.);opt.step()
  if step==1 or step%50==0:
   delta=float((watch-start_weight).abs().max().detach());assert delta>0;status(state='finetuning_acc_backbone',mode=mode,step=step,total_steps=500,loss=float(loss.detach()),kd=float(kd.detach()),backbone_max_update=delta,device=DEVICE)
  if step%50==0:
   zv=encode_indices(dev);q=r.pair_auc(zv,d['user'][dev],d['word'][dev]);history.append(dict(step=step,dev_crossword_auc=q,loss=float(loss.detach()),kd=float(kd.detach())));(OUT/(mode+'_history.json')).write_text(json.dumps(history,indent=2))
   if q>best:best=q;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};selected[mode]=dict(step=step,dev_crossword_auc=q,backbone_update=delta);torch.save(dict(state=best_state,selected=selected[mode]),OUT/(mode+'_private.pt'))
 net.load_state_dict(best_state);outputs[mode]=encode_indices(np.arange(N));print('FINETUNED',mode,selected[mode],flush=True);del opt;gc.collect()
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
(OUT/'protocol.json').write_text(json.dumps(dict(student='same local pureACCResNet18 initialization;update layer4 and fc;freeze earlier blocks and BN running statistics',teacher='VibraVoxAudio ECAPA finetuned only on local fit/develop then frozen;Audio embeddings fit only',loss_control='cosineCE +0.1SupCon +0.1 originalACC feature preservation',loss_guided='control +0.1 balanced positive/negative pairwise similarity relation distillation;no direct coordinate equality',training='same batches,seed42,500steps,layer4lr1e-5,fclr5e-5,headlr1e-3',selection=selected,backend='identical RBF train20/calibrate20 per identity;all8attack sources;final recording untouched',limits='closed registered nine identity setting;historical word labels partly inferred;single training seed'),indent=2));status(state='complete',registered_users=9)
