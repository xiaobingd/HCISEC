"""Frozen VoxCeleb ECAPA192 -> ACC192, isolated-user K-shot pilot.

New paired-waveform cohort, not an assumed mapping to old numbered spectra.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import signal,ndimage
sys.path.insert(0,'/root')
import run_identity_kd_pilot_20261103 as p
WHO=['bo','bro','jin','ke','mei','qi','qis','xin','yue']
INNER=[1,2,3,6,7];BASE=[1,2,3,6,7,8,9];TARGET=5;IMPOSTOR=4;KS=[1,3,5,10,20]
EXPECTED={'embedding_model.ckpt':'0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2','classifier.ckpt':'fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535','hyperparams.yaml':'6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41','label_encoder.txt':'e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9','mean_var_norm_emb.ckpt':'cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(root,name,x):p.save(root/name,x)
class Student(p.Encoder):
 def __init__(self):super().__init__();self.body.fc=nn.Linear(512,192)

def prepare(root,source):
 fields={k:[] for k in ['acc','wave','user','word','role','id']};records=[];hashes={};all_audio={};all_acc={};audit={};excluded=[]
 for label,name in enumerate(WHO,1):
  ap=source/(name+'_acc_final.npy');mp=source/(name+'_audio_final.npy');a=np.load(ap);m=np.load(mp);assert a.shape==m.shape and a.ndim==3 and a.shape[1:]==(20896,1);assert np.isfinite(a).all() and np.isfinite(m).all();hashes[name]=dict(acc=sha(ap),audio=sha(mp));audit[name]=dict(rows=len(a),retained=len(a)-100)
  for i in range(len(a)):
   ah=hashlib.sha256(a[i].tobytes()).hexdigest();mh=hashlib.sha256(m[i].tobytes()).hexdigest();all_audio.setdefault(mh,[]).append((name,i));all_acc.setdefault(ah,[]).append((name,i))
   if i<100:excluded.append(name+':'+str(i));continue
   w=10+(i-100)%20;role='fit' if w<=17 else 'develop' if w<=20 else 'enroll' if w<=23 else 'final'
   # Historical waveform sample rate is inferred from the supplied resampling
   # script:20896points/2seconds=10448Hz. Never label this as pristine raw IMU.
   ax=a[i,:,0].astype('float64');ax-=ax.mean();ax=signal.resample_poly(ax,125,1306);assert len(ax)==2000
   f,_,z=signal.stft(ax,fs=1000,nperseg=128,noverlap=118,boundary='zeros');band=(f>=40)&(f<=500);db=20*np.log10(abs(z[band])+1e-10);feat=ndimage.zoom(db,(80/db.shape[0],80/db.shape[1]),order=1).astype('float32');assert feat.shape==(80,80)
   mic=m[i,:,0].astype('float64');mic-=mic.mean();mic=signal.resample_poly(mic,1000,653).astype('float32')/32.;assert len(mic)==32000
   fields['acc'].append(feat);fields['wave'].append(mic);fields['user'].append(label);fields['word'].append(w);fields['role'].append(role);fields['id'].append(name+':'+str(i));records.append(dict(sample_id=name+':'+str(i),subject_id=label,name=name,row=i,word_id=w,role=role,pair_status='same_named_array_and_row;originalcapturepair_unverified',acc_row_sha256=ah,audio_row_sha256=mh))
 audio_cross=[items for items in all_audio.values() if len(set(u for u,i in items))>1];acc_cross=[items for items in all_acc.values() if len(set(u for u,i in items))>1]
 retained=set(fields['id']);assert not any(u+':'+str(i) in retained for group in audio_cross+acc_cross for u,i in group),'surviving exact cross-identity duplicate'
 out={k:np.asarray(v) for k,v in fields.items()};assert len(set(out['id']))==len(out['id'])
 for u in range(1,10):
  assert int(((out['user']==u)&(out['role']=='fit')).sum())==80
  assert int(((out['user']==u)&(out['role']=='develop')).sum())==30
  assert int(((out['user']==u)&(out['role']=='enroll')).sum())==30
 manifest=dict(records=records,source_hashes=hashes,excluded_ids=excluded,rule='Uniformly exclude original rows0:100 for every identity,not performance-based;xin/yue contain100 identical cross-identity Audio rows in that region',audio_duplicate_groups=audio_cross,acc_duplicate_groups=acc_cross)
 save(root,'manifest_private.json',manifest);save(root,'audit.json',dict(status='EXPLORATORY_ONLY',source='paired legacy time-domain arrays;not pristine capture waveforms',identity_mapping={str(i+1):n for i,n in enumerate(WHO)},rows=len(records),per_user=audit,retained_cross_identity_exact_duplicates=0,original_cross_identity_audio_duplicate_groups=len(audio_cross),uniformly_excluded_rows=900,limitations=['20word round-major order inferred,not per-record verified','same-row ACC/Audio pairing not traced to original timestamp','historical waveform normalization/filtering already applied','10448Hz from source processing script,not perfile metadata','single-session,one directed held-out identity pair'],outer=dict(target='mei',impostor='ke',base=[WHO[u-1] for u in BASE]),inner=dict(fit=[WHO[u-1] for u in INNER],target='xin',impostor='yue')))
 save(root,'protocol.json',dict(seed=42,methods=['ACC_only','ECAPA_pair','ECAPA_proto','ECAPA_relation','NEG_wrong_user'],student='ResNet18mono192L2;same architecture/initialization/minibatches for everymethod',teacher='Frozen official VoxCeleb ECAPA-TDNN192;no local fine-tuning;hashchecked;no weights transfer between incompatible architectures',frontend='all users same operations:legacyACC10448->1000Hz DCremove;STFT128/hop10,40-500Hz,resize80x80;referencepercentiles Base fit only. Audio10448->16000Hz,DCremove,fixeddivide32,noimageinversion',word_roles=dict(fit=list(range(10,18)),develop=[18,19,20],enroll=[21,22,23],final=list(range(24,30))),user_roles=dict(inner_fit=INNER,inner_target=8,inner_impostor=9,outer_base=BASE,outer_target=TARGET,outer_impostor=IMPOSTOR),backend='ACC-only Kshotcosine;K1/3/5/10/20;10fixedregistrationseeds;eachK thresholdfrominnerunseen pair,thenfrozen',selection='beta.1/.5 frominnerK5 only;checkpoints byBase develop pairAUC;outerresults neverused',scope='raw-array rebuilt cohort;NOT samecodedusers/featuresas previousnumbered-arraypilot;exploratorynotstrictword-disjoint'))
 np.savez_compressed(root/'inputs_private.npz',**out);return out

def scales(raw,users):
 ix=np.isin(raw['user'],users)&(raw['role']=='fit');lo,hi=np.percentile(raw['acc'][ix],[1,99]);out=dict(raw);out['acc']=np.clip(2*(raw['acc']-lo)/(hi-lo)-1,-1,1).astype('float32');return out,dict(lo=float(lo),hi=float(hi),fit_ids=raw['id'][ix].tolist())
def teacher_load(root):
 for f,h in EXPECTED.items():assert sha(root/'model'/f)==h,(f,'official hash mismatch')
 sys.path.insert(0,str(root/'deps'));from speechbrain.pretrained import EncoderClassifier
 model=EncoderClassifier.from_hparams(source=str(root/'model'),savedir=str(root/'model/loaded'),overrides={'pretrained_path':str(root/'model')},run_opts={'device':p.DEVICE});model.eval()
 for q in model.parameters():q.requires_grad_(False)
 assert all(not q.requires_grad for q in model.parameters());return model
@torch.no_grad()
def targets(model,data,users,root,stage):
 # No outer target/impostor Audio inference; only fit/develop of Base users.
 ix=np.flatnonzero(np.isin(data['user'],users)&np.isin(data['role'],['fit','develop']));out=np.zeros((len(data['user']),192),dtype='float32')
 for start in range(0,len(ix),16):
  jj=ix[start:start+16];z=model.encode_batch(torch.as_tensor(data['wave'][jj],device=p.DEVICE)).squeeze(1);out[jj]=F.normalize(z,dim=-1).cpu().numpy()
 assert all(not q.requires_grad for q in model.parameters())
 save(root,stage+'_teacher_inference_ids_private.json',data['id'][ix].tolist());return out
def train(data,fitusers,tfeat,key,beta,folder,steps):
 folder.mkdir(parents=True,exist_ok=True);p.seed();net=Student().to(p.DEVICE);head=nn.Linear(192,len(fitusers)).to(p.DEVICE);params=list(net.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);rng=np.random.RandomState(42);wrong_rng=np.random.RandomState(4342)
 fit=np.flatnonzero(np.isin(data['user'],fitusers)&(data['role']=='fit'));dev=np.flatnonzero(np.isin(data['user'],fitusers)&(data['role']=='develop'));words=list(range(10,18));lookup={(u,w):fit[(data['user'][fit]==u)&(data['word'][fit]==w)] for u in fitusers for w in words};assert all(len(ix)>=2 for ix in lookup.values());proto={u:p.norm(tfeat[fit[data['user'][fit]==u]].mean(0)) for u in fitusers};best=None;hist=[]
 for step in range(1,steps+1):
  net.train();ww=rng.choice(words,2,replace=False);ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in fitusers for w in ww]);y=torch.as_tensor([fitusers.index(u) for u in data['user'][ix]],device=p.DEVICE);z=net(torch.as_tensor(data['acc'][ix,None],device=p.DEVICE));ce=F.cross_entropy(head(z),y);sc=p.sc_loss(z,y);kd=z.new_zeros(())
  if beta:
   if key=='ECAPA_proto':target=np.stack([proto[u] for u in data['user'][ix]])
   else:
    ti=ix.copy()
    if key=='NEG_wrong_user':
     for j,i in enumerate(ix):ti[j]=wrong_rng.choice(lookup[fitusers[(fitusers.index(data['user'][i])+1)%len(fitusers)],data['word'][i]])
    target=tfeat[ti]
   t=torch.as_tensor(target,device=p.DEVICE)
   if key=='ECAPA_relation':
    same=y[:,None]==y[None,:];wi=torch.as_tensor(data['word'][ix],device=p.DEVICE);pos=same&(wi[:,None]!=wi[None,:]);neg=~same;err=(z@z.T-t@t.T).square();kd=.5*(err[pos].mean()+err[neg].mean())
   else:kd=(1-(z*t).sum(1)).mean()
  loss=ce+.1*sc+beta*kd;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
  if step%50==0:
   q=p.pair_diag(p.encode(net,data['acc'][dev]),data['user'][dev],data['word'][dev]);hist.append(dict(step=step,ce=float(ce.detach()),supcon=float(sc.detach()),kd=float(kd.detach()),develop=q))
   if best is None or q['auc']>best[0]:best=(q['auc'],copy.deepcopy(net.state_dict()),step)
 net.load_state_dict(best[1]);net.eval();torch.save(dict(state=best[1],selected_step=best[2]),folder/'student_private.pt');save(folder,'training.json',dict(seed=42,steps=steps,method=key,beta=beta,fit_users=fitusers,fit_ids=data['id'][fit].tolist(),develop_ids=data['id'][dev].tolist(),selected_step=best[2],history=hist));print('STUDENT_DONE',folder.name,key,beta,best[2],flush=True);return net

def main(a):
 root=Path(a.out);root.mkdir(exist_ok=True);save(root,'status.json',dict(state='preparing_waveform_pairs'));raw=prepare(root,Path(a.source));teacher=teacher_load(root);data,scale=scales(raw,INNER);save(root,'inner_scale_private.json',scale);tf=targets(teacher,data,INNER,root,'inner');devix=np.flatnonzero(np.isin(data['user'],INNER)&(data['role']=='develop'));q=p.pair_diag(tf[devix],data['user'][devix],data['word'][devix]);save(root,'teacher_gate.json',dict(quality=q,minimum_auc=.6,passed=q['auc']>=.6,teacher_trainable_parameters=sum(q.numel() for q in teacher.parameters() if q.requires_grad),model_revision='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'))
 if q['auc']<.6:save(root,'status.json',dict(state='stopped_teacher_gate'));return
 methods=['ACC_only','ECAPA_pair','ECAPA_proto','ECAPA_relation','NEG_wrong_user'];selected={};history={};innernets={};save(root,'status.json',dict(state='inner_development'))
 for method in methods:
  cc=[]
  for beta in ([0.] if method=='ACC_only' else [.1,.5]):
   folder=root/'inner'/(method+'_B'+str(beta));net=train(data,INNER,tf,method,beta,folder,a.steps);quality,t,rows,_=p.trial(net,data,8,9,5);rank=(quality['hter'],-quality['auc'],beta);cc.append((rank,beta,net,quality,folder.name));history[folder.name]=dict(beta=beta,quality=quality)
  _,beta,net,quality,name=min(cc,key=lambda x:x[0]);selected[method]=dict(beta=beta,quality=quality,candidate=name,thresholds={})
  for k in KS:
   q,t,rows,_=p.trial(net,data,8,9,k);selected[method]['thresholds'][str(k)]=t
 save(root,'locked_before_final.json',dict(seed=42,steps=a.steps,selected=selected,inner_candidates=history,KS=KS,registration_seeds=list(range(4200,4210)),outer_target=TARGET,outer_impostor=IMPOSTOR,base=BASE,outer_final_seen=False))
 data,scale=scales(raw,BASE);save(root,'outer_scale_private.json',scale);tf=targets(teacher,data,BASE,root,'outer');save(root,'status.json',dict(state='outer_training'));result={};dump={}
 for method,cfg in selected.items():
  net=train(data,BASE,tf,method,cfg['beta'],root/'outer'/method,a.steps);result[method]={}
  for k in KS:
   q,t,rows,scores=p.trial(net,data,TARGET,IMPOSTOR,k,cfg['thresholds'][str(k)]);result[method][str(k)]=dict(quality=q,trials=rows,beta=cfg['beta'])
   for j,(g,n) in enumerate(scores):dump[method+'_K'+str(k)+'_g_'+str(j)]=g;dump[method+'_K'+str(k)+'_n_'+str(j)]=n
  print('OUTER_METHOD_DONE',method,{k:v['quality'] for k,v in result[method].items()},flush=True)
 np.savez_compressed(root/'scores_private.npz',**dump);save(root,'results.json',dict(seed=42,target='mei',impostor='ke',scope='exploratory rebuilt-waveform single directed identity pair;no certifiedwordlabels;notnineusermacro',results=result));save(root,'status.json',dict(state='complete_exploratory'));print('ECAPA_PILOT_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--source',required=True);ap.add_argument('--steps',type=int,default=200);a=ap.parse_args()
 try:main(a)
 except Exception as e:
  Path(a.out).mkdir(exist_ok=True);save(Path(a.out),'status.json',dict(state='failed',error=repr(e)));raise
