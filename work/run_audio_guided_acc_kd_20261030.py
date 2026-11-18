"""Protocol A: real Audio privileged teachers -> ACC-only students, no CycleGAN."""
import sys,json,copy,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,roc_curve
sys.path.insert(0,'/root')
import run_nine_sameword_balanced_20261030 as util
old=util.old;D=old.D;ROOT=Path('/autodl-fs/data/mei_audio_guided_acc_kd_20261030');SEEDS=[42,43,44];STEPS=400;ROLES={'train':list(range(18)),'enroll':[18,19],'validation':[20,21,22],'calibration':[23,24],'test':[25,26,27,28,29]};KS=[3,5,10,20]

def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=D)
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def threshold(g,n):
 g=np.sort(g);n=np.sort(n);v=np.unique(np.r_[g,n]);v=np.r_[v,np.nextafter(v[-1],np.inf)];frr=np.searchsorted(g,v)/len(g);far=1-np.searchsorted(n,v)/len(n);i=np.lexsort((-v,np.round(np.maximum(far,frr),12),np.round((far+frr)/2,12)))[0];return float(v[i])

class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,32,3,2,1),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU());self.proj=nn.Linear(128,128)
 def forward(self,x):
  h=self.conv(x).flatten(2);return F.normalize(self.proj(torch.cat([h.mean(2),h.std(2,unbiased=False)],1)),dim=1)
class Classifier(nn.Module):
 def __init__(self):super().__init__();self.encoder=Encoder();self.head=nn.Linear(128,9);self.aam=False
 def forward(self,x):z=self.encoder(x);return z,16*F.linear(z,F.normalize(self.head.weight,dim=1)) if self.aam else self.head(z)
class Fusion(nn.Module):
 def __init__(self,acc,audio):
  super().__init__();self.acc=copy.deepcopy(acc.encoder);self.audio=copy.deepcopy(audio.encoder);self.fuse=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,128));self.head=nn.Linear(128,9)
  for q in list(self.acc.parameters())+list(self.audio.parameters()):q.requires_grad_(False)
 def forward(self,x,m):z=F.normalize(self.fuse(torch.cat([self.acc(x),self.audio(m)],1)),dim=1);return z,self.head(z)

def data_load():
 old.records();out={r:{k:[] for k in ['acc','audio','labels','words','ids']} for r in ROLES}
 for u in range(1,10):
  s=old.M['sources'][str(u)];words=np.asarray(s['content_groups']);acc=np.load(str(old.DATA/('acc.%d.npy'%u))).astype('float32')[:,:80,:80];audio=np.load(str(old.DATA/('audio.%d.npy'%u))).astype('float32')[:,:80,:80]
  for role,groups in ROLES.items():
   ix=np.flatnonzero(np.isin(words,groups))
   if role=='enroll':ix=np.asarray(sorted(ix,key=lambda j:((j//10 if j<100 else (j-100)//20),words[j])))
   out[role]['acc'].append(acc[ix]);out[role]['audio'].append(audio[ix]);out[role]['labels'].append(np.full(len(ix),u-1));out[role]['words'].append(words[ix]);out[role]['ids'].extend([str(u)+':'+str(j) for j in ix])
 allids=[]
 for role,v in out.items():
  for k in ['acc','audio','labels','words']:v[k]=np.concatenate(v[k])
  allids+=v['ids']
 assert len(allids)==len(set(allids));assert [len(out[r]['labels']) for r in ['train','enroll','validation','calibration']]==[1620,180,270,180]
 return out

@torch.no_grad()
def evaluate_embeddings(model,data,role,domain='acc'):
 z=[];logits=[];x=data[role][domain]
 for i in range(0,len(x),64):
  a=tensor(x[i:i+64,None]);value=model(a,tensor(data[role]['audio'][i:i+64,None])) if domain=='fusion' else model(a)
  z.append(value[0].cpu().numpy());logits.append(value[1].cpu().numpy())
 return np.concatenate(z),np.concatenate(logits)

def inference(model,data,role,domain='acc'):
 if domain!='fusion':return evaluate_embeddings(model,data,role,domain)
 with torch.no_grad():
  zz=[];ll=[]
  for i in range(0,len(data[role]['acc']),64):
   z,l=model(tensor(data[role]['acc'][i:i+64,None]),tensor(data[role]['audio'][i:i+64,None]));zz.append(z.cpu().numpy());ll.append(l.cpu().numpy())
 return np.concatenate(zz),np.concatenate(ll)

def prototypes(z,labels,k=None):return normalize(np.stack([z[np.flatnonzero(labels==u)[:k]].mean(0) for u in range(9)]))
def validation_quality(model,data,domain):
 model.eval();ix=np.concatenate([np.flatnonzero((data['train']['labels']==u)&(data['train']['words']==w))[:1] for u in range(9) for w in ROLES['train']]);anchor={k:data['train'][k][ix] for k in ['acc','audio','labels']};ze,_=inference(model,{'anchor':anchor},'anchor',domain);z,l=inference(model,data,'validation',domain);centers=prototypes(ze,anchor['labels']);s=z@centers.T;labs=data['validation']['labels'];words=data['validation']['words'];per=[]
 for u in range(9):
  accepted=np.zeros(len(labs),dtype=bool);y=labs==u
  for w in np.unique(words):
   va=words==w;tr=~va;t=threshold(s[tr,u][y[tr]],s[tr,u][~y[tr]]);accepted[va]=s[va,u]>=t
  tar=float(accepted[y].mean());far=float(accepted[~y].mean());per.append(dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s[:,u]))))
 hter=float(np.mean([r['hter'] for r in per]));tar=float(np.mean([r['tar'] for r in per]));far=float(np.mean([r['far'] for r in per]));auc=float(np.mean([r['auc'] for r in per]));ce=float(F.cross_entropy(tensor(l),torch.as_tensor(labs,device=D)))
 return dict(hter=hter,tar=tar,far=far,auc=auc,ce=ce,top1=float((l.argmax(1)==labs).mean()))
def rank(q):return (round(q['hter'],12),round(max(1-q['tar'],q['far']),12),-q['auc'],q['ce'])
def teacher_quality(model,data,domain):
 model.eval();z,l=inference(model,data,'validation',domain);labs=data['validation']['labels'];return dict(ce=float(F.cross_entropy(tensor(l),torch.as_tensor(labs,device=D))),auc=float(np.mean([roc_auc_score(labs==u,l[:,u]) for u in range(9)])),top1=float((l.argmax(1)==labs).mean()))

def aam_loss(net,z,labels):
 c=F.linear(z,F.normalize(net.head.weight,dim=1)).clamp(-1+1e-6,1-1e-6);m=.2;phi=c*np.cos(m)-torch.sqrt(1-c*c)*np.sin(m);phi=torch.where(c>np.cos(np.pi-m),phi,c-np.sin(np.pi-m)*m);one=F.one_hot(labels,9).float();return F.cross_entropy(16*(one*phi+(1-one)*c),labels)

def train_model(data,s,folder,key,domain='acc',mode='ce',targets=None,lam=0.,temperature=2.,frozen_acc=None,frozen_audio=None):
 seed(s);model=Fusion(frozen_acc,frozen_audio).to(D) if domain=='fusion' else Classifier().to(D)
 if domain!='fusion':model.aam=mode=='aam'
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.001);rng=np.random.RandomState(s);lookup={(u,w):np.flatnonzero((data['train']['labels']==u)&(data['train']['words']==w)) for u in range(9) for w in ROLES['train']};best=None;history=[]
 for step in range(1,STEPS+1):
  model.train();ww=rng.choice(ROLES['train'],4,replace=False);ix=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in range(9) for w in ww]);labels=torch.as_tensor(data['train']['labels'][ix],device=D)
  if domain=='fusion':z,l=model(tensor(data['train']['acc'][ix,None]),tensor(data['train']['audio'][ix,None]))
  else:z,l=model(tensor(data['train'][domain][ix,None]))
  loss=aam_loss(model,z,labels) if mode=='aam' else F.cross_entropy(l,labels)
  if targets is not None and lam>0:loss=loss+lam*temperature**2*F.kl_div(F.log_softmax(l/temperature,dim=1),F.softmax(targets[ix]/temperature,dim=1),reduction='batchmean')
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step%25==0:
   # Checkpoint prototypes come only from encoder TRAIN; enrollment is untouched here.
   q=teacher_quality(model,data,domain) if domain in ['audio','fusion'] else validation_quality(model,data,'acc');history.append(dict(step=step,training_loss=float(loss.detach()),quality=q));rr=(q['ce'],-q['auc'],step) if domain in ['audio','fusion'] else rank(q)+(step,)
   if best is None or rr<best[0]:best=(rr,copy.deepcopy(model.state_dict()),step,q)
 model.load_state_dict(best[1]);model.eval()
 for p in model.parameters():p.requires_grad_(False)
 meta=dict(key=key,seed=s,domain=domain,mode=mode,lambda_kd=lam,temperature=temperature,steps=STEPS,selected_step=best[2],quality=best[3],history=history)
 torch.save(dict(state=best[1],metadata=meta),str(folder/(key+'_private.pt')));print('TRAINED',s,key,best[2],flush=True);return model,meta

def metrics(g,n,t):
 r=old.sel.exp.metrics(g,n,t);fpr,tpr,_=roc_curve(np.r_[np.ones(len(g)),np.zeros(len(n))],np.r_[g,n]);i=np.argmin(np.abs(fpr-(1-tpr)));r['diagnostic_eer']=float((fpr[i]+1-tpr[i])/2);return r
def lowfar_threshold(g,n):
 v=np.unique(np.r_[g,n]);v=np.r_[v,np.nextafter(v[-1],np.inf)];return float(next(t for t in v if (n>=t).mean()<=.05+1e-12))

def backend_lock(z,data,folder,key):
 ze=z['enroll'];zv=z['validation'];zc=z['calibration'];el=data['enroll']['labels'];vl=data['validation']['labels'];cw=data['validation']['words'];cl=data['calibration']['labels'];locks={};models={}
 for k in KS:
  centers=prototypes(ze,el,k);cal=zc@centers.T
  for u in range(9):locks['cosine_K%d_U%d'%(k,u+1)]=dict(backend='cosine',K=k,user=u+1,threshold=threshold(cal[cl==u,u],cal[cl!=u,u]),lowfar_threshold=lowfar_threshold(cal[cl==u,u],cal[cl!=u,u]))
 for u in range(9):
  y=el==u;best=None
  for c in [1.,10.,100.]:
   for gamma in ['scale',.1]:
    fitted=old.sel.rbf.fit([ze],y,c,gamma);ss=old.sel.rbf.score(fitted,[zv]);accepted=np.zeros(len(vl),dtype=bool)
    for w in np.unique(cw):
     tr=cw!=w;va=~tr;t=threshold(ss[tr][vl[tr]==u],ss[tr][vl[tr]!=u]);accepted[va]=ss[va]>=t
    tar=float(accepted[vl==u].mean());far=float(accepted[vl!=u].mean());q=dict(hter=(1-tar+far)/2,tar=tar,far=far,auc=float(roc_auc_score(vl==u,ss)),ce=0.);rr=rank(q)+(c,0 if gamma==.1 else 1)
    if best is None or rr<best[0]:best=(rr,fitted,c,gamma,q)
  models[str(u+1)]=best[1];cal=old.sel.rbf.score(best[1],[zc]);locks['rbf_K20_U%d'%(u+1)]=dict(backend='rbf',K=20,user=u+1,C=best[2],gamma=best[3],validation=best[4],threshold=threshold(cal[cl==u],cal[cl!=u]),lowfar_threshold=lowfar_threshold(cal[cl==u],cal[cl!=u]))
 old.sel.joblib.dump(models,str(folder/(key+'_rbf_private.joblib')));return locks

def evaluate_final(net,data,folder,key,locks):
 ze,_=inference(net,data,'enroll');zt,_=inference(net,data,'test');el=data['enroll']['labels'];tl=data['test']['labels'];models=old.sel.joblib.load(str(folder/(key+'_rbf_private.joblib')));rows={};dump={}
 for name,c in locks.items():
  u=c['user']-1;s=zt@prototypes(ze,el,c['K'])[u] if c['backend']=='cosine' else old.sel.rbf.score(models[str(u+1)],[zt]);g=s[tl==u];n=s[tl!=u];r=metrics(g,n,c['threshold']);r.update(c);r['lowfar_calibrated_test_tar']=float((g>=c['lowfar_threshold']).mean());r['lowfar_calibrated_test_far']=float((n>=c['lowfar_threshold']).mean());r['attack_by_user']={str(v+1):dict(n=int((tl==v).sum()),accepted=int((s[tl==v]>=c['threshold']).sum())) for v in range(9) if v!=u};assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12;rows[name]=r;dump[name]=s
 np.savez_compressed(str(folder/(key+'_scores_private.npz')),**dump,test_labels=tl);return rows

def seed_run(s,data):
 folder=ROOT/('seed%d'%s);folder.mkdir(exist_ok=True);models={};metadata={}
 b0,m0=train_model(data,s,folder,'B0_CE');models['B0_CE']=b0;metadata['B0_CE']=m0
 audio,ma=train_model(data,s+1000,folder,'T1_Audio',domain='audio');fusion,mf=train_model(data,s+2000,folder,'T2_Fusion',domain='fusion',frozen_acc=b0,frozen_audio=audio)
 b1,m1=train_model(data,s,folder,'B1_AAM',mode='aam');models['B1_AAM']=b1;metadata['B1_AAM']=m1;candidate_metadata={};chosen={}
 for family,teacher,domain in [('SelfKD',b0,'acc'),('S1_AudioKD',audio,'audio'),('S2_FusionKD',fusion,'fusion')]:
  _,ll=inference(teacher,data,'train',domain);target_logits=tensor(ll);candidates=[(rank(m0['quality'])+(0.,0.),b0,m0,'B0_CE')];candidate_metadata[family]=[]
  for lam in [.1,.5,1.]:
   for temp in [2.,4.]:
    key=family+'_L%s_T%d'%(str(lam).replace('.','p'),temp);net,meta=train_model(data,s,folder,key,targets=target_logits,lam=lam,temperature=temp);candidate_metadata[family].append(meta);candidates.append((rank(meta['quality'])+(lam,temp),net,meta,key))
  selected=min(candidates,key=lambda x:x[0]);models[family]=selected[1];metadata[family]=selected[2];chosen[family]=selected[3]
 # Check fused teacher strength using same validation classification criteria, not final Audio.
 acc_teacher_q=teacher_quality(b0,data,'acc');qualification=dict(acc=acc_teacher_q,audio=ma['quality'],fusion=mf['quality'],fusion_beats_both_val_ce=mf['quality']['ce']<min(acc_teacher_q['ce'],ma['quality']['ce']))
 locks={}
 for key,net in models.items():
  z={r:inference(net,data,r)[0] for r in ['enroll','validation','calibration']};locks[key]=backend_lock(z,data,folder,key)
 old.save(folder/'locked_before_final.json',dict(seed=s,chosen=chosen,metadata=metadata,candidate_metadata=candidate_metadata,teacher_qualification=qualification,routes=locks,final_seen=False))
 results={key:evaluate_final(net,data,folder,key,locks[key]) for key,net in models.items()};out=dict(seed=s,chosen=chosen,teacher_qualification=qualification,routes=results);old.save(folder/'results_public.json',out);print('SEED_COMPLETE',s,flush=True);return out

def main():
 ROOT.mkdir(exist_ok=True);data=data_load();old.save(ROOT/'manifest_private.json',{r:dict(ids=v['ids'],labels=v['labels'].tolist(),words=v['words'].tolist()) for r,v in data.items()})
 old.save(ROOT/'protocol_public.json',dict(name='Audio-guided ACC privileged training protocolA',seeds=SEEDS,groups=ROLES,counts_per_user=dict(encoder_train=180,enrollment=20,validation=30,calibration=20,test='47-50'),architecture='independent CNN16/32/64 GroupNorm GELU globalmean/std ->128L2;CE9 orAAM margin.2scale16',training='400steps each candidate;AdamW3e-4 wd.001 clip1;batch72=9users4words2repeats;same Student init and minibatch order perseed;no SupCon orCycleGAN',teachers='ACC B0CE;realAudio sameCNN;fusion frozen learned ACC/Audio encoders +MLP256->128->128;teacher CE validation selection only',kd='frozen teacher train-only logits;T^2KL teacher||student;lambda0/.1/.5/1,T2/4;lambda0 cachedB0;selfKD same grid;no embedding force',selection='student checkpoint and lambda/T by validation macro prototypeHTER across3 heldout words thenworst thenAUC thenCE;never final;allroute locks before final',backend='cosineK3/5/10/20 independentenrollment;RBFK20 fit20genuine160negative enrollment only,validation grid,calibration jointHTER;lowFAR thresholdcalonly',limitations=['protocolA all9identities known,not unseen-user registration','single-session historical data with old final observations;exploratory','new group split0-17/18-19/20-22/23-24/25-29;not matched training resources to earlierCycleGAN','word labelsconditional on10/20period and missingtail mapping','teachers/fusion have extra training budget;selfKD control helps but not exact allresource matching','KD grid largerthanCE baseline;validation search may overfit','protocolB identity-disjoint must retrain allmodels/calibration ifA showsstablegain']))
 rows={}
 for s in SEEDS:
  old.save(ROOT/'status_public.json',dict(state='running',seed=s,completed=list(rows)));rows[str(s)]=seed_run(s,data);old.save(ROOT/'results_public.json',rows)
 old.save(ROOT/'status_public.json',dict(state='complete',completed=list(rows)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(exist_ok=True);old.save(ROOT/'status_public.json',dict(state='failed',error=repr(e)));raise

