"""Small controlled ACC temporal learners; audio is training supervision only."""
import os, sys, json, time, argparse, random
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('MKL_NUM_THREADS','1')
sys.path.insert(0,str(Path('work/torch_runtime_20261018').resolve()))
import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F
DEVICE='cpu'

class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.first=nn.Conv1d(251,64,1)
  self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(64,64,5,padding=2*d,dilation=d),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv1d(64,64,1)) for d in (1,2,4)])
 def forward(self,x):
  z=self.first(x)
  for block in self.blocks:z=z+.3*block(z)
  return z

class Model(nn.Module):
 def __init__(self,route,mean):
  super().__init__();self.route=route;self.encoder=Encoder()
  self.activity=nn.Conv1d(64,1,1)
  self.audio=nn.Sequential(nn.Conv1d(64,64,3,padding=1),nn.SiLU(),nn.Conv1d(64,80,1))
  self.acc=nn.Conv1d(64,251,1);self.project=nn.Conv1d(64,16,1)
  with torch.no_grad():self.audio[-1].bias.copy_(mean)
 def forward(self,x):
  z=self.encoder(x);a=self.activity(z).squeeze(1)
  gate=1 if self.route=='reconstruction' else (.2+.8*a.sigmoid()).unsqueeze(1)
  return self.audio(z*gate),a,z,self.acc(z),self.project(z)

def auc(y,p):
 y=np.asarray(y).astype(bool).ravel();p=np.asarray(p).ravel();n=y.sum();m=len(y)-n
 return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m)) if n*m else None
def corr(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel()
 return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.
def detail(y):return y-np.mean(y,axis=-2,keepdims=True)-np.mean(y,axis=-1,keepdims=True)+np.mean(y,axis=(-2,-1),keepdims=True)
def load(root,s):
 with np.load(root/('data_'+s+'.npz')) as d:return {k:d[k] for k in d.files}
def tensor(a):return torch.from_numpy(a.astype('float32')).to(DEVICE)
def alignment(q,t,mask):
 # Temporal content negatives are from the SAME individual, never identity negatives.
 q=F.normalize(q[:,:,::4].transpose(1,2).reshape(-1,16),dim=1)
 t=F.normalize(t[:,:,::4].transpose(1,2).reshape(-1,16),dim=1)
 good=mask[:,::4].reshape(-1)>0;q=q[good];t=t[good]
 if len(q)<2:return q.sum()*0
 return F.cross_entropy(q@t.T/.2,torch.arange(len(q),device=q.device))
def macro(values,ids):return float(np.mean([np.mean(np.asarray(values)[ids==g]) for g in np.unique(ids)]))
def evaluate(model,tr,va,root,route):
 model.eval()
 with torch.no_grad():
  yt,at,zt,xt,qt=model(tensor(tr['x']));yp,ap,zp,xp,qp=model(tensor(va['x']))
 yp=yp.cpu().numpy();ap=ap.sigmoid().cpu().numpy();zt=zt.cpu().numpy();zp=zp.cpu().numpy()
 ids=va['ids'];truth=va['y'];template=np.mean(tr['y'],axis=0)
 rows=[]
 for g in np.unique(ids):
  idx=np.flatnonzero(ids==g);other=np.flatnonzero(ids!=g);y=truth[idx];p=yp[idx]
  matched=np.mean(abs(p-y));wrong=np.mean([np.mean(abs(p[:,None]-truth[other][None]))])
  nonzero=va['audio_logrms'][idx]>-6
  rows.append(dict(recording=str(g),windows=len(idx),audio_mae=float(matched),mean_template_mae=float(np.mean(abs(y-template))),detail_corr=corr(detail(y),detail(p)),envelope_corr=corr(y.mean(1),p.mean(1)),wrong_minus_matched_mae=float(wrong-matched),activity_auc=auc(va['activity'][idx],ap[idx]),activity_nonzero_auc=auc(va['activity'][idx][nonzero],ap[idx][nonzero]),output_to_real_std=float(p.std(0).mean()/max(y.std(0).mean(),1e-8))))
 # Identical train-only ridge decoder for all routes; frozen learned frame features.
 ztrain=zt.transpose(0,2,1).reshape(-1,64);zval=zp.transpose(0,2,1).reshape(-1,64)
 xtrain=tr['x'].transpose(0,2,1).reshape(-1,251);xval=va['x'].transpose(0,2,1).reshape(-1,251)
 zm=ztrain.mean(0);zs=np.maximum(ztrain.std(0),.01);xx=(ztrain-zm)/zs;vv=(zval-zm)/zs;xm=xtrain.mean(0)
 w=solve(xx.T@xx+10*np.eye(64),xx.T@(xtrain-xm),assume_a='pos');prediction=vv@w+xm
 probe={}
 for name,sl in [('spectrum',slice(0,195)),('energy',slice(195,None))]:
  base=np.mean((xval[:,sl]-xm[sl])**2);err=np.mean((prediction[:,sl]-xval[:,sl])**2)
  probe[name+'_normalized_mse']=float(err/base)
 learned=np.concatenate([zp.mean(2),zp.std(2)],1)
 features=np.concatenate([va['raw_descriptor'],learned],1)
 assert np.array_equal(features[:,:502],va['raw_descriptor'])
 np.savez_compressed(root/(route+'_predictions.npz'),audio=yp,activity=ap,latent=zp,representation=features,ids=ids)
 result=dict(route=route,per_record=rows,macro={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k not in ('recording','windows')},acc_retention_probe=probe,representation_dim=features.shape[1],raw_descriptor_retained_exactly=True)
 (root/(route+'_results.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def main():
 global DEVICE
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=160);p.add_argument('--seed',type=int,default=42);p.add_argument('--resume-completed',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--root',type=Path,default=Path('outputs/activity_alignment_20261018'));p.add_argument('--data-root',type=Path,default=Path('outputs/activity_alignment_20261018'));p.add_argument('--routes',nargs='+',default=['reconstruction','activity_align','activity_align_preserve']);args=p.parse_args()
 DEVICE=args.device;torch.set_num_threads(2);root=args.root;root.mkdir(parents=True,exist_ok=True);tr=load(args.data_root,'train');va=load(args.data_root,'val')
 teacher=tr['y'].reshape(-1,16,5,81).mean(2);tm=teacher.mean((0,2),keepdims=True);ts=np.maximum(teacher.std((0,2),keepdims=True),.05);teacher=(teacher-tm)/ts
 groups=[np.flatnonzero(tr['ids']==g) for g in np.unique(tr['ids'])]
 protocol=dict(seed=args.seed,epochs=args.epochs,updates_per_epoch=4,batch_size=8,selection='fixed endpoint, validation evaluated only after all updates; no validation-driven selection',routes=['reconstruction','activity_align','activity_align_preserve'],loss_weights=dict(activity=.1,temporal_content_nce=.03,acc_preservation=.15),input_augmentation='joint temporal roll training only; 15 frames masked each side of wrap seam; no absolute-time embeddings',identity_limit='one person, no identity discrimination claim; content negatives are not identity negatives',split='20train/4 repeatedly-used exploratoryval recordings; 58/16 existing windows; test/reserve untouched',alignment='same-frame lowdimensional microphone descriptor InfoNCE, training only, does not equate ACC and audio spectra',retention_probe='same train-only ridge10 frozen latent to ACC for every route',representation='ungated exact raw502 plus latent mean/std128',torch=torch.__version__)
 (root/'training_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 protocol.update(routes=args.routes,device=DEVICE,frozen_route_pretraining='80epochs x4updates; random8frame input mask, original ACC reconstruction; freeze encoder and ACC head before audio supervision; extra compute, not compute-matched')
 (root/'training_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 all_results=[]
 for route in protocol['routes']:
  np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed);start=time.time()
  model=Model(route,tensor(tr['y'].mean((0,2))).cpu()).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  if args.resume_completed and (root/(route+'_weights.pt')).exists():
   model.load_state_dict(torch.load(root/(route+'_weights.pt'),map_location=DEVICE,weights_only=True));result=evaluate(model,tr,va,root,route);all_results.append(result);print('RESUMED',route,result['macro'],result['acc_retention_probe'],flush=True);continue
  tx,ty,ta,tt=map(tensor,(tr['x'],tr['y'],tr['activity'],teacher));history=[]
  if route=='frozen_preserve_align':
   for pre in range(80):
    for step in range(4):
     chosen=np.random.choice(len(groups),8,replace=False);idx=[np.random.choice(groups[g]) for g in chosen];original=tx[idx];corrupted=original.clone()
     for i in range(8):
      b=np.random.randint(0,74);corrupted[i,:,b:b+8]=0
     z=model.encoder(corrupted);recon=model.acc(z);loss=((recon[:,:195]-original[:,:195])**2).mean()+((recon[:,195:]-original[:,195:])**2).mean()
     opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
    if (pre+1)%20==0:print(route,'pretrain',pre+1,float(loss.detach()),flush=True)
   for m in (model.encoder,model.acc):m.requires_grad_(False)
   np.random.seed(args.seed);random.seed(args.seed)
  for epoch in range(args.epochs):
   model.train();losses=[]
   for step in range(4):
    chosen=np.random.choice(len(groups),8,replace=False);idx=[np.random.choice(groups[g]) for g in chosen]
    x,y,a,t=tx[idx].clone(),ty[idx].clone(),ta[idx].clone(),tt[idx].clone();mask=torch.ones_like(a)
    for i in range(len(idx)):
     shift=np.random.randint(-30,31)
     x[i]=torch.roll(x[i],int(shift),-1);y[i]=torch.roll(y[i],int(shift),-1);a[i]=torch.roll(a[i],int(shift),-1);t[i]=torch.roll(t[i],int(shift),-1)
     seam=shift if shift>=0 else 81+shift
     mask[i,max(0,seam-15):min(81,seam+15)]=0
    pred,act,z,recon,q=model(x);weight=mask if route=='reconstruction' else mask*(.5+a)
    loss=(abs(pred-y)*weight[:,None]).sum()/(weight.sum()*80)
    if route!='reconstruction':
     loss=loss+.1*(F.binary_cross_entropy_with_logits(act,a,reduction='none')*mask).sum()/mask.sum()+.03*alignment(q,t,mask)
    if route=='activity_align_preserve':
     preserve=((recon[:,:195]-x[:,:195])**2*mask[:,None]).sum()/(195*mask.sum())+((recon[:,195:]-x[:,195:])**2*mask[:,None]).sum()/(56*mask.sum())
     loss=loss+.15*preserve
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss.detach()))
   if (epoch+1)%20==0:
    row=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);print(route,row,flush=True)
  torch.save(model.state_dict(),root/(route+'_weights.pt'))
  result=evaluate(model,tr,va,root,route);all_results.append(result)
  (root/(route+'_history.json')).write_text(json.dumps(history,indent=2));print('RESULT',route,result['macro'],result['acc_retention_probe'],flush=True)
 (root/'results.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
if __name__=='__main__':main()
