"""Delay-tolerant generation with controlled global ACC-conditioning ablations."""
import os,sys,json,time,argparse
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL');os.environ.setdefault('MKL_NUM_THREADS','1')
sys.path.insert(0,str(Path('work/torch_runtime_20261018').resolve()))
import numpy as np
from scipy.linalg import solve
import torch
from torch import nn
from torch.nn import functional as F
class FrameNorm(nn.Module):
 def __init__(self,width):super().__init__();self.norm=nn.LayerNorm(width)
 def forward(self,x):return self.norm(x.transpose(1,2)).transpose(1,2)
class Encoder(nn.Module):
 def __init__(self,width):
  super().__init__();self.first=nn.Conv1d(641,width,1);self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,3,padding=d,dilation=d),FrameNorm(width),nn.SiLU(),nn.Conv1d(width,width,1)) for d in [1,2,4]])
 def forward(self,x):
  z=self.first(x)
  for b in self.blocks:z=z+.3*b(z)
  return z
class Model(nn.Module):
 def __init__(self,route,mean):
  super().__init__();self.route=route;self.temporal=Encoder(48);self.preserve=Encoder(32);self.activity=nn.Conv1d(48,1,1);self.acc=nn.Conv1d(32,641,1)
  self.q=nn.Conv1d(48,24,1);self.k=nn.Conv1d(48,24,1);self.v=nn.Conv1d(48,48,1);self.film=nn.Linear(64,96);self.project=nn.Conv1d(48,16,1)
  self.audio=nn.Sequential(nn.Conv1d(48,64,3,padding=1),nn.SiLU(),nn.Conv1d(64,80,1))
  with torch.no_grad():self.audio[-1].bias.copy_(mean);self.film.bias.zero_()
 def forward(self,x,condition_zero=False):
  z=self.temporal(x);act=self.activity(z).squeeze(1);h=self.preserve(x)
  # Only an ACC query and nearby ACC keys; never a microphone query.
  q=self.q(z)[:,:,20:101];kz=self.k(z);vz=self.v(z);keys=torch.stack([kz[:,:,d:d+81] for d in range(8,31)],2);values=torch.stack([vz[:,:,d:d+81] for d in range(8,31)],2)
  offsets=torch.arange(8,31,device=x.device,dtype=x.dtype);prior=-.5*((offsets-20)/8)**2
  weights=((q.unsqueeze(2)*keys).sum(1)/(24**.5)+prior[None,:,None]).softmax(1)
  aligned=(values*weights[:,None]).sum(2)
  descriptor=torch.cat([h.mean(2),h.std(2,unbiased=False)],1)
  if self.route=='temporal_only' or condition_zero:descriptor=descriptor*0
  scale,bias=self.film(descriptor).chunk(2,1);conditioned=aligned*(1+.1*scale.tanh()[:,:,None])+.1*bias[:,:,None]
  gated=conditioned*(.2+.8*act[:,20:101].sigmoid())[:,None]
  return self.audio(gated),act,z,h,self.acc(h),self.project(aligned),weights
def corr(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel();return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.
def detail(y):return y-y.mean(-2,keepdims=True)-y.mean(-1,keepdims=True)+y.mean((-2,-1),keepdims=True)
def load(p):
 with np.load(p) as d:return {k:d[k] for k in d.files}
def ridge_probe(zt,zv,xt,xv):
 zt=zt.transpose(0,2,1).reshape(-1,zt.shape[1]);zv=zv.transpose(0,2,1).reshape(-1,zv.shape[1]);xt=xt.transpose(0,2,1).reshape(-1,641);xv=xv.transpose(0,2,1).reshape(-1,641)
 mu=zt.mean(0);sd=np.maximum(zt.std(0),.01);a=(zt-mu)/sd;b=(zv-mu)/sd;xm=xt.mean(0)
 w=solve(a.T@a+10*np.eye(a.shape[1]),a.T@(xt-xm),assume_a='pos');p=b@w+xm;out={}
 for name,s in [('magnitude',slice(0,195)),('energy',slice(195,251)),('complex',slice(251,None))]:out[name]=float(np.mean((p[:,s]-xv[:,s])**2)/np.mean((xm[s]-xv[:,s])**2))
 return out
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--data',type=Path,default=Path('outputs/delay_generation_20261018'));pa.add_argument('--output',type=Path,required=True);pa.add_argument('--device',default='cpu');pa.add_argument('--epochs',type=int,default=120);pa.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);args=pa.parse_args()
 torch.set_num_threads(2);device=args.device;root=args.output;root.mkdir(parents=True,exist_ok=True);tr=load(args.data/'data_train.npz');va=load(args.data/'data_val.npz');assert not set(tr['ids'])&set(va['ids'])
 T=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
 tx,ty,ta=map(T,[tr['x'],tr['y'],tr['activity']]);vx=T(va['x']);teacher=tr['y'].reshape(-1,16,5,81).mean(2);tm=teacher.mean((0,2),keepdims=True);ts=np.maximum(teacher.std((0,2),keepdims=True),.05);teacher=T((teacher-tm)/ts)
 groups=[np.flatnonzero(tr['ids']==g) for g in np.unique(tr['ids'])];positive=float(np.mean([tr['activity'][ii].mean() for ii in groups]));template=tr['y'].mean(0)
 protocol=dict(seeds=args.seeds,epochs=args.epochs,updates_per_epoch=4,batch_size=8,device=device,torch=torch.__version__,routes=['temporal_only','dual_unconstrained','dual_preserve'],selection='fixed endpoint, no validation checkpoint selection; existing reused4val recordings',sampler='8distinctTRAIN recordings/update; uniform window per recording; identical numpy seed schedule across routes',augmentation='none; no synthetic phase seams or pitch/gain/axis changes',losses=dict(audio_L1=1.,audio_envelope=.1,frequency_gradient=.1,trainstandardized16band_teacher_MSE=.02,balanced_soft_activity_BCE=.05,ACC_groupwise_MSE=.15),activity_positive_fraction=positive,attention='ACC-only query at audio-time+200ms; keys+80..300ms with Gaussian relative offset prior; convolution receptivefield +/-70ms; futurecontext480ms; no absolute-time embedding or audio inference input',branch='same48temporal+32candidate globalACC encoders instantiated; temporalonly zeroes global conditioning; dual_no_recon and dual_preserve same full architecture, latter adds ACC reconstruction loss',identity='singleperson; globalbranch is candidate feature conditioning, NOT a trained identity embedding; no speaker teacher loss or verification accuracy',retention='fixedTRAINridge10 from frozen central81frame latents to641input features; baseline48dim, dual80dim; primary controlled comparison dual_no_recon vsdual_preserve80dim',condition_controls='validationACC shifted400ms withincontext, differentrecordACC, globalbranchzeroed; no favorable alignment onVAL',split='20train4reusedval,58/16 windows; test4+reserve28unread')
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');allresults=[]
 for seed in args.seeds:
  for route in protocol['routes']:
   rng=np.random.default_rng(seed);torch.manual_seed(seed);model=Model(route,ty.mean((0,2)).cpu()).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.003);start=time.time();history=[]
   for epoch in range(args.epochs):
    model.train();ll=[]
    for step in range(4):
     gg=rng.choice(len(groups),8,replace=False);idx=[rng.choice(groups[g]) for g in gg];x,y,a=tx[idx],ty[idx],ta[idx]
     p,act,z,h,recon,q,weights=model(x)
     loss=abs(p-y).mean()+.1*abs(p.mean(1)-y.mean(1)).mean()+.1*abs(torch.diff(p,dim=1)-torch.diff(y,dim=1)).mean()+.02*((q-teacher[idx])**2).mean()
     bce=-.5*(a*F.logsigmoid(act)/positive+(1-a)*F.logsigmoid(-act)/(1-positive));loss=loss+.05*bce.mean()
     if route=='dual_preserve':loss=loss+.15*sum(((recon[:,s]-x[:,s])**2).mean() for s in [slice(0,195),slice(195,251),slice(251,None)])
     optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();ll.append(float(loss.detach()))
    if (epoch+1)%30==0:
     row=dict(epoch=epoch+1,loss=float(np.mean(ll)),seconds=round(time.time()-start,1));history.append(row);print(seed,route,row,flush=True)
   model.eval()
   with torch.no_grad():
    trainout=model(tx);out=model(vx);shiftout=model(torch.roll(vx,40,-1));zeroout=model(vx,condition_zero=True)
   p=out[0].cpu().numpy();sp=shiftout[0].cpu().numpy();zp=zeroout[0].cpu().numpy();activity=out[1].sigmoid().cpu().numpy();weights=out[-1].cpu().numpy()
   zt=trainout[2] if route=='temporal_only' else torch.cat([trainout[2],trainout[3]],1);zv=out[2] if route=='temporal_only' else torch.cat([out[2],out[3]],1)
   probe=ridge_probe(zt[:,:,:81].cpu().numpy(),zv[:,:,:81].cpu().numpy(),tr['x'][:,:,:81],va['x'][:,:,:81]);rows=[];ids=va['ids'];permutation=np.arange(len(ids));ug=np.unique(ids)
   for k,g in enumerate(ug):permutation[np.flatnonzero(ids==g)]=np.flatnonzero(ids==ug[(k+1)%len(ug)])
   with torch.no_grad():wrong=model(vx[permutation])[0].cpu().numpy()
   for g in ug:
    ii=ids==g;y=va['y'][ii];err=np.mean(abs(p[ii]-y));rows.append(dict(recording=str(g),audio_mae=float(err),mean_template_mae=float(np.mean(abs(y-template))),detail_corr=corr(detail(p[ii]),detail(y)),envelope_corr=corr(p[ii].mean(1),y.mean(1)),shift400ms_minus_matched_mae=float(np.mean(abs(sp[ii]-y))-err),wrong_record_ACC_minus_matched_mae=float(np.mean(abs(wrong[ii]-y))-err),zero_global_branch_minus_matched_mae=float(np.mean(abs(zp[ii]-y))-err),soft_activity_mse=float(np.mean((activity[ii]-va['activity'][ii])**2)),attention_mean_offset_ms=float(np.mean(np.sum(weights[ii]*np.arange(80,301,10)[None,:,None],axis=1)))))
   result=dict(seed=seed,route=route,macro={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='recording'},acc_retention_probe=probe,per_record=rows,seconds=round(time.time()-start,1));allresults.append(result)
   name=route+'_'+str(seed);torch.save(model.state_dict(),root/(name+'_weights.pt'));(root/(name+'_history.json')).write_text(json.dumps(history,indent=2));np.savez_compressed(root/(name+'_predictions.npz'),audio=p,activity=activity,attention=weights,latent=zv.cpu().numpy(),ids=ids)
   (root/'results.json').write_text(json.dumps(allresults,indent=2),encoding='utf-8');print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
