"""Full bidirectional cycle pilot; no latent shortcut across the audio bottleneck."""
import os,json,argparse,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
class FrameNorm(nn.Module):
 def __init__(self,c):super().__init__();self.norm=nn.LayerNorm(c)
 def forward(self,x):return self.norm(x.transpose(1,2)).transpose(1,2)
class Generator(nn.Module):
 def __init__(self,a,b,audio=False):
  super().__init__();self.audio=audio;self.enc=nn.Conv1d(a,64,1);self.level=nn.Conv1d(a,b,1);self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(64,64,3,padding=d,dilation=d),FrameNorm(64),nn.SiLU(),nn.Conv1d(64,64,1)) for d in [1,2,4,8,16]]);self.out=nn.Conv1d(64,b,1)
 def forward(self,x):
  z=self.enc(x)
  for b in self.blocks:z=z+.25*b(z)
  y=self.level(x)+.25*self.out(z);return y.tanh() if self.audio else y
class Discriminator(nn.Module):
 def __init__(self):
  super().__init__();layers=[];c=1
  for width in [16,32,48]:layers.extend([nn.utils.spectral_norm(nn.Conv2d(c,width,3,stride=2,padding=1)),nn.LeakyReLU(.2)]);c=width
  layers.append(nn.Conv2d(c,1,3,padding=1));self.net=nn.Sequential(*layers)
 def forward(self,x):return self.net(x[:,None])
def acc_loss(p,x):return .5*(abs(p[:,:195]-x[:,:195]).mean()+abs(p[:,195:]-x[:,195:]).mean())
def audio_corrupt(y,kind='train'):
 if kind=='train':
  noisy=y+.01*torch.randn_like(y);quant=(noisy.clamp(-1,1)+1).mul(31.5).round().div(31.5)-1
  return noisy+(quant-noisy).detach()
 if kind=='quant8':return ((y.clamp(-1,1)+1)*127.5).round()/127.5-1
 if kind=='noise005':return y+.005*torch.randn_like(y)
 if kind=='blur3':return F.avg_pool2d(y[:,None],3,stride=1,padding=1,count_include_pad=False)[:,0]
 raise ValueError(kind)
def corr(a,b):
 a=a.ravel();b=b.ravel();return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.
def detail(a):return a-a.mean(-1,keepdims=True)-a.mean(-2,keepdims=True)+a.mean((-2,-1),keepdims=True)
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--data',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);pa.add_argument('--epochs',type=int,default=160);pa.add_argument('--seeds',nargs='+',type=int,default=[42,43]);args=pa.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 data={s:dict(np.load(args.data/('data_'+s+'.npz'))) for s in ['train','val']};tr,va=data['train'],data['val'];assert tr['x'].shape[1:]==(251,81);assert not set(tr['ids'])&set(va['ids']);T=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda');x,y=T(tr['x']),T(tr['y']);vx,vy=T(va['x']),T(va['y']);groups=[np.flatnonzero(tr['ids']==g) for g in np.unique(tr['ids'])];xs=x.std((0,2)).clamp_min(.05)[None,:,None]
 protocol=dict(source='same20train4REUSEDval,58/16 original.8s windows;251=195separateXYZ magnitudes+56energy;4test+28reserveUNREAD',scope='completepairedCycleGANpilot, new temporal generators and2DPatchGAN, NOT exact trainb reproduction; no identityclaims',chain='G:ACC251x81->audio80x81; F:audio80x81->ACC251x81; F only receivesG output, no originalACC/latent skip across cycle; reverseaudio cycle included',seeds=args.seeds,epochs=args.epochs,updates_per_epoch=4,batch_size=8,selection='fixedendpoint, noVALselection',common=dict(audio_cycle=10,paired_ACC=2,adversarial=.05,lr=.0005,weight_decay=.001),routes=dict(cycle_balanced=dict(acc_cycle=10,paired_audio=10,corrupt=False),acc_priority=dict(acc_cycle=30,paired_audio=2,corrupt=False),robust_priority=dict(acc_cycle=30,paired_audio=2,corrupt=True)),corruption='robustroute TRAIN6bitSTE quantization +Gaussian std.01 ongeneratedaudio; smalltrain-scaledACCnoise onreversecycle; G/audioadversarial evaluatedclean',controls='cleanvsquant8/noise.005/blur3 generatedaudio thenF, sameperturbationonREALaudio beforeF, wrongrecordgeneratedaudio beforeF; modifications are diagnostic, notproofhiddenencoding',sampler='same seed schedule,8distinctrecordings/update, uniformwindowperrecord, no augmentation',frontend='unchangedactivity_alignment_20261018 TRAINscales, identicalallroutes; nottheprevious641complexcontextfrontend',torch=torch.__version__)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2));allresults=[]
 for seed in args.seeds:
  for route,cfg in protocol['routes'].items():
   torch.manual_seed(seed);rng=np.random.default_rng(seed);g=Generator(251,80,True).cuda();f=Generator(80,251).cuda();da=Discriminator().cuda();db=Discriminator().cuda();opt=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=.0005,betas=(.5,.999),weight_decay=.001);od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0003,betas=(.5,.999));start=time.time();history=[]
   for ep in range(args.epochs):
    for m in [g,f,da,db]:m.train()
    losses=[]
    for step in range(4):
     gg=rng.choice(len(groups),8,replace=False);ii=[rng.choice(groups[i]) for i in gg];a,b=x[ii],y[ii]
     for d in [da,db]:d.requires_grad_(True)
     with torch.no_grad():ga=g(a);fb=f(b)
     od.zero_grad();dl=.5*((da(a)-1).square().mean()+da(fb).square().mean()+(db(b)-1).square().mean()+db(ga).square().mean());dl.backward();od.step()
     for d in [da,db]:d.requires_grad_(False)
     ga=g(a);fb=f(b);bottleneck=audio_corrupt(ga) if cfg['corrupt'] else ga;reverse=fb+.01*xs*torch.randn_like(fb) if cfg['corrupt'] else fb;rec=f(bottleneck);ry=g(reverse)
     cycle=cfg['acc_cycle']*acc_loss(rec,a)+10*abs(ry-b).mean();paired=cfg['paired_audio']*abs(ga-b).mean()+2*acc_loss(fb,a);adv=.05*min(1.,max(0.,(ep-20)/20))*((db(ga)-1).square().mean()+(da(fb)-1).square().mean());loss=cycle+paired+adv
     opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
    if (ep+1)%40==0:
     row=dict(epoch=ep+1,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);print(seed,route,row,flush=True)
   g.eval();f.eval();torch.manual_seed(seed+10000)
   with torch.no_grad():
    ga=g(vx);fb=f(vy);rec=f(ga);ry=g(fb);variants={k:f(audio_corrupt(ga,k)).cpu().numpy() for k in ['quant8','noise005','blur3']};realvariants={k:f(audio_corrupt(vy,k)).cpu().numpy() for k in variants};ug=np.unique(va['ids']);perm=np.arange(len(va['ids']))
    for k,name in enumerate(ug):perm[np.flatnonzero(va['ids']==name)]=np.flatnonzero(va['ids']==ug[(k+1)%len(ug)])
    wrong=f(ga[perm]).cpu().numpy()
   ga,fb,rec,ry=[z.cpu().numpy() for z in [ga,fb,rec,ry]];rows=[];mean=tr['x'].mean(0);amean=tr['y'].mean(0)
   for name in ug:
    ii=va['ids']==name;a,b=va['x'][ii],va['y'][ii];rr=rec[ii];cycle_mae=float(np.mean(abs(rr-a)));raw_error=float(np.mean(abs(fb[ii]-a)))
    row=dict(recording=str(name),audio_mae=float(np.mean(abs(ga[ii]-b))),audio_template_mae=float(np.mean(abs(amean-b))),audio_detail_corr=corr(detail(ga[ii]),detail(b)),paired_ACC_mae=raw_error,ACC_cycle_mae=cycle_mae,ACC_template_mae=float(np.mean(abs(mean-a))),ACC_cycle_corr=corr(rr,a),ACC_magnitude_cycle_corr=corr(rr[:,:195],a[:,:195]),ACC_magnitude_detail_corr=corr(detail(rr[:,:195]),detail(a[:,:195])),ACC_energy_cycle_corr=corr(rr[:,195:],a[:,195:]),reverse_audio_cycle_mae=float(np.mean(abs(ry[ii]-b))),wrong_generatedaudio_cycle_mae_increase=float(np.mean(abs(wrong[ii]-a))-cycle_mae))
    row.update(ACC_magnitude_cycle_mae=float(np.mean(abs(rr[:,:195]-a[:,:195]))),ACC_energy_cycle_mae=float(np.mean(abs(rr[:,195:]-a[:,195:]))),ACC_template_corr=corr(np.broadcast_to(mean,a.shape),a),ACC_template_detail_corr=corr(detail(np.broadcast_to(mean[:195],a[:,:195].shape)),detail(a[:,:195])))
    for k,v in variants.items():row[k+'_cycle_mae']=float(np.mean(abs(v[ii]-a)));row[k+'_cycle_mae_increase']=row[k+'_cycle_mae']-cycle_mae;row[k+'_real_audio_ACC_mae_increase']=float(np.mean(abs(realvariants[k][ii]-a))-raw_error)
    rows.append(row)
   result=dict(seed=seed,route=route,macro={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='recording'},per_record=rows,seconds=round(time.time()-start,1));allresults.append(result);name=route+'_'+str(seed);torch.save(dict(g=g.state_dict(),f=f.state_dict()),root/(name+'_weights.pt'));(root/(name+'_history.json')).write_text(json.dumps(history,indent=2));np.savez_compressed(root/(name+'_predictions.npz'),generated_audio=ga,generated_ACC=fb,cycle_ACC=rec,reverse_cycle_audio=ry,cycle_quant8=variants['quant8'],cycle_noise005=variants['noise005'],cycle_blur3=variants['blur3'],ids=va['ids']);(root/'results.json').write_text(json.dumps(allresults,indent=2));print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
