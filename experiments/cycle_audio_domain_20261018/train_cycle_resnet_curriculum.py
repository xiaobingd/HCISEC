"""User ResNet decoder + real-pair warmup, then the complete constrained cycle."""
import os,sys,json,argparse,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent.parent));sys.path.insert(0,str(Path('outputs/imu_cyclegan_trial_20261016').resolve()))
from model_drop import GeneratorAccToAudio
from train_cycle_audio_domain import AudioDiscriminator,Discriminator,Generator,acc_loss,adv,fm,smooth,corr,detail,load
def no_instance_norm(m):
 for name,child in list(m.named_children()):
  if isinstance(child,nn.InstanceNorm2d):setattr(m,name,nn.Identity())
  else:no_instance_norm(child)
class ResNetAudio(nn.Module):
 def __init__(self):
  super().__init__();self.projection=nn.Conv1d(251,80,1);self.decoder=GeneratorAccToAudio(input_dim=1,output_dim=1,hidden_dim=16,num_residual_blocks=2,dropout=0.,upsample='resize');no_instance_norm(self.decoder)
 def forward(self,x):return self.decoder(self.projection(x)[:,None])[:,0,:80,:81]
def evaluate(g,f,va,device='cuda'):
 g.eval();f.eval();T=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device);x,y=T(va['x']),T(va['y'])
 with torch.no_grad():
  ga=g(x);fb=f(y);rec=f(ga);blur=f(smooth(ga));ug=np.unique(va['ids']);perm=np.arange(len(va['ids']))
  for k,name in enumerate(ug):perm[np.flatnonzero(va['ids']==name)]=np.flatnonzero(va['ids']==ug[(k+1)%len(ug)])
  wrong=f(ga[perm]);reverse=g(fb)
 ga,fb,rec,blur,wrong,reverse=[z.cpu().numpy() for z in [ga,fb,rec,blur,wrong,reverse]];rows=[]
 for name in ug:
  ii=va['ids']==name;a,b=va['x'][ii],va['y'][ii];p=ga[ii];rr=rec[ii];cm=float(np.mean(abs(rr-a)));rows.append(dict(recording=str(name),audio_mae=float(np.mean(abs(p-b))),audio_detail_corr=corr(detail(p),detail(b)),audio_envelope_corr=corr(p.mean(1),b.mean(1)),audio_profile_mae=float(np.mean(abs(p.mean(2)-b.mean(2)))),audio_spectral_second_difference_ratio=float(np.mean(abs(np.diff(p,n=2,axis=1)))/np.mean(abs(np.diff(b,n=2,axis=1)))),audio_sample_variance_ratio=float(np.mean(np.var(p,axis=0))/max(np.mean(np.var(b,axis=0)),1e-9)),ACC_cycle_mae=cm,ACC_cycle_corr=corr(rr,a),ACC_magnitude_detail_corr=corr(detail(rr[:,:195]),detail(a[:,:195])),ACC_energy_corr=corr(rr[:,195:],a[:,195:]),paired_ACC_mae=float(np.mean(abs(fb[ii]-a))),reverse_audio_cycle_mae=float(np.mean(abs(reverse[ii]-b))),wrong_generatedaudio_cycle_mae_increase=float(np.mean(abs(wrong[ii]-a))-cm),blur3_cycle_mae_increase=float(np.mean(abs(blur[ii]-a))-cm)))
 return dict(macro={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='recording'},per_record=rows),dict(generated_audio=ga,cycle_ACC=rec,generated_ACC=fb,cycle_blur3=blur,ids=va['ids'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=120);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);tr,va=[load(args.data/('data_'+s+'.npz')) for s in ['train','val']];assert not set(tr['ids'])&set(va['ids']);T=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda');x,y=T(tr['x']),T(tr['y']);groups=[np.flatnonzero(tr['ids']==g) for g in np.unique(tr['ids'])];results=[];prewarm=[]
 protocol=dict(source='same251ACC/80Mel,20train4REUSEDval58/16windows;4test+28reserveUNREAD; no preprocessingchange',architecture='G=251->80learnedfrequencyprojection->user model_drop.GeneratorAccToAudio hidden16,2residualblocks,resizeconvolution,InstanceNorm removed; no directlinear outputskip. F=previous temporal80->251. Same2scaleaudioD/ACCPatchGAN',model_source_sha256='a50c38a48fc9dafe7ac6659d4705b1a90ef95f703724d1713987962f890a4c73',stages='120epochsreal-pairprewarm THEN120epochsfullbidirectionalcycle; each4updates8record-balanced; fixedendpoint; noVALselection',seeds=args.seeds,losses=dict(paired_audio=20,paired_ACC=5,audio_adversarial=1,ACC_adversarial=.05,frequency_gradient=1,envelope=1,profile=2,feature_matching=2,stage2_ACC_cycle='ramp10to20over40epochs',stage2_audio_cycle=10,stage2_TRAIN_teacher=5),teacher='cachedstage1G outputs forTRAIN only; no VALteacher conditioning at inference',perturbation='stage2eachbatch50%3x3Melaverage beforeF onbothgeneratedandrealMel, alwaysGaussian.005 ongeneratedMel cycle; evaluatecleanF(G) andblurcontrol; rawG shown',scope='architecture+curriculum+loss package exploration, not component-isolated comparison. Pairedprewarm alone is NOT fullcycle. Finalstage ISfullcycle, no crosscyclebypass, noidentityclaims',torch=torch.__version__)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for seed in args.seeds:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);noise_rng=np.random.default_rng(seed+555);g=ResNetAudio().cuda();f=Generator(80,251).cuda();da=Discriminator().cuda();db=AudioDiscriminator().cuda();opt=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=.0003,betas=(.5,.999),weight_decay=.001);od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0002,betas=(.5,.999));start=time.time();history=[];teacher=None
  for stage in ['real_pair_prewarm','full_cycle']:
   for ep in range(args.epochs):
    for m in [g,f,da,db]:m.train()
    losses=[]
    for step in range(4):
     gg=rng.choice(len(groups),8,replace=False);ii=[rng.choice(groups[i]) for i in gg];a,b=x[ii],y[ii];S=smooth if stage=='full_cycle' and noise_rng.random()<.5 else lambda a:a
     for d in [da,db]:d.requires_grad_(True)
     with torch.no_grad():ga=g(a);fb=f(S(b))
     od.zero_grad();dl=.5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0));dl.backward();od.step()
     for d in [da,db]:d.requires_grad_(False)
     ga=g(a);fb=f(S(b));predD=db(ga);loss=20*abs(ga-b).mean()+5*acc_loss(fb,a)+min(1.,.1+ep/20)*(adv(predD,1)+.05*(da(fb)-1).square().mean())+abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)).mean()+abs(ga.mean(1)-b.mean(1)).mean()+2*abs(ga.mean(2)-b.mean(2)).mean()+2*fm(predD,db(b))
     if stage=='full_cycle':
      rec=f(S(ga)+.005*torch.randn_like(ga));reverse=g(fb);loss=loss+(10+10*min(1.,ep/40))*acc_loss(rec,a)+10*abs(reverse-b).mean()+5*abs(ga-teacher[ii]).mean()
     opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
    if (ep+1)%30==0:
     row=dict(stage=stage,epoch=ep+1,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);print(seed,row,flush=True)
   metric,preds=evaluate(g,f,va);metric.update(seed=seed,route='resnet_curriculum',stage=stage,seconds=round(time.time()-start,1));name='resnet_curriculum_'+str(seed)
   if stage=='real_pair_prewarm':
    prewarm.append(metric);(root/'prewarm_results.json').write_text(json.dumps(prewarm,indent=2));np.savez_compressed(root/(name+'_prewarm_predictions.npz'),**preds);g.eval()
    with torch.no_grad():teacher=g(x).detach()
    print('PREWARM',json.dumps(metric),flush=True)
   else:
    results.append(metric);(root/'results.json').write_text(json.dumps(results,indent=2));np.savez_compressed(root/(name+'_predictions.npz'),**preds);torch.save(dict(g=g.state_dict(),f=f.state_dict(),da=da.state_dict(),db=db.state_dict()),root/(name+'_weights.pt'));print('RESULT',json.dumps(metric),flush=True)
   (root/(name+'_history.json')).write_text(json.dumps(history,indent=2))
if __name__=='__main__':main()
