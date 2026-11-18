"""Matched fine-tuning: audio-domain constraints without abandoning the full cycle."""
import os,json,time,argparse,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'cycle_return_20261018'))
from train_cycle_return import Generator,Discriminator,acc_loss,audio_corrupt,corr,detail
class FeatureDiscriminator(nn.Module):
 def __init__(self):
  super().__init__();c=1;self.blocks=nn.ModuleList()
  for width in [16,32,48]:self.blocks.append(nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(c,width,3,stride=2,padding=1)),nn.LeakyReLU(.2)));c=width
  self.head=nn.Conv2d(c,1,3,padding=1)
 def forward(self,x):
  z=x[:,None];features=[]
  for block in self.blocks:z=block(z);features.append(z)
  return self.head(z),features
class AudioDiscriminator(nn.Module):
 def __init__(self):super().__init__();self.scales=nn.ModuleList([FeatureDiscriminator(),FeatureDiscriminator()])
 def forward(self,x):return [self.scales[0](x),self.scales[1](F.avg_pool2d(x[:,None],2,ceil_mode=True)[:,0])]
def adv(out,target):return sum((o[0]-target).square().mean() for o in out)/len(out)
def fm(pred,real):
 return sum(abs(a.mean(-1)-b.detach().mean(-1)).mean()+.25*abs(a.std(-1,unbiased=False)-b.detach().std(-1,unbiased=False)).mean() for p,r in zip(pred,real) for a,b in zip(p[1],r[1]))/6
def smooth(x):return F.avg_pool2d(x[:,None],3,stride=1,padding=1,count_include_pad=False)[:,0]
def load(p):
 with np.load(p) as d:return {k:d[k] for k in d.files}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=120);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 tr,va=[load(args.data/('data_'+s+'.npz')) for s in ['train','val']];assert not set(tr['ids'])&set(va['ids']);T=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda');x,y=T(tr['x']),T(tr['y']);vx,vy=T(va['x']),T(va['y']);groups=[np.flatnonzero(tr['ids']==g) for g in np.unique(tr['ids'])]
 configs=dict(continuation=dict(adv=.05,pair_audio=2,pair_ACC=2,freq=0,envelope=0,profile=0,feature_matching=0,smooth=False),audio_anchor=dict(adv=1.,pair_audio=10,pair_ACC=5,freq=1,envelope=1,profile=2,feature_matching=2,smooth=False),audio_anchor_smooth=dict(adv=1.,pair_audio=10,pair_ACC=5,freq=1,envelope=1,profile=2,feature_matching=2,smooth=True))
 protocol=dict(source='same20train4REUSEDval,58/16 windows;test4/auth28 unread;same251ACC/80Mel frontend/scales ascycle_return',task='completeACC->Gaudio->FACC and reverseaudio cycle, no crosscyclelatent bypass',initialization='ALLroutes same acc_priority seed42/43 G/F fromcycle_return; newmatched discriminators,40D-only warmupupdates; no selectingcheckpoint byVAL',routes=configs,common=dict(acc_cycle=30,audio_cycle=10,ACC_adversarial=.05,epochs=args.epochs,updates_per_epoch=4,batch_size=8,lr=.0002,weight_decay=.001),seeds=args.seeds,discriminators='oneACC PatchGAN and two-scale audioPatchGAN with features, sameALLroutes includingcontinuation',bottleneck='smoothroute explicit differentiable3x3 normalizedMel averageS: F(S(G(x))) andF(S(realMel)); rawG evaluatedforaudioquality, no cosmeticpostfilter onshownG',feature_matching='pairedDfeature temporalmeans/std, preservesfrequencyprofile, notspeakeridentityloss',selection='fixedendpoint; noVALselection; seedsdescribeoptimizervariationnotindependent subjects',limits='new pairedCycleGANfine-tune, notoriginaltrainb reproduction or audiblewaveform; noidentityaccuracy',torch=torch.__version__)
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
 for seed in args.seeds:
  cp=torch.load(args.initial/('acc_priority_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True)
  for route,cfg in configs.items():
   torch.manual_seed(seed);rng=np.random.default_rng(seed);g=Generator(251,80,True).cuda();f=Generator(80,251).cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f']);da=Discriminator().cuda();db=AudioDiscriminator().cuda();opt=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=.0002,betas=(.5,.999),weight_decay=.001);od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0002,betas=(.5,.999));S=smooth if cfg['smooth'] else lambda a:a;start=time.time();history=[]
   for warm in range(40):
    gg=rng.choice(len(groups),8,replace=False);ii=[rng.choice(groups[i]) for i in gg];a,b=x[ii],y[ii]
    with torch.no_grad():ga=g(a);fb=f(S(b))
    od.zero_grad();loss=.5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0));loss.backward();od.step()
   rng=np.random.default_rng(seed)
   for ep in range(args.epochs):
    losses=[]
    for step in range(4):
     gg=rng.choice(len(groups),8,replace=False);ii=[rng.choice(groups[i]) for i in gg];a,b=x[ii],y[ii]
     for d in [da,db]:d.requires_grad_(True)
     with torch.no_grad():ga=g(a);fb=f(S(b))
     od.zero_grad();dl=.5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0));dl.backward();od.step()
     for d in [da,db]:d.requires_grad_(False)
     ga=g(a);fb=f(S(b));rec=f(S(ga));reverse=g(fb);predD=db(ga)
     loss=30*acc_loss(rec,a)+10*abs(reverse-b).mean()+cfg['pair_ACC']*acc_loss(fb,a)+cfg['pair_audio']*abs(ga-b).mean()+min(1.,.25+ep/20)*(.05*(da(fb)-1).square().mean()+cfg['adv']*adv(predD,1))
     if cfg['freq']:loss=loss+cfg['freq']*abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)).mean()+cfg['envelope']*abs(ga.mean(1)-b.mean(1)).mean()+cfg['profile']*abs(ga.mean(2)-b.mean(2)).mean()+cfg['feature_matching']*fm(predD,db(b))
     opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
    if (ep+1)%30==0:
     row=dict(epoch=ep+1,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);print(seed,route,row,flush=True)
   for m in [g,f,da,db]:m.eval()
   torch.manual_seed(seed+10000)
   with torch.no_grad():
    ga=g(vx);fb=f(S(vy));rec=f(S(ga));reverse=g(fb);rawcycle=f(ga);variants={k:f(S(audio_corrupt(ga,k))).cpu().numpy() for k in ['quant8','noise005','blur3']};ug=np.unique(va['ids']);perm=np.arange(len(va['ids']))
    for k,name in enumerate(ug):perm[np.flatnonzero(va['ids']==name)]=np.flatnonzero(va['ids']==ug[(k+1)%len(ug)])
    wrong=f(S(ga[perm])).cpu().numpy();real_score=float(sum(o[0].mean() for o in db(vy))/2);fake_score=float(sum(o[0].mean() for o in db(ga))/2)
   ga,fb,rec,reverse,rawcycle=[z.cpu().numpy() for z in [ga,fb,rec,reverse,rawcycle]];rows=[]
   for name in ug:
    ii=va['ids']==name;a,b=va['x'][ii],va['y'][ii];pp=ga[ii];rr=rec[ii];cm=float(np.mean(abs(rr-a)));row=dict(recording=str(name),audio_mae=float(np.mean(abs(pp-b))),audio_detail_corr=corr(detail(pp),detail(b)),audio_envelope_corr=corr(pp.mean(1),b.mean(1)),audio_profile_mae=float(np.mean(abs(pp.mean(2)-b.mean(2)))),audio_spectral_second_difference_ratio=float(np.mean(abs(np.diff(pp,n=2,axis=1)))/np.mean(abs(np.diff(b,n=2,axis=1)))),audio_sample_variance_ratio=float(np.mean(np.var(pp,axis=0))/max(np.mean(np.var(b,axis=0)),1e-9)),ACC_cycle_mae=cm,ACC_cycle_corr=corr(rr,a),ACC_magnitude_detail_corr=corr(detail(rr[:,:195]),detail(a[:,:195])),ACC_energy_corr=corr(rr[:,195:],a[:,195:]),paired_ACC_mae=float(np.mean(abs(fb[ii]-a))),reverse_audio_cycle_mae=float(np.mean(abs(reverse[ii]-b))),without_S_cycle_mae=float(np.mean(abs(rawcycle[ii]-a))),wrong_generatedaudio_cycle_mae_increase=float(np.mean(abs(wrong[ii]-a))-cm))
    for k,v in variants.items():row[k+'_cycle_mae_increase']=float(np.mean(abs(v[ii]-a))-cm)
    rows.append(row)
   result=dict(seed=seed,route=route,macro={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='recording'},per_record=rows,D_real_mean=real_score,D_fake_mean=fake_score,seconds=round(time.time()-start,1));results.append(result);name=route+'_'+str(seed);torch.save(dict(g=g.state_dict(),f=f.state_dict(),da=da.state_dict(),db=db.state_dict()),root/(name+'_weights.pt'));(root/(name+'_history.json')).write_text(json.dumps(history,indent=2));np.savez_compressed(root/(name+'_predictions.npz'),generated_audio=ga,generated_ACC=fb,cycle_ACC=rec,reverse_audio_cycle=reverse,cycle_blur3=variants['blur3'],ids=va['ids']);(root/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
