"""Warm-start complete cycles: remove blur anchors, learn conditional Mel detail.

This is Mel-domain reconstruction, not a waveform vocoder or LAU-Net reproduction.
"""
import os,sys,json,argparse,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'cycle_generalization_20261018'))
from train_cycle_generalization import AudioG,ReturnF,evaluate,load
from train_cycle_audio_domain import AudioDiscriminator,Discriminator,acc_loss,adv,fm,smooth,corr,detail

ROUTES={
    'continue':dict(repair=False,refiner=False,conditional=False),
    'objective_repair':dict(repair=True,refiner=False,conditional=False),
    'detail_refiner':dict(repair=True,refiner=True,conditional=False),
    'conditional_refiner':dict(repair=True,refiner=True,conditional=True),
}

class DetailBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(width,width,3,padding=dilation,dilation=dilation,groups=width),nn.SiLU(),nn.Conv2d(width,width*2,1),nn.GLU(dim=1),nn.Conv2d(width,width,1))
    def forward(self,z):return z+.2*self.net(z)

class DetailG(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.acc_projection=nn.Conv1d(251,4*80,1)
        self.stem=nn.Conv2d(5,16,3,padding=1)
        self.blocks=nn.Sequential(*[DetailBlock(16,d) for d in [(1,1),(1,2),(1,4),(1,8),(2,1),(4,1)]])
        self.head=nn.Conv2d(16,1,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        base=self.base(x)
        coords=self.acc_projection(x).reshape(len(x),4,80,x.shape[-1])
        z=self.blocks(F.silu(self.stem(torch.cat([base[:,None],coords],1))))
        return (base+.35*torch.tanh(self.head(z)[:,0])).clamp(-1,1)

class PairD(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Conv1d(251,80,1);layers=[];c=2
        for width in [16,32,48]:
            layers.extend([nn.utils.spectral_norm(nn.Conv2d(c,width,3,stride=2,padding=1)),nn.LeakyReLU(.2)]);c=width
        self.net=nn.Sequential(*layers,nn.Conv2d(c,1,3,padding=1))
    def forward(self,x,y):return self.net(torch.stack([self.projection(x),y],1))

def otsu(values):
    lo,hi=np.percentile(values,[1,99]);hist,edges=np.histogram(values,bins=128,range=(lo,hi))
    p=hist/max(hist.sum(),1);omega=np.cumsum(p);mu=np.cumsum(p*(edges[:-1]+edges[1:])/2)
    score=(mu[-1]*omega-mu)**2/np.maximum(omega*(1-omega),1e-12)
    k=int(np.argmax(score[:-1]));return float((edges[k]+edges[k+1])/2)

def log_energy_numpy(y,alo,ahi):
    db=(y+1)*(ahi-alo)/2+alo
    return np.log10(np.maximum(np.mean(10**((db-ahi)/10),axis=1),1e-12))

def contrast(z):
    return z-F.avg_pool2d(z[:,None],(9,1),stride=1,padding=(4,0),count_include_pad=False)[:,0]

def speech_loss(pred,target,alo,ahi,threshold):
    dbspan=ahi-alo
    amp=lambda z:torch.pow(10.,(z-1)*dbspan/40)
    pe=amp(pred);te=amp(target)
    env=lambda a:torch.log10(a.square().mean(1).clamp_min(1e-12))
    ep,et=env(pe),env(te);mask=(et>threshold).to(target.dtype)
    frames=abs(pred-target).mean(1)
    balanced=.5*((frames*mask).sum()/mask.sum().clamp_min(1)+(frames*(1-mask)).sum()/(1-mask).sum().clamp_min(1))
    weights=.5+mask[:,None]
    local=(abs(contrast(pred)-contrast(target))*weights).mean()
    freq=(abs(torch.diff(pred,dim=1)-torch.diff(target,dim=1))*weights).mean()
    temporal=abs(torch.diff(pred,dim=2)-torch.diff(target,dim=2)).mean()
    sc=0.
    for k in [1,3,7]:
        pp=F.avg_pool1d(pe,k,stride=1,padding=k//2,count_include_pad=False)
        tt=F.avg_pool1d(te,k,stride=1,padding=k//2,count_include_pad=False)
        sc=sc+(pp-tt).square().mean((1,2)).sqrt().div(tt.square().mean((1,2)).sqrt().clamp_min(.001)).mean()/3
    return 20*balanced+4*local+2*freq+temporal+2*abs(ep-et).mean()/(dbspan/10)+2*sc

def extra_metrics(preds,data,alo,ahi,threshold):
    rows=[];pred=preds['generated_audio'];target=data['y']
    ep=log_energy_numpy(pred,alo,ahi);et=log_energy_numpy(target,alo,ahi)
    def hp(a):
        # Exactly the same fixed frequency mean as the training contrast, with edge counts.
        sums=np.pad(np.cumsum(a,axis=1),((0,0),(1,0),(0,0)))
        j=np.arange(a.shape[1]);left=np.maximum(j-4,0);right=np.minimum(j+5,a.shape[1])
        return a-(sums[:,right]-sums[:,left])/(right-left)[None,:,None]
    ph,th=hp(pred),hp(target)
    for name in np.unique(data['ids']):
        ii=data['ids']==name;a,b=ep[ii]>threshold,et[ii]>threshold
        tp=np.sum(a&b);fp=np.sum(a&~b);fn=np.sum(~a&b)
        error=abs(pred[ii]-target[ii]).mean(1)
        rows.append(dict(recording=str(name),active_audio_mae=float(error[b].mean()) if b.any() else None,inactive_audio_mae=float(error[~b].mean()) if (~b).any() else None,activity_false_positive_rate=float(fp/max(np.sum(~b),1)),activity_recall=float(tp/max(np.sum(b),1)),activity_F1=float(2*tp/max(2*tp+fp+fn,1)),physical_log_energy_corr=corr(ep[ii],et[ii]),local_spectral_contrast_corr=corr(ph[ii],th[ii])))
    return dict(macro={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k!='recording'},per_record=rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=180);p.add_argument('--updates',type=int,default=4);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);p.add_argument('--routes',nargs='+',default=list(ROUTES));args=p.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    train=load(args.source/'data/mei180_train.npz');assert len(np.unique(train['ids']))==180
    dp=json.loads((args.source/'data_protocol.json').read_text());alo,ahi=dp['normalization']['audio_lo_hi']
    threshold=otsu(log_energy_numpy(train['y'],alo,ahi).ravel())
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
    x,y=tensor(train['x']),tensor(train['y']);groups=[np.flatnonzero(train['ids']==g) for g in np.unique(train['ids'])]
    protocol=dict(data='exact previous clean180TRAIN/60REUSEDval,720/240originalwindows; source-key protection/dedup unchanged. TEST/auth arrays not read.',frontend='EXACT prior shared251ACC/80Mel arrays and frozen pooled20personal+20meiTRAIN calibration; no source edits/per-user normalization',routes={r:ROUTES[r] for r in args.routes},initialization='Same seed mei180_long G/F checkpoints. New matched discriminators across routes,40D-only warmups. Refiner last layer zero => initial G EXACT oldG. No original optimizer/D states available, so continue is a matched new continuation, not exact resumed old trajectory.',epochs=args.epochs,updates_per_epoch=args.updates,batch_size=8,seeds=args.seeds,lr=.00015,weight_decay=.001,cycles='ALL complete bidirectionalcycles, raw F(G) mainmetric. No ACC/latent bypass across audio. ACCcycle20, reverseaudio10, pairedACC5 throughout.',baseline='old-like audio pair20+freqgrad/envelope/profile/FM, audioGAN1/ACCGAN.05,50%3x3Mel S beforeF(real/generated),Gaussian.005;TRAIN-only cached initialG anchor5 (NOT original discarded prewarmteacher)',repair='Remove anchor+MelS, retainGaussian.005. Replace audio loss with voice/background-balanced pairedL1,frequencycontrast9bins,frequency/timegradients,physicalMelenergy envelope,multi-time-scaleMel amplitude spectralconvergence(pool1/3/7). AudioGAN.25/ACCGAN.05,FM1. Package change, not singlecomponent causality.',refiner='Same oldG plus learnedACC251->4x80 spectralcoordinates and fullresolution16channel depthwise2Dresiduals, no downsampling/normalization;6dilationblocks,lastzerohead,.35bounded additive correction. Learned spectralcoordinates NOT physicalACC/Mel frequency equality.',conditional='Only conditional_refiner adds TRAIN matched(acc,audio) discriminator with wrong-record realaudio negatives. These are content-pair negatives within SAME person, never identity negatives. It never feeds real audio into inferenceG.',audio_energy_threshold=threshold,threshold_scope='Otsu fitted ONLY TRAIN fixedphysicalMelenergy; labels are automatic energy proxies, not human speech annotation',normalization_audio=[alo,ahi],selection='Fixed180endpoints, noVALcheckpointselection. Two optimizationseeds, one same-person reusedVAL; noidentityclaims.',references=['https://arxiv.org/abs/1910.11480','https://www.isca-archive.org/interspeech_2025/song25d_interspeech.html'],reference_limits='Inspired by structured spectral supervision, NOT waveformMR-STFT or LAU-Net reproduction. LAU-Net uses MIC+ACC, ours inferenceACC-only. Outputs areMelnotwaveforms.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    for seed in args.seeds:
        cp=torch.load(args.source/'results_long'/('mei180_long_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True)
        for route in args.routes:
            cfg=ROUTES[route];torch.manual_seed(seed);base=AudioG().cuda();base.load_state_dict(cp['g']);base.eval()
            with torch.no_grad():anchor=torch.cat([base(x[i:i+24]).detach() for i in range(0,len(x),24)]);before=base(x[:2]).detach()
            g=DetailG(base).cuda() if cfg['refiner'] else base
            with torch.no_grad():assert float(abs(g(x[:2])-before).max())<1e-6
            f=ReturnF().cuda();f.load_state_dict(cp['f'])
            torch.manual_seed(seed+2026);da,db=Discriminator().cuda(),AudioDiscriminator().cuda();pd=PairD().cuda() if cfg['conditional'] else None
            allD=[da,db]+([pd] if pd is not None else [])
            opt=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=.00015,betas=(.5,.999),weight_decay=.001)
            od=torch.optim.AdamW([p for d in allD for p in d.parameters()],lr=.0002,betas=(.5,.999))
            rng=np.random.default_rng(seed);perturb=np.random.default_rng(seed+555);noisegen=torch.Generator(device='cuda').manual_seed(seed+777)
            start=time.time();history=[];name=route+'_'+str(seed)
            def batch():
                gg=rng.choice(len(groups),8,replace=False);return [rng.choice(groups[j]) for j in gg]
            def d_loss(a,b,ga,fb):
                loss=.5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0))
                if pd is not None:loss=loss+.5*((pd(a,b)-1).square().mean()+.5*pd(a,ga).square().mean()+.5*pd(a,b.roll(1,0)).square().mean())
                return loss
            for _ in range(40):
                ii=batch();a,b=x[ii],y[ii]
                with torch.no_grad():ga=g(a);fb=f(b)
                od.zero_grad();dl=d_loss(a,b,ga,fb);dl.backward();od.step()
            rng=np.random.default_rng(seed)
            for ep in range(1,args.epochs+1):
                for m in [g,f]+allD:m.train()
                losses=[]
                for _ in range(args.updates):
                    ii=batch();a,b=x[ii],y[ii];S=smooth if not cfg['repair'] and perturb.random()<.5 else lambda z:z
                    for d in allD:d.requires_grad_(True)
                    with torch.no_grad():ga=g(a);fb=f(S(b))
                    od.zero_grad();dl=d_loss(a,b,ga,fb);dl.backward();od.step()
                    for d in allD:d.requires_grad_(False)
                    ga=g(a);fb=f(S(b));rec=f(S(ga)+.005*torch.randn(ga.shape,device='cuda',generator=noisegen));reverse=g(fb);dg=db(ga)
                    loss=20*acc_loss(rec,a)+10*abs(reverse-b).mean()+5*acc_loss(fb,a)+.05*(da(fb)-1).square().mean()
                    if cfg['repair']:
                        loss=loss+speech_loss(ga,b,alo,ahi,threshold)+.25*adv(dg,1)+fm(dg,db(b))
                    else:
                        loss=loss+20*abs(ga-b).mean()+abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)).mean()+abs(ga.mean(1)-b.mean(1)).mean()+2*abs(ga.mean(2)-b.mean(2)).mean()+2*fm(dg,db(b))+adv(dg,1)+5*abs(ga-anchor[ii]).mean()
                    if pd is not None:loss=loss+.5*(pd(a,ga)-1).square().mean()
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
                if ep%30==0 or ep==args.epochs:
                    row=dict(epoch=ep,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);(root/(name+'_history.json')).write_text(json.dumps(history,indent=2));print(name,json.dumps(row),flush=True)
            val=load(args.source/'data/mei_val.npz');assert not set(train['ids'])&set(val['ids'])
            templates=(train['x'].mean(0),train['y'].mean(0));vm,preds=evaluate(g,f,val,templates);em=extra_metrics(preds,val,alo,ahi,threshold);tm,_=evaluate(g,f,train,templates)
            metric=dict(route=route,seed=seed,train=tm,validation=vm,extra_validation=em,seconds=round(time.time()-start,1),parameters=sum(p.numel() for p in g.parameters()))
            results.append(metric);(root/'results.json').write_text(json.dumps(results,indent=2));np.savez_compressed(root/(name+'_predictions.npz'),**preds)
            torch.save(dict(g=g.state_dict(),f=f.state_dict(),da=da.state_dict(),db=db.state_dict(),pairD=pd.state_dict() if pd is not None else None),root/(name+'_weights.pt'));print('RESULT',json.dumps(metric),flush=True)
            del g,f,base,da,db,pd,anchor;torch.cuda.empty_cache()
if __name__=='__main__':main()
