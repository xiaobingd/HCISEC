"""Conservative learned detail and bounded activity correction inside the full cycle."""
import os,sys,json,argparse,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_mei_audio_repair import DetailG,AudioG,ReturnF,AudioDiscriminator,Discriminator,acc_loss,adv,evaluate,extra_metrics,load,contrast

class ConservativeG(DetailG):
    def __init__(self,base,alo,ahi,threshold,activity):
        super().__init__(base);self.base.requires_grad_(False);self.span=ahi-alo;self.threshold=threshold
        self.energy_head=nn.Conv1d(16,1,3,padding=1) if activity else None
        if self.energy_head is not None:nn.init.zeros_(self.energy_head.weight);nn.init.zeros_(self.energy_head.bias)
    def forward(self,x):
        base=self.base(x);coords=self.acc_projection(x).reshape(len(x),4,80,x.shape[-1])
        z=self.blocks(F.silu(self.stem(torch.cat([base[:,None],coords],1))))
        delta=self.head(z)[:,0];delta=contrast(delta)
        power=torch.pow(10.,(base-1)*self.span/20)
        energy=torch.log10(power.mean(1).clamp_min(1e-12))
        gate=torch.sigmoid((energy-self.threshold)/.15)[:,None]
        refined=power*torch.exp(2*torch.tanh(delta)*gate)
        refined=refined*power.sum(1,keepdim=True)/refined.sum(1,keepdim=True).clamp_min(1e-12)
        if self.energy_head is not None:
            shift=1.5*torch.tanh(self.energy_head(z.mean(2)))
            refined=refined*torch.pow(10.,shift)
        # Energy conservation is before fixed global clipping; audit actual output energy afterwards.
        return (1+20*torch.log10(refined.clamp_min(1e-12))/self.span).clamp(-1,1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--epochs',type=int,default=150);p.add_argument('--updates',type=int,default=4);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);args=p.parse_args()
    root=args.study/'conservative_results';root.mkdir(exist_ok=True);torch.set_num_threads(2)
    protocol=json.loads((args.study/'results/protocol.json').read_text());alo,ahi=protocol['normalization_audio'];threshold=protocol['audio_energy_threshold'];span=ahi-alo
    protocol.update(routes=['energy_detail','activity_detail'],epochs=args.epochs,followup='Exploratory after first objective-repair seed improved spectraldetail but increased falseactivity. Both routes freeze initialbaseG and train fullresolution residual/F; conservativeGAIN redistributes power in fixed Mel bins, preserves frameMelpower BEFORE finalclip. Trainable activityroute adds bounded+-15dB frameenergy correction. Raw output itself goes intoF, no cosmeticpostfilter/nohiddenbypass.',loss='pairMel20 + active-localcontrast4 + activefreqgradient1 + timegradient1 + initialTRAINoutputanchor1 + ACCcycle20/reverseaudio10/pairedACC5 + audioGAN.1/ACCGAN.05; activityroute adds balancedautomaticenergyBCE2 + physicalenergyregression5. No S or old prewarmteacher.',budget='150epochs4updates sameeachroute; matchedseedinitialG/F, zeroheads oldoutputparity, sameDseed initialization. BaseGfrozen, newrefinertrainable; optimizationpackage differsfirst4routes.',lr='newGrefiner.0003,F.0001,D.0002',note='Energy labels are TRAIN-threshold automatic proxies, not human speech truth. Gain module is learned insideG, not image sharpening. No identityaccuracy or waveform claim.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));data=load(args.source/'data/mei180_train.npz');tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda');x,y=tensor(data['x']),tensor(data['y']);groups=[np.flatnonzero(data['ids']==g) for g in np.unique(data['ids'])];results=[]
    env=lambda a:torch.log10(torch.pow(10.,(a-1)*span/20).mean(1).clamp_min(1e-12))
    for seed in args.seeds:
        cp=torch.load(args.source/'results_long'/('mei180_long_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True)
        for activity in [False,True]:
            route='activity_detail' if activity else 'energy_detail';name=route+'_'+str(seed);torch.manual_seed(seed)
            base=AudioG().cuda();base.load_state_dict(cp['g']);base.eval()
            with torch.no_grad():anchor=torch.cat([base(x[i:i+24]).detach() for i in range(0,len(x),24)])
            g=ConservativeG(base,alo,ahi,threshold,activity).cuda();f=ReturnF().cuda();f.load_state_dict(cp['f'])
            with torch.no_grad():
                parity=float(abs(g(x[:2])-base(x[:2])).max())
                assert parity<2e-5,parity
            torch.manual_seed(seed+2026);da,db=Discriminator().cuda(),AudioDiscriminator().cuda();allD=[da,db]
            opt=torch.optim.AdamW([dict(params=[p for p in g.parameters() if p.requires_grad],lr=.0003),dict(params=f.parameters(),lr=.0001)],betas=(.5,.999),weight_decay=.001)
            od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0002,betas=(.5,.999));rng=np.random.default_rng(seed);noise=torch.Generator(device='cuda').manual_seed(seed+777);start=time.time();history=[]
            for _ in range(40):
                ii=[rng.choice(groups[j]) for j in rng.choice(len(groups),8,replace=False)];a,b=x[ii],y[ii]
                with torch.no_grad():ga=g(a);fb=f(b)
                od.zero_grad();dl=.5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0));dl.backward();od.step()
            rng=np.random.default_rng(seed)
            for ep in range(1,args.epochs+1):
                g.train();g.base.eval();f.train();losses=[]
                for _ in range(args.updates):
                    ii=[rng.choice(groups[j]) for j in rng.choice(len(groups),8,replace=False)];a,b=x[ii],y[ii]
                    for d in allD:d.requires_grad_(True)
                    with torch.no_grad():ga=g(a);fb=f(b)
                    od.zero_grad();dl=.5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0));dl.backward();od.step()
                    for d in allD:d.requires_grad_(False)
                    ga=g(a);fb=f(b);rec=f(ga+.005*torch.randn(ga.shape,device='cuda',generator=noise));reverse=g(fb);target_e=env(b);pred_e=env(ga);mask=(target_e>threshold).float();weighted=lambda z:(z*mask[:,None]).sum()/(mask.sum().clamp_min(1)*z.shape[1])
                    loss=20*abs(ga-b).mean()+4*weighted(abs(contrast(ga)-contrast(b)))+weighted(abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)))+abs(torch.diff(ga,dim=2)-torch.diff(b,dim=2)).mean()+abs(ga-anchor[ii]).mean()+20*acc_loss(rec,a)+10*abs(reverse-b).mean()+5*acc_loss(fb,a)+.1*adv(db(ga),1)+.05*(da(fb)-1).square().mean()
                    if activity:
                        bce=F.binary_cross_entropy_with_logits((pred_e-threshold)/.15,mask,reduction='none');balanced=.5*((bce*mask).sum()/mask.sum().clamp_min(1)+(bce*(1-mask)).sum()/(1-mask).sum().clamp_min(1))
                        loss=loss+2*balanced+5*abs(pred_e-target_e).mean()/(span/10)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in g.parameters() if p.requires_grad]+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
                if ep%30==0 or ep==args.epochs:
                    row=dict(epoch=ep,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);(root/(name+'_history.json')).write_text(json.dumps(history,indent=2));print(name,json.dumps(row),flush=True)
            val=load(args.source/'data/mei_val.npz');assert not set(data['ids'])&set(val['ids']);templates=(data['x'].mean(0),data['y'].mean(0));vm,preds=evaluate(g,f,val,templates);em=extra_metrics(preds,val,alo,ahi,threshold);tm,_=evaluate(g,f,data,templates)
            result=dict(route=route,seed=seed,train=tm,validation=vm,extra_validation=em,seconds=round(time.time()-start,1),parameters=sum(p.numel() for p in g.parameters()));results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2));np.savez_compressed(root/(name+'_predictions.npz'),**preds)
            torch.save(dict(g=g.state_dict(),f=f.state_dict(),da=da.state_dict(),db=db.state_dict()),root/(name+'_weights.pt'));print('RESULT',json.dumps(result),flush=True)
            del g,f,base,da,db,anchor;torch.cuda.empty_cache()
if __name__=='__main__':main()
