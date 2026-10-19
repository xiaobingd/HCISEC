"""Paired anti-smoothing ablation on full-context complete cycles.

Focal Fourier error adapted to Mel-image patches (not waveform frequencies).
No image postprocessing, user-dependent calibration, extra data, or VAL selection.
"""
import os,sys,json,time,argparse,zipfile
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mei_audio_repair_20261018'))
from train_mei_full_context import FullG,ReturnF,legacy_predictions,metrics
from train_mei_audio_repair import AudioDiscriminator,Discriminator,acc_loss,adv,contrast,load,DetailBlock,log_energy_numpy

class SharpG(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.projection=nn.Conv1d(251,160,1)
        self.stem=nn.Conv2d(3,12,3,padding=1)
        self.blocks=nn.Sequential(*[DetailBlock(12,d) for d in [(1,1),(1,2),(2,1),(1,4)]])
        self.head=nn.Conv2d(12,1,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        y=self.base(x);c=self.projection(x).reshape(len(x),2,80,x.shape[-1])
        h=self.blocks(F.silu(self.stem(torch.cat([y[:,None],c],1))))
        return (y+.35*torch.tanh(self.head(h)[:,0])).clamp(-1,1)

class HighpassG(SharpG):
    def __init__(self,base):
        super().__init__(base)
        self.base.requires_grad_(False)
    def forward(self,x):
        y=self.base(x);c=self.projection(x).reshape(len(x),2,80,x.shape[-1])
        h=self.blocks(F.silu(self.stem(torch.cat([y[:,None],c],1))))
        delta=.175*torch.tanh(self.head(h)[:,0])
        delta=delta-F.avg_pool2d(delta[:,None],(9,1),stride=1,padding=(4,0),count_include_pad=False)[:,0]
        return (y+delta).clamp(-1,1)

def focal_fourier(pred,target):
    # Tile local Mel images, comparing complex coefficients including phase.
    # Detached adaptive weights emphasize difficult components within each patch.
    losses=[]
    for size in (16,32):
        pp=pred.unfold(1,size,size//2).unfold(2,size,size//2)
        tt=target.unfold(1,size,size//2).unfold(2,size,size//2)
        diff=torch.fft.fft2(pp-tt,norm='ortho')
        sq=diff.real.square()+diff.imag.square()
        weight=sq.detach().sqrt()
        weight=weight/weight.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-8)
        losses.append((weight*sq).mean())
    return sum(losses)/len(losses)

def paired_structure(pred,target):
    # Scale-invariant local correspondence avoids L1 favoring flat medians.
    # Paired contrast magnitude is constrained separately to avoid tiny-amplitude
    # solutions. This is a loss on actual targets, not unpaired texture matching.
    pp=pred.unfold(1,16,8).unfold(2,16,8).flatten(-2)
    tt=target.unfold(1,16,8).unfold(2,16,8).flatten(-2)
    pp=pp-pp.mean(-1,keepdim=True);tt=tt-tt.mean(-1,keepdim=True)
    pv=pp.square().mean(-1).add(1e-5).sqrt();tv=tt.square().mean(-1).add(1e-5).sqrt()
    weight=(tv.detach()/.15).clamp(0,1)
    cosine=(pp*tt).mean(-1)/(pv*tv).clamp_min(1e-5)
    shape=((1-cosine)*weight).sum()/weight.sum().clamp_min(1)
    amplitude=(abs(torch.log(pv.clamp_min(.02))-torch.log(tv.clamp_min(.02)))*weight).sum()/weight.sum().clamp_min(1)
    return shape,amplitude

def detail_metrics(pred,target,threshold,alo,ahi):
    active=log_energy_numpy(target,alo,ahi)>threshold
    def hp(z):
        t=torch.as_tensor(z)
        return (t-F.avg_pool2d(t[:,None],(9,1),stride=1,padding=(4,0),count_include_pad=False)[:,0]).numpy()
    a,b=hp(pred),hp(target);mask=np.broadcast_to(active[:,None],a.shape)
    ga=np.diff(pred,axis=1);gb=np.diff(target,axis=1)
    # False fine detail: local sign maxima occurring without a nearby target maximum.
    pa=(pred[:,1:-1]>pred[:,:-2]+.025)&(pred[:,1:-1]>pred[:,2:]+.025)
    pb=(target[:,1:-1]>target[:,:-2]+.025)&(target[:,1:-1]>target[:,2:]+.025)
    dil=pb.copy();dil[:,1:]|=pb[:,:-1];dil[:,:-1]|=pb[:,1:]
    am=active[:,None,:]
    return dict(active_highpass_mae=float(abs(a-b)[mask].mean()),active_highpass_rms_ratio=float(np.sqrt(np.mean(a[mask]**2))/max(np.sqrt(np.mean(b[mask]**2)),1e-9)),gradient_mae=float(abs(ga-gb).mean()),gradient_magnitude_ratio=float(abs(ga).mean()/max(abs(gb).mean(),1e-9)),near_target_peak_precision=float((pa&dil&am).sum()/max((pa&am).sum(),1)),near_target_peak_recall=float((pb&(pa|np.concatenate([pa[:,1:],np.zeros_like(pa[:,:1])],1)|np.concatenate([np.zeros_like(pa[:,:1]),pa[:,:-1]],1))&am).sum()/max((pb&am).sum(),1)),peak_scope='Mel local maxima margin .025, +/-1 frequency bin, active target proxy only; not physical harmonic/F0 or identity metric.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=180);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);p.add_argument('--routes',nargs='+',choices=['continue','focal','focal_residual','structure','structure_residual','highpass'],default=['continue','focal','focal_residual']);args=p.parse_args()
    root=args.output;root.mkdir(exist_ok=True);torch.set_num_threads(2)
    proto=json.loads((args.prior/'results/protocol.json').read_text());alo,ahi=proto['normalization_audio'];threshold=proto['audio_energy_threshold'];span=ahi-alo
    tr=load(args.prior/'full_context_data/full_train.npz');va=load(args.prior/'full_context_data/full_validation.npz');legacy=load(args.source/'data/mei_val.npz');assert not set(tr['ids'])&set(va['ids'])
    x=torch.as_tensor(tr['x'],device='cuda');y=torch.as_tensor(tr['y'],device='cuda')
    protocol=dict(scope='Repeatedly explored same 180TRAIN/60VAL recordings; no TEST/auth read. Shared frozen frontend unchanged. Offline full2s. Same 180 common 65-frame interiors as preceding experiment. Fixed end point; no VAL selection.',routes=args.routes,seeds=args.seeds,epochs=args.epochs,updates_per_epoch=4,batch_recordings=8,initial='Same-seed final full-context G/F weights; new D 40 warmup all routes.',loss='All prior full-context losses and complete bidirectional cycles retained. Focal adds 8*mean patch16/32 focal complex Fourier error + 4*paired local contrast L1. Residual additionally trainable full-resolution zero-head .35 bounded branch, no downsample/normalization. Structure routes instead add 4*local centered cosine error + 2*local log std error in 16x16 target patches (8 stride), target std weights; exploratory followup after first focal observations.',reference='https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.html',limits='Mel-image Fourier coefficients are not waveform spectral frequencies. Architecture/objective ablation, not proof of physical harmonic or identity recovery.')
    protocol['highpass_followup']='After first structure route had more peaks but poorer correspondence: freeze same-seed prior full-context base G; learn ONLY high-pass residual .175*tanh(head) minus9bin localmean (max increment .35), F still trainable. Adds 1*local shape+0.5*logstd instead of structure4/2; all prior completecycle losses unchanged. Base freeze / constrained branch / weaker structure is a package, not isolated component attribution.'
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));results=json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else []
    env=lambda a:torch.log10(torch.pow(10.,(a-1)*span/20).mean(1).clamp_min(1e-12))
    for seed in args.seeds:
        for route in protocol['routes']:
            if any(r['seed']==seed and r['route']==route for r in results):
                print('SKIP_COMPLETED',route,seed,flush=True);continue
            torch.manual_seed(seed);cp=torch.load(args.prior/'full_context_results'/('full_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True)
            g=FullG().cuda();f=ReturnF().cuda();g.load_state_dict(cp['g']);f.load_state_dict(cp['f'])
            if route.endswith('_residual') or route=='highpass':
                base=g;g=(HighpassG(base) if route=='highpass' else SharpG(base)).cuda()
                with torch.no_grad():assert torch.equal(g(x[:2]),base(x[:2]))
            # Reset RNG after optional branch initialization for common D, batches and noise.
            torch.manual_seed(seed+2027)
            da,db=Discriminator().cuda(),AudioDiscriminator().cuda()
            opt=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=.0001,betas=(.5,.999),weight_decay=.001);od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0002,betas=(.5,.999));rng=np.random.default_rng(seed);noise=torch.Generator(device='cuda').manual_seed(seed+888)
            def dl(a,b,ga,fb):return .5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0))
            for _ in range(40):
                ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
                with torch.no_grad():ga=g(a);fb=f(b)
                od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
            history=[];start=time.time()
            for ep in range(1,args.epochs+1):
                g.train();f.train();losses=[]
                for _ in range(4):
                    ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
                    for z in [da,db]:z.requires_grad_(True)
                    with torch.no_grad():ga=g(a);fb=f(b)
                    od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
                    for z in [da,db]:z.requires_grad_(False)
                    ga=g(a);fb=f(b);rec=f(ga+.005*torch.randn(ga.shape,device='cuda',generator=noise));ry=g(fb)
                    loss=20*abs(ga-b).mean()+20*acc_loss(rec,a)+10*abs(ry-b).mean()+5*acc_loss(fb,a)+.1*adv(db(ga),1)+.05*(da(fb)-1).square().mean()+.5*abs(contrast(ga)-contrast(b)).mean()+abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)).mean()+abs(torch.diff(ga,dim=2)-torch.diff(b,dim=2)).mean()+3*abs(env(ga)-env(b)).mean()/(span/10)
                    if route.startswith('focal'):loss=loss+8*focal_fourier(ga,b)+4*abs(contrast(ga)-contrast(b)).mean()
                    if route.startswith('structure'):
                        shape,amplitude=paired_structure(ga,b);loss=loss+4*shape+2*amplitude
                    if route=='highpass':
                        shape,amplitude=paired_structure(ga,b);loss=loss+shape+.5*amplitude
                    assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
                if ep%30==0 or ep==args.epochs:
                    row=dict(epoch=ep,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);(root/(route+'_'+str(seed)+'_history.json')).write_text(json.dumps(history,indent=2));print(route,seed,json.dumps(row),flush=True)
            d,preds,full=legacy_predictions(g,f,va,legacy);score=metrics(d,preds,alo,ahi,threshold);sharp=detail_metrics(preds['generated_audio'],d['y'],threshold,alo,ahi)
            ids=np.unique(d['ids']);names=[n for n in ids if str(n).startswith('0-9__')][:2]+[n for n in ids if str(n).startswith('word__')][:2];lookup={str(n):i for i,n in enumerate(full['ids'])};ii=[lookup[str(n)] for n in names];vi=[np.flatnonzero(va['ids']==n)[0] for n in names]
            np.savez_compressed(root/(route+'_'+str(seed)+'_plots.npz'),generated_audio=full['generated_audio'][ii],cycle_ACC=full['cycle_ACC'][ii],real_audio=va['y'][vi],real_ACC=va['x'][vi],ids=np.array(names))
            np.savez_compressed(root/(route+'_'+str(seed)+'_common.npz'),**preds)
            torch.save(dict(g=g.state_dict(),f=f.state_dict()),root/(route+'_'+str(seed)+'_weights.pt'))
            # Training indicators do not participate in checkpoint choice.
            train_legacy=load(args.source/'data/mei180_train.npz')
            td,tp,_=legacy_predictions(g,f,tr,train_legacy);train_mae=float(abs(tp['generated_audio']-td['y']).mean())
            result=dict(seed=seed,route=route,metrics=score,sharp=sharp,train_audio_mae=train_mae);results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2));print('RESULT',route,seed,json.dumps(dict(macro=score['macro'],sharp=sharp,train_audio_mae=train_mae)),flush=True)
            del g,f,da,db,cp;torch.cuda.empty_cache()
    # Private compact review archive excludes full arrays and weights.
    with zipfile.ZipFile(root/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for pattern in ['results.json','protocol.json','*_plots.npz','*_history.json']:
            for path in root.glob(pattern):z.write(path,path.name)
    print('DONE',flush=True)
if __name__=='__main__':main()
