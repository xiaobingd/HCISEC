"""Research-inspired full-cycle pilot: direct vs velocity vs data-prediction FM.

Mel-domain project adaptation, NOT a reproduction of RMFSR/Stream.FM/PASE.
No external samples, pretrained audio model, TEST, auth sources, or MIC inference.
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
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mei_sharp_cycle_20261019'))
from train_mei_full_context import FullG,ReturnF,legacy_predictions,metrics
from train_mei_audio_repair import AudioDiscriminator,Discriminator,acc_loss,adv,contrast,load,DetailBlock
from train_mei_sharp_cycle import detail_metrics

class FlowG(nn.Module):
    def __init__(self,base,route,steps=5):
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.route=route;self.steps=steps;self.sigma=.2
        self.acc=nn.Conv1d(251,4*80,1);self.stem=nn.Conv2d(6,24,3,padding=1)
        self.blocks=nn.ModuleList([DetailBlock(24,d) for d in [(1,1),(1,2),(2,1),(1,4),(4,1),(1,8)]])
        self.time=nn.Sequential(nn.Linear(8,48),nn.SiLU(),nn.Linear(48,24))
        self.head=nn.Conv2d(24,2 if route=='decoupled' else 1,3,padding=1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.noise_generator=torch.Generator(device='cuda').manual_seed(20261019)
    def context(self,x):
        base=self.base(x);c=self.acc(x).reshape(len(x),4,80,x.shape[-1]);return base,c
    def raw(self,z,base,c,t):
        if not torch.is_tensor(t):t=torch.full((len(z),),float(t),device=z.device)
        freq=torch.tensor([1.,2.,4.,8.],device=z.device);angle=t[:,None]*freq[None]*np.pi
        emb=self.time(torch.cat([angle.sin(),angle.cos()],1))[:,:,None,None]
        h=F.silu(self.stem(torch.cat([z[:,None],base[:,None],c],1)))
        for block in self.blocks:h=block(h+emb)
        return self.head(h)
    def predict(self,z,base,c,t):
        raw=self.raw(z,base,c,t)
        if self.route=='decoupled':return .7*torch.tanh(raw[:,0])-self.sigma*raw[:,1]
        delta=.7*torch.tanh(raw[:,0])
        return delta if self.route=='velocity' else (base+delta).clamp(-1,1)
    def sample_context(self,base,c,eps=None):
        if self.route=='direct':return self.predict(base,base,c,1.)
        if eps is None:eps=torch.randn(base.shape,device=base.device,generator=self.noise_generator)
        z=base+self.sigma*eps
        for step in range(self.steps):
            t=1-step/self.steps;v=self.predict(z,base,c,t)
            if self.route=='data_prediction':z=z+(v-z)/(t*self.steps)
            else:z=z+v/self.steps
        return z.clamp(-1,1)
    def forward(self,x):return self.sample_context(*self.context(x))
    def core_loss(self,base,c,target,gen,aux_rng):
        if self.route=='direct':return (gen-target).square().mean()
        # Reverse time 1->0 and a condition-centered Gaussian prior.
        t=torch.sigmoid(.4+torch.randn((len(base),),device=base.device,generator=aux_rng))
        eps=torch.randn(base.shape,device=base.device,generator=aux_rng)
        start=base+self.sigma*eps;z=(1-t[:,None,None])*target+t[:,None,None]*start
        if self.route=='decoupled':
            raw=self.raw(z,base,c,t)
            return (.7*torch.tanh(raw[:,0])-(target-base)).square().mean()+self.sigma**2*(raw[:,1]-eps).square().mean()
        pred=self.predict(z,base,c,t)
        target_field=target-start if self.route=='velocity' else target
        return (pred-target_field).square().mean()

def sampler_contract():
    base=torch.randn(2,80,201,device='cuda');target=torch.randn_like(base);eps=torch.randn_like(base);start=base+.2*eps
    for steps in (1,5):
        z=start.clone()
        for j in range(steps):z=z+(target-z)/((1-j/steps)*steps)
        assert torch.allclose(z,target,atol=2e-6)
        z=start.clone()
        for _ in range(steps):z=z+(target-start)/steps
        assert torch.allclose(z,target,atol=2e-6)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=180);p.add_argument('--seeds',nargs='+',type=int,default=[42,43]);p.add_argument('--routes',nargs='+',choices=['direct','velocity','data_prediction','decoupled'],default=['direct','velocity','data_prediction']);args=p.parse_args();root=args.output;root.mkdir(exist_ok=True);torch.set_num_threads(2);sampler_contract()
    tr=load(args.prior/'full_context_data/full_train.npz');va=load(args.prior/'full_context_data/full_validation.npz');legacy=load(args.source/'data/mei_val.npz');assert not set(tr['ids'])&set(va['ids'])
    oldproto=json.loads((args.prior/'results/protocol.json').read_text());alo,ahi=oldproto['normalization_audio'];threshold=oldproto['audio_energy_threshold'];span=ahi-alo
    x=torch.as_tensor(tr['x'],device='cuda');y=torch.as_tensor(tr['y'],device='cuda');env=lambda a:torch.log10(torch.pow(10.,(a-1)*span/20).mean(1).clamp_min(1e-12))
    protocol=dict(routes=args.routes,seeds=args.seeds,epochs=args.epochs,updates_per_epoch=4,batch_records=8,scope='Same clean mei180TRAIN/60 repeatedly explored VAL full2s shared frozen frontend, same common65frame interiors x180; no TEST/auth read, no external samples or pretrained speech weights. Offline future ACC context.',model='Frozen previous same-seed fullcontext G, 24channel no-downsample time-conditioned predictor shared architecture; ReturnF trainable. No crosscycle bypass. Direct=1NFE, velocity/DP=5NFE. Not compute-matched.',training='New D40warmup all routes, lr.0001; all completeACCcycle20/audio10/pairedACC5, pairedMel2 + coreMSE20 (direct actualoutput; velocity conditional OT field; DP interpolant-conditioned targetdata); audioGAN.1/ACCGAN.05, contrast.5/freqgrad1/timegrad1/physicalenergy3. Noise.005 beforeF. Gaussianprior centeredonbase,sigma.2, t=sigmoid(N(.4,1)). Uniform5step Euler, no MeanFlow/JVP training. Fixedendpoint/noVALcheckpointselection.',evaluation='3fixed draws per trainedflow, no best-of-N, metrics averaged across draw metrics NOT averaged spectrograms. Plot draw0 fixedsame4sources. Deterministic direct only1draw. Stochastic controls use SAME noise for paired and wrong/shuffled inputs.',references=['https://arxiv.org/abs/2605.16251','https://arxiv.org/abs/2512.19442','https://arxiv.org/abs/2510.16997'],limits='Inspired project Mel adaptation, not waveform RMFSR/MeanFlow/Stream.FM reproduction or speakeridentity proof. DP/prior ideas tested, no PASE implementation. Changed matched objective from prior experiment; old model is reference, direct is currentsame-budget control.')
    protocol['decoupled_followup']='Inspired by FRSE dual-field concept (SpeechCommunication183,September2026,doi10.1016/j.specom.2026.103449); not exact reproduction. Same shared24channel trunk,2outputheads instead1. residual target=realMel-base; noise target=sampledGaussian, velocity=residual_hat-.2*noise_hat. Core=sum(residualMSE+.2^2*noiseMSE), no cross-term. Same5NFE/fullcycles. Additional exploratory phase; no selected VALcheckpoint.'
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));results=json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else []
    for seed in args.seeds:
        for route in args.routes:
            if any(r['seed']==seed and r['route']==route for r in results):print('SKIP_COMPLETED',route,seed,flush=True);continue
            torch.manual_seed(seed);cp=torch.load(args.prior/'full_context_results'/('full_'+str(seed)+'_weights.pt'),map_location='cuda',weights_only=True)
            base=FullG().cuda();base.load_state_dict(cp['g']);g=FlowG(base,route).cuda();f=ReturnF().cuda();f.load_state_dict(cp['f'])
            with torch.no_grad():
                b,c=g.context(x[:2]);zero=torch.zeros_like(b)
                if route not in ['velocity','decoupled']:assert torch.equal(g.sample_context(b,c,zero),b)
                assert g.sample_context(b,c,zero).shape==(2,80,201)
            torch.manual_seed(seed+2028);da,db=Discriminator().cuda(),AudioDiscriminator().cuda()
            opt=torch.optim.AdamW([p for p in list(g.parameters())+list(f.parameters()) if p.requires_grad],lr=.0001,betas=(.5,.999),weight_decay=.001);od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0002,betas=(.5,.999))
            rng=np.random.default_rng(seed);g.noise_generator.manual_seed(seed+800);aux_rng=torch.Generator(device='cuda').manual_seed(seed+801);cycle_rng=torch.Generator(device='cuda').manual_seed(seed+802)
            def dl(a,b,ga,fb):return .5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0))
            for _ in range(40):
                ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
                with torch.no_grad():ga=g(a);fb=f(b)
                od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
            g.noise_generator.manual_seed(seed+900);start=time.time();history=[]
            for ep in range(1,args.epochs+1):
                g.train();f.train();losses=[];corevals=[]
                for _ in range(4):
                    ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
                    for disc in (da,db):disc.requires_grad_(True)
                    with torch.no_grad():ga=g(a);fb=f(b)
                    od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
                    for disc in (da,db):disc.requires_grad_(False)
                    ba,c=g.context(a);ga=g.sample_context(ba,c);fb=f(b);rec=f(ga+.005*torch.randn(ga.shape,device='cuda',generator=cycle_rng));ry=g(fb)
                    core=g.core_loss(ba,c,b,ga,aux_rng)
                    loss=20*core+2*abs(ga-b).mean()+20*acc_loss(rec,a)+10*abs(ry-b).mean()+5*acc_loss(fb,a)+.1*adv(db(ga),1)+.05*(da(fb)-1).square().mean()+.5*abs(contrast(ga)-contrast(b)).mean()+abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)).mean()+abs(torch.diff(ga,dim=2)-torch.diff(b,dim=2)).mean()+3*abs(env(ga)-env(b)).mean()/(span/10)
                    assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in list(g.parameters())+list(f.parameters()) if p.requires_grad],5);opt.step();losses.append(float(loss.detach()));corevals.append(float(core.detach()))
                if ep%30==0 or ep==args.epochs:
                    row=dict(epoch=ep,loss=float(np.mean(losses)),core=float(np.mean(corevals)),seconds=round(time.time()-start,1));history.append(row);(root/(route+'_'+str(seed)+'_history.json')).write_text(json.dumps(history,indent=2));print(route,seed,json.dumps(row),flush=True)
            draw_results=[];n_draws=1 if route=='direct' else 3
            for draw in range(n_draws):
                g.noise_generator.manual_seed(12000+draw);d,preds,full=legacy_predictions(g,f,va,legacy);score=metrics(d,preds,alo,ahi,threshold);sharp=detail_metrics(preds['generated_audio'],d['y'],threshold,alo,ahi)
                draw_results.append(dict(draw=draw,metrics=score,sharp=sharp))
                np.savez_compressed(root/(route+'_'+str(seed)+'_draw'+str(draw)+'_common.npz'),**preds)
                if draw==0:
                    ids=np.unique(d['ids']);names=[n for n in ids if str(n).startswith('0-9__')][:2]+[n for n in ids if str(n).startswith('word__')][:2];lookup={str(n):i for i,n in enumerate(full['ids'])};ii=[lookup[str(n)] for n in names];vi=[np.flatnonzero(va['ids']==n)[0] for n in names]
                    np.savez_compressed(root/(route+'_'+str(seed)+'_plots.npz'),generated_audio=full['generated_audio'][ii],cycle_ACC=full['cycle_ACC'][ii],real_audio=va['y'][vi],real_ACC=va['x'][vi],ids=np.array(names))
            result=dict(seed=seed,route=route,draws=draw_results);results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2));torch.save(dict(g=g.state_dict(),f=f.state_dict()),root/(route+'_'+str(seed)+'_weights.pt'));print('RESULT',route,seed,json.dumps(dict(macro={k:float(np.mean([d['metrics']['macro'][k] for d in draw_results])) for k in draw_results[0]['metrics']['macro']},sharp={k:float(np.mean([d['sharp'][k] for d in draw_results])) for k in draw_results[0]['sharp'] if isinstance(draw_results[0]['sharp'][k],(float,int))})),flush=True)
            del g,f,da,db,cp;torch.cuda.empty_cache()
    with zipfile.ZipFile(root/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for pattern in ['results.json','protocol.json','*_plots.npz','*_history.json']:
            for path in root.glob(pattern):z.write(path,path.name)
    print('DONE',flush=True)
if __name__=='__main__':main()
