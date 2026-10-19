"""Fresh re-run of the existing three-stage .8s->2s baseline, no auth reads."""
import os,sys,argparse,json,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'mei_audio_repair_20261018'))
from train_mei_full_context import FullG,ReturnF,AudioDiscriminator,Discriminator,acc_loss,adv,contrast,load
from train_cycle_generalization import fit

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);args=p.parse_args();root=args.output;root.mkdir(exist_ok=True);torch.set_num_threads(2)
    tr=load(args.source/'data/mei180_train.npz');full=load(args.prior/'full_context_data/full_train.npz');assert set(tr['ids'])==set(full['ids']) and len(set(full['ids']))==180
    (root/'cyclegan_protocol.json').write_text(json.dumps(dict(seed=args.seed,source='Exact same180 unique mei training sources and cached shared frozen frontend as prior baseline; no auth, test, validation arrays loaded by this trainer.',stage1='New initialization: original project adapted ResNet G(hidden16,2blocks,resize,noIN) + temporal F64x5; original fit prewarm360x4 then completecycle360x4 batch8.',stage2='FullG changes output crop81->inputT201, same loaded learned weights; newD40warmup, same prior full_context loss120x4. All original coefficients/RNG/learningrates retained. No latest flow/sharpness variants. Fixed endpoints; no checkpoint selection on held-out data.',scope='Re-run of recent .3071-reference baseline, not legacy80x80trainb GAN. Source functions reused unchanged; freshly trained from seed42, not previous weights.'),indent=2))
    warm=argparse.Namespace(warm_epochs=360,cycle_epochs=360,updates=4)
    g0,f,meta=fit(dict(small=False,feature_cycle=False),args.seed,tr,root,'fresh_local',warm,cycle_epochs=360)
    g=FullG().cuda();g.load_state_dict(g0.state_dict());del g0
    cp_proto=json.loads((args.prior/'results/protocol.json').read_text());alo,ahi=cp_proto['normalization_audio'];span=ahi-alo
    x=torch.as_tensor(full['x'],device='cuda');y=torch.as_tensor(full['y'],device='cuda')
    torch.manual_seed(args.seed+2026);da,db=Discriminator().cuda(),AudioDiscriminator().cuda();opt=torch.optim.AdamW(list(g.parameters())+list(f.parameters()),lr=.00015,betas=(.5,.999),weight_decay=.001);od=torch.optim.AdamW(list(da.parameters())+list(db.parameters()),lr=.0002,betas=(.5,.999));rng=np.random.default_rng(args.seed);noise=torch.Generator(device='cuda').manual_seed(args.seed+777)
    def dl(a,b,ga,fb):return .5*((da(a)-1).square().mean()+da(fb).square().mean()+adv(db(b),1)+adv(db(ga),0))
    for _ in range(40):
        ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
        with torch.no_grad():ga=g(a);fb=f(b)
        od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
    env=lambda a:torch.log10(torch.pow(10.,(a-1)*span/20).mean(1).clamp_min(1e-12));start=time.time();history=[]
    for ep in range(1,121):
        g.train();f.train();losses=[]
        for _ in range(4):
            ii=rng.choice(len(x),8,replace=False);a,b=x[ii],y[ii]
            for z in [da,db]:z.requires_grad_(True)
            with torch.no_grad():ga=g(a);fb=f(b)
            od.zero_grad();loss=dl(a,b,ga,fb);loss.backward();od.step()
            for z in [da,db]:z.requires_grad_(False)
            ga=g(a);fb=f(b);rec=f(ga+.005*torch.randn(ga.shape,device='cuda',generator=noise));ry=g(fb)
            loss=20*abs(ga-b).mean()+20*acc_loss(rec,a)+10*abs(ry-b).mean()+5*acc_loss(fb,a)+.1*adv(db(ga),1)+.05*(da(fb)-1).square().mean()+.5*abs(contrast(ga)-contrast(b)).mean()+abs(torch.diff(ga,dim=1)-torch.diff(b,dim=1)).mean()+abs(torch.diff(ga,dim=2)-torch.diff(b,dim=2)).mean()+3*abs(env(ga)-env(b)).mean()/(span/10)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(g.parameters())+list(f.parameters()),5);opt.step();losses.append(float(loss.detach()))
        if ep%30==0:
            row=dict(epoch=ep,loss=float(np.mean(losses)),seconds=round(time.time()-start,1));history.append(row);(root/'full_history.json').write_text(json.dumps(history,indent=2));print('FULL',json.dumps(row),flush=True)
    torch.save(dict(g=g.state_dict(),f=f.state_dict(),seed=args.seed,local_fit=meta),root/'baseline_weights.pt');print('BASELINE_DONE',flush=True)
if __name__=='__main__':main()
