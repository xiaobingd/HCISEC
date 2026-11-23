"""Adapted V1 personal diffusion, resumable CPU training and ACC-only authentication."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.svm import SVC

p=argparse.ArgumentParser()
p.add_argument("--owners",type=int,nargs="+",required=True)
p.add_argument("--steps",type=int,required=True)
p.add_argument("--sample-steps",type=int,default=10)
p.add_argument("--smoke",action="store_true")
args=p.parse_args()
root=Path("/root")
out=root/"full_run_10x40_raw_acc"/"diffusion_v1_cpu"
out.mkdir(exist_ok=True)
source=root/"autodl-tmp/train_core/diffusion"
sys.path.insert(0,str(source))
from model_diff import BiometricUNet64
from train_ddpm import DiffusionSchedule,SSIM
torch.set_num_threads(2)
torch.manual_seed(42)
d=np.load(root/"full_run_10x40_raw_acc/diffusion_baseline/frontend_private.npz")
a=F.interpolate(torch.tensor(d["acc"]),size=(40,32),mode="bilinear",align_corners=False)
# Only upstream train/validation Audio is present in the source cache.
m=F.interpolate(torch.tensor(d["audio"]),size=(40,32),mode="bilinear",align_corners=False)
u=d["user"]
reg,cal,same,cross=[d[k] for k in ("register","calibration","sameword","crossword")]
train,valid=d["train"],d["valid"]
assert (train.astype(int)+valid+reg+cal+same+cross).max()==1
rawdata=np.load(root/"paired_raw_acc_private.npz")
rawstats=np.load(root/"stats27_private.npz")
assert np.array_equal(d["sample_id"],rawdata["sample_id"])
assert np.array_equal(d["sample_id"],rawstats["sample_id"])
s=rawdata["spectrum"].reshape(len(u),120,-1)
raw=np.column_stack([s.mean(-1),s.std(-1),*(v.mean(-1) for v in np.array_split(s,4,axis=-1)),
                     rawdata["amplitude"],rawstats["features"]])
raw=normalize(StandardScaler().fit(raw[reg]).transform(raw))
schedule=DiffusionSchedule(1000,device="cpu")
ssim=SSIM()
def noise(shape,seed):
    return torch.randn((1,*shape[1:]),generator=torch.Generator().manual_seed(seed)).expand(shape)
@torch.no_grad()
def sample(net,condition,seed):
    image=noise(condition.shape,seed).clone()
    steps=np.unique(np.linspace(999,0,args.sample_steps+1,dtype=int))[::-1].copy()
    for tval,nextval in zip(steps[:-1],steps[1:]):
        t=torch.full((len(image),),int(tval),dtype=torch.long)
        eps=net(image,t,condition)
        alpha=schedule.alphas_cumprod[tval]
        nextalpha=schedule.alphas_cumprod[nextval]
        x0=((image-(1-alpha).sqrt()*eps)/alpha.sqrt()).clamp(-1,1)
        image=nextalpha.sqrt()*x0+(1-nextalpha).sqrt()*eps
    return image.clamp(-1,1)
@torch.no_grad()
def validate(models,ix):
    values=[]
    for net,target,condition in ((models[0],m,a),(models[1],a,m)):
        net.eval()
        for batch in np.array_split(ix,max(1,(len(ix)+3)//4)):
            for tval in (100,500):
                xx,cc=target[batch],condition[batch]
                t=torch.full((len(batch),),tval,dtype=torch.long)
                eps=noise(xx.shape,9100+tval)
                alpha=schedule.alphas_cumprod[tval]
                xt=alpha.sqrt()*xx+(1-alpha).sqrt()*eps
                pred=net(xt,t,cc)
                loss=F.mse_loss(pred,eps)+.4*(1-ssim(schedule.predict_x0(xt,t,pred).clamp(-1,1),xx))
                values.append(float(loss))
    return float(np.mean(values))

protocol=dict(version="V1 train_ddpm.py linear beta1e-4..0.01,1000steps,noiseMSE+0.4SSIM at all t",
              source_sha256=hashlib.sha256((source/"train_ddpm.py").read_bytes()).hexdigest(),
              model_sha256=hashlib.sha256((source/"model_diff.py").read_bytes()).hexdigest(),
              note="network file is the surviving BiometricUNet64 with GroupNorm; earliest archive identity unverified",
              initialization="random,seed42; no ESMB/V3 checkpoint",
              owners=args.owners,steps=args.steps,batch=4,lr=1e-4,optimizer="AdamW default weight decay=.01",
              resolution=[40,32],sampling_steps=args.sample_steps,features="latent1536,guidance252,cycle88,bundle1876",
              normalization="global upstream train-only mean and 3std,clip[-1,1]",
              counts={k:int(d[k].sum()) for k in ("train","valid","register","calibration","sameword","crossword")},
              authentication="balanced RBF C10;calibration EER threshold;query ACC only",
              target_scope="U10 retained as attacker; target owners specified explicitly")
(out/"protocol_public.json").write_text(json.dumps(protocol,indent=2))
for owner in args.owners:
    folder=out/f"owner{owner}_steps{args.steps}"
    folder.mkdir(exist_ok=True)
    tr=np.flatnonzero(train&(u==owner));va=np.flatnonzero(valid&(u==owner))
    assert len(tr)==160 and len(va)==48
    torch.manual_seed(42)
    rng=np.random.default_rng(42)
    models=[BiometricUNet64(),BiometricUNet64()]
    optimizers=[torch.optim.AdamW(net.parameters(),lr=1e-4) for net in models]
    latest=folder/"latest_private.pt"
    bestpath=folder/"best_private.pt"
    start=0;best=float("inf");history=[]
    if latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=False)
        for net,state,opt,ost in zip(models,ck["models"],optimizers,ck["optimizers"]):
            net.load_state_dict(state);opt.load_state_dict(ost)
        start=ck["step"];best=ck["best"];rng.bit_generator.state=ck["rng"]
        torch.set_rng_state(ck["torch_rng"]);history=ck["history"]
    for step in range(start,args.steps):
        ix=rng.choice(tr,4,replace=False)
        t=torch.randint(0,1000,(len(ix),))
        losses=[]
        # The directions are independent; sequential backward lowers peak memory.
        for net,opt,target,condition in ((models[0],optimizers[0],m,a),(models[1],optimizers[1],a,m)):
            net.train();xx,cc=target[ix],condition[ix]
            xt,eps=schedule.forward_diffusion(xx,t)
            pred=net(xt,t,cc)
            loss=F.mse_loss(pred,eps)+.4*(1-ssim(schedule.predict_x0(xt,t,pred).clamp(-1,1),xx))
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1)
            opt.step();losses.append(float(loss.detach()))
        if (step+1)%100==0 or step+1==args.steps:
            value=validate(models,va)
            if value<best:
                best=value
                torch.save(dict(models=[net.state_dict() for net in models],step=step+1,validation=best),bestpath)
            history.append(dict(step=step+1,train_loss=sum(losses),validation=value))
            torch.save(dict(models=[net.state_dict() for net in models],
                            optimizers=[opt.state_dict() for opt in optimizers],step=step+1,best=best,
                            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history),latest)
            (folder/"training_public.json").write_text(json.dumps(history,indent=2))
            print("TRAIN",owner,step+1,value,flush=True)
    ck=torch.load(bestpath,map_location="cpu",weights_only=False)
    for net,state in zip(models,ck["models"]):net.load_state_dict(state);net.eval()
    indices=np.flatnonzero(reg|cal|same|cross)
    blocks=folder/"feature_blocks_private"
    blocks.mkdir(exist_ok=True)
    for offset in range(0,len(indices),4):
        path=blocks/f"{offset:05}.npz"
        if path.exists():continue
        ix=indices[offset:offset+4]
        with torch.no_grad():
            aa=a[ix]
            fake=sample(models[0],aa,1881)
            recovered=sample(models[1],fake,1882)
            if offset==0:
                single=sample(models[0],aa[:1],1881)
                assert torch.allclose(single,fake[:1],atol=2e-4,rtol=2e-4)
            latent=[];guidance=[]
            for tval in (100,500,900):
                lat=[];guid=[]
                for seed in (7801,7802):
                    alpha=schedule.alphas_cumprod[tval]
                    xt=alpha.sqrt()*fake+(1-alpha).sqrt()*noise(aa.shape,seed)
                    t=torch.full((len(ix),),tval,dtype=torch.long)
                    eps,f=models[0](xt,t,aa,return_features=True)
                    null=models[0](xt,t,torch.zeros_like(aa))
                    h=f["bottleneck"]
                    lat.append(torch.cat([h.mean((2,3)),h.std((2,3))],1))
                    delta=eps-null;flat=delta.flatten(1)
                    stat=torch.stack([flat.mean(1),flat.std(1),flat.abs().max(1).values,flat.square().mean(1).sqrt()],1)
                    guid.append(torch.cat([delta.mean(3).flatten(1),delta.std(3).flatten(1),stat],1))
                latent.append(torch.stack(lat).mean(0));guidance.append(torch.stack(guid).mean(0))
            diff=(aa-recovered).abs();flat=diff.flatten(1)
            stat=torch.stack([flat.mean(1),flat.std(1),flat.max(1).values,flat.square().mean(1).sqrt(),
                              *[torch.quantile(flat,v,dim=1) for v in (.25,.5,.75,.95)]],1)
            cycle=torch.cat([diff.mean(3).flatten(1),diff.std(3).flatten(1),stat],1)
            latent=torch.cat(latent,1);guidance=torch.cat(guidance,1)
            assert [latent.shape[1],guidance.shape[1],cycle.shape[1]]==[1536,252,88]
            assert all(torch.isfinite(v).all() for v in (latent,guidance,cycle))
            np.savez_compressed(path,indices=ix,sample_id=d["sample_id"][ix],
                                latent=latent.numpy(),guidance=guidance.numpy(),cycle=cycle.numpy())
        if offset%80==0:print("FEATURES",owner,offset,len(indices),flush=True)
        if args.smoke:
            print("SMOKE_OK",owner,flush=True)
            break
    if args.smoke:
        continue
    values={name:np.zeros((len(u),size),dtype="float32") for name,size in (("latent",1536),("guidance",252),("cycle",88))}
    for path in sorted(blocks.glob("*.npz")):
        chunk=np.load(path);ix=chunk["indices"]
        assert np.array_equal(chunk["sample_id"],d["sample_id"][ix])
        for name in values:values[name][ix]=chunk[name]
    values["bundle"]=np.c_[values["latent"],values["guidance"],values["cycle"]]
    views={"raw":raw}
    for name,v in values.items():
        vv=normalize(StandardScaler().fit(v[reg]).transform(v))
        views[name]=vv;views["raw_"+name]=np.c_[raw,vv]/np.sqrt(2)
    result=[]
    for name,v in views.items():
        yy=u==owner
        clf=SVC(C=10,class_weight="balanced").fit(v[reg],yy[reg])
        f,t,th=roc_curve(yy[cal],clf.decision_function(v[cal]),drop_intermediate=False)
        tau=th[min(range(len(th)),key=lambda j:(abs(f[j]+t[j]-1),f[j]))]
        for condition,mask in (("sameword",same),("crossword",cross)):
            score=clf.decision_function(v[mask])
            f,t,_=roc_curve(yy[mask],score,drop_intermediate=False)
            j=np.argmin(abs(f+t-1))
            far=np.mean(score[~yy[mask]]>=tau);frr=np.mean(score[yy[mask]]<tau)
            result.append(dict(user=owner,method=name,condition=condition,eer=(f[j]+1-t[j])/2,
                               hter=(far+frr)/2,far=far,frr=frr,auc=roc_auc_score(yy[mask],score)))
    (folder/"results_public.json").write_text(json.dumps(result,indent=2))
    print("AUTH_DONE",owner,flush=True)

