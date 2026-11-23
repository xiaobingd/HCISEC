"""Matched V1 ACC-to-ACC response control; private caches stay on instance."""
import sys
from pathlib import Path
sys.argv=["control","--owners","1","--steps","400"]
exec(Path("/root/run_diffusion_v1_cpu.py").read_text().split("protocol=dict")[0])
out=root/"full_run_10x40_raw_acc"/"diffusion_self_control"
out.mkdir(exist_ok=True)
torch.manual_seed(42)
net=BiometricUNet64()
dummy=BiometricUNet64()
del dummy
opt=torch.optim.AdamW(net.parameters(),lr=1e-4)
rng=np.random.default_rng(42)
tr=np.flatnonzero(train&(u==1));va=np.flatnonzero(valid&(u==1))
best=float("inf");start=0;history=[]
latest=out/"latest_private.pt"
if latest.exists():
    ck=torch.load(latest,map_location="cpu",weights_only=False)
    net.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
    start=ck["step"];best=ck["best"];history=ck["history"]
    rng.bit_generator.state=ck["rng"];torch.set_rng_state(ck["torch_rng"])
for step in range(start,400):
    ix=rng.choice(tr,4,replace=False);t=torch.randint(0,1000,(4,))
    net.train();xx=a[ix]
    xt,eps=schedule.forward_diffusion(xx,t);pred=net(xt,t,xx)
    loss=F.mse_loss(pred,eps)+.4*(1-ssim(schedule.predict_x0(xt,t,pred).clamp(-1,1),xx))
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
    torch.randn_like(xx) # Match the other direction's RNG consumption in the cross-modal run.
    if (step+1)%100==0:
        vv=[]
        net.eval()
        with torch.no_grad():
            for batch in np.array_split(va,12):
                for tv in (100,500):
                    xx=a[batch];eps=noise(xx.shape,9100+tv)
                    alpha=schedule.alphas_cumprod[tv]
                    xt=alpha.sqrt()*xx+(1-alpha).sqrt()*eps
                    tt=torch.full((len(batch),),tv,dtype=torch.long)
                    pred=net(xt,tt,xx)
                    vv.append(float(F.mse_loss(pred,eps)+.4*(1-ssim(schedule.predict_x0(xt,tt,pred).clamp(-1,1),xx))))
        value=float(np.mean(vv))
        if value<best:
            best=value;torch.save(dict(model=net.state_dict(),step=step+1,validation=value),out/"best_private.pt")
        history.append(dict(step=step+1,validation=value))
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step+1,best=best,
                        rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=history),latest)
        (out/"training_public.json").write_text(json.dumps(history,indent=2))
        print("SELF_TRAIN",step+1,value,flush=True)
net.load_state_dict(torch.load(out/"best_private.pt",map_location="cpu",weights_only=False)["model"]);net.eval()
indices=np.flatnonzero(reg|cal|same|cross)
blocks=out/"feature_blocks_private";blocks.mkdir(exist_ok=True)
for offset in range(0,len(indices),4):
    path=blocks/f"{offset:05}.npz"
    if path.exists():continue
    ix=indices[offset:offset+4]
    with torch.no_grad():
        aa=a[ix];proxy=sample(net,aa,1881);parts=[]
        for tv in (100,500,900):
            features=[]
            for seed in (7801,7802):
                alpha=schedule.alphas_cumprod[tv]
                xt=alpha.sqrt()*proxy+(1-alpha).sqrt()*noise(aa.shape,seed)
                tt=torch.full((len(ix),),tv,dtype=torch.long)
                delta=net(xt,tt,aa)-net(xt,tt,torch.zeros_like(aa))
                flat=delta.flatten(1)
                stat=torch.stack([flat.mean(1),flat.std(1),flat.abs().max(1).values,flat.square().mean(1).sqrt()],1)
                features.append(torch.cat([delta.mean(3).flatten(1),delta.std(3).flatten(1),stat],1))
            parts.append(torch.stack(features).mean(0))
        v=torch.cat(parts,1)
        assert v.shape[1]==252 and torch.isfinite(v).all()
        np.savez_compressed(path,indices=ix,sample_id=d["sample_id"][ix],guidance=v.numpy())
    if offset%80==0:print("SELF_FEATURE",offset,len(indices),flush=True)
print("SELF_DONE",flush=True)

