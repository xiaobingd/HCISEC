"""Matched sampling control for sparse identity-teacher supervision."""
from pathlib import Path
prefix=Path("/root/train_temporal_audio.py").read_text().split("rows=[]")[0]
exec(prefix)
out=root/"full_run_10x40_raw_acc"/"teacher_coverage_fix"
out.mkdir(exist_ok=True)
covered=[np.flatnonzero(available&(u==owner)) for owner in range(1,11)]
allfit=[np.flatnonzero(fit&(u==owner)) for owner in range(1,11)]
assert all(len(v)>=2 for v in covered)
rows=[]
diagnostics=[]
for seed in (42,43,44):
    for method in ("B_matched_sampling","D_covered_audio"):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        net=Student(True)
        head=nn.Linear(192,10,bias=False)
        params=list(net.parameters())
        opt=torch.optim.AdamW([*params,*head.parameters()],lr=.001)
        gradient_log=[]
        for step in range(400):
            idx=np.concatenate([np.r_[rng.choice(covered[i],2,replace=False),
                                       rng.choice(allfit[i],2,replace=False)] for i in range(10)])
            z,pred=net(idx)
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y[idx])
            if method=="D_covered_audio":
                aux=F.smooth_l1_loss(F.adaptive_avg_pool1d(pred,7),F.adaptive_avg_pool1d(targets[idx],7))
                keep=available[idx]
                a=z[keep]@z[keep].T
                b=teacher[idx[keep]]@teacher[idx[keep]].T
                mask=~torch.eye(len(a),dtype=torch.bool)
                kd=F.mse_loss(a[mask],b[mask])
                # Balance norms on shared temporal body, using training batches only.
                shared=list(net.body.parameters())
                gi=torch.autograd.grad(loss,shared,retain_graph=True)
                ga=torch.autograd.grad(aux,shared,retain_graph=True)
                gk=torch.autograd.grad(kd,shared,retain_graph=True)
                norm=lambda g: torch.sqrt(sum(v.square().sum() for v in g))
                ni,na,nk=[float(norm(g)) for g in (gi,ga,gk)]
                wa=min(10.,.1*ni/max(na,1e-8))
                wk=min(10.,.3*ni/max(nk,1e-8))
                gradient_log.append([wa,wk,int(keep.sum()),na*wa/max(ni,1e-8),nk*wk/max(ni,1e-8)])
                loss=loss+wa*aux+wk*kd
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*params,*head.parameters()],1)
            opt.step()
        with torch.inference_mode():
            emb=net.eval()(np.arange(len(u)))[0].numpy()
        for target in range(1,11):
            labels=u==target
            svm=SVC(C=10,class_weight="balanced").fit(emb[reg],labels[reg])
            f,t,h=roc_curve(labels[cal],svm.decision_function(emb[cal]),drop_intermediate=False)
            tau=h[min(range(len(h)),key=lambda i:(abs(f[i]+t[i]-1),f[i]))]
            s=svm.decision_function(emb[query])
            fq,tq,_=roc_curve(labels[query],s,drop_intermediate=False)
            j=np.argmin(abs(fq+tq-1))
            far=np.mean(s[~labels[query]]>=tau);frr=np.mean(s[labels[query]]<tau)
            rows.append(dict(seed=seed,method=method,user=target,eer=(fq[j]+1-tq[j])/2,
                             hter=(far+frr)/2,far=far,frr=frr))
        if gradient_log:
            diagnostics.append(dict(seed=seed,mean_weights_and_coverage=np.mean(gradient_log,axis=0).tolist()))
        torch.save(net.state_dict(),out/f"{method}_{seed}_private.pt")
        (out/"results_public.json").write_text(json.dumps(rows,indent=2))
        (out/"gradient_public.json").write_text(json.dumps(diagnostics,indent=2))
        rr=[v for v in rows if v["seed"]==seed and v["method"]==method and v["user"]!=10]
        print(seed,method,[round(100*np.mean([v[k] for v in rr]),2) for k in ("eer","hter","far","frr")],flush=True)

