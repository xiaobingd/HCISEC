"""Dense training-only teacher extraction, held-word audit and learned projection."""
import copy
from pathlib import Path
exec(Path("/root/train_temporal_audio.py").read_text().split("rows=[]")[0])
from sklearn.metrics import roc_auc_score
out=root/"full_run_10x40_raw_acc"/"dense_learnable_teacher"
out.mkdir(exist_ok=True)
fitix=np.flatnonzero(fit)
poolpath=out/"dense_pool_private.npz"
if not poolpath.exists():
    import sys
    sys.path.insert(0,"/root")
    import run_ecapa_pilot as pilot
    pilot.p.DEVICE="cpu"
    model=pilot.teacher_load(root/"full_run_10x40_raw_acc")
    emb=model.mods.embedding_model.eval()
    emb.load_state_dict(torch.load(root/"full_run_10x40_raw_acc"/"adapted_audio"/"audio_teacher_private.pt",map_location="cpu",weights_only=True))
    pooled=[]
    def capture(module,inputs):
        pooled.append(inputs[0].detach().clone())
    hook=emb.fc.register_forward_pre_hook(capture)
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(fitix),8):
            ids=fitix[start:start+8]
            wave=torch.tensor(waves[start:start+8])
            wave=wave-wave.mean(1,keepdim=True)
            lengths=torch.ones(len(ids))
            f=model.mods.mean_var_norm(model.mods.compute_features(wave),lengths)
            outputs.append(F.normalize(emb(f,lengths).squeeze(1),dim=1))
            if start%160==0:print("DENSE_AUDIO",start,len(fitix),flush=True)
    hook.remove()
    np.savez_compressed(poolpath,indices=fitix,sample_id=d["sample_id"][fitix],
                        pooled=torch.cat(pooled).numpy(),embedding=torch.cat(outputs).numpy())
    torch.save(emb.fc.state_dict(),out/"projection_initial_private.pt")
    del model,emb
cache=np.load(poolpath)
assert np.array_equal(cache["indices"],fitix)
assert np.array_equal(cache["sample_id"],d["sample_id"][fitix])
assert cache["pooled"].shape==(2080,6144,1)
pool=torch.tensor(cache["pooled"])
initial=torch.tensor(cache["embedding"])
old=np.load(root/"full_run_10x40_raw_acc"/"adapted_audio"/"teacher_targets_private.npz")
seenwords=set(w[old["indices"]])
eligible=np.array([word for word in sorted(set(w[fit])) if word not in seenwords])
held=np.random.default_rng(1123).choice(eligible,6,replace=False)
ttrain=~np.isin(w[fitix],held)
assert not np.intersect1d(old["indices"],fitix[~ttrain]).size
trainwords=sorted(set(w[fitix[ttrain]]))
tgroups=np.stack([[np.flatnonzero(ttrain&(u[fitix]==owner)&(w[fitix]==word))
                   for word in trainwords] for owner in range(1,11)])
labels=y[fitix]
def audit(z):
    a=z.numpy()
    values=[]
    for owner in range(1,11):
        regmask=(u[fitix]==owner)&np.isin(w[fitix],held[:3])&(r[fitix]<3)
        center=a[regmask].mean(0);center/=np.linalg.norm(center)
        mask=np.isin(w[fitix],held[3:])&(r[fitix]>=3)
        score=a[mask]@center
        yy=u[fitix][mask]==owner
        values.append(dict(user=owner,crossword_auc=float(roc_auc_score(yy,score)),
                           margin=float(score[yy].mean()-score[~yy].mean())))
    return values

class AudioTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Conv1d(6144,192,1)
        state=torch.load(out/"projection_initial_private.pt",map_location="cpu",weights_only=True)
        self.projection.load_state_dict({k.replace("conv.","",1):v for k,v in state.items()})
        self.branch=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,192))
        nn.init.zeros_(self.branch[-1].weight);nn.init.zeros_(self.branch[-1].bias)
    def forward(self,j):
        z=self.projection(pool[j]).squeeze(-1)
        return F.normalize(z+self.branch(z),dim=1)

rows=[];audits=[];logs=[]
for seed in (42,43,44):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    teacher_net=AudioTeacher()
    head=nn.Linear(192,10,bias=False)
    opt=torch.optim.AdamW([dict(params=teacher_net.projection.parameters(),lr=1e-5),
                          dict(params=teacher_net.branch.parameters(),lr=1e-4),
                          dict(params=head.parameters(),lr=1e-3)])
    for step in range(300):
        ww=rng.choice(len(trainwords),4,replace=False)
        batch=tgroups[:,ww,:]
        j=np.take_along_axis(batch,rng.integers(8,size=(10,4,1)),axis=2).ravel()
        z=teacher_net(j);yy=labels[j]
        sim=z@z.T/.1
        diagonal=torch.eye(len(j),dtype=torch.bool)
        positive=(yy[:,None]==yy)&~diagonal
        logp=sim-torch.logsumexp(sim.masked_fill(diagonal,-torch.inf),dim=1,keepdim=True)
        con=-(logp*positive).sum(1).div(positive.sum(1)).mean()
        loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),yy)+.1*con+.01*(1-(z*initial[j]).sum(1)).mean()
        opt.zero_grad();loss.backward()
        nn.utils.clip_grad_norm_([*teacher_net.parameters(),*head.parameters()],1)
        opt.step()
    with torch.no_grad():
        learned=torch.cat([teacher_net(np.arange(i,min(i+64,len(fitix)))) for i in range(0,len(fitix),64)])
    audits.append(dict(seed=seed,frozen=audit(initial),learned=audit(learned)))
    torch.save(teacher_net.state_dict(),out/f"teacher_{seed}_private.pt")
    print("TEACHER_AUDIT",seed,np.mean([v["crossword_auc"] for v in audits[-1]["frozen"]]),
          np.mean([v["crossword_auc"] for v in audits[-1]["learned"]]),flush=True)
    (out/"audit_public.json").write_text(json.dumps(audits,indent=2))
    for method in ("ACC","dense_frozen","dense_learned"):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        net=Student(True);head=nn.Linear(192,10,bias=False)
        opt=torch.optim.AdamW([*net.parameters(),*head.parameters()],lr=.001)
        tf=torch.zeros(len(u),192)
        tf[fitix]=learned if method=="dense_learned" else initial
        for step in range(400):
            ww=rng.choice(groups.shape[1],4,replace=False)
            batch=groups[:,ww,:]
            idx=np.take_along_axis(batch,rng.integers(8,size=(10,4,1)),axis=2).ravel()
            z,_=net(idx)
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y[idx])
            if method!="ACC":
                same=y[idx,None]==y[idx][None,:]
                word=torch.tensor(w[idx])
                positive=same&(word[:,None]!=word[None,:])
                negative=~same&(word[:,None]==word[None,:])
                error=(z@z.T-tf[idx]@tf[idx].T).square()
                kd=.5*(error[positive].mean()+error[negative].mean())
                shared=list(net.body.parameters())
                gi=torch.autograd.grad(loss,shared,retain_graph=True)
                gk=torch.autograd.grad(kd,shared,retain_graph=True)
                norm=lambda g:float(torch.sqrt(sum(v.square().sum() for v in g)))
                weight=min(10.,.3*norm(gi)/max(norm(gk),1e-8))
                loss=loss+weight*kd
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_([*net.parameters(),*head.parameters()],1);opt.step()
        with torch.no_grad():e=net.eval()(np.arange(len(u)))[0].numpy()
        for owner in range(1,11):
            yy=u==owner
            svm=SVC(C=10,class_weight="balanced").fit(e[reg],yy[reg])
            f,t,h=roc_curve(yy[cal],svm.decision_function(e[cal]),drop_intermediate=False)
            tau=h[min(range(len(h)),key=lambda i:(abs(f[i]+t[i]-1),f[i]))]
            s=svm.decision_function(e[query]);f,t,_=roc_curve(yy[query],s,drop_intermediate=False)
            j=np.argmin(abs(f+t-1));far=np.mean(s[~yy[query]]>=tau);frr=np.mean(s[yy[query]]<tau)
            rows.append(dict(seed=seed,method=method,user=owner,eer=(f[j]+1-t[j])/2,
                             hter=(far+frr)/2,far=far,frr=frr))
        torch.save(net.state_dict(),out/f"{method}_{seed}_private.pt")
        (out/"results_public.json").write_text(json.dumps(rows,indent=2))
        rr=[v for v in rows if v["seed"]==seed and v["method"]==method and v["user"]!=10]
        print("AUTH",seed,method,[round(100*np.mean([v[k] for v in rr]),2) for k in ("eer","hter","far","frr")],flush=True)

