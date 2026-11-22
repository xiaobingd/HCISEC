"""Training-only diagnosis of the completed temporal auxiliary pilot."""
from pathlib import Path
exec(Path("/root/train_temporal_audio.py").read_text().split("rows=[]")[0])
original=out
diagnostic=[]
print("TEACHER_COVERAGE",int(available.sum()),int(fit.sum()),flush=True)
for seed in (42,43,44):
    rng=np.random.default_rng(seed)
    net=Student(True)
    net.load_state_dict(torch.load(original/f"C_audio_temporal_{seed}_private.pt",map_location="cpu",weights_only=True))
    net.eval()
    trainix=np.flatnonzero(fit)
    with torch.no_grad():
        z,pred=net(trainix)
        pred=F.adaptive_avg_pool1d(pred,7)
        target=F.adaptive_avg_pool1d(targets[trainix],7)
        mse=float(F.mse_loss(pred,target))
        zero=float(target.square().mean())
        # Same person and same word, another repetition.
        shuffled=np.array([rng.choice(np.flatnonzero(fit&(u==u[i])&(w==w[i])&(r!=r[i]))) for i in trainix])
        wrong=F.adaptive_avg_pool1d(targets[shuffled],7)
        wrong_mse=float(F.mse_loss(pred,wrong))
        centers=F.normalize(torch.stack([z[torch.tensor(u[trainix]==owner)].mean(0) for owner in range(1,11)]),dim=1)
    measures=[]
    shared=list(net.body.parameters())
    for step in range(20):
        words=rng.choice(groups.shape[1],4,replace=False)
        batch=groups[:,words,:]
        idx=np.take_along_axis(batch,rng.integers(8,size=(10,4,1)),axis=2).ravel()
        z,p=net(idx)
        identity=F.cross_entropy(16*F.linear(z,centers),y[idx])
        aux=F.smooth_l1_loss(F.adaptive_avg_pool1d(p,7),F.adaptive_avg_pool1d(targets[idx],7))
        gi=torch.autograd.grad(identity,shared,retain_graph=True)
        ga=torch.autograd.grad(aux,shared)
        a=torch.cat([g.ravel() for g in gi]);b=torch.cat([g.ravel() for g in ga])
        measures.append([float(F.cosine_similarity(a[None],b[None])),float(.1*b.norm()/a.norm().clamp_min(1e-8)),int(available[idx].sum())])
    measures=np.array(measures)
    result=dict(seed=seed,training_mse=mse,zero_mse=zero,same_user_same_word_shuffled_mse=wrong_mse,
                training_r2=1-mse/zero,gradient_cosine=float(measures[:,0].mean()),
                negative_gradient_fraction=float((measures[:,0]<0).mean()),
                weighted_aux_identity_norm_ratio=float(measures[:,1].mean()),
                mean_teacher_samples_per_batch=float(measures[:,2].mean()),
                teacher_enabled_fraction=float((measures[:,2]>=3).mean()))
    diagnostic.append(result)
    print(json.dumps(result),flush=True)
(original/"diagnostic_public.json").write_text(json.dumps(diagnostic,indent=2))

