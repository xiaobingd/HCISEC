"""Train-only Audio-guided enrollment-query margins; paired controls."""
from pathlib import Path
exec(Path("/root/train_dense_learnable_teacher.py").read_text().split("rows=[];audits=[];logs=[]")[0])
out=root/"full_run_10x40_raw_acc"/"coupled_authentication"
out.mkdir(exist_ok=True)
teacher_folder=root/"full_run_10x40_raw_acc"/"dense_learnable_teacher"
rows=[];coverage=[]
for seed in (42,43,44):
    result_folder=out
    out=teacher_folder
    teacher_net=AudioTeacher()
    out=result_folder
    teacher_net.load_state_dict(torch.load(teacher_folder/f"teacher_{seed}_private.pt",map_location="cpu",weights_only=True))
    with torch.no_grad():
        tf=torch.zeros(len(u),192)
        tf[fitix]=torch.cat([teacher_net(np.arange(i,min(i+64,len(fitix)))) for i in range(0,len(fitix),64)])
    for method in ("CE","episode","coupled_margin","wrong_negative_relation"):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        net=Student(True);head=nn.Linear(192,10,bias=False)
        opt=torch.optim.AdamW([*net.parameters(),*head.parameters()],lr=.001)
        logs=[]
        for step in range(400):
            ww=rng.choice(groups.shape[1],4,replace=False)
            batch=groups[:,ww,:]
            idx=np.take_along_axis(batch,rng.integers(8,size=(10,4,1)),axis=2).ravel()
            z,_=net(idx)
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y[idx])
            if method!="CE":
                episode=z.reshape(10,4,192)
                center=F.normalize(episode[:,:2].mean(1),dim=1)
                query_z=episode[:,2:].reshape(20,192)
                query_labels=torch.arange(10).repeat_interleave(2)
                score=query_z@center.T
                loss=loss+.1*F.cross_entropy(16*score,query_labels)
                if method!="episode":
                    a=tf[idx].reshape(10,4,192)
                    ac=F.normalize(a[:,:2].mean(1),dim=1)
                    ts=a[:,2:].reshape(20,192)@ac.T
                    negs=torch.stack([torch.tensor([k for k in range(10) if k!=int(owner)]) for owner in query_labels])
                    selected=score.detach().gather(1,negs).topk(2,dim=1).indices
                    hard=negs.gather(1,selected)
                    target_score=ts.gather(1,query_labels[:,None])
                    tm=target_score-ts.gather(1,hard)
                    if method=="wrong_negative_relation":
                        wrong=negs.gather(1,(selected+1)%9)
                        tm=target_score-ts.gather(1,wrong)
                    student_margin=score.gather(1,query_labels[:,None])-score.gather(1,hard)
                    reliable=tm>0
                    target_margin=.05+.1*tm.clamp(0,2)
                    kd=(F.relu(target_margin-student_margin)*reliable).sum()/reliable.sum().clamp_min(1)
                    shared=list(net.body.parameters())
                    gi=torch.autograd.grad(loss,shared,retain_graph=True)
                    gk=torch.autograd.grad(kd,shared,retain_graph=True)
                    norm=lambda g:float(torch.sqrt(sum(v.square().sum() for v in g)))
                    weight=min(10.,.1*norm(gi)/max(norm(gk),1e-8))
                    logs.append([float(reliable.float().mean()),weight])
                    loss=loss+weight*kd
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_([*net.parameters(),*head.parameters()],1);opt.step()
        with torch.no_grad():e=net.eval()(np.arange(len(u)))[0].numpy()
        for owner in range(1,11):
            yy=u==owner
            svm=SVC(C=10,class_weight="balanced").fit(e[reg],yy[reg])
            center=e[reg&yy].mean(0);center/=np.linalg.norm(center)
            scores=dict(rbf=svm.decision_function(e),cosine=e@center)
            for backend,s in scores.items():
                f,t,h=roc_curve(yy[cal],s[cal],drop_intermediate=False)
                tau=h[min(range(len(h)),key=lambda i:(abs(f[i]+t[i]-1),f[i]))]
                f,t,_=roc_curve(yy[query],s[query],drop_intermediate=False)
                j=np.argmin(abs(f+t-1));far=np.mean(s[query&~yy]>=tau);frr=np.mean(s[query&yy]<tau)
                rows.append(dict(seed=seed,method=method,backend=backend,user=owner,
                                 eer=(f[j]+1-t[j])/2,hter=(far+frr)/2,far=far,frr=frr))
        if logs:coverage.append(dict(seed=seed,method=method,mean_reliable_and_weight=np.mean(logs,0).tolist()))
        torch.save(net.state_dict(),out/f"{method}_{seed}_private.pt")
        (out/"results_public.json").write_text(json.dumps(rows,indent=2))
        (out/"coverage_public.json").write_text(json.dumps(coverage,indent=2))
        for backend in ("rbf","cosine"):
            rr=[v for v in rows if v["seed"]==seed and v["method"]==method and v["backend"]==backend and v["user"]!=10]
            print(seed,method,backend,[round(100*np.mean([v[k] for v in rr]),2) for k in ("eer","hter","far","frr")],flush=True)

