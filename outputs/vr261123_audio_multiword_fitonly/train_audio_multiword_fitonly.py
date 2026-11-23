"""Fit-word-only ECAPA teacher -> matched ACC students -> multiword authentication."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff, measure

ROOT=Path('/root')
OUT=ROOT/'full_run_10x40_raw_acc/audio_multiword_fitonly'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def supcon(z,y):
    sim=z@z.T/.1
    diagonal=torch.eye(len(z),dtype=torch.bool)
    pos=(y[:,None]==y[None,:])&~diagonal
    logp=sim-torch.logsumexp(sim.masked_fill(diagonal,-torch.inf),dim=1,keepdim=True)
    return -(logp*pos).sum(1).div(pos.sum(1)).mean()

def relation(z,t,y,w):
    positive=(y[:,None]==y[None,:])&(w[:,None]!=w[None,:])
    negative=(y[:,None]!=y[None,:])&(w[:,None]==w[None,:])
    error=(z@z.T-t.detach()@t.detach().T).square()
    assert positive.any() and negative.any()
    return .5*(error[positive].mean()+error[negative].mean())

class Teacher(nn.Module):
    def __init__(self,projection):
        super().__init__()
        self.projection=nn.Conv1d(6144,192,1)
        self.projection.load_state_dict({k.replace('conv.','',1):v for k,v in projection.items()})
        self.branch=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,192))
        nn.init.zeros_(self.branch[-1].weight);nn.init.zeros_(self.branch[-1].bias)
    def forward(self,pool):
        z=self.projection(pool).squeeze(-1)
        return F.normalize(z+self.branch(z),dim=1)

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    d=np.load(ROOT/'paired_raw_acc_private.npz')
    st=np.load(ROOT/'stats27_private.npz')
    assert np.array_equal(d['sample_id'],st['sample_id'])
    ids=d['sample_id'];u,w,r=np.array([s.split('_') for s in ids],dtype=int).T
    assert len(set(zip(u,w,r)))==len(ids)==3200
    fit=d['role']=='fit';fi=np.flatnonzero(fit)
    rw=np.array([20,21,22,23]);cw=np.array([8,9,18,19,24]);qw=np.array([25,26,27,28,29])
    assert (fit.astype(int)+np.isin(w,rw)+np.isin(w,cw)+np.isin(w,qw)).max()==1
    assert len(fi)==2080 and len(set(w[fi]))==26
    spec=d['spectrum'].reshape(len(u),120,-1)
    features=np.column_stack([spec.mean(-1),spec.std(-1),
        *(v.mean(-1) for v in np.array_split(spec,4,axis=-1)),d['amplitude'],st['features']])
    basepath=ROOT/'adapted_acc_private.pt'
    ck=torch.load(basepath,map_location='cpu',weights_only=True)
    x=torch.tensor((features-ck['mean'].numpy())/ck['scale'].numpy(),dtype=torch.float32)
    y=torch.tensor(u-1,dtype=torch.long)
    fitwords=np.unique(w[fi])
    lookup={(us,int(word)):np.flatnonzero(fit&(u==us)&(w==word)) for us in range(1,11) for word in fitwords}
    assert all(len(v)==8 for v in lookup.values())
    protocol=dict(dataset='same 10 users; hardware raw ACC; closed cross-word; all9 attackers retained',
        train_words=fitwords.tolist(),train_recordings=2080,registration_words=rw.tolist(),
        calibration_words=cw.tolist(),query_words=qw.tolist(),registration_per_user=24,
        primary='3 words, expanded_all8 query, quantile threshold; compare matched CE and AudioKD',
        secondary='1/5 words and additional_last4 recordings; no query-based choice of length',
        encoder='762->256->192 same initial pure-ACC700-step checkpoint',initial_sha256=sha(basepath),
        teacher='official hash-verified VoxCeleb ECAPA, fresh initialization; frozen backbone; fit-only projection+192->64->192 residual adaptation300steps seed4242',
        teacher_coverage='all2080 fit Audio only; no registration/calibration/query Audio forwarded or used for adaptation',
        teacher_optimizer='projection1e-5/residual1e-4/classifier1e-3 AdamW, CE+.1SupCon+.01anchor, batch40',
        methods=['CE','AudioKD','ShuffledAudioKD'],seeds=[42,43,44],student_steps=400,
        student_batch='same as previous CE control:10 users x(3 support words x2 +2 query words x2)=100',
        student_loss='CE; Audio arms add same-user-cross-word/different-user-same-word relation MSE; detached weight min(10,.1*CE encoder grad norm/KD encoder grad norm)',
        negative_control='fixed permutation across fit Audio records, independent RNG; preserves target marginal distribution, breaks input identity pairing',
        student_optimizer='same AdamW lr1e-4 weight_decay.01,clip1; all trainable student parameters matched',
        selection='fixed teacher300/student400; no model or weight search, fixed prior quantile rule q10/q95 midpoint',
        authentication='ACC only, RBF C10 balanced, mean per-record scores across1/3/5distinct words; all thresholds locked before query metrics',
        grouping='fixed seed1123, no source-ID reuse within evaluation set; 40available ->40/39/40used for1/3/5words',
        scope='historically explored vocabulary/cohort; not new blind collection or open-set evidence',
        exclusions='no old locally adapted teacher caches, no CycleGAN, no quality filtering or user removal')
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    # Fresh official backbone. Old local teachers are excluded because historical registration included current calibration words.
    original_model=ROOT/'full_run_10x40_raw_acc/model'
    expected='0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2'
    assert sha(original_model/'embedding_model.ckpt')==expected
    token=hashlib.sha256(ids[fi].tobytes()).hexdigest()
    progresspath=OUT/'extraction_progress_private.json'
    completed=0
    if progresspath.exists():
        progress=json.loads(progresspath.read_text())
        assert progress['sample_id_digest']==token and progress['official_model_sha256']==expected
        completed=progress['completed']
    mode='r+' if completed else 'w+'
    poolmap=np.lib.format.open_memmap(OUT/'official_pool_private.npy',mode=mode,dtype='float32',shape=(2080,6144,1))
    embmap=np.lib.format.open_memmap(OUT/'official_embedding_private.npy',mode=mode,dtype='float32',shape=(2080,192))
    if completed<len(fi):
        import run_ecapa_pilot as pilot
        pilot.p.DEVICE='cpu'
        model=pilot.teacher_load(ROOT/'full_run_10x40_raw_acc')
        emb=model.mods.embedding_model.eval()
        torch.save(emb.fc.state_dict(),OUT/'official_projection_private.pt')
        captured=[]
        hook=emb.fc.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
        waves=d['wave'][fi].astype('float32',copy=False)
        starttime=time.monotonic();startcount=completed
        with torch.inference_mode():
            for start in range(completed,len(fi),8):
                stop=min(start+8,len(fi));wave=torch.tensor(waves[start:stop])
                wave=wave-wave.mean(1,keepdim=True)
                lengths=torch.ones(len(wave))
                f=model.mods.mean_var_norm(model.mods.compute_features(wave),lengths)
                z=F.normalize(emb(f,lengths).squeeze(1),dim=1)
                assert len(captured)==1 and captured[0].shape==(len(wave),6144,1)
                poolmap[start:stop]=captured.pop().numpy();embmap[start:stop]=z.numpy()
                if stop%80==0 or stop==len(fi):
                    poolmap.flush();embmap.flush()
                    progresspath.write_text(json.dumps(dict(completed=stop,sample_id_digest=token,official_model_sha256=expected)))
                    print('AUDIO_EXTRACT',stop,len(fi),'seconds',round(time.monotonic()-starttime,1),'new',stop-startcount,flush=True)
        hook.remove();del model,emb,waves
    pool=torch.tensor(np.array(poolmap));initial=torch.tensor(np.array(embmap))
    projection=torch.load(OUT/'official_projection_private.pt',map_location='cpu',weights_only=True)
    torch.manual_seed(4242);rng=np.random.default_rng(4242)
    teacher=Teacher(projection)
    teacherpath=OUT/'fitonly_teacher_private.pt'
    tg=np.stack([[np.flatnonzero((u[fi]==us)&(w[fi]==word)) for word in fitwords] for us in range(1,11)])
    if teacherpath.exists():teacher.load_state_dict(torch.load(teacherpath,map_location='cpu',weights_only=True))
    else:
        head=nn.Linear(192,10,bias=False)
        opt=torch.optim.AdamW([dict(params=teacher.projection.parameters(),lr=1e-5),
            dict(params=teacher.branch.parameters(),lr=1e-4),dict(params=head.parameters(),lr=1e-3)])
        for step in range(300):
            sub=tg[:,rng.choice(len(fitwords),4,replace=False)]
            j=np.take_along_axis(sub,rng.integers(8,size=(10,4,1)),axis=2).ravel()
            z=teacher(pool[j]);yy=y[fi[j]]
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),yy)+.1*supcon(z,yy)+.01*(1-(z*initial[j]).sum(1)).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([*teacher.parameters(),*head.parameters()],1);opt.step()
            if (step+1)%100==0:print('TEACHER_FIT',step+1,float(loss.detach()),flush=True)
        torch.save(teacher.state_dict(),teacherpath)
    teacher.eval()
    for param in teacher.parameters():param.requires_grad_(False)
    with torch.inference_mode():targets=torch.cat([teacher(pool[i:i+64]) for i in range(0,len(fi),64)])
    assert not targets.requires_grad
    np.savez_compressed(OUT/'teacher_targets_private.npz',indices=fi,sample_id=ids[fi],targets=targets.numpy())
    tf=torch.zeros(len(u),192);tf[fi]=targets
    shuffled=torch.zeros_like(tf)
    permutation=np.random.default_rng(14242).permutation(len(fi))
    shuffled[fi]=targets[permutation]
    del pool,initial,teacher,poolmap,embmap
    def encoder():
        net=nn.Sequential(nn.Linear(features.shape[1],256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
        net.load_state_dict(ck['encoder']);return net
    def episode(rng):
        chosen=rng.choice(fitwords,5,replace=False)
        support=np.array([np.concatenate([rng.choice(lookup[us,int(word)],2,replace=False) for word in chosen[:3]]) for us in range(1,11)])
        query=np.array([np.concatenate([rng.choice(lookup[us,int(word)],2,replace=False) for word in chosen[3:]]) for us in range(1,11)])
        assert not np.intersect1d(support,query).size
        return np.concatenate([support,query],axis=1).ravel()
    embeddings={};training=[];replay=[]
    for seed in (42,43,44):
        for method in protocol['methods']:
            path=OUT/f'{method}_{seed}_private.pt'
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            net=encoder();head=nn.Linear(192,10,bias=False)
            with torch.no_grad():
                z=F.normalize(net(x),dim=1)
                head.weight.copy_(F.normalize(torch.stack([z[fit&(u==us)].mean(0) for us in range(1,11)]),dim=1))
            params=list(net.parameters());opt=torch.optim.AdamW([*params,*head.parameters()],lr=1e-4)
            trace=[]
            if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            else:
                for step in range(400):
                    ix=episode(rng);z=F.normalize(net(x[ix]),dim=1);yy=y[ix]
                    ce=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),yy)
                    weight=0.;kd=z.new_zeros(());ratio=0.
                    if method!='CE':
                        target=(tf if method=='AudioKD' else shuffled)[ix]
                        kd=relation(z,target,yy,torch.tensor(w[ix]))
                        gi=torch.autograd.grad(ce,params,retain_graph=True)
                        gk=torch.autograd.grad(kd,params,retain_graph=True)
                        ni=float(torch.sqrt(sum(v.square().sum() for v in gi)))
                        nk=float(torch.sqrt(sum(v.square().sum() for v in gk)))
                        weight=min(10.,.1*ni/max(nk,1e-8));ratio=weight*nk/max(ni,1e-8)
                    loss=ce+weight*kd
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([*params,*head.parameters()],1);opt.step()
                    if (step+1)%100==0:
                        trace.append(dict(step=step+1,ce=float(ce.detach()),kd=float(kd.detach()),weight=weight,aux_gradient_ratio=ratio))
                        print('STUDENT',seed,method,step+1,float(loss.detach()),flush=True)
                torch.save(net.state_dict(),path)
            if method=='CE':
                old=torch.load(ROOT/'full_run_10x40_raw_acc/strict_episodic_6x4'/f'CE_{seed}_private.pt',map_location='cpu',weights_only=True)['encoder']
                err=max(float((v-old[k]).abs().max()) for k,v in net.state_dict().items());replay.append(dict(seed=seed,parameter_max_error=err))
                assert err<1e-6,('CE_REPLAY_MISMATCH',seed,err)
            with torch.inference_mode():embeddings[seed,method]=F.normalize(net.eval()(x),dim=1).numpy()
            training.append(dict(seed=seed,method=method,trace=trace,steps=400))
            (OUT/'training_public.json').write_text(json.dumps(training,indent=2))
    order=np.random.default_rng(1123).permutation(5)
    index={(int(us),int(word),int(rep)):i for i,(us,word,rep) in enumerate(zip(u,w,r))}
    def trials(words,repeats,k):
        base=[(int(word),int(rep)) for rep in repeats for word in words[order]]
        groups=[base[i:i+k] for i in range(0,len(base)-k+1,k)]
        first=np.array([[index[us,word,rep] for word,rep in group] for us in range(1,11) for group in groups])
        identity=np.repeat(np.arange(1,11),len(groups))
        assert len(np.unique(first))==first.size
        assert all(len(set(w[v]))==k for v in first)
        return first,identity
    reg=np.isin(w,rw)&(r<6);locks=[];scorecache={}
    with threadpool_limits(limits=2):
        for (seed,method),view in embeddings.items():
            scores=np.column_stack([SVC(C=10,class_weight='balanced').fit(view[reg],u[reg]==us).decision_function(view) for us in range(1,11)])
            scorecache[seed,method]=scores
            for k in (1,3,5):
                ci,cu=trials(cw,range(8),k);cs=scores[ci].mean(1)
                for us in range(1,11):
                    tau=cutoff(cs[:,us-1],cu==us,'quantile_gap')
                    locks.append(dict(seed=seed,method=method,words=k,user=us,threshold=tau,
                        development=measure(cs[:,us-1],cu==us,tau)))
    (OUT/'thresholds_public.json').write_text(json.dumps(locks,indent=2,allow_nan=False))
    print('AUTHENTICATION_LOCKED',flush=True)
    rows=[]
    for lock in locks:
        scores=scorecache[lock['seed'],lock['method']]
        for name,reps in [('expanded_all8',range(8)),('additional_last4',range(4,8))]:
            qi,qu=trials(qw,reps,lock['words']);qs=scores[qi].mean(1)[:,lock['user']-1]
            rows.append(dict(**{k:lock[k] for k in ('seed','method','words','user')},query=name,
                **measure(qs,qu==lock['user'],lock['threshold'])))
    previous=json.loads((ROOT/'full_run_10x40_raw_acc/threeword_expanded/results_public.json').read_text());errors=[]
    for row in rows:
        if row['method']=='CE' and row['words']==3:
            prior=next(v for v in previous if v['seed']==row['seed'] and v['user']==row['user'] and v['registration']==24
                and v['calibration']==40 and v['rule']=='quantile_gap' and v['query']==row['query'])
            errors += [abs(row[k]-prior[k]) for k in ('far','frr','hter','eer','auc')]
    assert max(errors)<1e-10 and len(rows)==540 and len(locks)==270
    (OUT/'results_public.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (OUT/'verification_public.json').write_text(json.dumps(dict(status='PASS',teacher_fit_records=2080,
        teacher_nonfit_audio_forwarded=0,official_teacher_sha256=expected,ce_checkpoint_replay=replay,
        ce_threeword_metric_max_error=max(errors),query_workpoints=len(rows),threshold_count=len(locks),
        all_query_thresholds_locked=True,source_id_reuse=False,all_users_retained=True),indent=2))
    print('AUDIO_MULTIWORD_COMPLETE',flush=True)

if __name__=='__main__':main()

