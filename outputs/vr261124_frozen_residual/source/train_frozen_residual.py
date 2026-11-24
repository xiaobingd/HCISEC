"""Identity-supervised shared/private ACC decomposition; no evaluation Audio."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from train_audio_multiword_fitonly import Teacher, relation

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'phrase_auth'
OUT=ROOT/'frozen_residual'

def encoder():
    return nn.Sequential(nn.Linear(762,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))

class Split(nn.Module):
    def __init__(self,warm):
        super().__init__()
        self.encoder=encoder();self.encoder.load_state_dict(warm)
        for parameter in self.encoder.parameters():parameter.requires_grad_(False)
        self.shared=nn.Sequential(nn.Linear(192,32),nn.GELU(),nn.Linear(32,96))
        self.decoder=nn.Linear(96,192,bias=False)
        nn.init.zeros_(self.decoder.weight)
    def forward(self,x):
        h=F.normalize(self.encoder(x),dim=1).detach()
        s=F.normalize(self.shared(h),dim=1)
        delta=self.decoder(s)
        f=F.normalize(h+0.1*delta,dim=1)
        return h,s,F.normalize(h[:,:96],dim=1),f,h

def decorrelation(s,p):
    a=s-s.mean(0);b=p-p.mean(0)
    a=a/(a.square().sum(0).sqrt().clamp_min(1e-4))
    b=b/(b.square().sum(0).sqrt().clamp_min(1e-4))
    return (a.T@b).square().mean()

def norm(grads):
    return sum(v.square().sum() for v in grads if v is not None).sqrt()

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    d=np.load(BASE/'frontend_private.npz');old=np.load(ROOT/'paired_raw_acc_private.npz')
    assert np.array_equal(d['sample_id'],old['sample_id'])
    u,w=d['user'],d['word'];fit=d['fit'];fi=np.flatnonzero(fit)
    assert len(fi)==1280 and len(set(d['sample_id']))==3200
    words=np.unique(w[fit]);assert len(words)==16
    x=torch.tensor(StandardScaler().fit(d['features'][fit]).transform(d['features']),dtype=torch.float32)
    y=torch.tensor(u-1,dtype=torch.long)
    groups=np.array([[np.flatnonzero(fit&(u==us)&(w==word)) for word in words] for us in range(1,11)])
    assert groups.shape==(10,16,8)
    warm=torch.load(BASE/'fresh_warm_private.pt',map_location='cpu',weights_only=True)
    legacy=np.flatnonzero(old['role']=='fit');take=np.searchsorted(legacy,fi)
    assert np.array_equal(legacy[take],fi)
    official=ROOT/'audio_multiword_fitonly'
    provenance=json.loads((official/'extraction_progress_private.json').read_text())
    assert provenance['sample_id_digest']==hashlib.sha256(old['sample_id'][legacy].tobytes()).hexdigest()
    pool=torch.tensor(np.array(np.load(official/'official_pool_private.npy',mmap_mode='r')[take]))
    teacher=Teacher(torch.load(official/'official_projection_private.pt',map_location='cpu',weights_only=True))
    teacher.load_state_dict(torch.load(BASE/'fresh_fitonly_teacher_private.pt',map_location='cpu',weights_only=True));teacher.eval()
    with torch.inference_mode():target=torch.cat([teacher(pool[i:i+64]) for i in range(0,1280,64)])
    tf=torch.zeros(3200,192);tf[fi]=target
    # A single identity derangement preserves word/repetition matching and genuine groups.
    perm=np.random.default_rng(241124).permutation(10)
    mapping={int(perm[i]+1):int(perm[(i+1)%10]+1) for i in range(10)}
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,d['repeat']))}
    wrong=torch.zeros_like(tf)
    for i in fi:wrong[i]=tf[lookup[mapping[int(u[i])],int(w[i]),int(d['repeat'][i])]]
    del pool,target,teacher
    protocol=dict(train_words=words.tolist(),train_records=1280,users=10,seeds=[42,43,44],steps=400,
        initialization='same fresh16fit-only700step ACC checkpoint and existing fit-only adapted ECAPA; no evaluation Audio',
        methods=['ACC','SplitNoAudio','SplitAudio','SplitWrongAudio'],dimensions='ACC192;shared96;private96;concat192',
        identity='fused CE + .1 shared CE + .1 private CE; normalized cosine classifier scale16',
        auxiliary='0.01 centered standardized cross-correlation squared +0.05 reconstructionMSE of detached normalized ACC latent',
        audio='relationMSE shared branch only; same-user cross-word and different-user same-word equally weighted;encoder gradient norm budget .1 of identity loss, coefficient cap10',
        negative_control='fixed identity derangement across all fit records; preserves word and repeat correspondence; no fixed identity',
        optimizer='AdamW1e-4 weight_decay.01,clip1,batch100;10users x5words x2repetitions; same batches per seed/arm',
        scope='closed known users,unseen words; no user removal; all query and calibration words excluded from representation learning',
        evaluation='unchanged all20 role splits per family,1/3record challenges,RBF C10 balanced,q10/q95 midpoint; no selection from query',
        caveat='private means ACC supplementary branch,not proof of exclusive identity information; original ACC has fewer parameters,matched no-Audio split controls added capacity')
    protocol.update(initialization='per-seed completed ACC400 frozen; train only192->32->96->192 residual',identity='fused CE+.1 shared CE; private CE constant for encoder',auxiliary='none',audio='shared relation MSE;trainable residual gradient ratio .1 cap10',dimensions='fused normalize(h+.1 delta),shared96,diagnostic frozen first96',optimizer='frozen ACC;residual only400steps AdamW1e-4;baseline not retrained')
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    logs=[];views={}
    for seed in (42,43,44):
        for method in protocol['methods']:
            warm=torch.load(ROOT/'shared_private'/f'ACC_{seed}_private.pt',map_location='cpu',weights_only=True)
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            net=encoder() if method=='ACC' else Split(warm)
            if method=='ACC':net.load_state_dict(warm)
            heads=nn.ModuleList([nn.Linear(192,10,bias=False)] if method=='ACC' else
                [nn.Linear(192,10,bias=False),nn.Linear(96,10,bias=False),nn.Linear(96,10,bias=False)])
            with torch.no_grad():
                zz=[F.normalize(net(x[fi]),dim=1)] if method=='ACC' else list(net(x[fi])[3:4])+list(net(x[fi])[1:3])
                for head,z in zip(heads,zz):head.weight.copy_(F.normalize(torch.stack([z[u[fi]==us].mean(0) for us in range(1,11)]),dim=1))
            params=[v for v in net.parameters() if v.requires_grad];ep=params
            opt=torch.optim.AdamW([*params,*heads.parameters()],lr=1e-4)
            cp=OUT/f'{method}_{seed}_private.pt';trace=[]
            if method=='ACC':
                torch.save(net.state_dict(),cp)
            elif cp.exists():net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
            else:
                for step in range(400):
                    chosen=rng.choice(16,5,replace=False)
                    ix=np.array([np.concatenate([rng.choice(groups[us,wo],2,replace=False) for wo in chosen]) for us in range(10)]).ravel()
                    assert set(w[ix])<=set(words)
                    if method=='ACC':zz=[F.normalize(net(x[ix]),dim=1)];reg=zz[0].new_zeros(())
                    else:
                        h,s,p,f,recon=net(x[ix]);zz=[f,s,p]
                        reg=h.new_zeros(())
                    losses=[F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y[ix]) for head,z in zip(heads,zz)]
                    ident=losses[0]+(0 if method=='ACC' else .1*(losses[1]+losses[2]))
                    kd=ident.new_zeros(());weight=0.;ratio=0.
                    if method in ('SplitAudio','SplitWrongAudio'):
                        kd=relation(s,(tf if method=='SplitAudio' else wrong)[ix],y[ix],torch.tensor(w[ix]))
                        ni=norm(torch.autograd.grad(ident,ep,retain_graph=True))
                        nk=norm(torch.autograd.grad(kd,ep,retain_graph=True,allow_unused=True))
                        weight=min(10.,.1*float(ni)/max(float(nk),1e-8));ratio=weight*float(nk)/max(float(ni),1e-8)
                        assert nk>0 and torch.isfinite(kd)
                    loss=ident+reg+weight*kd
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([*params,*heads.parameters()],1);opt.step()
                    if (step+1)%100==0:
                        trace.append(dict(step=step+1,identity=float(ident.detach()),regularization=float(reg.detach()),kd=float(kd.detach()),weight=weight,encoder_kd_ratio=ratio))
                        print('SPLIT_TRAIN',seed,method,step+1,flush=True)
                torch.save(net.state_dict(),cp)
            with torch.inference_mode():
                if method=='ACC':views[f'{seed}_ACC']=F.normalize(net.eval()(x),dim=1).numpy()
                else:
                    h,s,p,f,_=net.eval()(x)
                    for name,z in [('fused',f),('shared',s),('private',p)]:views[f'{seed}_{method}_{name}']=z.numpy()
            logs.append(dict(seed=seed,method=method,parameters=sum(v.numel() for v in params),trace=trace))
            (OUT/'training_public.json').write_text(json.dumps(logs,indent=2))
    np.savez_compressed(OUT/'views_private.npz',**views)
    (OUT/'training_verification_public.json').write_text(json.dumps(dict(status='PASS',views=len(views),steps_per_new_arm=400,
        teacher_audio_rows=1280,nonfit_audio_rows=0,all_users_retained=True,training_prompt_overlap=0,
        sample_id_digest=hashlib.sha256(d['sample_id'].tobytes()).hexdigest()),indent=2))
    print('SHARED_PRIVATE_TRAIN_COMPLETE',flush=True)

if __name__=='__main__':main()
