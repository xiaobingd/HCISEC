"""Exploratory frozen-E4 residual experiment; Audio is used on fit words only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.svm import SVC
from evaluate_crossword_encoder_distill import metric,threshold,norm,KEYS

ROOT=Path('outputs/mei_audio_margin_residual_20261117')
USERS=[1,2,3,4,6,7,8,9]
METHODS=['uniform','acc_weight','audio_weight','wrong_audio_weight']

def main():
    torch.set_num_threads(2); ROOT.mkdir(parents=True,exist_ok=True)
    d=np.load('outputs/mei_crossword_representation_20261116/vectors_private.npz')
    a=np.load('outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz')
    assert np.array_equal(d['sample_id'],a['sample_id'])
    u,w=d['user'],d['word'];fit=np.flatnonzero(w<18)
    assert np.array_equal(fit,a['indices']) and int(a['done'])==len(fit)
    assert not a['embedding'][w>=18].any()
    base=d['E4'].astype('float32'); audio=norm(a['embedding'].astype('float32'))
    lookup={(i,j):np.flatnonzero((u==i)&(w==j)) for i in range(1,10) for j in range(30)}
    assert all(len(lookup[i,j])>=8 for i in range(1,10) for j in range(30))
    raw=d['raw'].astype('float32')
    alpha=json.loads(Path('outputs/mei_crossword_representation_20261116/fusion_selection_public.json').read_text(encoding='utf-8'))['methods']['E4']['selected']['alpha']
    # All weights are functions of fit-word recordings. Use two disjoint repeats for reliability.
    def centers(z,part):
        return norm(np.stack([norm(z[np.concatenate([lookup[i,j][part] for j in range(18)])].mean(0)) for i in range(1,10)]))
    acc_a,acc_b=centers(base,slice(0,4)),centers(base,slice(4,8))
    aud_a,aud_b=centers(audio,slice(0,4)),centers(audio,slice(4,8))
    def reliability(c1,c2):
        m=c1@c2.T
        gap=np.diag(m)[:,None]-.5*(c1@c1.T+c2@c2.T)
        stable=np.abs((c1@c1.T)-(c2@c2.T))
        return np.clip(gap,0,None)*np.exp(-5*stable)
    wa=reliability(acc_a,acc_b);wt=reliability(aud_a,aud_b)
    # Audio sees only training words; wrong control preserves the same weight values.
    rng=np.random.RandomState(20261117);perm=rng.permutation(9)
    wrong=wt[:,perm].copy();wrong[np.arange(9),np.arange(9)]=0
    weight={'uniform':np.ones((9,9)), 'acc_weight':wa,'audio_weight':wt,'wrong_audio_weight':wrong}
    for x in weight.values():
        np.fill_diagonal(x,0)
        x*=72/max(x.sum(),1e-8)
    torch.manual_seed(42)
    init=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,128))
    for method in METHODS:
        torch.manual_seed(42)
        net=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,128))
        net.load_state_dict(init.state_dict())
        opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3)
        gen=np.random.RandomState(42)
        bw=torch.tensor(weight[method],dtype=torch.float32)
        for step in range(400):
            words=gen.choice(18,2,replace=False)
            ix=np.stack([[gen.choice(lookup[i,j],2,replace=False) for j in words] for i in range(1,10)])
            reg=torch.tensor(base[ix[:,0]].mean(1))
            genuine=torch.tensor(base[ix[:,1]].mean(1))
            def adjust(t):return F.normalize(t+.1*net(t),dim=-1)
            rr,qq=adjust(reg),adjust(genuine)
            sim=qq@rr.T
            margin=(sim.diag()[:,None]-sim)
            loss=(F.relu(.2-margin)*bw).sum()/bw.sum()
            # Anchor keeps the residual small and avoids changing easy geometry.
            anchor=(1-(rr*F.normalize(reg,dim=-1)).sum(-1)).mean()
            total=loss+.2*anchor
            opt.zero_grad();total.backward();opt.step()
        with torch.no_grad():
            z=np.concatenate([F.normalize(torch.tensor(base[k:k+512])+.1*net(torch.tensor(base[k:k+512])),dim=-1).numpy() for k in range(0,len(u),512)])
        np.save(ROOT/(method+'_embedding_private.npy'),z)
    evaluate(u,w,raw,base,alpha)

def evaluate(u,w,raw,base,alpha):
    z={'frozen_E4':base,**{m:np.load(ROOT/(m+'_embedding_private.npy')) for m in METHODS}}
    views={m:np.c_[raw,alpha*v]/np.sqrt(1+alpha**2) for m,v in z.items()}
    lookup={(i,j):np.flatnonzero((u==i)&(w==j)) for i in USERS for j in range(21,30)}
    def reg(i,j,f):return lookup[i,j][f*4:f*4+4]
    def query(j,f):return np.concatenate([lookup[i,j][(1-f)*4:(1-f)*4+4] for i in USERS])
    rows=[]
    for name,v in views.items():
        for owner in USERS:
            models={}
            def score(rw,qw,f):
                if (rw,f) not in models:
                    ix=np.concatenate([reg(i,rw,f) for i in USERS])
                    models[rw,f]=SVC(C=10,gamma='scale',class_weight='balanced').fit(v[ix],u[ix]==owner)
                return models[rw,f].decision_function(v[query(qw,f)])
            ycal=[];scal=[]
            for qw in (21,22,23):
                yy=u[query(qw,0)]==owner
                for rw in (21,22,23):
                    if qw!=rw:
                        ycal.extend([yy,yy]);scal.extend([score(qw,qw,0),score(rw,qw,0)])
            tau=threshold(np.concatenate(ycal),np.concatenate(scal))
            for f in (0,1):
                for qw in range(24,30):
                    yy=u[query(qw,f)]==owner
                    for rw in range(24,30):
                        if rw==qw:continue
                        for cond,rword in [('sameword',qw),('crossword',rw)]:
                            rows.append(dict(method=name,user=f'U{owner}',condition=cond,**metric(yy,score(rword,qw,f),tau)))
    def aggregate(rr):return {k:float(np.mean([x[k] for x in rr])) for k in KEYS}
    summary=[];peruser=[]
    for m in views:
        for c in ('sameword','crossword'):
            ss=[x for x in rows if x['method']==m and x['condition']==c]
            summary.append(dict(method=m,condition=c,**aggregate(ss)))
            for i in USERS:peruser.append(dict(method=m,condition=c,user=f'U{i}',**aggregate([x for x in ss if x['user']==f'U{i}'])))
    (ROOT/'results_public.json').write_text(json.dumps(dict(summary=summary,per_user=peruser),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
