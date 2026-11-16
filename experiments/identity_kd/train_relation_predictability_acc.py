"""Fresh ACC-only encoder: words0..11, fixed1400 steps, no validation selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crossword_representation_losses import IdentityEncoder

OUT=Path('outputs/mei_relation_predictability_20261117')
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main(a):
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True,parents=True)
    d=np.load('outputs/mei_common_frontend_private.npz',allow_pickle=False)
    u,w,ids=d['user'],d['word'],d['sample_id'];fit=np.flatnonzero((w>=0)&(w<12));val=np.flatnonzero((w>=12)&(w<18))
    assert not set(fit)&set(val) and len(set(ids))==len(ids)
    spectrum=d['spectrum'];mean=spectrum[fit].mean((0,3),keepdims=True);std=spectrum[fit].std((0,3),keepdims=True).clip(1e-4)
    # Only fit+validation ACC loaded into model inputs. No Audio cache opened.
    used=np.r_[fit,val];x=torch.tensor((spectrum[used]-mean)/std)
    local={int(i):j for j,i in enumerate(used)}
    lookup={(user,word):np.array([local[int(i)] for i in np.flatnonzero((u==user)&(w==word))]) for user in range(1,10) for word in range(12)}
    assert all(len(v)>=8 for v in lookup.values())
    protocol=dict(state='training',training_words=list(range(12)),independent_validation_words=list(range(12,18)),
        users=list(range(1,10)),seed=42,steps=a.steps,checkpoint='fixed final step;no validation selection',
        initialization='fresh random weights;no old E0 or other pretrained ACC model',
        architecture='existing IdentityEncoder CNN32/64/128,temporal mean+std256,Linear128,L2',
        loss='cosine CE scale16 only;no Audio/SupCon/KD',optimizer='AdamW lr.0003 wd.001 clip1',
        batch='9users x2shared trainingwords x2repeats=36',normalization='fit0..11 only,peraxis-frequency mean/std',
        audio_in_training=False,calibration_or_final18_29_used=False,
        predictor='predeclared Ridge alpha10;no identity or word one-hot;fit only0..11 Audio targets',
        controls=['ACC raw relation','ACC identity-prior relation','Audio fixed dyad-prior memory',
                  'Ridge Audio supervision','task-residual Ridge','held-out dyad supervision','held-out endpoint supervision',
                  'command10/11-only predictor','all size5/7/9 user cohorts','99 training identity permutations'],
        user_scope='known users,unseen words;endpoint prediction holdout does not make encoder unseen-user',
        limits=['nominal word labels','same session','single encoder seed','historically explored cohort'])
    dump(OUT/'protocol_public.json',protocol)
    torch.manual_seed(42);rng=np.random.RandomState(1042)
    net=IdentityEncoder();head=nn.Linear(128,9,bias=False)
    before={k:v.clone() for k,v in net.state_dict().items()}
    params=list(net.parameters())+list(head.parameters())
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-3)
    history=[];started=time.time()
    for step in range(1,a.steps+1):
        words=rng.choice(12,2,replace=False)
        ii=np.concatenate([rng.choice(lookup[user,word],2,replace=False) for user in range(1,10) for word in words])
        assert np.all(w[used[ii]]<12)
        z=net(x[ii]);y=torch.tensor(u[used[ii]]-1,dtype=torch.long)
        loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)
        opt.zero_grad();loss.backward();grad=float(torch.linalg.vector_norm(net.backbone.network[0].weight.grad))
        if step==1:assert grad>0
        nn.utils.clip_grad_norm_(params,1);opt.step()
        if step==1 or step%100==0:
            history.append(dict(step=step,ce=float(loss.detach()),first_conv_gradient_norm=grad,seconds=time.time()-started))
            dump(OUT/'training_history_public.json',history);dump(OUT/'status.json',dict(stage='fresh_ACC_training',step=step,total=a.steps))
            print('ACC_TRAIN',step,'CE',round(float(loss.detach()),4),flush=True)
    changed=max(float((v-before[k]).abs().max()) for k,v in net.state_dict().items());assert changed>0
    torch.save(dict(encoder=net.state_dict(),training_indices=fit,seed=42,steps=a.steps),OUT/'acc_fresh_private.pt')
    net.eval()
    with torch.no_grad():embedding=np.concatenate([net(x[j:j+64]).numpy() for j in range(0,len(x),64)])
    np.savez_compressed(OUT/'acc_embeddings_private.npz',embedding=embedding,indices=used,user=u[used],word=w[used],sample_id=ids[used])
    np.savez_compressed(OUT/'normalization_private.npz',mean=mean,std=std,fit_indices=fit)
    protocol.update(state='encoder_complete',fit_records=len(fit),validation_records=len(val),encoder_max_change=changed,
                    checkpoint_sha256=hashlib.sha256((OUT/'acc_fresh_private.pt').read_bytes()).hexdigest())
    dump(OUT/'protocol_public.json',protocol);dump(OUT/'status.json',dict(stage='encoder_complete'))
    print('FRESH_ENCODER_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1400);main(p.parse_args())
