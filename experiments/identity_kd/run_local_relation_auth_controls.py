"""B0-B3, three seeds, fixed ACC authentication experiment, no validation tuning."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from crossword_representation_losses import IdentityEncoder
from local_relation_control_losses import stable_graph,local_relation_kl,combine_gradients

OUT=Path('outputs/mei_local_relation_auth_controls_20261117')
SEEDS=[42,43,44];METHODS=['B0','B1','B2','B3']

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def encode(net,x):
    net.eval()
    with torch.no_grad():return np.concatenate([net(x[j:j+64]).numpy() for j in range(0,len(x),64)])

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True,parents=True)
    d=np.load('outputs/mei_common_frontend_private.npz');t=np.load('outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz')
    user,word,ids=d['user'],d['word'],d['sample_id']
    assert len(set(ids))==len(ids) and np.array_equal(ids,t['sample_id'])
    assert np.array_equal(t['indices'],np.flatnonzero((word>=0)&(word<18))) and int(t['done'])==len(t['indices'])
    cells={(u,w):np.flatnonzero((user==u)&(word==w)) for u in range(1,10) for w in range(18)}
    assert all(len(v)>=8 for v in cells.values())
    fit=np.concatenate([cells[u,w][:6] for u in range(1,10) for w in range(12)])
    cal=np.concatenate([cells[u,w][6:8] for u in range(1,10) for w in range(12)])
    val=np.concatenate([cells[u,w][:8] for u in range(1,10) for w in range(12,18)])
    assert len(fit)==648 and len(cal)==216 and len(val)==432
    assert not set(fit)&set(cal) and not set(fit)&set(val) and not set(cal)&set(val)
    used=np.sort(np.r_[fit,cal,val]);index={int(i):j for j,i in enumerate(used)}
    fi=np.array([index[int(i)] for i in fit]);ci=np.array([index[int(i)] for i in cal]);vi=np.array([index[int(i)] for i in val])
    u,w=user[used],word[used];spec=d['spectrum'];mean=spec[fit].mean((0,3),keepdims=True);std=spec[fit].std((0,3),keepdims=True).clip(1e-4)
    x=torch.tensor((spec[used]-mean)/std)
    teacher=np.zeros((len(used),192),dtype='float32');teacher[fi]=t['embedding'][fit]
    assert np.all(np.linalg.norm(teacher[fi],axis=1)>.99) and not teacher[np.r_[ci,vi]].any()
    audio_graph=stable_graph(teacher[fi].reshape(9,12,6,-1));assert audio_graph.any()
    lookup={(us,ww):fi[(u[fi]==us)&(w[fi]==ww)] for us in range(1,10) for ww in range(12)}
    rank={int(i):j for vv in lookup.values() for j,i in enumerate(vv)}
    np.savez_compressed(OUT/'split_normalization_private.npz',used_indices=used,fit_indices=fit,cal_indices=cal,
        validation_indices=val,fit_local=fi,cal_local=ci,val_local=vi,mean=mean,std=std,user=u,word=w,sample_id=ids[used])
    protocol=dict(state='training',seeds=SEEDS,methods=dict(B0='ACC CE',B1='CE+frozen ACC self local relation',
        B2='CE+correct Audio local relation',B3='CE+wrong Audio local relation,same mask as B2'),
        users=list(range(1,10)),encoder_fit_words=list(range(12)),fit_repeats='first6 per user-word',
        cal_words=list(range(12)),cal_queries='repeat6:8,not in encoder fit',
        validation_words=list(range(12,18)),validation_repeats='first8:K4 two-fold swap',
        encoder='fresh IdentityEncoder CNN32/64/128 temporal mean/std256 Linear128 L2',
        warmup=400,continuation=400,total_steps_each_method=800,branch_optimizer_reset=True,
        optimizer='AdamW lr3e-4 wd1e-3 clip1',checkpoint='fixed final,no validation selection',
        batch='9users x2shared words x2repeats=36;same continuation batches within seed',
        normalization='only648 encoder-fit ACC records',
        graph='Top2 intersection:all training data,repeat first3,last3,even words,odd words',
        graph_source='B1 warmup ACC;B2 frozen ReDimNet2-B3;B3 same frozen mask as B2',
        local_loss='row KL tau.2;crossword sameidentity positives;only stable-neighbor sameword negatives',
        gradient_weight='exact current encoder norm ratio rho.03,lambda cap1,no tuning,no EMA',
        wrong_teacher='different nonzero identity shifts per batch word,repeat rank preserved;B2 mask retained',
        calibration='per owner/backend/method/seed Cal-EER threshold,pooled same/cross equal weight;locked for both validation conditions/folds',
        backend='main balancedRBF C10 gammaScale;cosine prototype diagnostic',
        margin='genuine mean minus hardest attacker-identity mean;also worstsample;divide by Cal impostor std',
        no_validation_audio=True,no_words18_29_used=True,
        limits=['known users','12..17 previously inspected;exploratory validation,not new confirmation',
                'nominal word labels','single session','three seeds are optimizer repeats,not new subjects'])
    dump(OUT/'protocol_public.json',protocol)
    summaries=[]
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.RandomState(seed+1000)
        def batch():
            words=rng.choice(12,2,replace=False)
            ii=np.concatenate([rng.choice(lookup[us,ww],2,replace=False) for us in range(1,10) for ww in words])
            assert np.all(w[ii]<12) and set(ii).issubset(set(fi))
            return ii
        net=IdentityEncoder();head=nn.Linear(128,9,bias=False)
        opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=3e-4,weight_decay=1e-3)
        for step in range(1,401):
            ii=batch();z=net(x[ii]);y=torch.tensor(u[ii]-1,dtype=torch.long)
            ce=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)
            opt.zero_grad();ce.backward();nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1);opt.step()
            if step%100==0:
                dump(OUT/'status.json',dict(stage='warmup',seed=seed,step=step,total=400));print('WARMUP',seed,step,flush=True)
        initial=copy.deepcopy(net.state_dict());initialhead=copy.deepcopy(head.state_dict());batchstate=rng.get_state()
        self_teacher=np.zeros((len(used),128),dtype='float32');self_teacher[fi]=encode(net,x[fi])
        self_graph=stable_graph(self_teacher[fi].reshape(9,12,6,-1))
        dump(OUT/f'graphs_seed{seed}_public.json',dict(seed=seed,
            per_user=[dict(user=f'U{us+1}',ACC_self_neighbors=int(self_graph[us].sum()),Audio_neighbors=int(audio_graph[us].sum())) for us in range(9)],
            all_users_still_in_CE=True,mask_only_filters_relations_not_users=True))
        for method in METHODS:
            net=IdentityEncoder();net.load_state_dict(initial);head=nn.Linear(128,9,bias=False);head.load_state_dict(initialhead)
            net.train();rng.set_state(batchstate)
            parameters=list(net.parameters())+list(head.parameters());opt=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=1e-3)
            graph=torch.tensor(self_graph if method=='B1' else audio_graph,dtype=torch.bool)
            trace=[];start=time.time();batch_digest=hashlib.sha256()
            for step in range(1,401):
                ii=batch();batch_digest.update(ii.tobytes());z=net(x[ii]);y=torch.tensor(u[ii]-1,dtype=torch.long);ww=torch.tensor(w[ii],dtype=torch.long)
                ce=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y);kd=z.sum()*0;active=0;detail={}
                opt.zero_grad()
                if method=='B0':ce.backward()
                else:
                    target=self_teacher[ii] if method=='B1' else teacher[ii]
                    if method=='B3':
                        rr=np.random.RandomState(seed+90000+step);words=np.unique(w[ii]);shifts=rr.choice(np.arange(1,9),2,replace=False);mapping=dict(zip(words,shifts))
                        jj=np.array([lookup[1+(int(u[i])-1+int(mapping[w[i]]))%9,int(w[i])][rank[int(i)]] for i in ii])
                        assert np.all(u[jj]!=u[ii]) and np.array_equal(w[jj],w[ii]);target=teacher[jj]
                    kd,active=local_relation_kl(z,torch.tensor(target),y,ww,graph)
                    detail=combine_gradients(ce,kd,net,head)
                nn.utils.clip_grad_norm_(parameters,1);opt.step()
                if step==1 or step%100==0:
                    trace.append(dict(step=step,ce=float(ce.detach()),relation=float(kd.detach()),active_anchors=active,**detail))
                    dump(OUT/f'{method}_seed{seed}_trace_public.json',trace)
                    dump(OUT/'status.json',dict(stage='branch_training',method=method,seed=seed,step=step,total=400,seconds=time.time()-start))
                    print('BRANCH',seed,method,step,'CE',round(float(ce.detach()),4),'seconds',round(time.time()-start),flush=True)
            changed=max(float((v-initial[k]).abs().max()) for k,v in net.state_dict().items());assert changed>0
            torch.save(dict(encoder=net.state_dict(),seed=seed,method=method,fit_indices=fit,warmup=400,continuation=400),OUT/f'{method}_seed{seed}_private.pt')
            embedding=encode(net,x);np.savez_compressed(OUT/f'{method}_seed{seed}_features_private.npz',embedding=embedding)
            summaries.append(dict(seed=seed,method=method,encoder_change_from_warmup=changed,batch_sha256=batch_digest.hexdigest(),
                checkpoint_sha256=hashlib.sha256((OUT/f'{method}_seed{seed}_private.pt').read_bytes()).hexdigest()))
            dump(OUT/'training_summary_public.json',summaries)
            print('BRANCH_COMPLETE',seed,method,flush=True)
        assert len(set(r['batch_sha256'] for r in summaries if r['seed']==seed))==1
    protocol['state']='all12_checkpoints_locked';dump(OUT/'protocol_public.json',protocol)
    dump(OUT/'status.json',dict(stage='all12_checkpoints_locked'));print('ALL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
