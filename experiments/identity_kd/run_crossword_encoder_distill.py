"""Independent innovation-2 pilot: real encoder updates, cross-word identities.

First run seed42. No changes to frozen innovation-1 assets.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from crossword_representation_losses import IdentityEncoder,crossword_supcon,relation_kl


def dump(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def encode(model,x,indices):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x[indices[s:s+64]]).numpy() for s in range(0,len(indices),64)])


def development(z,user,word):
    # Fast template geometry metric, matched query in both conditions.
    values={'crossword':[],'sameword':[]}
    for fold in [0,1]:
        for owner in range(1,10):
            for qw in [18,19,20]:
                qq=np.concatenate([np.flatnonzero((user==u)&(word==qw))[(1-fold)*4:(1-fold)*4+4] for u in range(1,10)])
                y=user[qq]==owner
                for rw in [18,19,20]:
                    if rw==qw:continue
                    for condition,w in [('sameword',qw),('crossword',rw)]:
                        rr=np.flatnonzero((user==owner)&(word==w))[fold*4:fold*4+4]
                        scores=z[qq]@norm(z[rr].mean(0))
                        values[condition].append(roc_auc_score(y,scores))
    return {k:float(np.mean(v)) for k,v in values.items()}


def main(a):
    torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    data=np.load(a.frontend);teacher_cache=np.load(a.teacher)
    assert np.array_equal(data['sample_id'],teacher_cache['sample_id'])
    user=data['user'];word=data['word'];fit=np.flatnonzero(word<18)
    dev=np.flatnonzero(np.isin(word,[18,19,20]))
    assert np.array_equal(fit,teacher_cache['indices'])
    assert int(teacher_cache['done'])==len(fit)
    teacher=teacher_cache['embedding'];assert not teacher[word>=18].any()
    spectrum=data['spectrum'].astype('float32')
    mean=spectrum[fit].mean((0,3),keepdims=True);std=spectrum[fit].std((0,3),keepdims=True).clip(1e-4)
    x=torch.tensor((spectrum-mean)/std)
    raw=np.concatenate([spectrum.mean(-1).reshape(len(user),-1),spectrum.std(-1).reshape(len(user),-1),
        *[spectrum[...,j].mean(-1).reshape(len(user),-1) for j in np.array_split(np.arange(28),4)]],1)
    scaler=StandardScaler().fit(raw[fit]);raw=norm(scaler.transform(raw)).astype('float32')
    np.savez_compressed(out/'normalization_private.npz',mean=mean,std=std,raw_mean=scaler.mean_,raw_scale=scaler.scale_)
    lookup={(u,w):np.flatnonzero((user==u)&(word==w)) for u in range(1,10) for w in range(18)}
    ranks={int(i):rank for ii in lookup.values() for rank,i in enumerate(ii)}
    def batch(rng):
        ww=rng.choice(18,2,replace=False)
        ii=np.concatenate([rng.choice(lookup[u,w],2,replace=False) for u in range(1,10) for w in ww])
        assert np.all(word[ii]<18)
        return ii
    def wrong_indices(ii,step):
        # Different word-specific identity permutations: destroys cross-word
        # positive relationships, unlike a global relabeling of whole identities.
        rng=np.random.RandomState(a.seed+90000+step)
        ww=np.unique(word[ii]);shifts=rng.choice(np.arange(1,9),len(ww),replace=False)
        mapping=dict(zip(ww,shifts));jj=[]
        for i in ii:
            other=1+(int(user[i])-1+int(mapping[word[i]]))%9
            candidates=lookup[other,int(word[i])]
            j=int(candidates[ranks[int(i)]%len(candidates)])
            assert user[j]!=user[i] and word[j]==word[i]
            jj.append(j)
        return np.array(jj)
    dump(out/'protocol_public.json',dict(state='running',seed=a.seed,
        stage='independent Audio-assisted representation study,not innovation1 modification',
        training_users=list(range(1,10)),final_targets=[1,2,3,4,6,7,8,9],final_attacker_pool=[1,2,3,4,6,7,8,9],
        fit_words=list(range(18)),develop_words=[18,19,20],cal_words=[21,22,23],final_words=list(range(24,30)),
        encoder='CNN32/64/128 GroupNorm,frequency strides;temporal mean+populationstd;Linear256to128,L2',
        warmup_steps=a.warmup,training_steps=a.steps,batch='9users x2 sharedwords x2 repeats=36',
        losses='CE;CE+.1CrossSupCon;row-similarityKL tau.2,not distance/angle RKD',
        cross_supcon='tau.1,sameidentitydifferentword positive,otheridentitysameword negative weight2',
        rho_candidates=[.03,.1,.3],rho_zero_option='E1 is always eligible against positive Audio candidates',
        gradient_balance='encoder-only norms;EMA .9;lambda cap100;record weighted ratio+cosine before totalclip1',
        wrong_teacher='different cross-identity permutation per batch word,repeat rank preserved,not global identity relabeling',
        self_teacher='frozen same CE-warmup ACC encoder,train rows only',
        model_selection='Dev cross-word prototype macroAUC,then same-word AUC,then earlier checkpoint',
        optimizer='AdamW lr3e-4 wd1e-3;fixed final budget for all variants',
        validation='same fit/dev/cal/final word roles for all;no final scores during selection',
        limits=['nominal word order','closed identities,not unseen target registration','single-seed pilot','historically explored data']))
    torch.manual_seed(a.seed);model=IdentityEncoder();head=nn.Linear(128,9,bias=False)
    warm_path=out/'warmup_private.pt'
    if warm_path.exists():
        ck=torch.load(warm_path,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['encoder']);head.load_state_dict(ck['head'])
    else:
        opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=3e-4,weight_decay=1e-3)
        rng=np.random.RandomState(a.seed)
        for step in range(1,a.warmup+1):
            model.train();ii=batch(rng);z=model(x[ii]);y=torch.tensor(user[ii]-1,dtype=torch.long)
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),y)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1);opt.step()
            if step%100==0:
                print('WARMUP',step,float(loss.detach()),flush=True)
                dump(out/'status.json',dict(stage='warmup',step=step,total=a.warmup))
        torch.save(dict(encoder=model.state_dict(),head=head.state_dict()),warm_path)
    initial=copy.deepcopy(model.state_dict());initial_head=copy.deepcopy(head.state_dict())
    self_targets=np.zeros((len(user),128),dtype='float32');self_targets[fit]=encode(model,x,fit)
    # Development is allowed to choose models; no Cal/final vectors exported yet.
    def run(name,kind,rho):
        complete=out/(name+'_complete_public.json')
        if complete.exists():return json.loads(complete.read_text())
        torch.manual_seed(a.seed)
        net=IdentityEncoder();net.load_state_dict(initial)
        classifier=nn.Linear(128,9,bias=False);classifier.load_state_dict(initial_head)
        params=list(net.parameters())
        optimizer=torch.optim.AdamW(params+list(classifier.parameters()),lr=3e-4,weight_decay=1e-3)
        rng=np.random.RandomState(a.seed+1000);history=[];trace=[];best=None;best_state=None;start=0;ema_id=ema_kd=None
        resume=out/(name+'_resume_private.pt')
        if resume.exists():
            ck=torch.load(resume,map_location='cpu',weights_only=False)
            net.load_state_dict(ck['encoder']);classifier.load_state_dict(ck['head']);optimizer.load_state_dict(ck['optimizer'])
            rng.set_state(ck['rng']);torch.set_rng_state(ck['torch_rng']);start=ck['step'];history=ck['history'];trace=ck['trace']
            best=ck['best'];best_state=ck['best_state'];ema_id=ck['ema_id'];ema_kd=ck['ema_kd']
            for entry in trace:
                if rho==0 and entry.get('identity_gradient_norm')==0:
                    entry['identity_gradient_norm']=None  # older logger did not measure this field
        started=time.time()
        for step in range(start+1,a.steps+1):
            net.train();ii=batch(rng);z=net(x[ii]);y=torch.tensor(user[ii]-1,dtype=torch.long);w=torch.tensor(word[ii],dtype=torch.long)
            ce=F.cross_entropy(16*F.linear(z,F.normalize(classifier.weight,dim=1)),y)
            identity=ce if kind=='ce' else ce+.1*crossword_supcon(z,y,w)
            kd=z.new_zeros(());lam=0.;ratio=0.;cosine=None;ni=nk=0.
            if rho>0:
                target=self_targets[ii] if kind=='self' else teacher[wrong_indices(ii,step) if kind=='wrong' else ii]
                tt=torch.tensor(target)
                kd=relation_kl(z,tt,y,w)
                gi=torch.autograd.grad(identity,params,retain_graph=True)
                gk=torch.autograd.grad(kd,params,retain_graph=True)
                ni=float(torch.sqrt(sum(g.square().sum() for g in gi)))
                nk=float(torch.sqrt(sum(g.square().sum() for g in gk)))
                ema_id=ni if ema_id is None else .9*ema_id+.1*ni
                if nk>1e-8:
                    ema_kd=nk if ema_kd is None else .9*ema_kd+.1*nk
                    lam=min(100.,rho*ema_id/max(ema_kd,1e-8))
                ratio=lam*nk/max(ni,1e-8)
                cosine=float(sum((a*b).sum() for a,b in zip(gi,gk))/(ni*nk)) if ni*nk>1e-12 else None
                if step==1 and kind=='wrong':
                    difference=float(relation_kl(torch.tensor(teacher[ii]),tt,y,w))
                    assert difference>1e-6
                    dump(out/'wrong_teacher_audit_public.json',dict(real_wrong_relation_kl=difference,
                         every_source_identity_changed=True,word_labels_preserved=True,crossword_positive_structure_broken=True))
            elif step in [1,10] or step%100==0:
                gi=torch.autograd.grad(identity,params,retain_graph=True)
                ni=float(torch.sqrt(sum(g.square().sum() for g in gi)))
            loss=identity+lam*kd
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params+list(classifier.parameters()),1);optimizer.step()
            if step in [1,10] or step%100==0:
                trace.append(dict(step=step,identity=float(identity.detach()),kd=float(kd.detach()),lambda_kd=lam,
                                  identity_gradient_norm=ni,kd_gradient_norm=nk,weighted_gradient_ratio=ratio,gradient_cosine=cosine))
            if step%100==0 or step==a.steps:
                dv=np.zeros((len(user),128),dtype='float32');dv[dev]=encode(net,x,dev)
                quality=development(dv,user,word)
                key=(quality['crossword'],quality['sameword'],-step)
                if best is None or key>tuple(best):best=list(key);best_state=copy.deepcopy(net.state_dict())
                history.append(dict(step=step,loss=float(loss.detach()),**quality))
                state=dict(encoder=net.state_dict(),head=classifier.state_dict(),optimizer=optimizer.state_dict(),step=step,
                           history=history,trace=trace,best=best,best_state=best_state,rng=rng.get_state(),
                           torch_rng=torch.get_rng_state(),ema_id=ema_id,ema_kd=ema_kd)
                torch.save(state,resume)
                dump(out/(name+'_history_public.json'),history);dump(out/(name+'_gradient_public.json'),trace)
                dump(out/'status.json',dict(stage='representation_training',method=name,step=step,total=a.steps,seconds=time.time()-started))
                print('REPRESENTATION',name,step,'DevCrossAUC',quality['crossword'],'seconds',round(time.time()-started),flush=True)
        assert best_state is not None
        changed=max(float((best_state[k]-initial[k]).abs().max()) for k in initial)
        assert changed>0
        torch.save(best_state,out/(name+'_best_private.pt'))
        result=dict(method=name,kind=kind,rho=rho,selected_step=-best[2],develop_cross_auc=best[0],
                    develop_same_auc=best[1],encoder_max_parameter_change=changed,steps=a.steps)
        dump(complete,result)
        print('REPRESENTATION_COMPLETE',name,flush=True)
        return result
    records={}
    records['E0']=run('E0','ce',0)
    records['E1']=run('E1','identity',0)
    candidates=[run(f'E2_rho{rho:g}','audio',rho) for rho in [.03,.1,.3]]
    chosen=max(candidates,key=lambda r:(r['develop_cross_auc'],r['develop_same_auc'],-r['rho']))
    noaudio=records['E1']
    deploy=noaudio if (noaudio['develop_cross_auc'],noaudio['develop_same_auc']) >= (chosen['develop_cross_auc'],chosen['develop_same_auc']) else chosen
    records['E2']=chosen
    records['E3']=run('E3','wrong',chosen['rho'])
    records['E4']=run('E4','self',chosen['rho'])
    dump(out/'selection_public.json',dict(positive_audio_candidates=candidates,best_positive=chosen,
         development_selected_audio_rho=0 if deploy is noaudio else chosen['rho'],
         deployment_choice=deploy['method'],rule='Dev cross AUC/same AUC/lower rho;E1 is zero-rho option',
         model_records=records,final_not_used=True))
    # Export ACC-only inference features only after all choices above are locked.
    vectors={}
    for name,r in records.items():
        net=IdentityEncoder();net.load_state_dict(torch.load(out/(r['method']+'_best_private.pt'),weights_only=True))
        vectors[name]=encode(net,x,np.arange(len(user)))
    np.savez_compressed(out/'vectors_private.npz',raw=raw,user=user,word=word,sample_id=data['sample_id'],**vectors)
    dump(out/'status.json',dict(stage='representation_complete_ready_for_authentication'))
    print('REPRESENTATION_ALL_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--teacher',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--warmup',type=int,default=400);p.add_argument('--steps',type=int,default=1000)
    main(p.parse_args())
