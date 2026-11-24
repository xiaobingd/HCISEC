"""Pinned Audio-conditioned verifier plan; known encoders, held-out gate target.
Only fit words optimize G. All thresholds are serialized before query scoring.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def weighted_cosine(q,p,w):
    numerator=(q*p[:,None,:]*w[:,None,:]).sum(-1)
    denominator=((q.square()*w[:,None,:]).sum(-1)*(p.square()*w).sum(-1)[:,None]).clamp_min(1e-12).sqrt()
    return numerator/denominator

class Gate(nn.Module):
    def __init__(self,initial_bias=.5):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(576,32),nn.LayerNorm(32),nn.SiLU(),nn.Linear(32,192))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.log_scale=nn.Parameter(torch.tensor(float(np.log(20.))))
        self.bias=nn.Parameter(torch.tensor(float(initial_bias)))
    def forward(self,p,a,dispersion):
        return 1+.25*torch.tanh(self.net(torch.cat([p,a,dispersion],dim=-1)))

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def matched_noise(shape,mu,sd,rng):
    values=rng.normal(size=shape);axes=tuple(range(values.ndim-1))
    values=(values-values.mean(axes,keepdims=True))/np.maximum(values.std(axes,keepdims=True),1e-8)
    return (mu+sd*values).astype('float32')

def main(root):
    sys.path.insert(0,str(root))
    from calibrate_threeword import cutoff,measure
    from prepare_phrase_auth import FIT,REG,CAL,SHORT,ENGLISH,CHINESE
    from scipy.stats import rankdata
    torch.set_num_threads(2)
    out=root/'conditional_correspondence/audio_conditioned_verifier';out.mkdir(exist_ok=True)
    d=np.load(root/'phrase_auth/frontend_private.npz');u,w,r,ids=d['user'],d['word'],d['repeat'],d['sample_id']
    fit=np.isin(w,FIT);visible=np.isin(w,FIT+REG+CAL);vi=np.flatnonzero(visible)
    mapping=np.full(3200,-1,dtype=int);mapping[vi]=np.arange(len(vi))
    assert len(vi)==2000 and fit.sum()==1280 and len(set(ids))==3200
    af=np.load(out/'fit_ecapa_private.npz');ar=np.load(root/'conditional_correspondence/enrollment_identity_quality/registered_ecapa_private.npz')
    assert np.array_equal(af['indices'],np.flatnonzero(fit)) and np.array_equal(af['sample_id'],ids[fit]) and bool(af['length_aware'])
    regmask=np.isin(w,REG);assert np.array_equal(ar['indices'],np.flatnonzero(regmask)) and np.array_equal(ar['sample_id'],ids[regmask])
    audio=np.zeros((3200,192),dtype='float32');audio[af['indices']]=af['embedding'];audio[ar['indices']]=ar['embedding']
    trainwords=sorted(FIT)[:12];innerwords=sorted(FIT)[12:]
    methods=['A_zero_audio','B_correct_audio','C_other_audio','D_destroyed_audio']
    evalmethods=methods+['ungated_ACC','B_wrong_audio','B_audio_first2words','B_audio_last2words']
    protocol=dict(upstream_branch='experiment/audio-conditioned-verifier-20261124',upstream_commit='b2953752d7156c9b54d62a7adb16df5d2b5add46',
        users=10,seeds=[42,43,44],target_folds=10,methods=methods,main_query='oneACCqueryrecord,no3wordfusion',
        gate='576->32 LayerNorm SiLU ->192,1+.25tanh;lastlayerzero givesall1;per-channel registration variance raw',
        optimizer='AdamW.001 wd.01,400steps;BCE balanced4genuine/4ACChardnegative +.01mean((w-1)^2);scale20 initialization,biasfit-only initial midpoint;no margin loss',
        fit_words=FIT,gate_optimization_words=trainwords,fit_internal_validation_words=innerwords,
        exclusion='heldtarget identity excluded from supports,query positives,impostor candidates,Audio stats andG optimization;frozenACC encoder previously trainedall10fitidentities,not full unseenuseropen-set',
        episodes='all9allowed users each4supportwords x3reps,4differentquerywords x1rep;4hardnegativequeries fromother8users byungatedACCcos;identicalepisodeindices forA/B/C/D',
        controls='C per-episode no-fixed-point permutation ofallowed9Audio prototypes,evalnextotherregistereduser;DGaussian featurewise matched to all trainingepisode prototype means/std,no L2renormalization;Azeros',
        registration='sameACCcrosswordtop3per4REGwords=12 forallarms;Audio matchingthese12;variancefromsame12',
        calibration_words=CAL,query_families=dict(english_word=SHORT,english_long=ENGLISH,chinese_long=CHINESE),
        threshold='fixed (cal genuineQ10+impostorQ95)/2;alltarget/seed/method locks savedbefore query features scored',
        diagnostic_threshold='BwrongAudio andsame-userword halves reuse Bcorrect threshold,no perturbation recalibration;ungated baseline retains prior same-rule CAL threshold after numeric consistency check',
        validation='fixed400checkpoint;fit-inner heldwords only audit,not selecthyperparameters usingCAL/query',
        sensitivity='Bcorrectgate vswrongAudio;Bsameuser disjointAudioREGfirst2vs last2words;bothhalves6Audio samples versusmain12,diagnosticonly',
        limitations='knownfrozenencoder identities;historically inspected cohort/querywords;3secondACCwindows mayinclude neighborutterances;44100interpretation/physicalsyncnotindependentlyverified')
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    legacy_calibration=json.loads((root/'conditional_correspondence/enrollment_identity_quality_fixed_background/thresholds_private.json').read_text())
    # Only fit/register/calibration slices are retained for optimization and threshold locking.
    known={}
    for seed in (42,43,44):
        with np.load(root/'conditional_correspondence/stage3/views_private.npz') as vf:known[seed]=vf[f'{seed}_A_ACC'][vi].copy()
    episodes_audit=[];training=[];gates_public=[];locks=[];context={}
    labels=torch.tensor(np.tile([1]*4+[0]*4,(9,1)),dtype=torch.float32)
    for seed in (42,43,44):
        z=known[seed]
        def zz(ix):
            assert np.all(mapping[ix]>=0)
            return z[mapping[ix]]
        selection={}
        for us in range(1,11):
            parts=[]
            for word in REG:
                ix=np.flatnonzero((u==us)&(w==word));ix=ix[np.argsort(r[ix])]
                other=np.flatnonzero((u==us)&np.isin(w,REG)&(w!=word));p=unit(zz(other).mean(0))
                score=rankdata(zz(ix)@p)/8;order=np.lexsort((np.arange(8),-score));parts.extend(ix[order[:3]])
            selection[us]=np.array(parts)
        p_reg=np.stack([zz(selection[us]).mean(0) for us in range(1,11)])
        p_reg/=np.maximum(np.linalg.norm(p_reg,axis=1,keepdims=True),1e-8)
        a_reg=np.stack([unit(audio[selection[us]].mean(0)) for us in range(1,11)])
        disp_reg=np.stack([zz(selection[us]).var(0) for us in range(1,11)])
        context[seed]=dict(p=p_reg,selection=selection)
        for held in range(1,11):
            allowed=np.array([us for us in range(1,11) if us!=held]);rng=np.random.default_rng(seed*100+held)
            by={(int(us),int(word)):np.flatnonzero((u==us)&(w==word)) for us in allowed for word in FIT}
            def episode(supportwords,querywords,erng):
                support=np.array([np.concatenate([erng.choice(by[int(us),int(word)],3,replace=False) for word in supportwords]) for us in allowed])
                query=np.array([[erng.choice(by[int(us),int(word)]) for word in querywords] for us in allowed])
                assert not np.intersect1d(support,query).size and not np.any(u[support]==held) and not np.any(u[query]==held)
                p=np.stack([unit(zz(a).mean(0)) for a in support]);a=np.stack([unit(audio[b].mean(0)) for b in support]);disp=np.stack([zz(b).var(0) for b in support])
                flat=query.ravel();queries=[]
                for i,us in enumerate(allowed):
                    candidates=flat[u[flat]!=us];similarity=zz(candidates)@p[i]
                    hard=candidates[np.argsort(-similarity,kind='stable')[:4]]
                    queries.append(np.r_[query[i],hard])
                queries=np.array(queries)
                assert not np.any(u[queries]==held)
                assert (u[queries[:,:4]]==allowed[:,None]).all() and (u[queries[:,4:]]!=allowed[:,None]).all()
                return p,a,disp,zz(queries),support,queries
            data=[];support_indices=[];query_indices=[]
            for step in range(400):
                choices=rng.choice(trainwords,8,replace=False)
                p,a,disp,q,si,qi=episode(choices[:4],choices[4:],rng)
                data.append((p,a,disp,q));support_indices.append(si);query_indices.append(qi)
            pp,aa,dd,qq=[torch.tensor(np.stack([a[j] for a in data]),dtype=torch.float32) for j in range(4)]
            si=np.array(support_indices);qi=np.array(query_indices)
            assert np.isin(w[si],trainwords).all() and np.isin(w[qi],trainwords).all()
            np.savez_compressed(out/f'episodes_seed{seed}_held{held}_private.npz',support_indices=si,query_indices=qi)
            digest=hashlib.sha256(si.tobytes()+qi.tobytes()).hexdigest()
            episodes_audit.append(dict(seed=seed,held_target=held,episode_sha256=digest,steps=400,held_target_optimization_records=0,train_users=9))
            mu=aa.numpy().mean((0,1));sd=aa.numpy().std((0,1))
            wrong_rng=np.random.default_rng(7711+seed*100+held);permutations=[]
            for step in range(400):
                perm=wrong_rng.permutation(9)
                while np.any(perm==np.arange(9)):perm=wrong_rng.permutation(9)
                permutations.append(perm)
            wrong_audio=torch.stack([aa[step,perm] for step,perm in enumerate(permutations)])
            destroyed=torch.tensor(matched_noise((400,9,192),mu,sd,np.random.default_rng(9911+seed*100+held)))
            moment_error=max(float(abs(destroyed.numpy().mean((0,1))-mu).max()),float(abs(destroyed.numpy().std((0,1))-sd).max()))
            assert moment_error<1e-5
            episodes_audit[-1]['destroyed_audio_moment_max_error']=moment_error
            # Fit-internal validation never affects model/weight selection.
            vp,va,vd,vq,_,_=episode(trainwords[:4],innerwords,np.random.default_rng(8100+seed*100+held))
            vp,va,vd,vq=[torch.tensor(a,dtype=torch.float32) for a in (vp,va,vd,vq)]
            first=weighted_cosine(qq[0],pp[0],torch.ones_like(pp[0]));bias0=float((first[:,:4].mean()+first[:,4:].mean())/2)
            own=held-1;other=held%10
            pc=torch.tensor(p_reg[own:own+1]);dc=torch.tensor(disp_reg[own:own+1]);ac=torch.tensor(a_reg[own:own+1]);oc=torch.tensor(a_reg[other:other+1])
            noise=mu+sd*np.random.default_rng(12000+seed*100+held).normal(size=192)
            foldgates={}
            for method in methods:
                torch.manual_seed(seed*100+held);net=Gate(bias0);opt=torch.optim.AdamW(net.parameters(),lr=.001)
                checkpoint=out/f'{method}_seed{seed}_held{held}_private.pt';trace=[]
                if checkpoint.exists():net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                else:
                    for step in range(400):
                        condition=torch.zeros_like(aa[step]) if method=='A_zero_audio' else aa[step] if method=='B_correct_audio' else wrong_audio[step] if method=='C_other_audio' else destroyed[step]
                        gate=net(pp[step],condition,dd[step]);score=weighted_cosine(qq[step],pp[step],gate)
                        logits=net.log_scale.exp().clamp(1,100)*(score-net.bias)
                        bce=F.binary_cross_entropy_with_logits(logits,labels);reg=.01*(gate-1).square().mean();loss=bce+reg
                        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
                        if (step+1)%100==0:
                            with torch.inference_mode():
                                vcond=torch.zeros_like(va) if method=='A_zero_audio' else va if method=='B_correct_audio' else va.roll(1,0) if method=='C_other_audio' else torch.tensor(matched_noise((9,192),mu,sd,np.random.default_rng(555)))
                                vg=net(vp,vcond,vd);vs=weighted_cosine(vq,vp,vg)
                                val=F.binary_cross_entropy_with_logits(net.log_scale.exp().clamp(1,100)*(vs-net.bias),labels)
                            trace.append(dict(step=step+1,train_bce=float(bce.detach()),fit_inner_bce=float(val),gate_abs_deviation=float((gate-1).abs().mean().detach())))
                    torch.save(net.state_dict(),checkpoint)
                with torch.inference_mode():
                    condition=torch.zeros_like(ac) if method=='A_zero_audio' else ac if method=='B_correct_audio' else oc if method=='C_other_audio' else torch.tensor(noise[None,:],dtype=torch.float32)
                    g=net.eval()(pc,condition,dc).numpy()[0];assert np.isfinite(g).all() and g.min()>=.75 and g.max()<=1.25
                    foldgates[method]=g
                    if method=='B_correct_audio':
                        foldgates['B_wrong_audio']=net(pc,oc,dc).numpy()[0]
                        for name,words in [('B_audio_first2words',REG[:2]),('B_audio_last2words',REG[2:])]:
                            subset=selection[held][np.isin(w[selection[held]],words)];assert len(subset)==6
                            alt=torch.tensor(unit(audio[subset].mean(0))[None,:]);foldgates[name]=net(pc,alt,dc).numpy()[0]
                training.append(dict(seed=seed,held_target=held,method=method,steps=400,trace=trace))
            foldgates['ungated_ACC']=np.ones(192,dtype='float32')
            fold_thresholds={}
            for name,g in foldgates.items():
                gates_public.append(dict(seed=seed,user=held,method=name,mean_abs_deviation=float(abs(g-1).mean()),gate_min=float(g.min()),gate_max=float(g.max())))
                calidx=np.flatnonzero(np.isin(w,CAL));q=torch.tensor(zz(calidx)[None,:,:]);p=pc;ww=torch.tensor(g[None,:])
                with torch.inference_mode():score=weighted_cosine(q,p,ww).numpy()[0]
                # Preserve the exact prior cosine baseline for its regression check.
                if name=='ungated_ACC':score=(z@p_reg.T)[mapping[calidx],own]
                tau=cutoff(score,u[calidx]==held,'quantile_gap')
                if name in ('B_wrong_audio','B_audio_first2words','B_audio_last2words'):
                    tau=fold_thresholds['B_correct_audio']
                if name=='ungated_ACC':
                    reference=next(a['threshold'] for a in legacy_calibration if a['seed']==seed and a['user']==held and a['method']=='acc_crossword12' and a['backend']=='cosine' and a['words']==1)
                    assert abs(reference-tau)<1e-6
                    tau=reference
                fold_thresholds[name]=tau
                locks.append(dict(seed=seed,user=held,method=name,threshold=tau))
            context[seed,held]=dict(gates=foldgates,p=p_reg[own])
            (out/'training_public.json').write_text(json.dumps(training))
            (out/'episode_audit_public.json').write_text(json.dumps(episodes_audit))
            print('CONDITIONED_FOLD_DONE',seed,held,flush=True)
    assert len(locks)==240
    (out/'thresholds_private.json').write_text(json.dumps(locks));print('CONDITIONED_ALL_THRESHOLDS_LOCKED',flush=True)
    rows=[];sensitivity=[]
    families={'english_word':SHORT,'english_long':ENGLISH,'chinese_long':CHINESE}
    for seed in (42,43,44):
        with np.load(root/'conditional_correspondence/stage3/views_private.npz') as vf:query_view=vf[f'{seed}_A_ACC']
        baseline=query_view@context[seed]['p'].T
        for held in range(1,11):
            c=context[seed,held];p=torch.tensor(c['p'][None,:]);queryidx=np.flatnonzero(np.isin(w,SHORT+ENGLISH+CHINESE))
            correct=c['gates']['B_correct_audio'];wrong=c['gates']['B_wrong_audio']
            with torch.inference_mode():
                sq=torch.tensor(query_view[queryidx][None,:,:])
                sc=weighted_cosine(sq,p,torch.tensor(correct[None,:])).numpy()[0]
                sw=weighted_cosine(sq,p,torch.tensor(wrong[None,:])).numpy()[0]
            genuine=u[queryidx]==held
            sensitivity.append(dict(seed=seed,user=held,audio_wrong_gate_mean_abs_difference=float(abs(correct-wrong).mean()),
                sameuser_word_halves_gate_mean_abs_difference=float(abs(c['gates']['B_audio_first2words']-c['gates']['B_audio_last2words']).mean()),
                wrong_audio_score_mean_abs_change=float(abs(sc-sw).mean()),wrong_audio_genuine_score_mean_change=float((sw-sc)[genuine].mean()),
                wrong_audio_impostor_score_mean_change=float((sw-sc)[~genuine].mean())))
            for name,g in c['gates'].items():
                tau=next(a['threshold'] for a in locks if a['seed']==seed and a['user']==held and a['method']==name)
                for family,words in families.items():
                    qi=np.flatnonzero(np.isin(w,words))
                    with torch.inference_mode():score=weighted_cosine(torch.tensor(query_view[qi][None,:,:]),p,torch.tensor(g[None,:])).numpy()[0]
                    if name=='ungated_ACC':score=baseline[qi,held-1]
                    rows.append(dict(seed=seed,user=held,method=name,family=family,**measure(score,u[qi]==held,tau)))
    assert len(rows)==720
    (out/'results_private.json').write_text(json.dumps(rows));(out/'per_seed_user_public.json').write_text(json.dumps(rows))
    macro=[];users=[]
    for name in evalmethods:
        for family in families:
            sub=[a for a in rows if a['method']==name and a['family']==family];assert len(sub)==30
            macro.append(dict(method=name,family=family,**{key:float(np.mean([a[key] for a in sub])) for key in ('far','frr','hter','eer','auc')}))
            for us in range(1,11):
                part=[a for a in sub if a['user']==us]
                users.append(dict(method=name,family=family,user=us,**{key:float(np.mean([a[key] for a in part])) for key in ('far','frr','hter','eer','auc')}))
    (out/'summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=users),indent=2))
    (out/'gate_diagnostics_public.json').write_text(json.dumps(dict(weight_ranges=gates_public,sensitivity=sensitivity),indent=2))
    (out/'verification_public.json').write_text(json.dumps(dict(status='PASS',models=120,target_folds=30,workpoints=720,primary_workpoints=450,thresholds=240,
        held_target_optimization_records=0,hard_negative_indices_shared_across_arms=True,query_audio_used=False,all_thresholds_locked_before_query_scores=True,
        global_encoders_frozen=True,not_full_user_disjoint_encoder=True,parameters_per_gate=sum(p.numel() for p in Gate().parameters())),indent=2))
    print('AUDIO_CONDITIONED_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root.resolve())
