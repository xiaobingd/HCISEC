"""Fresh text-isolated ACC/AudioKD students: phrase once versus word fusion."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff,measure
from train_audio_multiword_fitonly import Teacher,supcon,relation
from prepare_phrase_auth import FIT,REG,CAL,SHORT,ENGLISH,CHINESE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'phrase_auth'

def main():
    torch.set_num_threads(2)
    data=np.load(OUT/'frontend_private.npz')
    old=np.load(ROOT/'paired_raw_acc_private.npz')
    assert np.array_equal(data['sample_id'],old['sample_id'])
    u,w,r=data['user'],data['word'],data['repeat'];fit=data['fit'];fi=np.flatnonzero(fit)
    assert set(w[fit])==set(FIT) and not set(FIT)&set(REG+CAL+SHORT+ENGLISH+CHINESE)
    scaler=StandardScaler().fit(data['features'][fit])
    x=torch.tensor(scaler.transform(data['features']),dtype=torch.float32)
    y=torch.tensor(u-1,dtype=torch.long)
    words=np.array(FIT)
    groups=np.stack([[np.flatnonzero(fit&(u==us)&(w==word)) for word in words] for us in range(1,11)])
    assert groups.shape==(10,16,8) and len(fi)==1280
    protocol=dict(users=10,train_words=FIT,train_records=1280,registration_words=REG,
        registration_per_user=24,calibration_words=CAL,query_single_words=SHORT,
        query_english_long=ENGLISH,query_chinese_long=CHINESE,
        training_isolation='all15 query prompts excluded from scaler, warm encoder, Audio projection adaptation and students',
        initialization='fresh random ACC weights;700-step CE+.1SupCon on16fitwords only; no prior26-word encoder',
        student='same762->256->192,400steps,100/step shared word-disjoint support/query sampler',
        teacher='fresh official ECAPA final projection+residual,300steps on1280newfit only; reuse immutable official frozen-backbone per-record cache subset after raw Audio equality check',
        loss='CE; AudioKD adds relationMSE same-user/different-word and different-user/same-word;encoder KD grad ratio .1 cappedweight10; fixed record-shuffle control',
        methods=['CE','AudioKD','ShuffledAudioKD'],seeds=[42,43,44],
        threshold='fixed previous q10 genuine/q95 impostor midpoint, separate target/method/seed/k, shared across query prompt families',
        backend='balanced RBF C10, ACC only at registration/calibration/query; mean1 or3 scores',
        primary='one complete Chinese prompt versus3independent English words; English long-prompt control separates some language effects',
        scope='closed known identities, unseen prompt classes; Chinese is also cross-language from English training/enrollment; history-explored cohort,not new blind sample',
        budget='full utterances preserved;min2sec padding; no source reuse within a challenge set;duration is active Audio length,not measured online latency',
        safety='no user filtering,query-based tuning,old adapted teacher/encoder reuse or query Audio inference')
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    def newnet():return nn.Sequential(nn.Linear(x.shape[1],256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
    warm=OUT/'fresh_warm_private.pt'
    warmtrace=[]
    if warm.exists():warmstate=torch.load(warm,map_location='cpu',weights_only=True)
    else:
        torch.manual_seed(42);rng=np.random.default_rng(42)
        net=newnet();head=nn.Linear(192,10,bias=False)
        opt=torch.optim.AdamW([*net.parameters(),*head.parameters()],lr=.001)
        for step in range(700):
            batch=groups[:,rng.choice(len(words),4,replace=False)]
            ix=np.take_along_axis(batch,rng.integers(8,size=(10,4,1)),axis=2).ravel()
            z=F.normalize(net(x[ix]),dim=1);yy=y[ix]
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),yy)+.1*supcon(z,yy)
            opt.zero_grad();loss.backward();opt.step()
            if (step+1)%100==0:
                warmtrace.append(dict(step=step+1,loss=float(loss.detach())))
                print('FRESH_WARM',step+1,float(loss.detach()),flush=True)
        warmstate=net.state_dict();torch.save(warmstate,warm)
    legacyfit=np.flatnonzero(old['role']=='fit')
    take=np.searchsorted(legacyfit,fi)
    assert np.array_equal(legacyfit[take],fi)
    official=ROOT/'audio_multiword_fitonly'
    provenance=json.loads((official/'extraction_progress_private.json').read_text())
    expected='0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2'
    assert provenance['official_model_sha256']==expected and provenance['completed']==2080
    assert provenance['sample_id_digest']==hashlib.sha256(old['sample_id'][legacyfit].tobytes()).hexdigest()
    pool=torch.tensor(np.array(np.load(official/'official_pool_private.npy',mmap_mode='r')[take]))
    initial=torch.tensor(np.array(np.load(official/'official_embedding_private.npy',mmap_mode='r')[take]))
    projection=torch.load(official/'official_projection_private.pt',map_location='cpu',weights_only=True)
    torch.manual_seed(4242);rng=np.random.default_rng(4242);teacher=Teacher(projection)
    with torch.no_grad():initerror=float((teacher(pool[:8])-initial[:8]).abs().max())
    assert initerror<1e-5
    tg=np.stack([[np.flatnonzero((u[fi]==us)&(w[fi]==word)) for word in words] for us in range(1,11)])
    tp=OUT/'fresh_fitonly_teacher_private.pt';teachertrace=[]
    if tp.exists():teacher.load_state_dict(torch.load(tp,map_location='cpu',weights_only=True))
    else:
        head=nn.Linear(192,10,bias=False)
        opt=torch.optim.AdamW([dict(params=teacher.projection.parameters(),lr=1e-5),
            dict(params=teacher.branch.parameters(),lr=1e-4),dict(params=head.parameters(),lr=1e-3)])
        for step in range(300):
            batch=tg[:,rng.choice(len(words),4,replace=False)]
            j=np.take_along_axis(batch,rng.integers(8,size=(10,4,1)),axis=2).ravel()
            z=teacher(pool[j]);yy=y[fi[j]]
            loss=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),yy)+.1*supcon(z,yy)+.01*(1-(z*initial[j]).sum(1)).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([*teacher.parameters(),*head.parameters()],1);opt.step()
            if (step+1)%100==0:
                teachertrace.append(dict(step=step+1,loss=float(loss.detach())))
                print('FRESH_TEACHER',step+1,float(loss.detach()),flush=True)
        torch.save(teacher.state_dict(),tp)
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    with torch.inference_mode():target=torch.cat([teacher(pool[i:i+64]) for i in range(0,len(fi),64)])
    assert torch.isfinite(target).all()
    tf=torch.zeros(3200,192);tf[fi]=target
    wrong=torch.zeros_like(tf);wrong[fi]=target[np.random.default_rng(14242).permutation(len(fi))]
    del pool,initial,teacher
    def episode(rng):
        chosen=rng.choice(len(words),5,replace=False)
        support=np.array([np.concatenate([rng.choice(groups[us,word],2,replace=False) for word in chosen[:3]]) for us in range(10)])
        query=np.array([np.concatenate([rng.choice(groups[us,word],2,replace=False) for word in chosen[3:]]) for us in range(10)])
        assert not np.intersect1d(support,query).size
        return np.concatenate([support,query],axis=1).ravel()
    views={};training=[]
    for seed in (42,43,44):
        for method in protocol['methods']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            net=newnet();net.load_state_dict(warmstate);head=nn.Linear(192,10,bias=False)
            with torch.no_grad():
                z=F.normalize(net(x),dim=1)
                head.weight.copy_(F.normalize(torch.stack([z[fit&(u==us)].mean(0) for us in range(1,11)]),dim=1))
            params=list(net.parameters());opt=torch.optim.AdamW([*params,*head.parameters()],lr=1e-4)
            path=OUT/f'{method}_{seed}_private.pt';trace=[]
            if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            else:
                for step in range(400):
                    ix=episode(rng);z=F.normalize(net(x[ix]),dim=1);yy=y[ix]
                    ce=F.cross_entropy(16*F.linear(z,F.normalize(head.weight,dim=1)),yy)
                    kd=z.new_zeros(());weight=0.;ratio=0.
                    if method!='CE':
                        kd=relation(z,(tf if method=='AudioKD' else wrong)[ix],yy,torch.tensor(w[ix]))
                        gi=torch.autograd.grad(ce,params,retain_graph=True);gk=torch.autograd.grad(kd,params,retain_graph=True)
                        ni=float(torch.sqrt(sum(v.square().sum() for v in gi)));nk=float(torch.sqrt(sum(v.square().sum() for v in gk)))
                        weight=min(10.,.1*ni/max(nk,1e-8));ratio=weight*nk/max(ni,1e-8)
                    loss=ce+weight*kd;opt.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_([*params,*head.parameters()],1);opt.step()
                    if (step+1)%100==0:
                        trace.append(dict(step=step+1,ce=float(ce.detach()),kd=float(kd.detach()),weight=weight,aux_gradient_ratio=ratio))
                        print('PHRASE_STUDENT',seed,method,step+1,flush=True)
                torch.save(net.state_dict(),path)
            with torch.inference_mode():views[seed,method]=F.normalize(net.eval()(x),dim=1).numpy()
            training.append(dict(seed=seed,method=method,trace=trace))
            (OUT/'training_public.json').write_text(json.dumps(dict(warm=warmtrace,teacher=teachertrace,students=training),indent=2))
    index={(int(us),int(word),int(rep)):i for i,(us,word,rep) in enumerate(zip(u,w,r))}
    order=np.random.default_rng(1123).permutation(5)
    def group(wordset,k):
        base=[(int(word),rep) for rep in range(8) for word in np.array(wordset)[order]]
        groups2=[base[i:i+k] for i in range(0,len(base)-k+1,k)]
        ix=np.array([[index[us,word,rep] for word,rep in g] for us in range(1,11) for g in groups2])
        users=np.repeat(np.arange(1,11),len(groups2))
        assert np.unique(ix).size==ix.size
        durations=data['duration_audio'][ix].sum(1)
        return ix,users,durations
    reg=np.isin(w,REG)&(r<6);scores={};locks=[]
    with threadpool_limits(limits=2):
        for (seed,method),view in views.items():
            scores[seed,method]=np.column_stack([SVC(C=10,class_weight='balanced').fit(view[reg],u[reg]==us).decision_function(view) for us in range(1,11)])
            for k in (1,3):
                ci,cu,_=group(CAL,k);cs=scores[seed,method][ci].mean(1)
                for us in range(1,11):
                    tau=cutoff(cs[:,us-1],cu==us,'quantile_gap')
                    locks.append(dict(seed=seed,method=method,words=k,user=us,threshold=tau,
                        development=measure(cs[:,us-1],cu==us,tau)))
    (OUT/'thresholds_public.json').write_text(json.dumps(locks,indent=2,allow_nan=False))
    print('PHRASE_THRESHOLDS_LOCKED',flush=True)
    rows=[];durations=[]
    for name,wordset in [('single_word',SHORT),('english_long',ENGLISH),('chinese_long',CHINESE)]:
        for k in (1,3):
            qi,qu,dur=group(wordset,k)
            durations.append(dict(condition=name,words=k,trials_per_user=int(len(qu)//10),source_records_per_user=int(qi.size//10),
                active_speech_median_s=float(np.median(dur)),active_speech_mean_s=float(np.mean(dur))))
            for lock in locks:
                if lock['words']!=k:continue
                qs=scores[lock['seed'],lock['method']][qi].mean(1)[:,lock['user']-1]
                rows.append(dict(**{key:lock[key] for key in ['seed','method','words','user']},condition=name,
                    **measure(qs,qu==lock['user'],lock['threshold'])))
    assert len(rows)==540 and len(locks)==180
    (OUT/'results_public.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (OUT/'duration_public.json').write_text(json.dumps(durations,indent=2))
    (OUT/'verification_public.json').write_text(json.dumps(dict(status='PASS',query_workpoints=540,thresholds=180,
        fresh_encoder_fit_only=True,teacher_cache_fit_records=1280,teacher_initial_projection_max_error=initerror,
        teacher_nonfit_audio_forwarded=0,query_prompts_in_train=0,scaler_fit_records=1280,
        source_id_reuse=False,all_thresholds_locked_before_query=True,all_users_retained=True),indent=2))
    print('PHRASE_AUTH_COMPLETE',flush=True)

if __name__=='__main__':main()
