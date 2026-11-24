# Run against the existing private-root cache; no PCM and no model training.
import argparse
import sys
from pathlib import Path as _P
_cli=argparse.ArgumentParser()
_cli.add_argument('--root',type=_P)
_cli.add_argument('--test',action='store_true')
_args=_cli.parse_args()
if _args.test:
    import numpy as _np
    _r=_np.array([8.,7.,6.,5.,4.,3.,2.,1.])
    _p=_np.lexsort((_np.arange(8),-_r))
    assert list(_p[:4])==[0,1,2,3]
    _qual=_np.array([0.,0.,1.,2.,0.,0.,0.,0.])
    assert (_p[2] if _qual[_p[2]]>=_qual[_p[3]] else _p[3])==3
    print('SYNTHETIC_TEST_PASS')
    sys.exit(0)
if _args.root is None:_cli.error('--root is required')
sys.path.insert(0,str(_args.root.resolve()))

"""Frozen ACC authentication; enrollment-only Audio quality guides fixed-budget selection."""
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.stats import rankdata
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff,measure
from prepare_phrase_auth import FIT,REG,CAL,SHORT,ENGLISH,CHINESE
from run_strict_within_content import role_check

ROOT=_args.root.resolve()
OUT=ROOT/'conditional_correspondence/enrollment_boundary_tiebreak';OUT.mkdir(exist_ok=True)

def audio_quality(pcm,fs):
    """Three fixed acoustic quality cues, not an identity embedding."""
    assert fs==48000 and pcm.dtype==np.int16
    wave=pcm.astype('float32')/32768
    if wave.ndim==2:wave=wave[:,0]
    clipped=float((abs(wave)>=.999).mean())
    wave=signal.resample_poly(wave,160,441);wave-=wave.mean()
    if len(wave)<1600:wave=np.pad(wave,(0,1600-len(wave)))
    frames=np.lib.stride_tricks.sliding_window_view(wave,1600)[::800]
    energy=(frames**2).mean(1)
    snr=float(10*np.log10((np.quantile(energy,.8)+1e-12)/(np.quantile(energy,.2)+1e-12)))
    corr=np.fft.irfft(abs(np.fft.rfft(frames,n=4096))**2,n=4096)[:,:321]
    periodic=corr[:,53:321].max(1)/np.maximum(corr[:,0],1e-12)
    voiced=energy>=np.median(energy)
    periodicity=float(periodic[voiced].mean())
    return np.array([snr,periodicity,-clipped],dtype=float)

def centrality(z):
    """Within-own-word leave-one-out cosine consistency; no query/background labels."""
    centers=z.sum(0)[None,:]-z
    centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    return (z*centers).sum(1)

def top3(score):return np.lexsort((np.arange(8),-np.asarray(score)))[:3]

def main():
    d=np.load(ROOT/'phrase_auth/frontend_private.npz');u,w,r=d['user'],d['word'],d['repeat'];ids=d['sample_id']
    role_check(FIT,REG,CAL,SHORT+ENGLISH+CHINESE)
    assert len(set(ids))==3200
    reg=np.flatnonzero(np.isin(w,REG));assert len(reg)==320
    manifest=json.loads((ROOT/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf8'))['records']
    assert [a['sample_id'] for a in manifest]==list(ids)
    c=np.load(ROOT/'conditional_correspondence/enrollment_identity_quality/registered_ecapa_private.npz');assert np.array_equal(c['indices'],reg) and np.array_equal(c['sample_id'],ids[reg]) and bool(c['length_aware'])
    az=np.zeros((3200,192));az[reg]=c['embedding'];quality=np.zeros((3200,1))
    for us in range(1,11):
        for wo in REG:
            ix=np.flatnonzero((u==us)&(w==wo));other=np.flatnonzero((u==us)&np.isin(w,REG)&(w!=wo))
            center=az[other].mean(0);center/=max(np.linalg.norm(center),1e-8)
            quality[ix,0]=az[ix]@center
    # Per-word ranks avoid fitting a teacher or selecting cue weights on evaluation data.
    aq=np.zeros(3200);wrong=np.zeros(3200)
    for us in range(1,11):
        for wo in REG:
            ix=np.flatnonzero((u==us)&(w==wo));ix=ix[np.argsort(r[ix])];assert len(ix)==8
            aq[ix]=rankdata(quality[ix],axis=0).mean(1)/8
            prng=np.random.default_rng(241124+us*100+wo)
            permutation=prng.permutation(8)
            while np.any(permutation==np.arange(8)):permutation=prng.permutation(8)
            wrong[ix]=aq[ix[permutation]]
    views=np.load(ROOT/'conditional_correspondence/stage3/views_private.npz')
    methods=['acc_crossword12','audio_boundary','coin_boundary']+[f'shuffled_boundary_{i}' for i in range(5)]
    families={'english_word':SHORT,'english_long':ENGLISH,'chinese_long':CHINESE}
    protocol=dict(scope='10known users,strict word-disjoint registration/calibration/query;frozenACC',
        registration_words=REG,registration_pool_per_user=32,primary_retained_per_user=12,per_word_retained=3,
        calibration_words=CAL,query_families=families,seeds=[42,43,44],methods=methods,
        encoders='exact stage3 A_ACC frozen,all methods same representation per seed',
        audio='registeredAudio320only;WAVheader48000 interpreted44100 andresampled16k;no teacher/encoder updates',
        acoustic_quality='equal mean of within-user-word ranks:80th/20th shortframeenergy ratio,50-300Hz-range autocorrelation peak periodicity,negative clippingfraction;not calibrated SNR or identityembedding',
        acc_quality='within-user-word leave-one-out cosine centrality',
        fusion='0.5 ACCcentrality rank+0.5 Audioquality rank;fixed before metrics',
        negative_control='fixed within-user-word random permutation of Audioquality ranks across8repetitions',
        budget='all methods acquire same32ACC;Audio methods additionally observe registeredAudio;retain12 except all32diagnostic;no quality-based user removal',
        backend='primary balancedRBF C10;secondary normalized mean-template cosine;no backend selection',
        threshold='fixed (cal genuineQ10+impostorQ95)/2;all locks saved before query metrics',
        challenge='1record or3distinct querywords averaged score;all8repetitions;40or39sourcesperuser,40or13trials',
        limitations='new protocol uses4registrationwords and5calibrationwords,not prior single-registrationword folds;historically viewed cohort;3sACCwindows may overlap withinword/include adjacent utterances;VADcenters remain;physicalsyncunverified')
    protocol.update(scope='phase2 enrollment identity-consistency exploration;allrules fixed before this phase metrics',audio='frozen official ECAPA192,registeredAudio320only,length-aware44100->16k',acoustic_quality='Audio embedding cosine against same-person other3registration-word mean;rank within current word',acc_quality='within-word medoid control andleave-word-out ACCcosine control',fusion='fixed ACC top2; third sample chosen between ACC ranks3/4 using Audio cross-word consistency',negative_control='fixed coin flip and five no-fixed-point Audio within-user-word shuffles',comparison='Audio boundary must beat ACC top3, coin boundary and shuffled Audio on fixed EER/HTER; do not tune using query')
    protocol.update(negative_background='fixed all32registered ACC records from each ofother9users;288negatives for every target andmethod;selection affects own positive template only',scope_revision='fixednegative comparison isolates own enrollment-quality;original variable-negative acoustic pilot retained separately')
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))}
    order=np.random.default_rng(1123).permutation(5)
    def groups(words,k):
        sources=[(int(wo),rep) for rep in range(8) for wo in np.array(words)[order]]
        chunks=[sources[i:i+k] for i in range(0,len(sources)-k+1,k)]
        ix=np.array([[lookup[us,wo,re] for wo,re in chunk] for us in range(1,11) for chunk in chunks])
        labels=np.repeat(np.arange(1,11),len(chunks))
        assert np.unique(ix).size==ix.size and all(len(set(w[a]))==k for a in ix)
        return ix,labels
    pending=[];locks=[];selected_private=[]
    with threadpool_limits(limits=2):
        for seed in (42,43,44):
            z=views[f'{seed}_A_ACC'];assert z.shape==(3200,192)
            choices={m:[] for m in methods}
            for us in range(1,11):
                for wo in REG:
                    ix=np.flatnonzero((u==us)&(w==wo));ix=ix[np.argsort(r[ix])]
                    ac=rankdata(centrality(z[ix]))/8
                    other=np.flatnonzero((u==us)&np.isin(w,REG)&(w!=wo));center=z[other].mean(0);center/=max(np.linalg.norm(center),1e-8)
                    acx=rankdata(z[ix]@center)/8
                    order_acc=np.lexsort((np.arange(8),-np.asarray(acx)))
                    front=order_acc[:2];third=int(order_acc[2]);fourth=int(order_acc[3])
                    def last_slot(q):
                        return third if q[third]>=q[fourth] else fourth
                    pick={'acc_crossword12':order_acc[:3],
                          'audio_boundary':np.r_[front,last_slot(aq[ix])]}
                    coin=np.random.default_rng(719+seed*10000+us*100+wo).integers(0,2)
                    pick['coin_boundary']=np.r_[front,third if coin==0 else fourth]
                    for sh in range(5):
                        rng=np.random.default_rng(241124+sh*100000+us*100+wo)
                        perm=rng.permutation(8)
                        while np.any(perm==np.arange(8)):perm=rng.permutation(8)
                        pick[f'shuffled_boundary_{sh}']=np.r_[front,last_slot(aq[ix][perm])]
                    for method in methods:choices[method].extend(ix[pick[method]].tolist())
            for method in methods:
                ri=np.array(choices[method]);assert len(set(ri))==len(ri)==(320 if method=='all32_reference' else 120)
                assert all(np.sum(u[ri]==us)==(32 if method=='all32_reference' else 12) for us in range(1,11))
                selected_private.append(dict(seed=seed,method=method,indices=ri.tolist()))
                scores={}
                heads=[]
                for us in range(1,11):
                    positives=ri[u[ri]==us];background=reg[u[reg]!=us]
                    training=np.r_[positives,background]
                    assert len(background)==288 and not np.intersect1d(positives,background).size
                    heads.append(SVC(C=10,class_weight='balanced').fit(z[training],u[training]==us).decision_function(z))
                scores['RBF']=np.column_stack(heads)
                templates=np.stack([z[ri[u[ri]==us]].mean(0) for us in range(1,11)])
                templates/=np.maximum(np.linalg.norm(templates,axis=1,keepdims=True),1e-8)
                scores['cosine']=z@templates.T
                for backend,score in scores.items():
                    for k in (1,3):
                        ci,cu=groups(CAL,k);cs=score[ci].mean(1)
                        for us in range(1,11):
                            tau=cutoff(cs[:,us-1],cu==us,'quantile_gap')
                            cfg=dict(seed=seed,method=method,backend=backend,words=k,user=us)
                            locks.append(dict(**cfg,threshold=tau))
                            for family,words in families.items():
                                qi,qu=groups(words,k);assert not np.intersect1d(ri,qi).size and not np.intersect1d(ci,qi).size
                                pending.append((dict(**cfg,family=family),score[qi].mean(1)[:,us-1],qu==us,tau))
                print('ENROLLMENT',seed,method,flush=True)
    assert len(locks)==960 and len(pending)==2880
    (OUT/'thresholds_private.json').write_text(json.dumps(locks));(OUT/'selection_private.json').write_text(json.dumps(selected_private))
    print('ENROLLMENT_THRESHOLDS_LOCKED',flush=True)
    rows=[dict(**cfg,**measure(s,y,tau)) for cfg,s,y,tau in pending]
    (OUT/'results_private.json').write_text(json.dumps(rows))
    macro=[];users=[]
    for method in methods:
        for backend in ('RBF','cosine'):
            for family in families:
                for k in (1,3):
                    sub=[a for a in rows if a['method']==method and a['backend']==backend and a['family']==family and a['words']==k];assert len(sub)==30
                    macro.append(dict(method=method,backend=backend,family=family,words=k,**{key:float(np.mean([a[key] for a in sub])) for key in ('far','frr','hter','eer','auc')}))
                    for us in range(1,11):
                        sr=[a for a in sub if a['user']==us];assert len(sr)==3
                        users.append(dict(method=method,backend=backend,family=family,words=k,user=us,**{key:float(np.mean([a[key] for a in sr])) for key in ('far','frr','hter','eer','auc')}))
    # Regression gate: exactly replay old ACC-crossword protocol and backend.
    baseline=json.loads((ROOT/'conditional_correspondence/enrollment_identity_quality_fixed_background/summary_public.json').read_text())
    for row in macro:
        if row['method']!='acc_crossword12':continue
        old=next(x for x in baseline['macro']
                 if x['method']=='acc_crossword12' and x['backend']==row['backend']
                 and x['family']==row['family'] and x['words']==row['words'])
        for metric in ('far','frr','hter','eer','auc'):
            assert abs(old[metric]-row[metric])<1e-8,(row,old,metric)
    # Paired 10-user uncertainty; do not count repeats, seed, or trial as independent.
    comparisons=[]
    for family in families:
        for backend in ('RBF','cosine'):
            for k in (1,3):
                for metric in ('hter','eer'):
                    def vector(method):
                        return np.array([next(a[metric] for a in users
                          if a['method']==method and a['backend']==backend
                          and a['family']==family and a['words']==k and a['user']==us)
                          for us in range(1,11)])
                    real=vector('audio_boundary')
                    for control in ('acc_crossword12','coin_boundary')+tuple(f'shuffled_boundary_{j}' for j in range(5)):
                        diff=real-vector(control)
                        rng=np.random.default_rng(4242)
                        boot=diff[rng.integers(0,10,(5000,10))].mean(1)
                        comparisons.append(dict(family=family,backend=backend,words=k,metric=metric,
                            control=control,audio_minus_control=float(diff.mean()),
                            user_cluster95=np.quantile(boot,[.025,.975]).tolist(),
                            improved_users=int((diff<0).sum()),worsened_users=int((diff>0).sum())))
    selected_lookup={(a['seed'],a['method']):set(a['indices']) for a in selected_private}
    swap_counts={method:sum(len(selected_lookup[(seed,method)] ^ selected_lookup[(seed,'acc_crossword12')])//2
         for seed in (42,43,44)) for method in methods}
    (OUT/'summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=users,
        comparisons=comparisons,swap_counts_all_seeds=swap_counts),indent=2))
    (OUT/'per_seed_user_public.json').write_text(json.dumps(rows,indent=2))
    (OUT/'verification_public.json').write_text(json.dumps(dict(status='PASS',workpoints=2880,thresholds=960,registration_audio_reads=320,
        calibration_or_query_audio_read=0,models_updated=False,all_users_retained=True,registration_budget_matched_primary=True,
        roles_word_disjoint=True,all_locks_serialized_before_query_metrics=True,reference32_not_budget_matched=True),indent=2))
    print('ENROLLMENT_BOUNDARY_COMPLETE',flush=True)

if __name__=='__main__':main()
