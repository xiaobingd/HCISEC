"""Fixed quantile calibration: registration/calibration source-count factorial."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff, measure

ROOT=Path('/root')
OUT=ROOT/'full_run_10x40_raw_acc/threeword_expanded'

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    d=np.load(ROOT/'paired_raw_acc_private.npz')
    st=np.load(ROOT/'stats27_private.npz')
    assert np.array_equal(d['sample_id'],st['sample_id'])
    ids=d['sample_id']
    assert len(ids)==len(set(ids))==3200
    u,w,r=np.array([s.split('_') for s in ids],dtype=int).T
    assert np.array_equal(np.unique(u),np.arange(1,11))
    assert np.array_equal(np.unique(w),np.arange(40))
    assert np.array_equal(np.unique(r),np.arange(8))
    assert all(np.sum(u==us)==320 for us in range(1,11))
    fit=d['role']=='fit'
    devwords=np.array([8,9,18,19,24]);testwords=np.array([25,26,27,28,29])
    regword=np.isin(w,[20,21,22,23])
    assert (fit.astype(int)+regword+np.isin(w,devwords)+np.isin(w,testwords)).max()==1
    spectrum=d['spectrum'];amp=d['amplitude']
    hashes=[hashlib.sha256(np.ascontiguousarray(spectrum[i]).tobytes()+
             np.ascontiguousarray(amp[i]).tobytes()).digest() for i in range(len(ids))]
    exact_duplicates=len(hashes)-len(set(hashes))
    spec=spectrum.reshape(len(u),120,-1)
    features=np.column_stack([spec.mean(-1),spec.std(-1),
        *(a.mean(-1) for a in np.array_split(spec,4,axis=-1)),amp,st['features']])
    ck=torch.load(ROOT/'adapted_acc_private.pt',map_location='cpu',weights_only=True)
    x=torch.tensor((features-ck['mean'].numpy())/ck['scale'].numpy(),dtype=torch.float32)
    lookup={(int(us),int(word),int(rep)):i for i,(us,word,rep) in enumerate(zip(u,w,r))}
    order=np.random.default_rng(1123).permutation(5)
    def group(words,repeats,legacy=False):
        base=[(int(word),int(rep)) for rep in repeats for word in words[order]]
        starts=range(0,len(base),3) if legacy else range(0,len(base)-2,3)
        groups=[[base[(i+j)%len(base)] for j in range(3)] for i in starts]
        first,users=[],[]
        for us in range(1,11):
            for g in groups:
                assert len({word for word,rep in g})==3
                first.append([lookup[us,word,rep] for word,rep in g]);users.append(us)
        first=np.array(first);users=np.array(users)
        if not legacy:assert len(np.unique(first))==first.size
        return first,users,dict(source_available_per_user=len(base),
            source_used_per_user=len(np.unique(first))//10,trials_per_user=len(groups),
            source_reuse=legacy,unused_per_user=len(base)-len(np.unique(first))//10)
    cals={n:group(devwords,range(n)) for n in (4,8)}
    queries={'legacy_first4':group(testwords,range(4),True),
             'additional_last4':group(testwords,range(4,8)),
             'expanded_all8':group(testwords,range(8))}
    legacycal=group(devwords,range(4),True)
    for c in [legacycal]+list(cals.values()):
        for q in queries.values():assert not np.intersect1d(c[0],q[0]).size
    protocol=dict(dataset='same 10 users x40 words x8 independent nominal recordings; 3200 total IDs',
        scope='increase authentication source recordings only, not subjects, word coverage, sessions or encoder training',
        model='frozen CE encoders42/43/44, fixed train scaler, same RBF C10 balanced',
        registration_words=[20,21,22,23],registration_conditions=[24,32],
        calibration_words=devwords.tolist(),calibration_conditions=[20,40],
        query_words=testwords.tolist(),calibration_grouping={str(n):cals[n][2] for n in (4,8)},
        query_grouping={name:q[2] for name,q in queries.items()},
        rules=['strict5','quantile_gap'],quantile_rule='(genuine q10 + impostor q95)/2; fixed from previous development selection',
        grouping='same seed1123 word order, no repeated source record in primary groups; unused last1/2 fixed by index, no quality selection',
        baseline='exact previous 7-group calibration/query replay at registration24; legacy group has one source reuse',
        main_comparison='2x2 registration/calibration factorial on identical expanded_all8 queries',
        uncertainty='all seeds reuse same physical data; no independence-based CI; additional repeats may have appeared in old quality-retry diagnostics, not blind new collection',
        isolation='all fit/reg/cal/query words disjoint; all thresholds serialized before query metrics; no user filtering, Audio, retry, quality gating or query tuning',
        exact_spectrum_amplitude_duplicate_rows=exact_duplicates,
        source_audit='unique sample IDs, complete user/word/repetition grid and exact spectrum+amplitude hashes; not a fresh raw-waveform dedup audit')
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    print('INVENTORY',3200,'EXACT_FEATURE_DUPLICATES',exact_duplicates,flush=True)
    locks=[];scores_by={}
    with threadpool_limits(limits=2):
        for seed in (42,43,44):
            net=nn.Sequential(nn.Linear(features.shape[1],256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
            state=torch.load(ROOT/'full_run_10x40_raw_acc/strict_episodic_6x4'/f'CE_{seed}_private.pt',map_location='cpu',weights_only=True)
            net.load_state_dict(state['encoder'])
            with torch.inference_mode():v=F.normalize(net.eval()(x),dim=1).numpy()
            for nr in (6,8):
                reg=regword&(r<nr)
                assert reg.sum()==10*4*nr
                score=np.column_stack([SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==us).decision_function(v)
                                       for us in range(1,11)])
                scores_by[seed,nr]=score
                for nc,cal in [(4,cals[4]),(8,cals[8])]+([(0,legacycal)] if nr==6 else []):
                    cs=score[cal[0]].mean(1)
                    for us in range(1,11):
                        for rule in ('strict5','quantile_gap'):
                            tau=cutoff(cs[:,us-1],cal[1]==us,rule)
                            locks.append(dict(seed=seed,user=us,registration=nr*4,calibration=nc*5,
                                grouping='legacy' if nc==0 else 'nonoverlap',rule=rule,threshold=tau,
                                development=measure(cs[:,us-1],cal[1]==us,tau)))
                print('CALIBRATED',seed,nr*4,flush=True)
    (OUT/'thresholds_public.json').write_text(json.dumps(locks,indent=2,allow_nan=False))
    print('ALL_THRESHOLDS_LOCKED',flush=True)
    rows=[]
    for lock in locks:
        nr=lock['registration']//4
        for qname,q in queries.items():
            if lock['grouping']=='legacy' and qname!='legacy_first4':continue
            qs=scores_by[lock['seed'],nr][q[0]].mean(1)[:,lock['user']-1]
            rows.append(dict(**{k:lock[k] for k in ('seed','user','registration','calibration','grouping','rule')},
                query=qname,**measure(qs,q[1]==lock['user'],lock['threshold'])))
    old=json.loads((ROOT/'full_run_10x40_raw_acc/threeword_calibration/results_public.json').read_text())
    errors=[]
    for row in rows:
        if row['grouping']=='legacy':
            prior=next(v for v in old if v['normalization']=='raw' and v['rule']==row['rule']
                       and v['seed']==row['seed'] and v['user']==row['user'])
            errors += [abs(row[k]-prior[k]) for k in ('far','frr','hter','eer','auc')]
    assert max(errors)<1e-10
    assert len(rows)==780 and len(locks)==300
    (OUT/'results_public.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (OUT/'verification_public.json').write_text(json.dumps(dict(status='PASS',
        sample_id_count=len(ids),unique_sample_ids=len(set(ids)),exact_feature_duplicate_rows=exact_duplicates,
        old_baseline_max_error=max(errors),query_workpoints=len(rows),thresholds=len(locks),
        primary_group_source_reuse=False,word_split_disjoint=True,all_thresholds_locked_before_query=True),indent=2))
    print('EXPANSION_COMPLETE',flush=True)

if __name__=='__main__':main()

