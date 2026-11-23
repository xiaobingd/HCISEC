"""Within-language prompt-disjoint authentication on all 20 enrollment/calibration splits."""
import hashlib
import itertools
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
from prepare_phrase_auth import SHORT,ENGLISH,CHINESE

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'phrase_auth'
OUT=ROOT/'strict_within_content'

def splits(prompts):
    assert len(set(prompts))==5
    return [(int(reg),int(cal),tuple(int(v) for v in prompts if v not in (reg,cal)))
            for reg in prompts for cal in prompts if cal!=reg]

def role_check(train_words,registration,calibration,query):
    sets=[set(train_words),set(registration),set(calibration),set(query)]
    for i in range(4):
        for j in range(i):
            if sets[i]&sets[j]:raise ValueError('Content overlap between roles')

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    d=np.load(BASE/'frontend_private.npz');u,w,r=d['user'],d['word'],d['repeat']
    oldprotocol=json.loads((BASE/'protocol_public.json').read_text())
    fit=d['fit'];assert fit.sum()==1280 and set(w[fit])==set(oldprotocol['train_words'])
    families={'english_word':SHORT,'english_long':ENGLISH,'chinese_long':CHINESE}
    assert not set(w[fit])&set(SHORT+ENGLISH+CHINESE)
    x=torch.tensor(StandardScaler().fit(d['features'][fit]).transform(d['features']),dtype=torch.float32)
    digest=hashlib.sha256(d['sample_id'].tobytes()).hexdigest()
    combos=np.array(list(itertools.combinations(range(8),3)))
    assert combos.shape==(56,3) and all(len(set(v))==3 for v in combos)
    protocol=dict(scope='10 known identities; within-language strict unseen-content authentication',
        families=families,encoder='reuse fresh16fitword-only CE/AudioKD/shuffled encoders; no new shared training',
        role_split='one prompt enrollment,one different prompt calibration,three other prompts query;all20ordered pairs equally reported',
        registration='all8recordings/person of enrollment prompt',
        calibration_single='all8recordings/person of calibration prompt',
        calibration_three='all56combinations of3different recordings of the ONE calibration prompt; only8unique sources/person,dependent calibration triplets',
        query_single='three held-out prompts x8recordings =24 trials/person',
        query_three='3distinct held-out prompts,one recording each at same repetition =8trials/person;24unique sources,none repeated',
        threshold='fixed prior (genuine q10 + impostor q95)/2;only current calibration prompt;all locks serialized before query metrics',
        backend='same RBF C10 balanced,all9attack sources,ACC only',
        sampling='all20rolesplits,no favorable fold/user/prompt selection',
        seeds=[42,43,44],methods=['CE','AudioKD','ShuffledAudioKD'],
        limitation='5prompt classes cannot provide1enrollment+3distinct calibration+3distinct query classes;calibration diversity differs from3prompt query',
        uncertainty='folds,seeds andchallenge-length comparisons reuse underlying data;not20independentdatasets ornewblind/open-set evidence',
        no_comparison_claim='previous same-content protocol had15enrollment/10calibration anddifferentqueries;absolute gaps are not a budget-matched causal effect',
        frontend_id_digest=digest)
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    locks=[];pending=[];checkpoints=[];splitrecords=[]
    for family,prompts in families.items():
        for fold,(regword,calword,querywords) in enumerate(splits(prompts)):
            role_check(w[fit],[regword],[calword],querywords)
            splitrecords.append(dict(family=family,fold=fold,registration_prompt=regword,
                                     calibration_prompt=calword,query_prompts=querywords))
    with threadpool_limits(limits=2):
        for seed in (42,43,44):
            for method in protocol['methods']:
                cp=BASE/f'{method}_{seed}_private.pt'
                checkpoints.append(dict(seed=seed,method=method,sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
                net=nn.Sequential(nn.Linear(762,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
                net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
                with torch.inference_mode():v=F.normalize(net.eval()(x),dim=1).numpy()
                for family,prompts in families.items():
                    pool=np.flatnonzero(np.isin(w,prompts));assert len(pool)==400
                    loc={int(index):j for j,index in enumerate(pool)}
                    # One enrollment head is reused across its four independent calibration choices.
                    for regword in prompts:
                        reg=np.flatnonzero(w==regword);assert len(reg)==80
                        scores=np.column_stack([SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==us).decision_function(v[pool])
                                                for us in range(1,11)])
                        for fold,(rw,cw,qw) in enumerate(splits(prompts)):
                            if rw!=regword:continue
                            cal=np.stack([np.flatnonzero((u==us)&(w==cw))[np.argsort(r[(u==us)&(w==cw)])]
                                          for us in range(1,11)])
                            query=np.stack([np.stack([np.flatnonzero((u==us)&(w==word))[np.argsort(r[(u==us)&(w==word)])]
                                                      for word in qw],axis=1) for us in range(1,11)])
                            assert cal.shape==(10,8) and query.shape==(10,8,3)
                            assert np.unique(query).size==query.size
                            assert not np.intersect1d(reg,cal).size and not np.intersect1d(reg,query).size and not np.intersect1d(cal,query).size
                            for k in (1,3):
                                ci=cal.reshape(-1,1) if k==1 else cal[:,combos].reshape(-1,3)
                                qi=query.reshape(-1,1) if k==1 else query.reshape(-1,3)
                                cu=np.repeat(np.arange(1,11),8 if k==1 else 56)
                                qu=np.repeat(np.arange(1,11),24 if k==1 else 8)
                                assert (u[ci]==cu[:,None]).all() and (u[qi]==qu[:,None]).all()
                                assert all(len(set(w[z]))==(1 if k==1 else 3) for z in qi)
                                cs=scores[np.array([[loc[int(i)] for i in z] for z in ci])].mean(1)
                                qs=scores[np.array([[loc[int(i)] for i in z] for z in qi])].mean(1)
                                for us in range(1,11):
                                    tau=cutoff(cs[:,us-1],cu==us,'quantile_gap')
                                    cfg=dict(seed=seed,method=method,family=family,fold=fold,words=k,user=us)
                                    locks.append(dict(**cfg,threshold=tau,development=measure(cs[:,us-1],cu==us,tau)))
                                    pending.append((cfg,qs[:,us-1],qu==us,tau))
                    print('WITHIN_FAMILY_DONE',seed,method,family,flush=True)
    assert len(locks)==10800
    (OUT/'thresholds_private.json').write_text(json.dumps(locks,allow_nan=False))
    (OUT/'split_protocol_public.json').write_text(json.dumps(splitrecords,indent=2))
    print('WITHIN_THRESHOLDS_LOCKED',flush=True)
    rows=[dict(**cfg,**measure(scores,labels,tau)) for cfg,scores,labels,tau in pending]
    assert len(rows)==10800
    (OUT/'fold_user_results_private.json').write_text(json.dumps(rows,allow_nan=False))
    (OUT/'verification_public.json').write_text(json.dumps(dict(status='PASS',workpoints=len(rows),
        families=3,ordered_prompt_role_splits_per_family=20,role_content_disjoint=True,
        shared_training_prompt_overlap=0,all9attack_sources=True,query_source_id_reuse=False,
        calibration_unique_sources_per_user=8,calibration_triplets_independent=False,
        all_thresholds_locked_before_query=True,checkpoints=checkpoints),indent=2))
    print('STRICT_WITHIN_CONTENT_COMPLETE',flush=True)

if __name__=='__main__':main()
