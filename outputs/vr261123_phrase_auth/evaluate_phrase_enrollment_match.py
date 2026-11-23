"""Same Chinese queries/budgets, English versus Chinese enrollment/calibration."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from prepare_phrase_auth import SHORT,ENGLISH,CHINESE
from calibrate_threeword import cutoff,measure

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'phrase_auth'
DEST=OUT/'matched_enrollment'

def main():
    DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    d=np.load(OUT/'frontend_private.npz');u,w,r=d['user'],d['word'],d['repeat']
    x=torch.tensor(StandardScaler().fit(d['features'][d['fit']]).transform(d['features']),dtype=torch.float32)
    order=np.random.default_rng(1123).permutation(5)
    lookup={(int(us),int(word),int(rep)):i for i,(us,word,rep) in enumerate(zip(u,w,r))}
    def group(wordset,repeats,k):
        base=[(int(word),rep) for rep in repeats for word in np.array(wordset)[order]]
        gs=[base[i:i+k] for i in range(0,len(base)-k+1,k)]
        idx=np.array([[lookup[us,word,rep] for word,rep in g] for us in range(1,11) for g in gs])
        assert idx.size==len(np.unique(idx))
        return idx,np.repeat(np.arange(1,11),len(gs))
    protocol=dict(scope='exploratory enrollment-family diagnosis; same Chinese queries for all conditions',
        registration='15 records per user, first3repeats of5prompts',calibration='next2repeats,10available',
        query='same Chinese5prompts repeats5..7 for every enrollment condition',
        variable='English single-word, English long-prompt or Chinese same-prompt enrollment+calibration',
        models='same frozen fresh16-word encoders as primary; no retraining or query tuning',
        interpretation='registration/calibration language and content change together; this does not separate their individual causes')
    (DEST/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    locks=[];pending=[]
    with threadpool_limits(limits=2):
        for seed in (42,43,44):
            for method in ['CE','AudioKD','ShuffledAudioKD']:
                net=nn.Sequential(nn.Linear(762,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
                net.load_state_dict(torch.load(OUT/f'{method}_{seed}_private.pt',map_location='cpu',weights_only=True))
                with torch.inference_mode():v=F.normalize(net.eval()(x),dim=1).numpy()
                for enrollment,wordset in [('single_word',SHORT),('english_long',ENGLISH),('chinese_long',CHINESE)]:
                    reg=np.isin(w,wordset)&(r<3)
                    scores=np.column_stack([SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==us).decision_function(v)
                                            for us in range(1,11)])
                    for k in (1,3):
                        ci,cu=group(wordset,range(3,5),k);qi,qu=group(CHINESE,range(5,8),k)
                        assert not np.intersect1d(np.flatnonzero(reg),qi).size and not np.intersect1d(ci,qi).size
                        cs=scores[ci].mean(1);qs=scores[qi].mean(1)
                        for us in range(1,11):
                            tau=cutoff(cs[:,us-1],cu==us,'quantile_gap')
                            cfg=dict(seed=seed,method=method,words=k,user=us,enrollment=enrollment)
                            locks.append(dict(**cfg,threshold=tau))
                            pending.append((cfg,qs[:,us-1],qu==us,tau))
    (DEST/'thresholds_public.json').write_text(json.dumps(locks,indent=2,allow_nan=False))
    rows=[dict(**cfg,**measure(score,labels,tau)) for cfg,score,labels,tau in pending]
    assert len(rows)==540
    previous=json.loads((OUT/'fixed_content/results_public.json').read_text());err=[]
    for row in rows:
        if row['enrollment']=='chinese_long':
            old=next(v for v in previous if v['condition']=='chinese_long' and all(v[k]==row[k] for k in ['seed','method','words','user']))
            err += [abs(row[k]-old[k]) for k in ['far','frr','hter','eer','auc']]
    assert max(err)==0
    (DEST/'results_public.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (DEST/'verification_public.json').write_text(json.dumps(dict(status='PASS',same_query_recordings=True,
        same_registration_calibration_counts=True,chinese_reference_max_error=max(err),
        query_workpoints=540,all_thresholds_locked_before_query=True),indent=2))
    print('MATCHED_PHRASE_ENROLLMENT_COMPLETE',flush=True)

if __name__=='__main__':main()
