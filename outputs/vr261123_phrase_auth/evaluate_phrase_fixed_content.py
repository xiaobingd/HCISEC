"""Secondary text-dependent enrollment: independent repeats of long prompts."""
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
DEST=OUT/'fixed_content'

def main():
    DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    d=np.load(OUT/'frontend_private.npz');u,w,r=d['user'],d['word'],d['repeat']
    x=torch.tensor(StandardScaler().fit(d['features'][d['fit']]).transform(d['features']),dtype=torch.float32)
    order=np.random.default_rng(1123).permutation(5)
    ix={(int(us),int(word),int(rep)):i for i,(us,word,rep) in enumerate(zip(u,w,r))}
    def group(words,repeats,k):
        base=[(int(word),int(rep)) for rep in repeats for word in np.array(words)[order]]
        gs=[base[i:i+k] for i in range(0,len(base)-k+1,k)]
        idx=np.array([[ix[us,word,rep] for word,rep in g] for us in range(1,11) for g in gs])
        users=np.repeat(np.arange(1,11),len(gs))
        assert np.unique(idx).size==idx.size
        return idx,users
    views={};locks=[];pending=[]
    protocol=dict(scope='secondary closed same-content authentication; all shared encoders and teacher excluded all query prompt classes',
        registration='same5prompt classes, repeats0..2,15peruser; shared encoder remains frozen',
        calibration='same5prompts repeats3..4,10availableperuser',
        query='same5prompts repeats5..7,15availableperuser',
        roles='recording indices disjoint; prompt classes overlap by design, not cross-word authentication',
        challenges='one complete utterance or3distinct prompt classes; no source-ID reuse within condition',
        backend='same RBF C10 balanced, same prior q10/q95 midpoint rule, no query tuning',
        comparisons='English single/English long/Chinese long share15/10/15 budget; do not attribute differences against primary24/40/40 protocol only to content overlap',
        methods=['CE','AudioKD','ShuffledAudioKD'],seeds=[42,43,44],no_extra_encoder_training=True)
    (DEST/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    with threadpool_limits(limits=2):
        for seed in (42,43,44):
            for method in protocol['methods']:
                net=nn.Sequential(nn.Linear(762,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
                net.load_state_dict(torch.load(OUT/f'{method}_{seed}_private.pt',map_location='cpu',weights_only=True))
                with torch.inference_mode():v=F.normalize(net.eval()(x),dim=1).numpy()
                for name,words in [('single_word',SHORT),('english_long',ENGLISH),('chinese_long',CHINESE)]:
                    reg=np.isin(w,words)&(r<3)
                    score=np.column_stack([SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==us).decision_function(v)
                                           for us in range(1,11)])
                    for k in (1,3):
                        ci,cu=group(words,range(3,5),k);qi,qu=group(words,range(5,8),k)
                        assert not np.intersect1d(np.flatnonzero(reg),ci).size
                        assert not np.intersect1d(np.flatnonzero(reg),qi).size
                        assert not np.intersect1d(ci,qi).size
                        cs=score[ci].mean(1);qs=score[qi].mean(1)
                        for us in range(1,11):
                            tau=cutoff(cs[:,us-1],cu==us,'quantile_gap')
                            cfg=dict(seed=seed,method=method,condition=name,words=k,user=us)
                            locks.append(dict(**cfg,threshold=tau,development=measure(cs[:,us-1],cu==us,tau)))
                            pending.append((cfg,qs[:,us-1],qu==us,tau))
    (DEST/'thresholds_public.json').write_text(json.dumps(locks,indent=2,allow_nan=False))
    rows=[dict(**cfg,**measure(score,labels,tau)) for cfg,score,labels,tau in pending]
    assert len(rows)==len(locks)==540
    (DEST/'results_public.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (DEST/'verification_public.json').write_text(json.dumps(dict(status='PASS',query_workpoints=540,
        source_role_indices_disjoint=True,prompt_classes_overlap=True,no_new_encoder_training=True,
        all_thresholds_locked_before_query=True,all_users_retained=True),indent=2))
    print('FIXED_PHRASE_COMPLETE',flush=True)

if __name__=='__main__':main()
