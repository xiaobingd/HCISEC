import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff,measure

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'conditional_correspondence/stage3'

def main():
    rows=json.loads((OUT/'fold_user_results_private.json').read_text());locks=json.loads((OUT/'thresholds_private.json').read_text())
    assert len(rows)==len(locks)==14400
    keys=['seed','method','family','fold','words','user']
    assert len({tuple(a[k] for k in keys) for a in rows})==14400
    for a in rows:
        assert a['genuine_trials']==(24 if a['words']==1 else 8) and a['attack_trials']==9*a['genuine_trials']
        assert abs(a['far']-a['false_accepts']/a['attack_trials'])<1e-12
        assert abs(a['frr']-a['false_rejects']/a['genuine_trials'])<1e-12
        assert abs(a['hter']-(a['far']+a['frr'])/2)<1e-12
    d=np.load(ROOT/'phrase_auth/frontend_private.npz');u,w=d['user'],d['word']
    v=np.load(OUT/'views_private.npz')['42_B_correct'];reg=np.flatnonzero(w==35)
    with threadpool_limits(limits=2):score=SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==1).decision_function(v)
    errors=[]
    for k in (1,3):
        ci=[];qi=[];cu=[];qu=[]
        for us in range(1,11):
            c=np.flatnonzero((u==us)&(w==36));q=np.stack([np.flatnonzero((u==us)&(w==wo)) for wo in (37,38,39)],1)
            ci.extend(c[:,None] if k==1 else c[list(itertools.combinations(range(8),3))]);qi.extend(q.reshape(-1,1) if k==1 else q)
            cu.extend([us]*(8 if k==1 else 56));qu.extend([us]*(24 if k==1 else 8))
        tau=cutoff(score[np.array(ci)].mean(1),np.array(cu)==1,'quantile_gap');m=measure(score[np.array(qi)].mean(1),np.array(qu)==1,tau)
        select=lambda a:a['seed']==42 and a['method']=='B_correct' and a['family']=='chinese_long' and a['fold']==0 and a['words']==k and a['user']==1
        row=next(a for a in rows if select(a));lock=next(a for a in locks if select(a))
        errors.append(abs(tau-lock['threshold']));errors.extend(abs(m[key]-row[key]) for key in m)
    assert max(errors)<1e-10
    ret=json.loads((OUT/'retrieval_public.json').read_text());assert len(ret)==2400
    for a in ret:assert (a['word'] in (8,18,24,30))==(a['scope']=='development_unseen_words')
    summary=[];userrows=[]
    for method in ('A_ACC','B_correct','C_wrong_repeat','D_wrong_user'):
        for scope in ('fit','development_unseen_words'):
            sub=[a for a in ret if a['method']==method and a['scope']==scope]
            summary.append(dict(method=method,scope=scope,top1=float(np.mean([a['top1'] for a in sub])),margin=float(np.mean([a['margin'] for a in sub]))))
            for us in range(1,11):
                s=[a for a in sub if a['user']==us];userrows.append(dict(method=method,scope=scope,user=us,top1=float(np.mean([a['top1'] for a in s])),margin=float(np.mean([a['margin'] for a in s]))))
    contrasts=[]
    for control in ('A_ACC','C_wrong_repeat','D_wrong_user'):
        def values(method):return np.array([next(a['top1'] for a in userrows if a['method']==method and a['scope']=='development_unseen_words' and a['user']==us) for us in range(1,11)])
        delta=values('B_correct')-values(control);rng=np.random.default_rng(4242)
        boot=delta[rng.integers(0,10,(5000,10))].mean(1)
        contrasts.append(dict(control=control,correct_audio_minus_control_top1=float(delta.mean()),user_cluster95=np.quantile(boot,[.025,.975]).tolist()))
    seedrows=[]
    for seed in (42,43,44):
        for method in ('A_ACC','B_correct','C_wrong_repeat','D_wrong_user'):
            sub=[a for a in ret if a['seed']==seed and a['method']==method and a['scope']=='development_unseen_words']
            seedrows.append(dict(seed=seed,method=method,top1=float(np.mean([a['top1'] for a in sub])),margin=float(np.mean([a['margin'] for a in sub]))))
    (OUT/'retrieval_summary_public.json').write_text(json.dumps(dict(macro=summary,per_user=userrows,development_contrasts=contrasts,development_per_seed=seedrows),indent=2))
    verification=json.loads((OUT/'verification_public.json').read_text());verification.update(independent_replay_max_error=max(errors),rate_count_checks=True,retrieval_groups=len(ret),physical_sync_verified=False)
    (OUT/'verification_public.json').write_text(json.dumps(verification,indent=2))
    sources=['train_crossword_mechanism.py','evaluate_crossword_mechanism.py','test_crossword_mechanism.py','verify_crossword_mechanism.py','report_crossword_mechanism.py',
        'train_audio_multiword_fitonly.py','calibrate_threeword.py','prepare_phrase_auth.py','run_strict_within_content.py']
    (OUT/'execution_manifest_public.json').write_text(json.dumps(dict(source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
        checkpoint_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*_private.pt'))}),indent=2))
    print('LOCAL_VERIFICATION_PASS',max(errors),flush=True)

if __name__=='__main__':main()
