"""Independent global-index authentication replay and aggregate consistency checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff,measure

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'shared_private'

def main():
    rows=json.loads((OUT/'fold_user_results_private.json').read_text())
    locks=json.loads((OUT/'thresholds_private.json').read_text())
    assert len(rows)==len(locks)==36000
    keys=['seed','method','family','fold','words','user']
    assert len({tuple(row[k] for k in keys) for row in rows})==36000
    for row in rows:
        assert row['genuine_trials']==(24 if row['words']==1 else 8)
        assert row['attack_trials']==9*row['genuine_trials']
        assert abs(row['far']-row['false_accepts']/row['attack_trials'])<1e-12
        assert abs(row['frr']-row['false_rejects']/row['genuine_trials'])<1e-12
        assert abs(row['hter']-(row['far']+row['frr'])/2)<1e-12
    d=np.load(ROOT/'phrase_auth/frontend_private.npz');views=np.load(OUT/'views_private.npz')
    diagnostics=[]
    for seed in (42,43,44):
        for method in ('SplitNoAudio','SplitAudio','SplitWrongAudio'):
            for scope,mask in [('fit',d['fit']),('held_english_word',np.isin(d['word'],[25,26,27,28,29])),
                ('held_english_long',np.isin(d['word'],[3,9,12,19,32])),('held_chinese_long',np.isin(d['word'],[35,36,37,38,39]))]:
                a=views[f'{seed}_{method}_shared'][mask].astype('float64');b=views[f'{seed}_{method}_private'][mask].astype('float64')
                a-=a.mean(0);b-=b.mean(0)
                cka=float(np.linalg.norm(a.T@b)**2/(np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)))
                def rank(z):
                    sv=np.linalg.svd(z,compute_uv=False)**2;prob=sv/sv.sum()
                    return float(np.exp(-(prob*np.log(np.maximum(prob,1e-30))).sum()))
                diagnostics.append(dict(seed=seed,method=method,scope=scope,linear_cka=cka,shared_effective_rank=rank(a),private_effective_rank=rank(b)))
    (OUT/'branch_diagnostics_public.json').write_text(json.dumps(diagnostics,indent=2))
    u,w=d['user'],d['word'];v=views['42_SplitAudio_fused']
    # First Chinese fold:35 enrollment,36 calibration,37/38/39 query.
    train=np.flatnonzero(w==35);cal=np.flatnonzero(w==36)
    assert len(train)==80 and len(cal)==80
    with threadpool_limits(limits=2):
        clf=SVC(C=10,class_weight='balanced').fit(v[train],u[train]==1)
        fullscore=clf.decision_function(v)
    import itertools
    triples=list(itertools.combinations(range(8),3))
    errors=[]
    for k in (1,3):
        ci=[];qi=[];cu=[];qu=[]
        for us in range(1,11):
            c=np.flatnonzero((u==us)&(w==36))
            q=np.stack([np.flatnonzero((u==us)&(w==wo)) for wo in (37,38,39)],1)
            ci.extend(c[:,None] if k==1 else c[triples]);cu.extend([us]*(8 if k==1 else 56))
            qi.extend(q.reshape(-1,1) if k==1 else q);qu.extend([us]*(24 if k==1 else 8))
        sc=fullscore[np.array(ci)].mean(1);sq=fullscore[np.array(qi)].mean(1)
        tau=cutoff(sc,np.array(cu)==1,'quantile_gap');m=measure(sq,np.array(qu)==1,tau)
        row=next(row for row in rows if row['seed']==42 and row['method']=='SplitAudio_fused' and row['family']=='chinese_long' and row['fold']==0 and row['user']==1 and row['words']==k)
        lock=next(row for row in locks if row['seed']==42 and row['method']=='SplitAudio_fused' and row['family']=='chinese_long' and row['fold']==0 and row['user']==1 and row['words']==k)
        errors.append(abs(tau-lock['threshold']));errors.extend(abs(m[key]-row[key]) for key in m)
    assert max(errors)<1e-10
    state=json.loads((OUT/'verification_public.json').read_text());state.update(independent_global_index_replay_max_error=max(errors),rate_count_checks=True,unique_workpoints=36000)
    (OUT/'verification_public.json').write_text(json.dumps(state,indent=2))
    sources=['train_shared_private.py','evaluate_shared_private.py','check_shared_private.py','verify_shared_private.py','report_shared_private.py',
        'train_audio_multiword_fitonly.py','calibrate_threeword.py','prepare_phrase_auth.py','run_strict_within_content.py']
    private=[ROOT/'phrase_auth/fresh_warm_private.pt',ROOT/'phrase_auth/fresh_fitonly_teacher_private.pt']+sorted(OUT.glob('*_private.pt'))
    manifest=dict(source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        checkpoint_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in private},
        official_audio_backbone_sha256=json.loads((ROOT/'audio_multiword_fitonly/extraction_progress_private.json').read_text())['official_model_sha256'],
        environment='Windows CPU; Python3.10; torch2.5.0; NumPy1.23.4; sklearn1.4.2;2threads')
    (OUT/'execution_manifest_public.json').write_text(json.dumps(manifest,indent=2))
    print('SHARED_PRIVATE_VERIFIED:36000rates/counts;independent replay error',max(errors))

if __name__=='__main__':main()
