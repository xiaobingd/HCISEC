"""Fixed authentication protocol for shared/private ablations, all folds/users."""
import itertools
import json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from calibrate_threeword import cutoff,measure
from prepare_phrase_auth import SHORT,ENGLISH,CHINESE
from run_strict_within_content import role_check,splits

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'conditional_correspondence/stage2'

def main():
    d=np.load(ROOT/'phrase_auth/frontend_private.npz');u,w,r=d['user'],d['word'],d['repeat']
    views=np.load(OUT/'views_private.npz')
    assert len(views.files)==12
    families={'english_word':SHORT,'english_long':ENGLISH,'chinese_long':CHINESE}
    combos=np.array(list(itertools.combinations(range(8),3)))
    pending=[];locks=[];roles=[]
    for family,prompts in families.items():
        for fold,(rw,cw,qw) in enumerate(splits(prompts)):
            role_check(w[d['fit']],[rw],[cw],qw)
            roles.append(dict(family=family,fold=fold,registration_prompt=rw,calibration_prompt=cw,query_prompts=qw))
    (OUT/'role_splits_public.json').write_text(json.dumps(roles,indent=2))
    with threadpool_limits(limits=2):
        for key in views.files:
            seed,method=key.split('_',1);seed=int(seed);v=views[key]
            assert v.shape[0]==3200 and np.isfinite(v).all()
            for family,prompts in families.items():
                pool=np.flatnonzero(np.isin(w,prompts));loc={int(ix):j for j,ix in enumerate(pool)}
                for rw in prompts:
                    reg=np.flatnonzero(w==rw)
                    scores=np.column_stack([SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==us).decision_function(v[pool]) for us in range(1,11)])
                    for fold,(regword,cw,qw) in enumerate(splits(prompts)):
                        if regword!=rw:continue
                        cal=np.stack([np.flatnonzero((u==us)&(w==cw))[np.argsort(r[(u==us)&(w==cw)])] for us in range(1,11)])
                        query=np.stack([np.stack([np.flatnonzero((u==us)&(w==wo))[np.argsort(r[(u==us)&(w==wo)])] for wo in qw],axis=1) for us in range(1,11)])
                        assert len(reg)==80 and cal.shape==(10,8) and query.shape==(10,8,3)
                        assert not np.intersect1d(reg,cal).size and not np.intersect1d(reg,query).size and not np.intersect1d(cal,query).size
                        for k in (1,3):
                            ci=cal.reshape(-1,1) if k==1 else cal[:,combos].reshape(-1,3)
                            qi=query.reshape(-1,1) if k==1 else query.reshape(-1,3)
                            cu=np.repeat(np.arange(1,11),8 if k==1 else 56)
                            qu=np.repeat(np.arange(1,11),24 if k==1 else 8)
                            assert np.unique(qi).size==qi.size and (u[qi]==qu[:,None]).all()
                            assert all(len(set(w[ix]))==k for ix in qi)
                            cs=scores[np.array([[loc[int(i)] for i in ix] for ix in ci])].mean(1)
                            qs=scores[np.array([[loc[int(i)] for i in ix] for ix in qi])].mean(1)
                            for us in range(1,11):
                                tau=cutoff(cs[:,us-1],cu==us,'quantile_gap')
                                cfg=dict(seed=seed,method=method,family=family,fold=fold,words=k,user=us)
                                locks.append(dict(**cfg,threshold=tau))
                                pending.append((cfg,qs[:,us-1],qu==us,tau))
                print('SPLIT_AUTH',key,family,flush=True)
    assert len(locks)==14400
    (OUT/'thresholds_private.json').write_text(json.dumps(locks,allow_nan=False))
    print('ALL_SPLIT_THRESHOLDS_LOCKED',flush=True)
    rows=[dict(**cfg,**measure(score,label,tau)) for cfg,score,label,tau in pending]
    (OUT/'fold_user_results_private.json').write_text(json.dumps(rows,allow_nan=False))
    fields=['far','frr','hter','eer','auc']
    assert all(all(k in row for k in fields) for row in rows)
    macros=[];users=[];seedusers=[]
    for method in sorted(set(row['method'] for row in rows)):
        for family in families:
            for k in (1,3):
                subset=[row for row in rows if row['method']==method and row['family']==family and row['words']==k]
                assert len(subset)==600
                macros.append(dict(method=method,family=family,words=k,**{f:float(np.mean([row[f] for row in subset])) for f in fields}))
                for us in range(1,11):
                    sub=[row for row in subset if row['user']==us];assert len(sub)==60
                    users.append(dict(method=method,family=family,words=k,user=us,**{f:float(np.mean([row[f] for row in sub])) for f in fields}))
                    for seed in (42,43,44):
                        ss=[row for row in sub if row['seed']==seed];assert len(ss)==20
                        seedusers.append(dict(method=method,family=family,words=k,user=us,seed=seed,**{f:float(np.mean([row[f] for row in ss])) for f in fields}))
    (OUT/'summary_public.json').write_text(json.dumps(dict(macro=macros,per_user=users),indent=2))
    (OUT/'per_seed_user_public.json').write_text(json.dumps(seedusers,indent=2))
    comparisons=[]
    for family in families:
        for k in (1,3):
            for control in ('A_ACC','C_wrong_repeat','D_wrong_user'):
                for metric in ('hter','eer','far','frr'):
                    def get(method):return np.array([next(v[metric] for v in users if v['method']==method and v['family']==family and v['words']==k and v['user']==us) for us in range(1,11)])
                    delta=get('B_correct')-get(control)
                    rng=np.random.default_rng(4244);boot=delta[rng.integers(0,10,size=(5000,10))].mean(1)
                    comparisons.append(dict(family=family,words=k,control=control,metric=metric,audio_minus_control=float(delta.mean()),
                        user_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),improved_users=int((delta<0).sum()),worsened_users=int((delta>0).sum())))
    (OUT/'comparisons_public.json').write_text(json.dumps(comparisons,indent=2))
    (OUT/'verification_public.json').write_text(json.dumps(dict(status='PASS',workpoints=14400,macro_rows=len(macros),per_user_rows=len(users),
        all_thresholds_locked_before_query=True,role_content_disjoint=True,all_users_and_attack_sources_retained=True,
        query_audio_used=False,query_source_reuse_within_condition=False,calibration_unique_recordings=8,
        calibration_three='56dependent combinations of same one prompt;query has3distinct prompts',
        uncertainty='bootstrap targets only,not repeated folds or trials;10known identities,exploratory reused cohort'),indent=2))
    print('SHARED_PRIVATE_AUTH_COMPLETE',flush=True)

if __name__=='__main__':main()
