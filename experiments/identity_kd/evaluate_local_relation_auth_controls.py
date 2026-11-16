"""ACC-only authentication, locked Cal thresholds, all users and all three seeds."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from evaluate_crossword_encoder_distill import metric,threshold,norm

OUT=Path('outputs/mei_local_relation_auth_controls_20261117')
KEYS=['eer','hter','tar','far','frr','auc','hard_identity_margin','worst_sample_margin','raw_hard_identity_margin']
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    protocol=json.loads((OUT/'protocol_public.json').read_text());assert protocol['state']=='all12_checkpoints_locked'
    sources=json.loads((OUT/'training_summary_public.json').read_text());assert len(sources)==12
    s=np.load(OUT/'split_normalization_private.npz');user,word=s['user'],s['word'];cal=s['cal_local'];val=s['val_local'];fit=s['fit_local']
    assert not set(cal)&set(fit) and not set(val)&set(fit) and not set(cal)&set(val)
    lookup={(u,w):np.flatnonzero((user==u)&(word==w)) for u in range(1,10) for w in range(18)}
    def reg(w,f,u):return lookup[u,w][f*4:f*4+4]
    def query(w,f,calibration=False):
        return np.concatenate([lookup[u,w][6:8] if calibration else lookup[u,w][(1-f)*4:(1-f)*4+4] for u in range(1,10)])
    rows=[];private={};private_locks={};point_count=0
    for record in sources:
        seed,method=record['seed'],record['method'];prefix=f'{method}_seed{seed}'
        assert hashlib.sha256((OUT/f'{prefix}_private.pt').read_bytes()).hexdigest()==record['checkpoint_sha256']
        z=np.load(OUT/f'{prefix}_features_private.npz')['embedding'];assert z.shape==(1296,128)
        for backend in ['rbf','cos']:
            models={};cache={}
            def score(owner,rw,qw,fold,calibration=False):
                key=owner,rw,qw,fold,calibration
                if key not in cache:
                    rr=reg(rw,fold,owner);qq=query(qw,fold,calibration)
                    assert not set(rr)&set(qq)
                    if calibration:assert set(qq).issubset(set(cal))
                    else:assert set(qq).issubset(set(val))
                    if backend=='cos':cache[key]=z[qq]@norm(z[rr].mean(0))
                    else:
                        mk=owner,rw,fold
                        if mk not in models:
                            train=np.concatenate([reg(rw,fold,u) for u in range(1,10)])
                            models[mk]=SVC(C=10,gamma='scale',class_weight='balanced').fit(z[train],user[train]==owner)
                        cache[key]=models[mk].decision_function(z[qq])
                return cache[key]
            for owner in range(1,10):
                yy=[];ss=[]
                for qw in range(12):
                    y=user[query(qw,0,True)]==owner
                    for rw in range(12):
                        if rw==qw:continue
                        yy.extend([y,y]);ss.extend([score(owner,qw,qw,0,True),score(owner,rw,qw,0,True)])
                ycal=np.concatenate(yy);scal=np.concatenate(ss);tau=threshold(ycal,scal)
                negmean=float(scal[~ycal].mean());negstd=max(float(scal[~ycal].std()),1e-6)
                key=f'{prefix}_{backend}_U{owner}';private[key+'_calibration']=scal
                private_locks[key]=dict(threshold=tau,impostor_mean=negmean,impostor_std=negstd)
                bycondition={c:dict(labels=[],scores=[],margin=[],worst=[],raw=[]) for c in ['sameword','crossword']}
                for fold in [0,1]:
                    for qw in range(12,18):
                        qq=query(qw,fold);y=user[qq]==owner
                        for rw in range(12,18):
                            if rw==qw:continue
                            for condition,registration in [('sameword',qw),('crossword',rw)]:
                                value=score(owner,registration,qw,fold)
                                other=[value[user[qq]==u].mean() for u in range(1,10) if u!=owner]
                                margin=float(value[y].mean()-max(other));worst=float(value[y].min()-value[~y].max())
                                box=bycondition[condition];box['labels'].append(y);box['scores'].append(value)
                                box['margin'].append(margin/negstd);box['worst'].append(worst/negstd);box['raw'].append(margin)
                                point_count+=1
                for condition,box in bycondition.items():
                    labels=np.concatenate(box['labels']);scores=np.concatenate(box['scores']);private[key+'_'+condition]=scores
                    values=metric(labels,scores,tau)
                    rows.append(dict(seed=seed,method=method,backend=backend,user=f'U{owner}',condition=condition,
                        **values,hard_identity_margin=float(np.mean(box['margin'])),worst_sample_margin=float(np.mean(box['worst'])),
                        raw_hard_identity_margin=float(np.mean(box['raw'])),
                        hard_margin_nonpositive_fraction=float(np.mean(np.array(box['margin'])<=0))))
            dump(OUT/'status.json',dict(stage='authentication',seed=seed,method=method,backend=backend));print('AUTH_COMPLETE',seed,method,backend,flush=True)
    np.savez_compressed(OUT/'scores_private.npz',**private);dump(OUT/'calibration_locks_private.json',private_locks)
    def aggregate(rr):return {k:float(np.mean([r[k] for r in rr])) for k in KEYS}
    per_seed=[];summary=[];per_user=[];paired=[]
    for method in ['B0','B1','B2','B3']:
        for backend in ['rbf','cos']:
            for condition in ['sameword','crossword']:
                rr=[r for r in rows if r['method']==method and r['backend']==backend and r['condition']==condition]
                seedrows=[]
                for seed in [42,43,44]:
                    item=dict(method=method,backend=backend,condition=condition,seed=seed,**aggregate([r for r in rr if r['seed']==seed]))
                    per_seed.append(item);seedrows.append(item)
                summary.append(dict(method=method,backend=backend,condition=condition,
                    mean=aggregate(seedrows),std={k:float(np.std([r[k] for r in seedrows],ddof=1)) for k in KEYS}))
                for u in range(1,10):per_user.append(dict(method=method,backend=backend,condition=condition,user=f'U{u}',
                    mean=aggregate([r for r in rr if r['user']==f'U{u}']),
                    std={k:float(np.std([r[k] for r in rr if r['user']==f'U{u}'],ddof=1)) for k in KEYS}))
    for backend in ['rbf','cos']:
        for condition in ['sameword','crossword']:
            for baseline in ['B0','B1','B3']:
                diffs=[]
                for seed in [42,43,44]:
                    b=next(r for r in per_seed if r['method']==baseline and r['backend']==backend and r['condition']==condition and r['seed']==seed)
                    a=next(r for r in per_seed if r['method']=='B2' and r['backend']==backend and r['condition']==condition and r['seed']==seed)
                    diffs.append(dict(seed=seed,**{k:a[k]-b[k] for k in ['eer','hter','far','hard_identity_margin']}))
                paired.append(dict(B2_minus=baseline,backend=backend,condition=condition,per_seed=diffs))
    error=0.
    for r in rows:
        key=f"{r['method']}_seed{r['seed']}_{r['backend']}_{r['user']}"
        values=private[key+'_'+r['condition']];owner=int(r['user'][1:])
        y=np.tile(np.repeat(np.arange(1,10),4)==owner,60)
        check=metric(y,values,private_locks[key]['threshold'])
        error=max(error,max(abs(check[k]-r[k]) for k in ['eer','hter','tar','far','frr','auc']))
    assert error==0 and len(rows)==432 and point_count==25920
    dump(OUT/'results_public.json',dict(summary=summary,per_seed=per_seed,paired_B2_differences=paired))
    compact=dict(metric_order=KEYS,row_order=['method','backend','condition','user','mean','std'],
        rows=[[r['method'],r['backend'],r['condition'],r['user'],[r['mean'][k] for k in KEYS],[r['std'][k] for k in KEYS]] for r in per_user])
    (OUT/'per_user_public.json').write_text(json.dumps(compact,separators=(',',':'))+'\n',encoding='utf-8')
    dump(OUT/'all_seed_user_private.json',rows)
    dump(OUT/'verification_public.json',dict(state='PASS',score_replay_error=error,aggregate_rows=len(rows),workpoints=point_count,
        all9_users_all3_seeds_all4_methods=True,all12_checkpoints_locked_before_auth=True,
        same_queries_in_same_cross=True,cal_queries_not_in_encoder_fit=True,
        thresholds_shared_across_conditions_and_folds=True,no_query_audio=True,no_words18_29_used=True,
        same_batch_order_within_seed=True,single_seed_differences_not_claimed_stable=True,
        limits='saved-score replay and split checks;12..17 previously explored;not new confirmatory data'))
    dump(OUT/'status.json',dict(stage='complete'));print('AUTH_ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
