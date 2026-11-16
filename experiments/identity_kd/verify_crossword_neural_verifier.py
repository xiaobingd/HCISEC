"""Replay saved models, legal calibration and every user/condition result."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from crossword_neural_verifier import ACCEncoder, Verifier
from run_crossword_neural_verifier import norm, pair_features, threshold, metrics, dump


def verify(frontend, out):
    torch.set_num_threads(2)
    out=Path(out); data=np.load(frontend); n=np.load(out/'normalization_private.npz')
    tasksfile=np.load(out/'tasks_private.npz')
    tasks={role:{k:tasksfile[role+'_'+k] for k in ['q','r','y','owner']}
           for role in ['train','dev','cal','sameword','crossword']}
    assert len(set(data['sample_id']))==len(data['sample_id'])
    role_pools=[set(np.r_[t['q'],t['r'].ravel()]) for t in [tasks[k] for k in ['train','dev','cal']]]
    role_pools.append(set(np.r_[tasks['sameword']['q'],tasks['crossword']['q'],tasks['sameword']['r'].ravel()]))
    assert all(not role_pools[i]&role_pools[j] for i in range(4) for j in range(i))
    for role in ['train','dev','cal']:
        t=tasks[role]
        assert not np.any(data['word'][t['q'],None]==data['word'][t['r']])
    spectrum=data['spectrum']
    raw=np.concatenate([spectrum.mean(-1).reshape(len(spectrum),-1),
                        spectrum.std(-1).reshape(len(spectrum),-1),
                        *[spectrum[...,j].mean(-1).reshape(len(spectrum),-1)
                          for j in np.array_split(np.arange(28),4)]],1)
    fit=np.flatnonzero(data['word']<18)
    scaler=StandardScaler().fit(raw[fit])
    assert np.array_equal(scaler.mean_,n['raw_mean'])
    assert np.array_equal(scaler.scale_,n['raw_scale'])
    raw=norm(scaler.transform(raw)).astype('float32')
    assert np.allclose(n['mean'],spectrum[fit].mean((0,3),keepdims=True))
    teacher=np.load(out/'redimnet2_train_embeddings_private.npz')
    assert int(teacher['done'])==len(fit)
    assert np.array_equal(teacher['indices'],fit)
    assert np.array_equal(teacher['sample_id'],data['sample_id'])
    assert not teacher['embedding'][data['word']>=18].any()
    x=torch.tensor((spectrum-n['mean'])/n['std'])
    enc=ACCEncoder().eval();enc.load_state_dict(torch.load(out/'encoder_private.pt',weights_only=True))
    with torch.no_grad():tokens=torch.cat([enc(v) for v in x.split(64)])
    base=LogisticRegression(C=1,max_iter=1000).fit(pair_features(raw,tasks['train']),tasks['train']['y'])
    rows=json.loads((out/'per_user_public.json').read_text(encoding='utf-8'))
    bykey={(r['method'],r['user'],r['condition']):r for r in rows}
    assert len(bykey)==108
    max_error=0.;checks=0
    def check(method,tau_fn,scorer):
        nonlocal max_error,checks
        for cond in ['sameword','crossword']:
            t=tasks[cond];scores=scorer(t)
            for u in range(1,10):
                jj=t['owner']==u; y=t['y'][jj]
                assert y.sum()==12 and (~y).sum()==96
                assert set(data['user'][t['q'][jj&~t['y']]])==set(range(1,10))-{u}
                m=metrics(y,scores[jj],tau_fn(u))
                old=bykey[method,'U'+str(u),cond]
                err=max(abs(m[k]-old[k]) for k in m)
                max_error=max(max_error,err);assert err<1e-7,(method,u,cond,err)
                checks+=1
    def bs(t):return base.decision_function(pair_features(raw,t)).astype('float32')
    def taus(s):return {u:threshold(tasks['cal']['y'][tasks['cal']['owner']==u],s[tasks['cal']['owner']==u]) for u in range(1,10)}
    th=taus(bs(tasks['cal']));check('raw_pair_score',lambda u:th[u],bs)
    register=np.unique(tasks['sameword']['r']);models={};thresholds={}
    for u in range(1,10):
        model=SVC(C=10,gamma='scale',class_weight='balanced').fit(raw[register],data['user'][register]==u)
        jj=tasks['cal']['owner']==u
        thresholds[u]=threshold(tasks['cal']['y'][jj],model.decision_function(raw[tasks['cal']['q'][jj]]))
        models[u]=model
    def rb(t):
        score=np.zeros(len(t['q']))
        for u in range(1,10):
            jj=t['owner']==u;score[jj]=models[u].decision_function(raw[t['q'][jj]])
        return score
    check('raw_rbf',lambda u:thresholds[u],rb)
    for mode in ['global','local','audio','shuffle']:
        net=Verifier(local=mode!='global').eval()
        net.load_state_dict(torch.load(out/(mode+'_best_private.pt'),weights_only=True))
        history=json.loads((out/(mode+'_history_public.json')).read_text())
        lock=json.loads((out/(mode+'_selection_public.json')).read_text())
        selected=max(history,key=lambda r:r['dev_macro_auc'])
        assert selected['step']==lock['step']
        def score(t):
            values=[];baseline=bs(t)
            with torch.no_grad():
                for start in range(0,len(baseline),16):
                    sl=slice(start,start+16)
                    values.append(net(tokens[t['q'][sl]],tokens[t['r'][sl]],torch.tensor(baseline[sl]))[0].numpy())
            return np.concatenate(values)
        th=taus(score(tasks['cal']));check(mode,lambda u:th[u],score)
    dump(out/'verification_public.json',dict(state='PASS',replayed_workpoints=checks,max_metric_error=max_error,
        calibration_replayed=True,checkpoint_selection_replayed=True,word_record_role_disjoint=True,
        teacher_zero_embeddings_outside_fit=True,scaler_fit_only=True,all_nine_users_all_eight_attackers=True,
        limits='replays inference and selection;does not retrain from seed or verify nominal transcript semantics'))
    print('ALL_108_WORKPOINTS_VERIFIED',max_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();verify(a.frontend,a.out)
