"""Frozen ACC: development-only cross-word selection of score normalization/calibration."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path('/root')
OUT = ROOT / 'full_run_10x40_raw_acc/threeword_calibration'
NORMS = ['raw', 'z', 't', 's', 'as3']
RULES = ['strict5', 'midpoint5', 'quantile_gap']

def cutoff(scores, labels, rule):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    if rule == 'quantile_gap':
        # Fixed before query: bridge the low genuine and upper impostor tails.
        return float((np.quantile(scores[labels], .10) + np.quantile(scores[~labels], .95)) / 2)
    values = np.unique(scores)
    if rule == 'strict5':
        candidates = np.r_[values, values[-1] + max(1., abs(values[-1]) * 1e-6)]
    else:
        assert rule == 'midpoint5'
        candidates = np.r_[values[0] - 1., (values[:-1] + values[1:]) / 2, values[-1] + 1.]
    accept = scores[None, :] >= candidates[:, None]
    far = accept[:, ~labels].mean(1)
    frr = (~accept[:, labels]).mean(1)
    feasible = np.flatnonzero(far <= .05 + 1e-12)
    index = min(feasible, key=lambda i: (frr[i], far[i], -candidates[i]))
    return float(candidates[index])

def measure(scores, labels, tau):
    labels = np.asarray(labels, dtype=bool)
    accept = scores >= tau
    fa, fr = int(accept[~labels].sum()), int((~accept[labels]).sum())
    ng, ni = int(labels.sum()), int((~labels).sum())
    fp, tp, _ = roc_curve(labels, scores, drop_intermediate=False)
    ix = np.argmin(np.abs(fp - (1 - tp)))
    return dict(far=fa/ni, frr=fr/ng, hter=(fa/ni+fr/ng)/2,
                eer=float((fp[ix]+1-tp[ix])/2), auc=float(roc_auc_score(labels, scores)),
                genuine_trials=ng, attack_trials=ni, false_accepts=fa, false_rejects=fr)

def normalization(score, zcohort, floor):
    # score[n_record, claimed_model]; zcohort[cohort_identity, claimed_model].
    result = {k: np.empty_like(score) for k in NORMS}
    result['raw'] = score.copy()
    for owner in range(10):
        other = np.arange(10) != owner
        z = zcohort[other, owner]
        t = score[:, other]
        zs = (score[:, owner] - z.mean()) / max(z.std(), floor)
        ts = (score[:, owner] - t.mean(1)) / np.maximum(t.std(1), floor)
        az = np.sort(z)[-3:]
        at = np.sort(t, axis=1)[:, -3:]
        result['z'][:, owner] = zs
        result['t'][:, owner] = ts
        result['s'][:, owner] = (zs + ts) / 2
        result['as3'][:, owner] = .5 * (
            (score[:, owner]-az.mean())/max(az.std(), floor) +
            (score[:, owner]-at.mean(1))/np.maximum(at.std(1), floor))
    assert all(np.isfinite(v).all() for v in result.values())
    return result

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    d = np.load(ROOT/'paired_raw_acc_private.npz')
    st = np.load(ROOT/'stats27_private.npz')
    assert np.array_equal(d['sample_id'], st['sample_id'])
    assert len(set(d['sample_id'])) == 3200
    u, w, r = np.array([s.split('_') for s in d['sample_id']], dtype=int).T
    fit = d['role'] == 'fit'
    reg = np.isin(w, [20,21,22,23]) & (r < 6)
    devwords, querywords = np.array([8,9,18,19,24]), np.array([25,26,27,28,29])
    assert (fit.astype(int)+reg+np.isin(w,devwords)+np.isin(w,querywords)).max() == 1
    assert fit.sum() == 2080 and reg.sum() == 240
    spec = d['spectrum'].reshape(len(u),120,-1)
    x = np.column_stack([spec.mean(-1),spec.std(-1),
         *(a.mean(-1) for a in np.array_split(spec,4,axis=-1)),d['amplitude'],st['features']])
    ck = torch.load(ROOT/'adapted_acc_private.pt',map_location='cpu',weights_only=True)
    x = torch.tensor((x-ck['mean'].numpy())/ck['scale'].numpy(),dtype=torch.float32)
    lookup = {(int(us),int(word),int(rep)):i for i,(us,word,rep) in enumerate(zip(u,w,r))}
    def trials(groups):
        indexes, identities = [], []
        for us in range(1,11):
            for group in groups:
                ids=[lookup[us,word,rep] for word,rep in group]
                assert len(set(ids))==3
                indexes.append(ids);identities.append(us)
        return np.array(indexes),np.array(identities)
    def original(words):
        ordered=words[np.random.default_rng(1123).permutation(5)]
        base=[(int(word),rep) for rep in range(4) for word in ordered]
        return trials([[base[(i+j)%20] for j in range(3)] for i in range(0,20,3)])
    dev = original(devwords)
    query = original(querywords)
    folds=[]
    for calwords in itertools.combinations(devwords.tolist(),2):
        validwords=[int(v) for v in devwords if v not in calwords]
        a,b=calwords
        cg=[]
        for rep in range(4):
            cg += [[(a,rep),(b,rep),(a,(rep+1)%4)],
                   [(b,rep),(a,rep),(b,(rep+1)%4)]]
        vg=[[(v,rep) for v in validwords] for rep in range(4)]
        cal, valid=trials(cg),trials(vg)
        assert not set(w[cal[0]].ravel()) & set(w[valid[0]].ravel())
        assert not np.intersect1d(cal[0],valid[0]).size
        folds.append((cal,valid))
    protocol=dict(dataset='10 users, raw hardware ACC, closed known identities; all nine attacker identities retained',
        model='frozen CE 42/43/44; same 762->256->192 encoder and RBF C10 balanced',
        registration_words=[20,21,22,23],registration_per_user=24,
        development_words=devwords.tolist(),query_words=querywords.tolist(),
        challenge='3 distinct query words, mean of normalized per-record RBF scores; previous exact 7 trials/identity retained',
        development_crossvalidation='10 exhaustive two-calibration-word versus three-validation-word splits; word sets disjoint',
        cv_calibration='8 triplets/identity, three independent recordings from two words; records reused across triplets',
        cv_validation='4 triplets/identity, three distinct held-out words; folds dependent, not independent participants',
        cohort='Z: nine non-target identity prototypes from 26 fit words only; T: nine non-target registration RBF models; no probe identity label',
        normalization='raw, Z, T, S=(Z+T)/2, adaptive S with independent top3 Z/T cohorts; RBF-score adaptation, not cosine backend',
        std_floor='max(1e-3, 0.1 * standard deviation of fit-only RBF score matrix)',
        threshold_rules={'strict5':'development FAR<=5%, min FRR, then min FAR, highest threshold (legacy)',
                         'midpoint5':'same decisions/objective on development, threshold at adjacent-score gap midpoint',
                         'quantile_gap':'(genuine q10 + impostor q95)/2, fixed quantiles; no claim of empirical FAR bound'},
        selection='single global recipe: CV FAR<=5%, minimum macro FRR, then HTER/FAR/declared order; fallback min max(FAR,FRR) if none feasible',
        safety='FAR constraint only on development CV; no guaranteed final FAR',
        scope='historically observed query vocabulary, exploratory calibration follow-up, not new blind validation',
        exclusions='no Audio, new training, quality gate, retries, target/attacker filtering or query tuning',
        source='S/AS normalization definition: https://www.fit.vut.cz/research/group/speech/public/publi/2017/matejka_interspeech2017_IS170803.pdf')
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    caches={};cv=[];locks=[]
    with threadpool_limits(limits=2):
        for seed in (42,43,44):
            net=nn.Sequential(nn.Linear(x.shape[1],256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,192))
            state=torch.load(ROOT/'full_run_10x40_raw_acc/strict_episodic_6x4'/f'CE_{seed}_private.pt',map_location='cpu',weights_only=True)
            net.load_state_dict(state['encoder'])
            with torch.inference_mode():v=F.normalize(net.eval()(x),dim=1).numpy()
            models=[SVC(C=10,class_weight='balanced').fit(v[reg],u[reg]==us) for us in range(1,11)]
            scores=np.column_stack([model.decision_function(v) for model in models])
            cohort=normalize(np.stack([v[fit&(u==us)].mean(0) for us in range(1,11)]))
            zcohort=np.column_stack([model.decision_function(cohort) for model in models])
            floor=max(1e-3,.1*float(scores[fit].std()))
            normalized=normalization(scores,zcohort,floor)
            caches[seed]=normalized
            for name in NORMS:
                sc=normalized[name]
                ds=sc[dev[0]].mean(1)
                for us in range(1,11):
                    label=dev[1]==us
                    for rule in RULES:
                        tau=cutoff(ds[:,us-1],label,rule)
                        locks.append(dict(seed=seed,user=us,normalization=name,rule=rule,
                            threshold=tau,development=measure(ds[:,us-1],label,tau)))
                    # A positive affine Z transform must preserve legacy decisions.
                    if name=='z':
                        raw=caches[seed]['raw'][dev[0]].mean(1)[:,us-1]
                        t0=cutoff(raw,label,'strict5');tz=cutoff(ds[:,us-1],label,'strict5')
                        assert np.array_equal(raw>=t0,ds[:,us-1]>=tz)
                    for fold,(cal,valid) in enumerate(folds):
                        cs=sc[cal[0]].mean(1)[:,us-1]
                        vs=sc[valid[0]].mean(1)[:,us-1]
                        for rule in RULES:
                            tau=cutoff(cs,cal[1]==us,rule)
                            cv.append(dict(seed=seed,user=us,fold=fold,normalization=name,rule=rule,
                                **measure(vs,valid[1]==us,tau)))
            print('SEED_CALIBRATED',seed,flush=True)
    recipes=[]
    for name in NORMS:
        for rule in RULES:
            part=[v for v in cv if v['normalization']==name and v['rule']==rule]
            recipes.append(dict(normalization=name,rule=rule,
                **{k:float(np.mean([v[k] for v in part])) for k in ['far','frr','hter','eer','auc']}))
    feasible=[v for v in recipes if v['far']<=.05+1e-12]
    selected=min(feasible,key=lambda v:(v['frr'],v['hter'],v['far'],NORMS.index(v['normalization']),RULES.index(v['rule']))) if feasible else min(recipes,key=lambda v:(max(v['far'],v['frr']),v['hter'],v['far']))
    selection=dict(selected=selected,development_cv=recipes,feasible_count=len(feasible),
                   selection_frozen_before_query_metrics=True)
    (OUT/'selection_public.json').write_text(json.dumps(selection,indent=2,allow_nan=False))
    (OUT/'thresholds_public.json').write_text(json.dumps(locks,indent=2,allow_nan=False))
    print('SELECTION_LOCKED',selected,flush=True)
    # Query labels first enter metric calculations after all choices and locks were serialized.
    rows=[]
    for lock in locks:
        seed,us=lock['seed'],lock['user']
        qs=caches[seed][lock['normalization']][query[0]].mean(1)[:,us-1]
        rows.append(dict(**{k:lock[k] for k in ['seed','user','normalization','rule']},
                         **measure(qs,query[1]==us,lock['threshold'])))
    old=json.loads((ROOT/'full_run_10x40_raw_acc/multiword_auth/results_public.json').read_text())
    errors=[]
    for row in rows:
        if row['normalization']=='raw' and row['rule']=='strict5':
            oldrow=next(v for v in old if v['seed']==row['seed'] and v['user']==row['user'] and v['backend']=='RBF' and v['words']==3 and not v['quality'])
            errors += [abs(row[k]-oldrow[k]) for k in ['far','frr','hter','eer','auc']]
    assert max(errors)<1e-10
    assert len(rows)==450 and len(cv)==4500
    (OUT/'results_public.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    # CV fold metrics are private working output; public report aggregates by user/seed.
    (OUT/'cv_metrics_private.json').write_text(json.dumps(cv))
    (OUT/'verification_public.json').write_text(json.dumps(dict(status='PASS',
        baseline_max_error=max(errors),configuration_count=len(rows),cv_fold_workpoints=len(cv),
        disjoint_cal_validation_words=True,z_cohort_fit_only=True,t_models_registration_only=True,
        z_affine_decision_check=True,choices_serialized_before_query=True),indent=2))
    print('CALIBRATION_COMPLETE',flush=True)

if __name__=='__main__':main()
