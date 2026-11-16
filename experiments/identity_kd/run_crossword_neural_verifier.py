"""Independent Mei9 verifier pilot. Private inputs/checkpoints never publish.

Fit words 0:18, model development 18:21, calibration 21:24,
registration/same-word 24:27, cross-word queries 27:30.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from crossword_neural_verifier import ACCEncoder, Verifier, objective

REVISION = 'c5bbe0b76e37df698c403f8844e41304ceab6307'
WEIGHT_SHA = '02d14efdc184573873b5c3f62bea847fcdb6919fbba19a8da7af7c1ba36b76a2'


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def resolve_audio_path(value):
    """Reverse legacy UTF-8-as-GBK path corruption; payload hash checked below."""
    path = Path(value)
    if path.exists():
        return path
    parts = []
    for part in path.parts:
        try:
            parts.append(part.encode('gbk').decode('utf-8'))
        except (UnicodeError, LookupError):
            parts.append(part)
    repaired = Path(*parts)
    if repaired.exists():
        return repaired
    raise FileNotFoundError('Audio path unavailable after legacy encoding repair: '+str(repaired))


def threshold(y, s):
    f, t, th = roc_curve(y, s, drop_intermediate=False)
    j = min(range(len(th)), key=lambda i: (abs(f[i]-(1-t[i])), f[i]))
    return float(th[j])


def metrics(y, s, tau):
    f, t, _ = roc_curve(y, s, drop_intermediate=False)
    j = np.argmin(abs(f-(1-t)))
    far = float((s[~y] >= tau).mean())
    frr = float((s[y] < tau).mean())
    return dict(auc=float(roc_auc_score(y, s)), eer=float((f[j]+1-t[j])/2),
                far=far, frr=frr, tar=1-frr, hter=(far+frr)/2,
                genuine_n=int(y.sum()), impostor_n=int((~y).sum()))


class Tasks:
    def __init__(self, user, word):
        self.user, self.word = user, word
        self.lookup = {(u, w): np.flatnonzero((user == u) & (word == w))
                       for u in range(1, 10) for w in range(30)}

    def sample(self, words, n, seed):
        rng = np.random.RandomState(seed)
        queries, supports, targets, owners = [], [], [], []
        for j in range(n):
            owner = 1 + (j//2) % 9
            genuine = j % 2 == 0
            # Cycle all eight attack sources; random draws can omit a source
            # in short development batches and skew per-owner calibration.
            negatives = [u for u in range(1, 10) if u != owner]
            speaker = owner if genuine else negatives[(j//18) % 8]
            sw, sw2, qw = rng.choice(words, 3, replace=False)
            rr = np.r_[rng.choice(self.lookup[owner, sw], 6, replace=False),
                       rng.choice(self.lookup[owner, sw2], 6, replace=False)]
            q = int(rng.choice(self.lookup[speaker, qw]))
            assert q not in rr and self.word[q] not in [sw, sw2]
            queries.append(q); supports.append(rr); targets.append(genuine); owners.append(owner)
        return dict(q=np.array(queries), r=np.array(supports),
                    y=np.array(targets), owner=np.array(owners))

    def evaluation(self, support_words, query_words, same=False):
        q, r, y, owner = [], [], [], []
        for u in range(1, 10):
            rr = np.concatenate([self.lookup[u, w][:4] for w in support_words])
            assert len(rr) == 12
            for v in range(1, 10):
                for w in query_words:
                    jj = self.lookup[v, w][4:8] if same else self.lookup[v, w][:4]
                    assert len(jj) == 4
                    for i in jj:
                        assert i not in rr
                        q.append(i); r.append(rr); y.append(u == v); owner.append(u)
        return dict(q=np.array(q), r=np.array(r), y=np.array(y), owner=np.array(owner))


def pair_features(z, tasks):
    q, r = z[tasks['q']], z[tasks['r']]
    sim = (q[:, None]*r).sum(-1)
    return np.c_[sim.mean(1), sim.std(1), sim.max(1), sim.min(1),
                 (q*norm(r.mean(1))).sum(-1)]


def teacher_embeddings(a, d, fit, out):
    path = out/'redimnet2_train_embeddings_private.npz'
    done, z = 0, None
    if path.exists():
        cache = np.load(path)
        assert np.array_equal(cache['sample_id'], d['sample_id'])
        assert np.array_equal(cache['indices'], fit)
        done, z = int(cache['done']), cache['embedding']
        if done == len(fit):
            return z
    sys.path.insert(0, str(Path(a.runtime)/('redimnet2-'+REVISION)))
    sys.path.insert(0, str(Path(a.deps).resolve()))
    from redimnet2.redimnet2 import ReDimNet2Wrap
    weights = Path(a.runtime)/'b3-vox2-lm.pt'
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == WEIGHT_SHA
    state = torch.load(weights, map_location='cpu', weights_only=False)
    model = ReDimNet2Wrap(**state['model_config'])
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    manifest = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    source = {r['sample_id']: r for r in manifest['records']}
    start = time.time()
    for j, i in enumerate(fit):
        if j < done:
            continue
        record = source[str(d['sample_id'][i])]
        assert record['word_id'] == int(d['word'][i])
        wave = np.load(resolve_audio_path(record['audio_file']), allow_pickle=False)
        assert hashlib.sha256(memoryview(np.ascontiguousarray(wave)).cast('B')).hexdigest() == record['audio_array_sha256']
        assert wave.shape == (96000, 2)
        wave = resample_poly(wave[:, 0].astype('float64'), 1, 3).astype('float32')
        wave -= wave.mean()
        with torch.no_grad():
            emb = F.normalize(model(torch.from_numpy(wave[None])).reshape(1, -1), dim=1).numpy()[0]
        if z is None:
            z = np.zeros((len(d['user']), len(emb)), dtype='float32')
        z[i] = emb
        if (j+1) % 50 == 0:
            np.savez_compressed(path, embedding=z, indices=fit, sample_id=d['sample_id'], done=j+1)
            print('TEACHER', j+1, '/', len(fit), 'seconds', round(time.time()-start), flush=True)
            dump(out/'status.json', dict(stage='teacher_extraction', done=j+1, total=len(fit)))
    np.savez_compressed(path, embedding=z, indices=fit, sample_id=d['sample_id'], done=len(fit))
    dump(out/'teacher_audit_public.json', dict(backbone='ReDimNet2-B3 vox2 lm',
         revision=REVISION, weight_sha256=WEIGHT_SHA, audio_records=len(fit),
         audio_words=list(range(18)), calibration_audio_reads=0, final_audio_reads=0,
         source_payload_hashes_checked=True, sample_rate=16000,
         channel='left, same rule for all users', backbone_frozen=True))
    return z


def run(a):
    torch.set_num_threads(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = np.load(a.frontend)
    d = {k: data[k] for k in ['spectrum', 'user', 'word', 'sample_id']}
    assert len(set(d['sample_id'])) == len(d['sample_id'])
    fit = np.flatnonzero(d['word'] < 18)
    tasks = Tasks(d['user'], d['word'])
    x = d['spectrum'].astype('float32')
    mean = x[fit].mean((0, 3), keepdims=True)
    std = x[fit].std((0, 3), keepdims=True).clip(1e-4)
    x = torch.from_numpy((x-mean)/std)
    raw = np.concatenate([data['spectrum'].mean(-1).reshape(len(x), -1),
                          data['spectrum'].std(-1).reshape(len(x), -1),
                          *[data['spectrum'][..., j].mean(-1).reshape(len(x), -1)
                            for j in np.array_split(np.arange(28), 4)]], 1)
    scaler = StandardScaler().fit(raw[fit]); raw = norm(scaler.transform(raw)).astype('float32')
    np.savez_compressed(out/'normalization_private.npz', mean=mean, std=std,
                        raw_mean=scaler.mean_, raw_scale=scaler.scale_)
    train = tasks.sample(list(range(18)), 3600, a.seed)
    dev = tasks.sample([18, 19, 20], 900, 18042)
    cal = tasks.sample([21, 22, 23], 1800, 21042)
    tests = dict(sameword=tasks.evaluation([24, 25, 26], [24, 25, 26], True),
                 crossword=tasks.evaluation([24, 25, 26], [27, 28, 29]))
    # Split by physical recording and word; supports repeat within a role only.
    pools = [set(np.r_[t['q'], t['r'].ravel()]) for t in [train, dev, cal]]
    final_pool = set(np.r_[tests['sameword']['q'], tests['crossword']['q'], tests['sameword']['r'].ravel()])
    pools.append(final_pool)
    assert all(not pools[i] & pools[j] for i in range(4) for j in range(i))
    np.savez_compressed(out/'tasks_private.npz', **{role+'_'+k: v for role, t in
                         dict(train=train, dev=dev, cal=cal, **tests).items() for k, v in t.items()})
    protocol_path=out/'protocol_public.json'
    prior_protocol=json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else {}
    all_methods=list(dict.fromkeys(prior_protocol.get('methods',[])+a.methods))
    dump(protocol_path, dict(stage='running', seed=a.seed,
         methods=all_methods, last_invocation_methods=a.methods, fit_words=list(range(18)), develop_words=[18,19,20],
         calibration_words=[21,22,23], registration_words=[24,25,26], cross_words=[27,28,29],
         encoder_steps=a.encoder_steps, verifier_steps=a.steps, budget='same task order and initialization per verifier',
         local_attention='one layer,128dim,4heads,bidirectional;temporal mean pooling;enrollment mean/std pooling',
         thresholds='per-user on Cal only;shared between same/cross final queries',
         registration=12, same_queries_per_user=12, cross_queries_per_user=12,
         train_support=12, all_eight_attack_sources=True, scaler_fit_only=True,
         limits=['closed nine known identities','nominal chronological word labels, not transcript-verified',
                 'single session; no controlled redon','historically explored cohort',
                 'HMD tracking velocity, not hardware raw acceleration',
                 'adaptation of 2026 neural rescoring;not original hybrid TD/TI enrollment']))
    dump(out/'status.json', dict(stage='encoder_training'))
    enc = ACCEncoder(); torch.manual_seed(a.seed); enc = ACCEncoder()
    head = nn.Linear(128, 9)
    enc_path = out/'encoder_private.pt'
    if enc_path.exists():
        enc.load_state_dict(torch.load(enc_path, weights_only=True))
    else:
        opt = torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=3e-4, weight_decay=1e-3)
        rng = np.random.RandomState(a.seed)
        for step in range(a.encoder_steps):
            ww = rng.choice(18, 2, replace=False)
            ii = np.concatenate([rng.choice(tasks.lookup[u,w], 2, replace=False)
                                 for u in range(1,10) for w in ww])
            z = F.normalize(enc(x[ii]).mean(1), dim=1)
            loss = F.cross_entropy(16*F.linear(z, F.normalize(head.weight, dim=1)),
                                   torch.tensor(d['user'][ii]-1,dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step()
            if (step+1)%100 == 0:
                print('ENCODER', step+1, float(loss.detach()), flush=True)
        torch.save(enc.state_dict(), enc_path)
    enc.eval().requires_grad_(False)
    with torch.no_grad():
        tokens = torch.cat([enc(v) for v in x.split(64)])
    del x
    # Raw baseline calibrated on training episodes only.
    base = LogisticRegression(C=1, max_iter=1000).fit(pair_features(raw, train), train['y'])
    def base_scores(t):
        return base.decision_function(pair_features(raw, t)).astype('float32')
    base_train = base_scores(train)
    teacher_targets = None
    if any(m in a.methods for m in ['audio', 'shuffle']):
        teacher = teacher_embeddings(a, d, fit, out)
        teacher_targets = np.zeros(len(train['q']), dtype='float32')
        # OOF scorer never fits the query word whose tasks it predicts.
        for fold in range(3):
            words = [w for w in range(18) if w%3 != fold]
            tr = tasks.sample(words, 3600, a.seed+100+fold)
            scorer = LogisticRegression(C=1, max_iter=1000).fit(pair_features(teacher, tr), tr['y'])
            selected = d['word'][train['q']]%3 == fold
            tt = {k:v[selected] for k,v in train.items()}
            teacher_targets[selected] = scorer.decision_function(pair_features(teacher, tt))
        print('TEACHER_OOF_AUC', roc_auc_score(train['y'], teacher_targets), flush=True)
        dump(out/'teacher_task_quality_public.json', dict(oof_auc=float(roc_auc_score(train['y'],teacher_targets)),
             scorer='logistic pair similarity statistics;3fold heldout query-word scorer',
             backbone='external pretrained frozen;no local backbone fit', tasks=len(teacher_targets)))
    result_path=out/'per_user_public.json'
    previous=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else []
    rerun=set(a.methods)|{'raw_pair_score','raw_rbf'}
    results=[r for r in previous if r['method'] not in rerun]
    def evaluate(name, scorer):
        cs = scorer(cal)
        thresholds = {u:threshold(cal['y'][cal['owner']==u],cs[cal['owner']==u]) for u in range(1,10)}
        for condition, t in tests.items():
            score = scorer(t)
            for u in range(1,10):
                sel = t['owner']==u
                results.append(dict(method=name,user='U'+str(u),condition=condition,
                     **metrics(t['y'][sel],score[sel],thresholds[u])))
        dump(out/'per_user_public.json',results)
    evaluate('raw_pair_score',base_scores)
    # RBF reference uses exactly the same final registration and Cal protocol.
    register = np.unique(tests['sameword']['r'])
    for u in range(1,10):
        clf = SVC(C=10,gamma='scale',class_weight='balanced').fit(raw[register],d['user'][register]==u)
        cc=cal['owner']==u; ss=clf.decision_function(raw[cal['q'][cc]])
        tau=threshold(cal['y'][cc],ss)
        for condition,t in tests.items():
            sel=t['owner']==u
            results.append(dict(method='raw_rbf',user='U'+str(u),condition=condition,
                 **metrics(t['y'][sel],clf.decision_function(raw[t['q'][sel]]),tau)))
    for mode in a.methods:
        torch.manual_seed(a.seed)
        net = Verifier(local=mode!='global')
        opt = torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3)
        rng = np.random.RandomState(a.seed)
        best, best_step, state = -1, 0, copy.deepcopy(net.state_dict())
        history=[]
        targets=teacher_targets
        if mode=='shuffle':
            shuffled=teacher.copy()
            shuffle_rng=np.random.RandomState(a.seed+1000)
            for u in range(1,10):
                jj=fit[d['user'][fit]==u]
                shuffled[jj]=teacher[shuffle_rng.permutation(jj)]
            # Same calibrated scoring protocol, train-only identity-preserving permutation.
            targets=np.zeros(len(train['q']),dtype='float32')
            for fold in range(3):
                tr=tasks.sample([w for w in range(18) if w%3!=fold],3600,a.seed+100+fold)
                sc=LogisticRegression(C=1,max_iter=1000).fit(pair_features(shuffled,tr),tr['y'])
                sel=d['word'][train['q']]%3==fold
                targets[sel]=sc.decision_function(pair_features(shuffled,{k:v[sel] for k,v in train.items()}))
        def score(t):
            net.eval(); values=[]; bs=base_scores(t)
            with torch.no_grad():
                for start in range(0,len(bs),16):
                    sl=slice(start,start+16)
                    values.append(net(tokens[t['q'][sl]],tokens[t['r'][sl]],torch.tensor(bs[sl]))[0].numpy())
            return np.concatenate(values)
        resume_path=out/(mode+'_resume_private.pt')
        start_step=0
        if resume_path.exists():
            saved=torch.load(resume_path,map_location='cpu',weights_only=False)
            net.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer'])
            start_step=saved['step'];best=saved['best'];best_step=saved['best_step']
            state=saved['best_state'];history=saved['history']
            rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng'])
        for step in range(start_step+1,a.steps+1):
            net.train(); ii=rng.choice(len(train['q']),16,replace=False)
            logit,delta=net(tokens[train['q'][ii]],tokens[train['r'][ii]],torch.tensor(base_train[ii]))
            target=torch.tensor(targets[ii]) if mode in ['audio','shuffle'] else None
            loss=objective(logit,delta,torch.tensor(train['y'][ii]),target)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
            if step%100==0 or step==a.steps:
                ds=score(dev)
                auc=float(np.mean([roc_auc_score(dev['y'][dev['owner']==u],ds[dev['owner']==u]) for u in range(1,10)]))
                history.append(dict(step=step,loss=float(loss.detach()),dev_macro_auc=auc))
                if auc>best:
                    best,best_step,state=auc,step,copy.deepcopy(net.state_dict())
                dump(out/'status.json',dict(stage='verifier_training',method=mode,step=step,total=a.steps,dev_auc=auc))
                dump(out/(mode+'_history_public.json'),history)
                torch.save(dict(state=net.state_dict(),optimizer=opt.state_dict(),step=step,
                     best=best,best_step=best_step,best_state=state,history=history,
                     rng=rng.get_state(),torch_rng=torch.get_rng_state()),resume_path)
                print('VERIFIER',mode,step,'loss',float(loss.detach()),'devAUC',auc,flush=True)
        net.load_state_dict(state);torch.save(state,out/(mode+'_best_private.pt'))
        dump(out/(mode+'_selection_public.json'),dict(step=best_step,dev_macro_auc=best,query_selection=False))
        evaluate(mode,score)
        print('METHOD_COMPLETE',mode,flush=True)
    dump(out/'per_user_public.json',results)
    summary=[]
    for method in sorted(set(r['method'] for r in results)):
        for condition in tests:
            rows=[r for r in results if r['method']==method and r['condition']==condition]
            summary.append(dict(method=method,condition=condition,**{k:float(np.mean([r[k] for r in rows])) for k in ['auc','eer','tar','far','frr','hter']}))
    dump(out/'results_public.json',dict(state='complete',summary=summary))
    dump(out/'status.json',dict(stage='complete'))
    print('ALL_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--frontend',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--runtime',default='work/redimnet2_runtime')
    p.add_argument('--deps',default='work/verifier_deps');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--encoder-steps',type=int,default=400);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--methods',nargs='+',choices=['global','local','audio','shuffle'],default=['global','local','audio','shuffle'])
    run(p.parse_args())
