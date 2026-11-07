"""Identity-disjoint residual KD and matched same/cross-word enrollment.

Each outer fold: four base identities, two calibration identities, one target,
one unseen attacker. Final identities never enter model/threshold selection.
Research runner; needs private data and /root/run_raw9_300.py dependencies.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import copy
import json
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/root')
import run_raw9_300 as r

ROOT = Path('/autodl-fs/data')
SRC = ROOT / 'identity_kd_all9_raw_full_multisource_20261103'
DEFAULT_OUT = ROOT / 'mei8_strict_open_residual_20261108'
MODES = ['identity_only', 'uniform_audio', 'targeted_audio']


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fold_users(target):
    order = [1 + (target - 1 + j) % 8 for j in range(8)]
    return dict(target=target, unseen_attacker=order[1], calibration=order[2:4], base=order[4:])


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(192, 64)
        self.b = nn.Linear(64, 192)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, x):
        delta = self.b(F.gelu(self.a(x)))
        delta = .1 * delta / torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1.)
        return F.normalize(x + delta, dim=1)


def sample_batch(rng, lookup, users, words):
    selected = rng.choice(words, 2, replace=False)
    return np.array([i for u in rng.permutation(users) for w in selected
                     for i in rng.choice(lookup[u, w], 2, replace=False)])


def encode(net, x, device):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.tensor(x[j:j+8].copy(), device=device)).cpu().numpy()
                               for j in range(0, len(x), 8)])


def train_backbone(x, d, f, seed, out, steps, device, status):
    completed = out / 'backbone_complete.json'
    net = r.Student().to(device)
    if completed.exists():
        net.load_state_dict(torch.load(out / 'best_backbone_private.pt', map_location=device)['state'])
        return net
    head = nn.Linear(192, len(f['base'])).to(device)
    opt = torch.optim.AdamW(list(net.parameters()) + list(head.parameters()), lr=1e-4, weight_decay=1e-4)
    rng = np.random.RandomState(seed)
    fit = np.flatnonzero(np.isin(d['user'], f['base']) & (d['role'] == 'fit'))
    dev = np.flatnonzero(np.isin(d['user'], f['base']) & (d['role'] == 'develop'))
    words = np.unique(d['word'][fit])
    lookup = {(u,w):fit[(d['user'][fit] == u) & (d['word'][fit] == w)] for u in f['base'] for w in words}
    labels = {u:j for j,u in enumerate(f['base'])}
    start = 0
    best = -1.
    resume = out / 'backbone_resume_private.pt'
    if resume.exists():
        ck = torch.load(resume, map_location=device)
        net.load_state_dict(ck['state']); head.load_state_dict(ck['head']); opt.load_state_dict(ck['optimizer'])
        rng.set_state(ck['rng']); torch.set_rng_state(ck['torch_rng'].cpu())
        start, best = ck['step'], ck['best']
    for step in range(start+1, steps+1):
        net.train()
        ii = sample_batch(rng, lookup, f['base'], words)
        y = torch.tensor([labels[u] for u in d['user'][ii]], device=device)
        z = net(torch.tensor(x[ii].copy(), device=device))
        loss = F.cross_entropy(head(z), y) + .1*r.p.sc_loss(z, y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.); opt.step()
        if step % 50 == 0:
            zz = encode(net, x[dev], device)
            q = r.pair_auc(zz, d['user'][dev], d['word'][dev])
            if q > best:
                best = q
                torch.save(dict(state=net.state_dict(),step=step,dev_pair_auc=q),out/'best_backbone_private.pt')
        if step == 1 or step % 10 == 0:
            torch.save(dict(state=net.state_dict(),head=head.state_dict(),optimizer=opt.state_dict(),rng=rng.get_state(),torch_rng=torch.get_rng_state(),step=step,best=best),resume)
            status(state='backbone_training',target=f['target'],seed=seed,step=step,total_steps=steps,loss=float(loss.detach()))
            print('BACKBONE',f['target'],seed,step,float(loss.detach()),flush=True)
    net.load_state_dict(torch.load(out/'best_backbone_private.pt',map_location=device)['state'])
    dump(completed,dict(base_users=f['base'],fit_n=len(fit),develop_n=len(dev),steps=steps,seed=seed,best_dev_pair_auc=best))
    return net


def selection_weights(z, t, d, fit, base):
    # All rules are learned from base fit recordings; never select bro/jin by name.
    words = np.unique(d['word'][fit])
    def matrix(v):
        centers = {w:norm(np.stack([v[fit[(d['user'][fit]==u)&(d['word'][fit]==w)]].mean(0) for u in base])) for w in words}
        out = [centers[w]@centers[q].T for w in words for q in words if w!=q]
        return np.mean(out,axis=0)
    am, tm = matrix(z), matrix(t)
    edges = [(i,j) for i in range(len(base)) for j in range(i+1,len(base))]
    av = np.array([am[i,j] for i,j in edges]); tv=np.array([tm[i,j] for i,j in edges])
    # Within-modality ranks avoid assuming equal cosine scales across encoders.
    ar = np.argsort(np.argsort(av))/(len(edges)-1)
    tr = np.argsort(np.argsort(tv))/(len(edges)-1)
    gap = ar-tr
    weights = np.ones_like(am)
    chosen = []
    for k in np.argsort(-gap)[:max(1,len(edges)//3)]:
        i,j=edges[k]
        audio_margin=min(tm[i,i],tm[j,j])-tm[i,j]
        if gap[k] > 0 and audio_margin > 0:
            weights[i,j]=weights[j,i]=2.
            chosen.append(dict(users=[base[i],base[j]],rank_gap=float(gap[k]),audio_margin=float(audio_margin)))
    return weights.astype('float32'),dict(base_users=base,selected_pairs=chosen,acc_relation=am.tolist(),audio_relation=tm.tolist(),rule='top third positive ACC-minus-Audio confusion rank gaps, positive Audio margin; negative pair weight2, otherwise1; per-class normalized loss')


def relational(z, t, y, pairweights):
    err=(z@z.T-t@t.T).square()
    eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
    same=(y[:,None]==y[None,:])&~eye
    diff=y[:,None]!=y[None,:]
    pw=pairweights[y[:,None],y[None,:]]
    # Equal global strength for uniform vs selected, regardless of mean weights.
    return .5*((err*same).sum()/same.sum().clamp_min(1)+(err*diff*pw).sum()/(diff*pw).sum().clamp_min(1))


def train_residual(z,t,d,f,seed,out,steps,mode,weights,status):
    path=out/(mode+'_embeddings_private.npy')
    if path.exists():return np.load(path)
    torch.manual_seed(seed);rng=np.random.RandomState(seed);net=Residual();head=nn.Linear(192,len(f['base']),bias=False)
    fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'))
    dev=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='develop'))
    words=np.unique(d['word'][fit]);labels={u:j for j,u in enumerate(f['base'])}
    lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in f['base'] for w in words}
    with torch.no_grad():head.weight.copy_(torch.tensor(np.stack([norm(z[fit[d['user'][fit]==u]].mean(0)) for u in f['base']])))
    opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=1e-3,weight_decay=1e-4)
    best=r.pair_auc(z[dev],d['user'][dev],d['word'][dev]);state=copy.deepcopy(net.state_dict());chosen=0
    pw=torch.tensor(weights if mode=='targeted_audio' else np.ones_like(weights))
    for step in range(1,steps+1):
        ii=sample_batch(rng,lookup,f['base'],words);y=torch.tensor([labels[u] for u in d['user'][ii]])
        x=torch.tensor(z[ii]);v=net(x);loss=F.cross_entropy(16*F.linear(v,F.normalize(head.weight,dim=1)),y)+.1*r.p.sc_loss(v,y)+.1*(1-(v*x).sum(1)).mean()
        if mode!='identity_only':loss=loss+.1*relational(v,torch.tensor(t[ii]),y,pw)
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0:
            with torch.no_grad():q=r.pair_auc(net(torch.tensor(z[dev])).numpy(),d['user'][dev],d['word'][dev])
            if q>best:best=q;state=copy.deepcopy(net.state_dict());chosen=step
            status(state='residual_training',target=f['target'],seed=seed,mode=mode,step=step,total_steps=steps)
    net.load_state_dict(state)
    with torch.no_grad():v=net(torch.tensor(z)).numpy()
    np.save(path,v);torch.save(state,out/(mode+'_private.pt'))
    dump(out/(mode+'_selection.json'),dict(selected_step=chosen,base_develop_pair_auc=best))
    return v


def eval_scores(z,d,client,attackers,condition,direction):
    # Same queries and registration size in both conditions; only enrollment words change.
    wa=[24,25,26] if direction==0 else [27,28,29]
    wb=[27,28,29] if direction==0 else [24,25,26]
    ew=wb if condition=='sameword' else wa
    def rows(u,w):
        ii=np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:8]
        assert len(ii)==8
        return ii
    enroll=np.concatenate([rows(client,w)[:4] for w in ew])
    query=np.concatenate([rows(u,w)[4:8] for u in [client]+list(attackers) for w in wb])
    assert not set(enroll)&set(query)
    prototype=norm(z[enroll].mean(0))
    return z[query]@prototype,d['user'][query]==client


def evaluate(z,d,f):
    results={}
    cs=[];cp=[]
    for cond in ['sameword','crossword']:
        for direction in [0,1]:
            for client in f['calibration']:
                attacks=[u for u in f['base']+f['calibration'] if u!=client]
                sc,po=eval_scores(z,d,client,attacks,cond,direction);cs.extend(sc);cp.extend(po)
    cs=np.asarray(cs);cp=np.asarray(cp);tau=r.eer_threshold(cs[cp],cs[~cp])
    for cond in ['sameword','crossword']:
        results[cond]={}
        for scope,attacks in [('unseen_attacker',[f['unseen_attacker']]),('all_seven_sources',[u for u in range(1,9) if u!=f['target']])]:
            ss=[];pp=[]
            for direction in [0,1]:
                sc,po=eval_scores(z,d,f['target'],attacks,cond,direction);ss.extend(sc);pp.extend(po)
            ss=np.asarray(ss);pp=np.asarray(pp);tar=float((ss[pp]>=tau).mean());far=float((ss[~pp]>=tau).mean())
            results[cond][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(pp,ss)),genuine_n=int(pp.sum()),impostor_n=int((~pp).sum()),threshold=tau)
    return results


def main(args):
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    def status(**v):dump(out/'status.json',dict(updated_unix=time.time(),**v))
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1);r.p.DEVICE=device
    a=np.load(SRC/'raw9_full_features_private.npz');keep=a['user']!=9
    d={k:a[k][keep] for k in ['user','word','role','sample_id']}
    raw=a['acc_vel_300'][keep].astype('float32')
    t=np.load(args.audio_cache)
    if len(t)==len(keep):t=t[keep]
    assert t.shape==(len(d['user']),192)
    external=np.load(args.acc_cache) if args.acc_cache else None
    if external is not None:
        if len(external)==len(keep):external=external[keep]
        assert external.shape==(len(d['user']),192)
        assert np.isfinite(external).all() and np.all(np.linalg.norm(external,axis=1)>.9)
    audit=[]
    for target in range(1,9):
        f=fold_users(target);assert len(set(f['base']+f['calibration']+[f['target'],f['unseen_attacker']]))==8
        fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'))
        assert np.all(np.linalg.norm(t[fit],axis=1)>.9)
        audit.append(dict(**f,base_fit_sample_ids=d['sample_id'][fit].tolist()))
    dump(out/'protocol_private.json',dict(folds=audit,seeds=args.seeds,backbone_steps=args.backbone_steps,residual_steps=args.residual_steps,scope='strict identity-disjoint target and unseen attacker; separate two-identity calibration',same_cross='identical12queries per direction;12enrollment records; only enrollment word group differs; two directions; one shared threshold per method pooled from calibration conditions; no target-dependent threshold',selection='only base fit/develop; no all-eight checkpoints or local-nine Audio teacher reused',teacher_cache=args.audio_cache,acc_cache=args.acc_cache,backbone_reuse_root=args.backbone_root,limits='inferred words; eight-user population selected after previous yue audit; three seeds; cosine prototype backend replaces trained closed-set RBF for genuine unseen-user enrollment'))
    results={}
    for seed in args.seeds:
        for target in args.targets:
            key=f'target{target}_seed{seed}';folder=out/key;folder.mkdir(exist_ok=True)
            if (folder/'results_private.json').exists():
                results[key]=json.loads((folder/'results_private.json').read_text());continue
            f=fold_users(target);torch.manual_seed(seed);np.random.seed(seed)
            fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit'))
            lo,hi=np.percentile(raw[fit],[1,99]);x=np.clip(2*(raw-lo)/(hi-lo)-1,-1,1).astype('float32')
            dump(folder/'scaling.json',dict(low=float(lo),high=float(hi),users=f['base']))
            backbone_folder=Path(args.backbone_root)/key if args.backbone_root else folder
            if args.backbone_root:
                assert (backbone_folder/'backbone_complete.json').exists()
                assert json.loads((backbone_folder/'backbone_complete.json').read_text())['base_users']==f['base']
            net=None if external is not None else train_backbone(x,d,f,seed,backbone_folder,args.backbone_steps,device,status)
            cache=folder/'backbone_embeddings_private.npy'
            if external is not None:z=external
            elif args.backbone_root and (backbone_folder/'backbone_embeddings_private.npy').exists():z=np.load(backbone_folder/'backbone_embeddings_private.npy')
            elif cache.exists():z=np.load(cache)
            else:z=encode(net,x,device);np.save(cache,z)
            del net,x
            weights,diag=selection_weights(z,t,d,fit,f['base']);dump(folder/'targeted_rule_private.json',diag)
            outputs={'raw_acc':z}
            for mode in MODES:outputs[mode]=train_residual(z,t,d,f,seed,folder,args.residual_steps,mode,weights,status)
            result=dict(fold=f,seed=seed,methods={mode:evaluate(v,d,f) for mode,v in outputs.items()})
            dump(folder/'results_private.json',result);results[key]=result;dump(out/'results_private.json',results)
            status(state='fold_complete',target=target,seed=seed,completed_folds=len(results),expected_folds=len(args.seeds)*len(args.targets));print('FOLD_COMPLETE',key,flush=True)
    dump(out/'results_private.json',results);status(state='complete',completed_folds=len(results))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(DEFAULT_OUT));ap.add_argument('--targets',nargs='+',type=int,default=list(range(1,9)));ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--backbone-steps',type=int,default=200);ap.add_argument('--residual-steps',type=int,default=1000);ap.add_argument('--audio-cache',default=str(ROOT/'mei9_vibravox_teacher_transfer_20261107/vibravox_audio_embeddings_private.npy'));ap.add_argument('--acc-cache');ap.add_argument('--backbone-root')
    main(ap.parse_args())
