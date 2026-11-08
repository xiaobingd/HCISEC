"""Predefined 12/60/120 genuine registration ablation; same 480 Base negatives."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import run_personal_registration_adaptation as r
p=r.p
WORDS=[0,1,2,3,4,5,7,10,11,12,13,14]

def main(a):
    torch.set_num_threads(1)
    root=Path('outputs/p1_multimodal_bundle');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=np.load(root/'metadata_private.npz');d={k:meta[k] for k in meta.files}
    orig=np.load('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz')
    audio=np.load(root/(a.teacher+'_fit_private.npy'))[orig['user']!=9].astype('float32')
    def enrollment(u,n):
        ix=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='fit'))[:n//12] for w in WORDS])
        assert len(ix)==n and np.all(np.linalg.norm(audio[ix],axis=1)>.9)
        return ix
    def query(u):
        ix=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:8] for w in range(24,30)])
        assert len(ix)==48
        return ix
    p.write(out/'protocol.json',dict(date='2026-11-09',teacher=a.teacher,registration_counts=[12,60,120],words=WORDS,negative_count=480,query='48/user; words24..29 first8 repeats; identical for all registration counts',roles='4Base encoder+mapper;2Cal;target;unseen attacker;disjoint identities',selection='fixed200steps, same old hyperparameters, independent Cal EER and CalFAR05; no test selection',caveat='exploratory8 users; inferred word labels; all-seven includes Base negatives; registration uses fit-labelled rows ONLY for held-out identities, not universal training'))
    results={}
    for seed in [42,43,44]:
        for u in range(1,9):
            key=f'target{u}_seed{seed}';folder=out/key;folder.mkdir(exist_ok=True)
            path=folder/'results_private.json'
            if path.exists():results[key]=json.loads(path.read_text());continue
            f=p.fold(u);assert len(set(f['base']+f['cal']+[u,f['attacker']]))==8
            z=np.load(root/key/'acc_private.npy').astype('float32');net=p.Mapper()
            net.load_state_dict(torch.load(Path(f'outputs/p1_enrollment_{a.teacher}_20261108')/key/'mapper_private.pt',weights_only=True,map_location='cpu'));net.eval()
            for v in net.parameters():v.requires_grad_(False)
            negatives=np.concatenate([enrollment(v,120) for v in f['base']]);assert len(negatives)==480
            fold={};audit={};params={}
            for n in [12,60,120]:
                users=f['cal']+[u];train=[];teachers=[];protos=[];audit[str(n)]={}
                for v in users:
                    en=enrollment(v,n);assert not set(en)&set(query(v));assert not set(d['word'][en])&set(d['word'][query(v)])
                    train.append(z[np.r_[en,negatives]]);protos.append(p.norm(z[en].mean(0)))
                    with torch.no_grad():teachers.append(net(torch.tensor(audio[en])).numpy())
                    audit[str(n)][str(v)]=dict(positive_ids=d['sample_id'][en].tolist(),negative_ids=d['sample_id'][negatives].tolist(),query_ids=d['sample_id'][query(v)].tolist())
                x=torch.tensor(np.stack(train));t=torch.tensor(np.stack(teachers));models={m:r.fit_models(x,t,m,200,npos=n) for m in r.MODES}
                for m,model in models.items():
                    for k,val in model.items():
                        if k!='mode':params[f'n{n}_{m}_{k}']=val
                def scores(v,attacks,m):
                    ix=np.concatenate([query(w) for w in [v]+attacks]);index=users.index(v)
                    ss=z[ix]@protos[index] if m=='cosine' else r.score(z[ix],models[m],index)
                    return ss,d['user'][ix]==v
                fold[str(n)]={}
                for m in ['cosine']+r.MODES:
                    ss=[];yy=[]
                    for v in f['cal']:
                        s,y=scores(v,[w for w in f['cal'] if w!=v],m);ss.extend(s);yy.extend(y)
                    ss=np.array(ss);yy=np.array(yy)
                    for rule in ['eer','cal_far05']:
                        tau=p.tau_eer(ss[yy],ss[~yy]) if rule=='eer' else r.far_threshold(ss[~yy]);metrics=dict(threshold=tau,cal_tar=float((ss[yy]>=tau).mean()),cal_far=float((ss[~yy]>=tau).mean()))
                        if rule=='cal_far05':assert metrics['cal_far']<=.05+1e-12
                        for scope,attacks in [('unseen_attacker',[f['attacker']]),('all_seven_sources',[v for v in range(1,9) if v!=u])]:
                            s,y=scores(u,attacks,m);tar=float((s[y]>=tau).mean());far=float((s[~y]>=tau).mean());metrics[scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(y,s)),genuine_n=int(y.sum()),impostor_n=int((~y).sum()))
                        fold[str(n)][m+'_'+rule]=metrics
            p.write(folder/'registration_audit_private.json',audit);np.savez(folder/'parameters_private.npz',**params);p.write(path,fold);results[key]=fold;p.write(out/'results_private.json',results);p.write(out/'status.json',dict(state='running',completed=len(results),total=24));print(a.teacher,key,flush=True)
    p.write(out/'status.json',dict(state='complete',folds=len(results)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',choices=['ecapa','eres'],default='ecapa');ap.add_argument('--out',default='outputs/personal_registration_120_ecapa_20261109');main(ap.parse_args())