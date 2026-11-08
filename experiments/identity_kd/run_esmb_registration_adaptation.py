"""ESMB frozen BC encoder, registration-only personal head and Audio constraint."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import run_personal_registration_adaptation as r
p=r.p
USERS=['0005','0006','0007','0021','0022','0023','0026','0027','0024','0025']

def main(a):
    torch.set_num_threads(1);root=Path('outputs/esmb_registration_20261109');out=root/a.stage;out.mkdir(exist_ok=True)
    data=np.load(root/'embeddings_private.npz');user=data['user'];j=np.tile(np.arange(200),10)
    base=[1,2,3,4];cal=[5,6,7,8];targets=[9,10];clients=cal+targets
    def enroll(u,n):
        # Nested 1/5/10 repeats in each fixed temporal block; never word labels.
        off=np.arange(10)[:n//12]
        return np.concatenate([np.flatnonzero((user==u)&(j==10*b+k)) for b in range(12) for k in off])
    def query(u):return np.flatnonzero((user==u)&(j>=160))
    audio=np.load(root/'registration_audio_private.npy') if a.stage=='full' else None
    results={};audit={}
    p.write(root/'protocol.json',dict(base=[USERS[u-1] for u in base],cal=[USERS[u-1] for u in cal],targets=[USERS[u-1] for u in targets],input='ESMB BC not VR ACC; channel0 mic/channel1BC',windows='2s every3s; 200/user; identical preprocessing; cached encoder Basefit80+dev20 only',registration='12/60/120 across first120 temporal windows; eachBase120 negatives total480',final='last40 windows index160..199; no final Audio accessed',content='NO verified text; temporal separation only, neither same-word nor cross-word claim',parameters='193 linear head+192 channel adapter;200steps;prior fixed hyperparameters',calibration='4Cal identities personally adapt; attack eachother only; predefined EER and CalFAR05 rules; no target test used',encoders='two existing frozen seed42 BC backbones beta0 and beta0.5; not 3 independent training seeds',audio='official frozen VoxCeleb ECAPA192; Base-only mapping fixed200steps; no local teacher finetuning in this experiment',limits='two previously explored target identities, one unseen attack identity per target, single source recording/user, adjacent same-record windows correlated'))
    for encoder in ['beta_0.0','beta_0.5']:
        z=data[encoder].astype('float32');net=None
        if audio is not None:
            torch.manual_seed(42);net=p.Mapper();opt=torch.optim.Adam(net.parameters(),lr=1e-3);rng=np.random.RandomState(42)
            fi=np.flatnonzero(np.isin(user,base)&(j<80));assert np.all(np.linalg.norm(audio[fi],axis=1)>.9)
            for step in range(200):
                ix=np.concatenate([rng.choice(fi[user[fi]==u],8,replace=False) for u in base]);za=net(torch.tensor(audio[ix]));zb=torch.tensor(z[ix]);same=torch.tensor(user[ix,None]==user[None,ix]);logit=za@zb.T/.1
                loss=(1-(za*zb).sum(1)).mean()-.1*((logit-torch.logsumexp(logit,1,keepdim=True))*same).sum(1).div(same.sum(1)).mean()
                opt.zero_grad();loss.backward();opt.step()
            net.eval()
            for parameter in net.parameters():parameter.requires_grad_(False)
            torch.save(net.state_dict(),out/(encoder+'_mapper_private.pt'))
        results[encoder]={};audit[encoder]={}
        for n in [12,60,120]:
            neg=np.concatenate([enroll(u,120) for u in base]);assert len(neg)==480
            train=[];teacher=[];protos=[];audit[encoder][str(n)]={}
            for u in clients:
                en=enroll(u,n);assert len(en)==n and not set(en)&set(query(u));train.append(z[np.r_[en,neg]]);protos.append(p.norm(z[en].mean(0)))
                if audio is not None:
                    assert np.all(np.linalg.norm(audio[en],axis=1)>.9)
                    with torch.no_grad():teacher.append(net(torch.tensor(audio[en])).numpy())
                audit[encoder][str(n)][USERS[u-1]]=dict(registration=en.tolist(),negative=neg.tolist(),query=query(u).tolist())
            x=torch.tensor(np.stack(train));t=torch.tensor(np.stack(teacher)) if teacher else torch.zeros((len(clients),n,192))
            modes=r.MODES if audio is not None else [m for m in r.MODES if m!='residual_audio']
            models={m:r.fit_models(x,t,m,200,npos=n) for m in modes}
            np.savez(out/f'{encoder}_n{n}_parameters_private.npz',**{m+'_'+k:v for m,model in models.items() for k,v in model.items() if k!='mode'})
            def scores(u,attacks,m):
                ix=np.concatenate([query(v) for v in [u]+attacks]);ci=clients.index(u);ss=z[ix]@protos[ci] if m=='cosine' else r.score(z[ix],models[m],ci)
                return ss,user[ix]==u
            results[encoder][str(n)]={}
            for m in ['cosine']+modes:
                cs=[];cy=[]
                for u in cal:
                    ss,yy=scores(u,[v for v in cal if v!=u],m);cs.extend(ss);cy.extend(yy)
                cs=np.array(cs);cy=np.array(cy)
                for rule in ['eer','cal_far05']:
                    tau=p.tau_eer(cs[cy],cs[~cy]) if rule=='eer' else r.far_threshold(cs[~cy]);entry=dict(threshold=tau,cal_tar=float((cs[cy]>=tau).mean()),cal_far=float((cs[~cy]>=tau).mean()),targets={})
                    if rule=='cal_far05':assert entry['cal_far']<=.05+1e-12
                    for u in targets:
                        entry['targets'][USERS[u-1]]={}
                        for scope,attacks in [('unseen_attacker',[v for v in targets if v!=u]),('all_nine_sources',[v for v in range(1,11) if v!=u])]:
                            ss,yy=scores(u,attacks,m);tar=float((ss[yy]>=tau).mean());far=float((ss[~yy]>=tau).mean());entry['targets'][USERS[u-1]][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(yy,ss)),genuine_n=int(yy.sum()),impostor_n=int((~yy).sum()))
                    results[encoder][str(n)][m+'_'+rule]=entry
            p.write(out/'results_private.json',results);p.write(out/'audit_private.json',audit);print(a.stage,encoder,n,flush=True)
    p.write(out/'status.json',dict(state='complete',stage=a.stage,audio_complete=audio is not None))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['acc','full'],default='acc');main(ap.parse_args())