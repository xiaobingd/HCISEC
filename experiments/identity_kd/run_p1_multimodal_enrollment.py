"""P1 E0/E1/E2: frozen encoders, Audio-supported enrollment, HMD-only queries.

Portable cached-feature runner. Inner identity holdout concerns the mapper;
the fixed outer Base-only HMD encoder has seen all four Base identities.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,roc_curve

LAMBDAS=[0.,.1,.25,.5,.75,1.]
NAMES=['bo','bro','jin','ke','mei','qi','qis','xin']
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def write(path,x):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(x,indent=2));p.replace(path)
def fold(target):
    order=[1+(target-1+j)%8 for j in range(8)]
    return dict(target=target,attacker=order[1],cal=order[2:4],base=order[4:])
class Mapper(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,192))
    def forward(self,x):return F.normalize(self.net(x),dim=1)
def train(net,za,zb,d,fit,users,seed,steps,callback=None):
    rng=np.random.RandomState(seed);words=np.unique(d['word'][fit]);lookup={(u,w):fit[(d['user'][fit]==u)&(d['word'][fit]==w)] for u in users for w in words}
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    for step in range(1,steps+1):
        ww=rng.choice(words,2,replace=False);ii=np.array([i for u in users for w in ww for i in rng.choice(lookup[u,w],2,replace=False)])
        y=torch.tensor(d['user'][ii]);w=torch.tensor(d['word'][ii]);z=net(torch.tensor(za[ii]));target=torch.tensor(zb[ii]);same=y[:,None]==y[None,:];different_word=w[:,None]!=w[None,:];pos=same&different_word;neg=~same
        logits=z@target.T/.1+(neg&~different_word)*np.log(2.);den=torch.logsumexp(logits.masked_fill(~(pos|neg),-1e9),dim=1,keepdim=True)
        loss=-((logits-den)*pos).sum(1).div(pos.sum(1)).mean()+.1*(1-(z*target).sum(1)).mean()
        opt.zero_grad();loss.backward();opt.step()
        if callback is not None and step%50==0:callback(step,net)
    return net
def inner_select(za,zb,d,base,fit,seed,budget):
    curves={step:[] for step in range(50,budget+1,50)}
    for held in base:
        users=[u for u in base if u!=held];trainix=fit[np.isin(d['user'][fit],users)]
        support=np.concatenate([np.flatnonzero((d['user']==held)&(d['word']==w)&(d['role']=='fit'))[:2] for w in range(6)])
        query=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='develop'))[:2] for u in base for w in [8,9,18,19]])
        pb=norm(zb[support].mean(0));pa=norm(za[support].mean(0));truth=d['user'][query]==held
        torch.manual_seed(seed);net=Mapper()
        def check(step,net):
            with torch.no_grad():mapped=net(torch.tensor(pa[None])).numpy()[0]
            auc=[]
            for weight in LAMBDAS:
                p=pb if weight==0 else mapped if weight==1 else norm((1-weight)*pb+weight*mapped)
                auc.append(float(roc_auc_score(truth,zb[query]@p)))
            curves[step].append(auc)
        train(net,za,zb,d,trainix,users,seed,budget,check)
    averages={s:np.asarray(v).mean(0) for s,v in curves.items()}
    # Select mapper checkpoint by E1, then fusion weight at that checkpoint.
    step=max(averages,key=lambda s:(averages[s][-1],-s));scores=averages[step]
    weight=LAMBDAS[max(range(len(LAMBDAS)),key=lambda j:(scores[j],-LAMBDAS[j]))]
    return step,weight,dict(inner_mean_auc_by_step={str(s):v.tolist() for s,v in averages.items()},lambda_grid=LAMBDAS,selected_step=step,selected_lambda=weight,limits='mapper identity holdout only;encoder has seen all outer Base identities')
def tau_eer(g,n):
    vals=np.unique(np.r_[g,n]);vals=np.r_[vals,np.nextafter(vals[-1],np.inf)]
    return float(min(vals,key=lambda t:(abs((g<t).mean()-(n>=t).mean()),(g<t).mean()+(n>=t).mean(),-t)))
def evaluate(net,za,zb,d,f,weight):
    mapped={}
    for u in f['cal']+[f['target']]:
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                wa=[24,25,26] if direction==0 else [27,28,29];wb=[27,28,29] if direction==0 else [24,25,26];ew=wb if cond=='sameword' else wa
                en=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:4] for w in ew])
                assert len(en)==12 and np.all(np.linalg.norm(za[en],axis=1)>.9)
                pb=norm(zb[en].mean(0));pa=norm(za[en].mean(0))
                with torch.no_grad():pm=net(torch.tensor(pa[None])).numpy()[0]
                mapped[u,cond,direction]={'E0':pb,'E1':pm,'E2':pb if weight==0 else pm if weight==1 else norm((1-weight)*pb+weight*pm)}
    def scores(client,attackers,cond,direction,method):
        wb=[27,28,29] if direction==0 else [24,25,26]
        query=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[4:8] for u in [client]+attackers for w in wb])
        # No Audio query access: scoring uses only fixed HMD embeddings/template.
        return zb[query]@mapped[client,cond,direction][method],d['user'][query]==client,d['user'][query]
    out={}
    for method in ['E0','E1','E2']:
        cs=[];cp=[]
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                for client in f['cal']:
                    s,p,_=scores(client,[u for u in f['base']+f['cal'] if u!=client],cond,direction,method);cs.extend(s);cp.extend(p)
        cs=np.asarray(cs);cp=np.asarray(cp);tau=tau_eer(cs[cp],cs[~cp]);out[method]={}
        for cond in ['sameword','crossword']:
            out[method][cond]={}
            for scope,attacks in [('unseen_attacker',[f['attacker']]),('all_seven_sources',[u for u in range(1,9) if u!=f['target']])]:
                ss=[];pp=[];who=[]
                for direction in [0,1]:
                    s,p,u=scores(f['target'],attacks,cond,direction,method);ss.extend(s);pp.extend(p);who.extend(u)
                ss=np.asarray(ss);pp=np.asarray(pp);who=np.asarray(who);tar=float((ss[pp]>=tau).mean());far=float((ss[~pp]>=tau).mean());fpr,tpr,_=roc_curve(pp,ss);k=np.argmin(np.abs(fpr-(1-tpr)));eer=float((fpr[k]+1-tpr[k])/2)
                out[method][cond][scope]=dict(tar=tar,far=far,frr=1-tar,hter=(1-tar+far)/2,auc=float(roc_auc_score(pp,ss)),descriptive_eer=eer,genuine_n=int(pp.sum()),impostor_n=int((~pp).sum()),threshold=tau,by_attack_user={NAMES[u-1]:float((ss[(who==u)&~pp]>=tau).mean()) for u in attacks})
    return out
def main(a):
    root=Path(a.bundle);out=Path(a.out);out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(1)
    meta=np.load(root/'metadata_private.npz');d={k:meta[k] for k in meta.files};keep_original=np.load(a.original)['user']!=9
    man=json.loads(Path(a.manifest).read_text(encoding='utf-8'));exact={r['sample_id'] for r in man['records'] if r['pair_mode']=='exact_timestamp'};valid=np.array([str(s) in exact for s in d['sample_id']]);provenance=json.loads((root/'provenance.json').read_text())
    support=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:4] for u in range(1,9) for w in range(24,30)]);assert valid[support].all()
    za=np.load(root/(a.teacher+'_fit_private.npy'))[keep_original].astype('float32');extra=np.load(root/(a.teacher+'_enrollment_private.npz'));lookup={str(s):i for i,s in enumerate(d['sample_id'])}
    for j,s in enumerate(extra['sample_id']):za[lookup[str(s)]]=extra['audio'][j]
    assert np.all(np.linalg.norm(za[support],axis=1)>.9)
    write(out/'protocol.json',dict(source_plan_commit='ec2ec9b1cf285c195b119d59777c879f4f17d2eb',experiment='P1 E0/E1/E2',teacher=a.teacher,backbones='frozen strict outerBase-only ResNet18,24existingcheckpoints',mapper='192-128GELU-192L2,AdamW1e-3,weightdecay1e-4',loss='crossmodal sameidentity differentword positives;differentidentity sameword negatives weight2;+.1pairedcosine',selection='four Base-only inner mapperidentityholdouts;checkpoint by E1meanAUC,lambda at that checkpoint by fusedmeanAUC,ties favor smallerlambda',budget=a.steps,K=12,threshold='eachmethodsharedsameword/crosswordEERthresholdfromindependentCalusers',identity_roles='4Base/2Cal/target/unseenattacker',limits=['word labels inferred,not semantically verified','eight-user population previously selected after yue audit','inneridentityholdout applies to mapper;fixedencoder saw all fourBaseidentities','not synchronization coupling evidence'],excluded_unconfirmed_fit_pairs=[str(s) for s,ok in zip(d['sample_id'],valid) if not ok],query_audio_used=False))
    result={}
    for seed in [42,43,44]:
        for target in range(1,9):
            key=f'target{target}_seed{seed}';folder=out/key;folder.mkdir(exist_ok=True);rp=folder/'results_private.json'
            if rp.exists():result[key]=json.loads(rp.read_text());continue
            f=fold(target);p=root/key/'acc_private.npy';assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance[key]['embedding_sha256'];assert provenance[key]['base_users']==f['base'];zb=np.load(p).astype('float32');assert za.shape==zb.shape
            fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit')&valid);assert np.all(np.linalg.norm(za[fit],axis=1)>.9)
            write(out/'status.json',dict(state='inner_mapper_selection',teacher=a.teacher,target=target,seed=seed,completed_folds=len(result)))
            step,weight,selection=inner_select(za,zb,d,f['base'],fit,seed,a.steps);torch.manual_seed(seed);net=Mapper();train(net,za,zb,d,fit,f['base'],seed,step);net.eval()
            r=dict(target=target,seed=seed,fold=f,selection=selection,methods=evaluate(net,za,zb,d,f,weight));write(rp,r);torch.save(net.state_dict(),folder/'mapper_private.pt');result[key]=r;write(out/'results_private.json',result)
            print('FOLD',a.teacher,key,'step',step,'lambda',weight,flush=True)
    write(out/'results_private.json',result);write(out/'status.json',dict(state='complete',folds=len(result)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--teacher',choices=['ecapa','eres'],default='ecapa');p.add_argument('--steps',type=int,default=500);p.add_argument('--manifest',default='outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_manifest_private.json');p.add_argument('--original',default='outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');main(p.parse_args())
