"""Per-account teacher ablations and regularized authentication corrections.

All development/selection remains outside authentication rows. Frozen historical
G/F and baseline head. All final scores are deferred until route choices lock.
"""
import copy, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

sys.path.insert(0,'/root')
import run_complete_baseline_teacher_20261028 as base
from target_auth_protocol_150 import build_target_protocol, split_user_data
from trainb_target_150 import fusion_inputs

OUT=Path('/autodl-fs/data/personal_teacher_auth_optimization_20261028')
OLD=Path('/autodl-fs/data/complete_baseline_teacher_20261028')
DEVUSERS=[1,2,4,5,6,7,9]
D=base.DEVICE

def tensor(x): return torch.as_tensor(x[:,None],dtype=torch.float32,device=D)

@torch.no_grad()
def generate(g,x):
    return np.concatenate([g(tensor(x[i:i+16])).squeeze(1).cpu().numpy() for i in range(0,len(x),16)])

@torch.no_grad()
def embed(teacher,x,m):
    aa=[]; mm=[]
    for i in range(0,len(x),48):
        aa.append(teacher.acc(tensor(x[i:i+48])).cpu().numpy())
        mm.append(teacher.audio(tensor(m[i:i+48])).cpu().numpy())
    return {'acc':np.concatenate(aa),'generated':np.concatenate(mm)}

def supcon(z,y):
    eye=torch.eye(len(y),dtype=torch.bool,device=D)
    pos=(y[:,None]==y[None,:])&~eye
    logits=(z@z.T/.1).masked_fill(eye,-1e4)
    lp=logits-torch.logsumexp(logits,1,keepdim=True)
    return -((lp*pos).sum(1)/pos.sum(1).clamp_min(1)).mean()

def teacher_dev(target,old_teacher,model,sources):
    # Reuse the EXACT previously recorded fit and selection indices.
    dev={k:[] for k in ('a','m','uid','word','isfit')}; ids=old_teacher['metadata']['source_indices']
    for u in DEVUSERS:
        name='bo' if u==1 else 'user%d'%u
        entry=ids[name]; fit=entry['fit']; val=entry['selection']; indices=fit+val
        assert set(indices)==set(sources[name]['cyclegan_indices'])
        assert not set(indices)&set(sources[name]['auth_indices'])
        a=np.load(str(base.PROJECT/'meiy'/str(u)/'acc.npy')).astype('float32')[:,:80,:80]
        m=np.load(str(base.PROJECT/'meiy'/str(u)/'audio.npy')).astype('float32')[:,:80,:80]
        dev['a'].append(a[indices]); dev['m'].append(m[indices])
        dev['uid'].append(np.full(150,u-1)); dev['word'].append(np.asarray(indices)//10)
        dev['isfit'].append(np.arange(150)<len(fit))
    dev={k:np.concatenate(v) for k,v in dev.items()}
    dev['g']=generate(model.g_acc2audio,dev['a'])
    return dev

def fit_teacher(target,dev,robust):
    base.seed(3800+target)
    model=base.Teacher().to(D)
    # Separate personal teacher for each target, with non-target identities retained.
    identities=nn.Linear(64,9).to(D)
    params=list(model.parameters())+list(identities.parameters())
    opt=torch.optim.AdamW(params,lr=.0005,weight_decay=.003)
    rng=np.random.RandomState(3800+target)
    tr=np.flatnonzero(dev['isfit']); va=np.flatnonzero(~dev['isfit'])
    pools={u:tr[dev['uid'][tr]==u-1] for u in DEVUSERS}
    # Same-word negative batches, different-word positives for each identity.
    common=sorted(set.intersection(*[set(dev['word'][ix]) for ix in pools.values()]))
    assert len(common)>=8
    lookup={(u,w):ix[dev['word'][ix]==w] for u,ix in pools.items() for w in common}
    best=None; history=[]
    yv=(dev['uid'][va]==target-1).astype(float)
    for step in range(1,601):
        model.train(); identities.train()
        words=rng.choice(common,8,replace=False)
        ix=np.asarray([rng.choice(lookup[u,w]) for u in DEVUSERS for w in words])
        a=tensor(dev['a'][ix]); m=tensor(dev['m'][ix]); g=tensor(dev['g'][ix])
        labels=torch.as_tensor(dev['uid'][ix],device=D)
        binary=(labels==target-1).float()
        # Noise without circular wrap-around of utterance boundaries.
        a=a+.003*torch.randn_like(a); m=m+.003*torch.randn_like(m)
        za=model.acc(a); zm=model.audio(m)
        z=[za,zm]; labs=[labels,labels]; by=[binary,binary]
        pair=(1-(za*zm).sum(1)).mean()
        if robust:
            zg=model.audio(g+.003*torch.randn_like(g)); z.append(zg); labs.append(labels); by.append(binary)
            pair=pair+.5*(1-(za*zg).sum(1)).mean()
        z=torch.cat(z); labs=torch.cat(labs); by=torch.cat(by)
        bce=F.binary_cross_entropy_with_logits(model.binary(z).squeeze(1),by,pos_weight=torch.tensor(6.,device=D))
        loss=bce+.3*F.cross_entropy(identities(z),labs)+.1*supcon(z,labs)+.1*pair
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params,1.); opt.step()
        if step%25==0:
            model.eval()
            e=embed(model,dev['a'][va],dev['m'][va])
            if robust: e['generated']=embed(model,dev['a'][va],dev['g'][va])['generated']
            with torch.no_grad():
                aucs={k:float(roc_auc_score(yv,model.binary(torch.as_tensor(v,device=D)).squeeze(1).cpu().numpy())) for k,v in e.items()}
            quality=float(np.mean(list(aucs.values())))
            history.append({'step':step,'selection_mean_auc':quality,'aucs':aucs})
            if best is None or quality>best[0]:best=(quality,step,copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2]); model.eval()
    for p in model.parameters():p.requires_grad_(False)
    name='domain_robust' if robust else 'identity_preserving'
    torch.save({'state':model.state_dict(),'target':target,'selected_step':best[1],
                'selection_mean_auc':best[0],'history':history,
                'source':'same recorded upstream fit/selection indices; no auth data',
                'same_word_negative_batches':True,'negative_identities_collapsed':False},OUT/('U%d_%s_teacher.pt'%(target,name)))
    return model

def fit_lr(x,y,c=.01):
    sc=StandardScaler().fit(x)
    cl=LogisticRegression(C=c,class_weight='balanced',max_iter=1500,solver='liblinear',random_state=42)
    cl.fit(sc.transform(x),y)
    return sc,cl

def score(pair,x):return pair[1].decision_function(pair[0].transform(x))

def metrics(sg,sn,t):
    far=float(np.mean(sn>=t)); frr=float(np.mean(sg<t))
    return {'far':far,'frr':frr,'tar':1-frr,'hter':(far+frr)/2,
            'auc':float(roc_auc_score(np.r_[np.ones(len(sg)),np.zeros(len(sn))],np.r_[sg,sn])),
            'genuine_n':len(sg),'impostor_n':len(sn),'threshold':float(t)}

def choose(candidate,labels):
    t=base.original.choose_threshold(candidate['target_dev_pool'],candidate['impostor_calibration'])
    y=labels['model_validation']; v=candidate['model_validation']
    m=metrics(v[y],v[~y],t)
    return (m['hter'],-m['auc']),t,m

@torch.no_grad()
def baseline_features(model,binary,arrays):
    out={}
    for name,x in arrays.items():
        ff=[]; ss=[]
        for i in range(0,len(x),16):
            f=fusion_inputs(model,tensor(x[i:i+16])); ff.append(f.cpu().numpy())
            ss.append(binary(model.fusion(f)).squeeze(1).cpu().numpy())
        out[name]={'feature':np.concatenate(ff),'score':np.concatenate(ss)}
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); results={}
    sources=split_user_data(42)
    saved_sources=json.loads((OLD/'source_manifest.json').read_text())
    assert sources==saved_sources
    for target in (1,4,5):
        print('TARGET',target,flush=True); base.seed(42)
        protocol=build_target_protocol(target,42,42)
        ck=torch.load(str(OLD/('U%d_baseline'%target)/'trainb_binary_auth_head.pth'),map_location=D,weights_only=False)
        model,_=base.original.create_multi_source_model(ck['result']['cyclegan_ckpt'],freeze_generators=True,proj_dim=128,device=D)
        model.load_state_dict(ck['model_state_dict']); model.eval()
        binary=nn.Linear(512,1).to(D); binary.load_state_dict(ck['binary_state_dict']); binary.eval()
        # Current experiment never updates these baseline/G/F parameters.
        for p in model.parameters():p.requires_grad_(False)
        for p in binary.parameters():p.requires_grad_(False)
        arrays=protocol.arrays; labels={k:(v==target-1) for k,v in protocol.labels.items()}
        development=[k for k in arrays if not k.startswith('final_')]
        b=baseline_features(model,binary,{k:arrays[k] for k in development})
        raw={k:b[k]['score'] for k in development}
        locked={}; diagnostics={}; score_models={}
        _,t,vm=choose(raw,labels); locked['baseline_original']={'t':t,'validation':vm,'scores':raw,'kind':'original'}
        # Baseline correction: regularized binary readout on all original features.
        best=None
        for c in (.001,.01,.1,1.):
            pair=fit_lr(b['auth_train']['feature'],labels['auth_train'],c)
            ss={k:score(pair,b[k]['feature']) for k in development}
            rank,t,vm=choose(ss,labels)
            if best is None or rank<best[0]:best=(rank,t,vm,pair,c,ss)
        locked['baseline_regularized']={'t':best[1],'validation':best[2],'scores':best[5],'kind':'ridge','c':best[4]}
        score_models['baseline_regularized']=best[3]
        old=torch.load(str(OLD/('U%d_teacher.pt'%target)),map_location=D,weights_only=False)
        oldmodel=base.Teacher().to(D); oldmodel.load_state_dict(old['state']); oldmodel.eval()
        dev=teacher_dev(target,old,model,sources)
        teachers={'old':oldmodel,'identity':fit_teacher(target,dev,False),'robust':fit_teacher(target,dev,True)}
        generated={k:generate(model.g_acc2audio,arrays[k]) for k in development}
        teacher_features={}
        audio_sources={u:np.load(str(base.PROJECT/'meiy'/str(u)/'audio.npy')).astype('float32')[:,:80,:80] for u in range(1,10)}
        real_audio={}
        for k in ('auth_train','model_validation'):
            samples=[]
            for sid in protocol.sample_ids[k]:
                owner,index=str(sid).split(':'); u=1 if owner=='bo' else int(owner.replace('user',''))
                samples.append(audio_sources[u][int(index)])
            real_audio[k]=np.stack(samples)
        for family,teacher in teachers.items():
            teacher_features[family]={k:embed(teacher,arrays[k],generated[k]) for k in development}
            # Score real and generated modalities using AUTH TRAIN only, evaluate validation only.
            for branch in ('acc','generated'):
                x=teacher_features[family]['auth_train'][branch]; pair=fit_lr(x,labels['auth_train'])
                v=score(pair,teacher_features[family]['model_validation'][branch]); y=labels['model_validation']
                diagnostics[family+'_'+branch]={'validation_auc':float(roc_auc_score(y,v)),
                    'mean_genuine_cos_to_train_center':float(np.mean(teacher_features[family]['model_validation'][branch][y] @ x[labels['auth_train']].mean(0))),
                    'mean_impostor_cos_to_train_center':float(np.mean(teacher_features[family]['model_validation'][branch][~y] @ x[labels['auth_train']].mean(0)))}
            real={k:embed(teacher,arrays[k],real_audio[k])['generated'] for k in real_audio}
            # Real Audio diagnostic has no role in route fitting or inference.
            rp=fit_lr(real['auth_train'],labels['auth_train']); y=labels['model_validation']
            diagnostics[family+'_real_audio']={
                'validation_auc':float(roc_auc_score(y,score(rp,real['model_validation']))),
                'real_readout_on_generated_auc':float(roc_auc_score(y,score(rp,teacher_features[family]['model_validation']['generated']))),
                'paired_generated_real_cosine':float(np.mean(np.sum(real['model_validation']*teacher_features[family]['model_validation']['generated'],axis=1)))}
            for branch in ('acc','generated','both'):
                def getx(k):
                    e=teacher_features[family][k]
                    return np.concatenate([e['acc'],e['generated']],1) if branch=='both' else e[branch]
                pair=fit_lr(getx('auth_train'),labels['auth_train'])
                ts={k:score(pair,getx(k)) for k in development}
                # Keep the pretrained baseline intact; choose a bounded teacher correction.
                center=float(raw['auth_train'].mean()); scale=float(raw['auth_train'].std()+1e-6)
                tc=float(ts['auth_train'].mean()); tscale=float(ts['auth_train'].std()+1e-6)
                best=None
                for alpha in (0.,.25,.5,1.):
                    ss={k:(raw[k]-center)/scale+alpha*(ts[k]-tc)/tscale for k in development}
                    rank,t,vm=choose(ss,labels)
                    if best is None or rank<best[0]:best=(rank,t,vm,alpha,ss)
                name=family+'_'+branch+'_gated'
                locked[name]={'t':best[1],'validation':best[2],'alpha':best[3],'kind':'gate',
                              'family':family,'branch':branch,'base_mean':center,'base_std':scale,
                              'teacher_mean':tc,'teacher_std':tscale,'scores':best[4]}
                score_models[name]=pair
        # Persist route settings BEFORE any final query evaluation.
        config={k:{a:b for a,b in v.items() if a!='scores'} for k,v in locked.items()}
        selected=min(config,key=lambda k:(config[k]['validation']['hter'],-config[k]['validation']['auc'],
                                          0 if k=='baseline_original' else 1 if k=='baseline_regularized' else 2))
        (OUT/('U%d_locked_routes.json'%target)).write_text(json.dumps(config,indent=2))
        (OUT/('U%d_selected_route.json'%target)).write_text(json.dumps({'selected':selected,'rule':'minimum validation HTER, then maximum validation AUC, then simpler model'},indent=2))
        joblib.dump(score_models,str(OUT/('U%d_readouts.joblib'%target)))
        print('ROUTES_LOCKED',target,diagnostics,flush=True)
        finalkeys=('final_genuine','final_impostor')
        fb=baseline_features(model,binary,{k:arrays[k] for k in finalkeys})
        fg={k:generate(model.g_acc2audio,arrays[k]) for k in finalkeys}
        fe={fam:{k:embed(teacher,arrays[k],fg[k]) for k in finalkeys} for fam,teacher in teachers.items()}
        rows={}; dumped={}
        for name,route in locked.items():
            ss={}
            for k in finalkeys:
                if route['kind']=='original':s=fb[k]['score']
                elif route['kind']=='ridge':s=score(score_models[name],fb[k]['feature'])
                else:
                    e=fe[route['family']][k]; branch=route['branch']
                    x=np.concatenate([e['acc'],e['generated']],1) if branch=='both' else e[branch]
                    s=(fb[k]['score']-route['base_mean'])/route['base_std']+route['alpha']*(score(score_models[name],x)-route['teacher_mean'])/route['teacher_std']
                ss[k]=s; dumped[name+'_'+k]=s
            row=metrics(ss['final_genuine'],ss['final_impostor'],route['t'])
            row['selection']=config[name]; row['attack_by_user']={}
            for uid in np.unique(protocol.labels['final_impostor']):
                s=ss['final_impostor'][protocol.labels['final_impostor']==uid]
                row['attack_by_user'][str(int(uid)+1)]={'n':len(s),'accepted':int(np.sum(s>=route['t'])),'far':float(np.mean(s>=route['t']))}
            rows[name]=row
        np.savez_compressed(str(OUT/('U%d_final_scores.npz'%target)),**dumped)
        results[str(target)]={'routes':rows,'diagnostics':diagnostics,'counts':protocol.manifest['counts'],
                              'selected_on_development':selected,'generator_updated':False,
                              'final_test_reused':'Exploratory comparison on the prior final set; no independent confirmation yet'}
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        print('TARGET_DONE',target,{k:(v['tar'],v['far'],v['auc']) for k,v in rows.items()},flush=True)
        del model,binary,teachers,dev
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
