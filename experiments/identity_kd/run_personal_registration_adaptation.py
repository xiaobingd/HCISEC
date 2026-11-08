"""Personal registration adaptation; frozen identity-disjoint universal model.

Twelve independent claimed-user tasks are vectorized, with no shared trainable
parameters. Only registration positives and same-word Base negatives update them.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import run_p1_multimodal_enrollment as p
from compare_background_normalization import far_threshold

MODES=['linear','residual','residual_self','residual_audio']
def fit_models(x,audio,mode,steps):
    # x: independent task x (12 genuine + 48 Base negatives) x 192.
    pos=F.normalize(x[:,:12].mean(1),dim=1);neg=F.normalize(x[:,12:].mean(1),dim=1)
    init=3*F.normalize(pos-neg,dim=1);w=nn.Parameter(init.clone());bias=nn.Parameter(-.5*((init*pos).sum(1)+(init*neg).sum(1)))
    gamma=nn.Parameter(torch.zeros_like(w))
    params=[dict(params=[w,bias],lr=.02)]
    if mode!='linear':params.append(dict(params=[gamma],lr=.01))
    opt=torch.optim.Adam(params)
    for step in range(steps):
        zz=x if mode=='linear' else F.normalize(x*(1+.1*torch.tanh(gamma[:,None,:])),dim=2)
        logits=(zz*w[:,None,:]).sum(2)+bias[:,None]
        per_task=.5*F.softplus(-logits[:,:12]).mean(1)+.5*F.softplus(logits[:,12:]).mean(1)+.001*w.square().mean(1)
        if mode!='linear':per_task=per_task+.01*torch.tanh(gamma).square().mean(1)
        if mode=='residual_self':per_task=per_task+.1*(1-(zz[:,:12]*x[:,:12]).sum(2)).mean(1)
        if mode=='residual_audio':per_task=per_task+.1*(1-(zz[:,:12]*audio).sum(2)).mean(1)
        # Sum preserves each task's standalone gradient scale; Adam is elementwise.
        opt.zero_grad();per_task.sum().backward();opt.step()
    return dict(w=w.detach().numpy(),bias=bias.detach().numpy(),gamma=gamma.detach().numpy(),mode=mode)
def score(z,params,index):
    if params['mode']!='linear':z=p.norm(z*(1+.1*np.tanh(params['gamma'][index])))
    return z@params['w'][index]+params['bias'][index]
def run_fold(net,za,zb,d,f,steps):
    tasks=[];train=[];audio=[];prototypes=[];audit=[]
    for u in f['cal']+[f['target']]:
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                wa=[24,25,26] if direction==0 else [27,28,29];wb=[27,28,29] if direction==0 else [24,25,26];ww=wb if cond=='sameword' else wa
                def enroll(v):return np.concatenate([np.flatnonzero((d['user']==v)&(d['word']==w)&(d['role']=='final'))[:4] for w in ww])
                en=enroll(u);negative=np.concatenate([enroll(v) for v in f['base']]);assert len(en)==12 and len(negative)==48
                assert not set(en)&set(negative);assert np.isin(d['user'][negative],f['base']).all();assert (d['user'][en]==u).all()
                with torch.no_grad():teacher=net(torch.tensor(za[en])).numpy()
                tasks.append((u,cond,direction));train.append(zb[np.r_[en,negative]]);audio.append(teacher);prototypes.append(p.norm(zb[en].mean(0)))
                audit.append(dict(user=u,condition=cond,direction=direction,positive_ids=d['sample_id'][en].tolist(),negative_ids=d['sample_id'][negative].tolist()))
    x=torch.tensor(np.stack(train));t=torch.tensor(np.stack(audio));models={m:fit_models(x,t,m,steps) for m in MODES};lookup={task:j for j,task in enumerate(tasks)}
    def query(u,attacks,cond,direction,mode):
        ww=[27,28,29] if direction==0 else [24,25,26];ii=np.concatenate([np.flatnonzero((d['user']==v)&(d['word']==w)&(d['role']=='final'))[4:8] for v in [u]+attacks for w in ww]);index=lookup[u,cond,direction]
        assert not set(d['sample_id'][ii])&set(audit[index]['positive_ids']+audit[index]['negative_ids'])
        values=zb[ii]@prototypes[index] if mode=='cosine' else score(zb[ii],models[mode],index)
        return values,d['user'][ii]==u
    results={};thresholds={}
    for mode in ['cosine']+MODES:
        for cal_scope in ['unseen','mixed_legacy']:
            cs=[];cp=[]
            for cond in ['sameword','crossword']:
                for direction in [0,1]:
                    for u in f['cal']:
                        pool=f['cal'] if cal_scope=='unseen' else f['base']+f['cal']
                        s,y=query(u,[v for v in pool if v!=u],cond,direction,mode);cs.extend(s);cp.extend(y)
            cs=np.asarray(cs);cp=np.asarray(cp)
            for rule in ['eer','cal_far05']:
                tau=p.tau_eer(cs[cp],cs[~cp]) if rule=='eer' else far_threshold(cs[~cp]);name=mode+'_'+rule+'_'+cal_scope;results[name]={};thresholds[name]=dict(tau=tau,cal_far=float((cs[~cp]>=tau).mean()),cal_tar=float((cs[cp]>=tau).mean()),cal_genuine_n=int(cp.sum()),cal_impostor_n=int((~cp).sum()))
                if rule=='cal_far05':assert thresholds[name]['cal_far']<=.05+1e-12
                for cond in ['sameword','crossword']:
                    results[name][cond]={}
                    for scope,attacks in [('unseen_attacker',[f['attacker']]),('all_seven_sources',[v for v in range(1,9) if v!=f['target']])]:
                        ss=[];yy=[]
                        for direction in [0,1]:
                            s,y=query(f['target'],attacks,cond,direction,mode);ss.extend(s);yy.extend(y)
                        ss=np.asarray(ss);yy=np.asarray(yy);tar=float((ss[yy]>=tau).mean());far=float((ss[~yy]>=tau).mean());results[name][cond][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(yy,ss)),genuine_n=int(yy.sum()),impostor_n=int((~yy).sum()))
    return results,thresholds,models,audit
def main(a):
    torch.set_num_threads(1);root=Path('outputs/p1_multimodal_bundle');out=Path(a.out);out.mkdir(exist_ok=True,parents=True);meta=np.load(root/'metadata_private.npz');d={k:meta[k] for k in meta.files};orig=np.load('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');keep=orig['user']!=9
    za=np.load(root/(a.teacher+'_fit_private.npy'))[keep].astype('float32');extra=np.load(root/(a.teacher+'_enrollment_private.npz'));indices={str(s):i for i,s in enumerate(d['sample_id'])}
    for j,s in enumerate(extra['sample_id']):za[indices[str(s)]]=extra['audio'][j]
    p.write(out/'protocol.json',dict(scope='8targets3seeds;frozen4Baseonly universalHMD;2Cal;target;unseenattacker',methods=['cosine','linear','residual','residual_self','residual_audio'],registration='12genuine and48Base background negatives;exactlysamewords/repeats;suffixqueriesdisjoint',steps=a.steps,head_parameters=193,adapter_parameters=192,adapter='norm(z*(1+.1tanh(gamma))),zero initialized;no universalencoder update',lr_head=.02,lr_adapter=.01,loss='balanced BCE+.001meanw²+.01meantanhgamma²;self/Audioauxweight.1',teacher='frozen existing '+a.teacher+' plus samefold BaseonlyP1mapper;registrationpositiveAudioonly',selection='fixed200steps,no targetdev/test stopping;methodCalEER and CalFAR05 both predefined',threshold='primary:two Calusers attackeachother,unseen to universalmodel/personhead;secondary:old mixedBase+Calpool;separate results,not selectedbytest',cal_adaptation='Calusers personally adaptusingownregistration+sameBasepool,thenCalquerychoosessharedthresholdacrosssame/cross',limits='only4Basebackgroundidentities;allseven includesseennegativeidentities;wordlabels inferred;yue excludedexploratorily'))
    allresults={}
    for seed in [42,43,44]:
        for target in range(1,9):
            key=f'target{target}_seed{seed}';folder=out/key;folder.mkdir(exist_ok=True);path=folder/'results_private.json'
            if path.exists():allresults[key]=json.loads(path.read_text());continue
            f=p.fold(target);prior=Path(f'outputs/p1_enrollment_{a.teacher}_20261108')/key;old=json.loads((prior/'results_private.json').read_text());net=p.Mapper();net.load_state_dict(torch.load(prior/'mapper_private.pt',map_location='cpu',weights_only=True));net.eval()
            for parameter in net.parameters():parameter.requires_grad_(False)
            assert not any(parameter.requires_grad for parameter in net.parameters())
            zb=np.load(root/key/'acc_private.npy').astype('float32');p.write(out/'status.json',dict(state='personal_registration_adaptation',target=target,seed=seed,completed_folds=len(allresults)))
            metrics,tau,models,audit=run_fold(net,za,zb,d,f,a.steps)
            for cond in ['sameword','crossword']:
                for scope in ['unseen_attacker','all_seven_sources']:
                    for k in ['tar','far','hter','auc']:assert metrics['cosine_eer_mixed_legacy'][cond][scope][k]==old['methods']['E0'][cond][scope][k]
            np.savez(folder/'personal_parameters_private.npz',**{m+'_'+k:v for m,model in models.items() for k,v in model.items() if k!='mode'})
            p.write(folder/'registration_audit_private.json',audit);r=dict(target=target,seed=seed,methods=metrics,calibration=tau);p.write(path,r);allresults[key]=r;p.write(out/'results_private.json',allresults);print('PERSONAL',a.teacher,key,flush=True)
    p.write(out/'status.json',dict(state='complete',folds=len(allresults)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',choices=['ecapa','eres'],default='ecapa');ap.add_argument('--out',default='outputs/personal_adaptation_ecapa_20261108');ap.add_argument('--steps',type=int,default=200);main(ap.parse_args())
