"""Enrollment-conditioned threshold: Base-trained g, Cal policy, HMD-only test."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import run_p1_multimodal_enrollment as p
from run_e3_enrollment_quality import quality,blend

def descriptor(net,za,zb,words,background,weight):
    q,pb,pa=quality(net,za,zb,words);template=blend(pb,pa,weight);bg=background@template
    return np.r_[q,np.median(bg),np.quantile(bg,.9),np.max(bg)],template,float(np.max(bg))
def learn(za,zb,d,f,fit,seed,steps,weight):
    data={m:dict(x=[],y=[],bg=[]) for m in ['E0','E2']}
    for held in f['base']:
        users=[u for u in f['base'] if u!=held];tr=fit[np.isin(d['user'][fit],users)];torch.manual_seed(seed);net=p.Mapper();p.train(net,za,zb,d,tr,users,seed,steps);net.eval()
        background=p.norm(np.stack([zb[fit[d['user'][fit]==u]].mean(0) for u in users]));pool=fit[d['user'][fit]==held];words=np.unique(d['word'][pool]);rng=np.random.RandomState(seed+held*1000)
        query=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='develop'))[:2] for u in f['base'] for w in [8,9,18,19]]);truth=d['user'][query]==held
        for episode in range(24):
            en=np.concatenate([rng.choice(pool[d['word'][pool]==w],4,replace=False) for w in rng.choice(words,3,replace=False)])
            for method,w in [('E0',0.),('E2',weight)]:
                x,template,bg=descriptor(net,za[en],zb[en],d['word'][en],background,w);scores=zb[query]@template;tau=p.tau_eer(scores[truth],scores[~truth]);data[method]['x'].append(x);data[method]['y'].append(tau);data[method]['bg'].append(bg)
    models={};metadata={}
    for method,v in data.items():
        x=np.asarray(v['x']);y=np.asarray(v['y']);sc=StandardScaler().fit(x);reg=Ridge(alpha=20.).fit(sc.transform(x),y);margin=float(np.clip(np.quantile(y-np.asarray(v['bg']),.1),0,.05));models[method]=(sc,reg,margin)
        metadata[method]=dict(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=reg.coef_.tolist(),intercept=float(reg.intercept_),background_margin=margin,episodes=len(y))
    return models,metadata
def evaluate(net,za,zb,d,f,fit,weight,models):
    background=p.norm(np.stack([zb[fit[d['user'][fit]==u]].mean(0) for u in f['base']]));registrations={}
    for u in f['cal']+[f['target']]:
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                wa=[24,25,26] if direction==0 else [27,28,29];wb=[27,28,29] if direction==0 else [24,25,26];ww=wb if cond=='sameword' else wa;en=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:4] for w in ww])
                for method,w in [('E0',0.),('E2',weight)]:
                    x,t,bg=descriptor(net,za[en],zb[en],d['word'][en],background,w);sc,reg,_=models[method];pred=float(np.clip(reg.predict(sc.transform(x[None]))[0],-1,1));registrations[u,cond,direction,method]=dict(template=t,pred=pred,bg=bg)
    def query(u,attackers,cond,direction,method):
        ww=[27,28,29] if direction==0 else [24,25,26];ii=np.concatenate([np.flatnonzero((d['user']==v)&(d['word']==w)&(d['role']=='final'))[4:8] for v in [u]+attackers for w in ww]);return zb[ii]@registrations[u,cond,direction,method]['template'],d['user'][ii]==u
    result={};policies={};thresholds={}
    for method in ['E0','E2']:
        cal=[];preds=[]
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                for u in f['cal']:
                    scores,y=query(u,[v for v in f['base']+f['cal'] if v!=u],cond,direction,method);reg=registrations[u,cond,direction,method];cal.append((scores,y,reg));preds.append(reg['pred'])
        scores=np.concatenate([v[0] for v in cal]);labels=np.concatenate([v[1] for v in cal]);tau=p.tau_eer(scores[labels],scores[~labels]);anchor=float(np.median(preds));basefar=float((scores[~labels]>=tau).mean());bgmargin=models[method][2]
        def threshold(reg,eta,floor):
            value=tau+eta*float(np.clip(reg['pred']-anchor,-.1,.1))
            return max(value,reg['bg']+bgmargin) if floor else value
        policies[method]={}
        for policy in ['fixed','adaptive','adaptive_guarded']:
            candidates=[]
            for eta in ([0.] if policy=='fixed' else [0.,.25,.5,1.]):
                for floor in ([False,True] if policy=='adaptive_guarded' else [False]):
                    margins=np.concatenate([s-threshold(reg,eta,floor) for s,y,reg in cal]);far=float((margins[~labels]>=0).mean());frr=float((margins[labels]<0).mean())
                    if far<=basefar+1e-12:candidates.append(((far+frr)/2,far,eta,floor))
            hter,far,eta,floor=min(candidates,key=lambda x:(x[0],x[1],x[2],x[3]));name=method+'_'+policy;policies[method][policy]=dict(tau=tau,cal_anchor=anchor,eta=eta,background_floor=floor,cal_far=far,cal_hter=hter,baseline_cal_far=basefar);result[name]={}
            for cond in ['sameword','crossword']:
                result[name][cond]={}
                for scope,attacks in [('unseen_attacker',[f['attacker']]),('all_seven_sources',[u for u in range(1,9) if u!=f['target']])]:
                    ss=[];yy=[];mm=[];tt=[]
                    for direction in [0,1]:
                        s,y=query(f['target'],attacks,cond,direction,method);t=threshold(registrations[f['target'],cond,direction,method],eta,floor);ss.extend(s);yy.extend(y);mm.extend(s-t);tt.append(t)
                    ss=np.asarray(ss);yy=np.asarray(yy);mm=np.asarray(mm);tar=float((mm[yy]>=0).mean());far=float((mm[~yy]>=0).mean());result[name][cond][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,raw_auc=float(roc_auc_score(yy,ss)),margin_auc=float(roc_auc_score(yy,mm)),genuine_n=int(yy.sum()),impostor_n=int((~yy).sum()));thresholds[name+'_'+cond]=tt
    return result,policies,thresholds
def main(a):
    torch.set_num_threads(1);root=Path('outputs/p1_multimodal_bundle');out=Path(a.out);out.mkdir(exist_ok=True,parents=True);meta=np.load(root/'metadata_private.npz');d={k:meta[k] for k in meta.files};old=np.load('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');keep=old['user']!=9
    manifest=json.loads(Path('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_manifest_private.json').read_text(encoding='utf-8'));exact={r['sample_id'] for r in manifest['records'] if r['pair_mode']=='exact_timestamp'};valid=np.array([str(s) in exact for s in d['sample_id']]);za=np.load(root/(a.teacher+'_fit_private.npy'))[keep].astype('float32');extra=np.load(root/(a.teacher+'_enrollment_private.npz'));lookup={str(s):j for j,s in enumerate(d['sample_id'])}
    for j,s in enumerate(extra['sample_id']):za[lookup[str(s)]]=extra['audio'][j]
    p.write(out/'protocol.json',dict(teacher=a.teacher,scope='registration-conditioned threshold;fixed E0/E2 templates and frozenP1 models',g='Ridge alpha20 on6registrationquality+3Basebackgroundrelations;96Baseepisodes',policy='Cal selects shrinketa in0/.25/.5/1 and optionalbackgroundfloor subjectto CalFAR<=fixedCalFAR;eta0noFloor fallback',offset_cap=.1,background='otherBase prototypes inBaseepisodes;all4Base prototypes inouterregistration;maxscorefloor+Base10percentilemarginclipped0..05',limits='only4Baseusers;Baseencoderseen;fewCalidentities;CalFARconstraintNOTunseenattackguarantee',same_cross='samegeneratorandpolicy;numericthresholdmaychangedeterministicallywithregistrationwordset',query='HMDonly;no targetquerylabel used for g or policy'))
    results={}
    for seed in [42,43,44]:
        for target in range(1,9):
            key=f'target{target}_seed{seed}';folder=out/key;folder.mkdir(exist_ok=True);path=folder/'results_private.json'
            if path.exists():results[key]=json.loads(path.read_text());continue
            f=p.fold(target);prior=Path(f'outputs/p1_enrollment_{a.teacher}_20261108')/key;source=json.loads((prior/'results_private.json').read_text());zb=np.load(root/key/'acc_private.npy').astype('float32');fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit')&valid)
            p.write(out/'status.json',dict(state='learning_base_threshold_model',target=target,seed=seed,completed_folds=len(results)));models,learned=learn(za,zb,d,f,fit,seed,source['selection']['selected_step'],source['selection']['selected_lambda']);net=p.Mapper();net.load_state_dict(torch.load(prior/'mapper_private.pt',map_location='cpu',weights_only=True));net.eval();metrics,policy,thresholds=evaluate(net,za,zb,d,f,fit,source['selection']['selected_lambda'],models)
            for method in ['E0','E2']:
                for cond in ['sameword','crossword']:
                    for scope in ['unseen_attacker','all_seven_sources']:
                        for k in ['tar','far','hter']:assert metrics[method+'_fixed'][cond][scope][k]==source['methods'][method][cond][scope][k]
            value=dict(target=target,seed=seed,fold=f,methods=metrics,cal_policy=policy,learned_threshold_model=learned,target_thresholds=thresholds);p.write(path,value);results[key]=value;p.write(out/'results_private.json',results);print('THRESHOLD_FOLD',a.teacher,key,flush=True)
    p.write(out/'status.json',dict(state='complete',folds=len(results)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',choices=['ecapa','eres'],default='ecapa');ap.add_argument('--out',default='outputs/enrollment_threshold_ecapa_20261108');main(ap.parse_args())
