"""Locked-template raw cosine vs enrollment-background Z-score vs Ridge."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import run_p1_multimodal_enrollment as p
from run_enrollment_threshold import descriptor

def far_threshold(negative,level=.05):
    vals=np.unique(negative);upper=np.nextafter(vals[-1],np.array(np.inf,dtype=vals.dtype));vals=np.r_[vals,upper]
    return float(min(t for t in vals if (negative>=t).mean()<=level+1e-12))
def evaluate(net,za,zb,d,f,fit,weight,learned):
    bg=p.norm(np.stack([zb[fit[d['user'][fit]==u]].mean(0) for u in f['base']]));reg={}
    for u in f['cal']+[f['target']]:
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                wa=[24,25,26] if direction==0 else [27,28,29];wb=[27,28,29] if direction==0 else [24,25,26];words=wb if cond=='sameword' else wa;en=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:4] for w in words])
                for method,w in [('E0',0.),('E2',weight)]:
                    x,t,b=descriptor(net,za[en],zb[en],d['word'][en],bg,w);model=learned[method];pred=float(np.clip(((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coef'])+model['intercept'],-1,1));scores=bg@t
                    reg[u,cond,direction,method]=dict(template=t,mu=float(scores.mean()),sigma=float(scores.std()),pred=pred,bg=b)
    def query(u,attacks,cond,direction,method):
        words=[27,28,29] if direction==0 else [24,25,26];ii=np.concatenate([np.flatnonzero((d['user']==v)&(d['word']==w)&(d['role']=='final'))[4:8] for v in [u]+attacks for w in words]);return zb[ii]@reg[u,cond,direction,method]['template'],d['user'][ii]==u
    def transform(scores,r,kind):return (scores.astype('float64')-r['mu'])/(r['sigma']+1e-6) if kind=='zscore' else scores
    results={};locked={};diagnosis={}
    for method in ['E0','E2']:
        cal=[];predictions=[]
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                for u in f['cal']:
                    s,y=query(u,[v for v in f['base']+f['cal'] if v!=u],cond,direction,method);r=reg[u,cond,direction,method];cal.append((s,y,r));predictions.append(r['pred'])
        labels=np.concatenate([y for s,y,r in cal]);anchor=float(np.median(predictions))
        for point in ['eer','cal_far05']:
            for kind in ['raw','zscore','ridge']:
                ss=np.concatenate([transform(s,r,kind) for s,y,r in cal]);tau=p.tau_eer(ss[labels],ss[~labels]) if point=='eer' else far_threshold(ss[~labels]);eta=0.;floor=False;calfar=float((ss[~labels]>=tau).mean())
                def threshold(r,e,fl):return max(tau+e*float(np.clip(r['pred']-anchor,-.1,.1)),r['bg']+learned[method]['background_margin']) if fl else tau+e*float(np.clip(r['pred']-anchor,-.1,.1))
                if kind=='ridge':
                    candidates=[]
                    for e in [0.,.25,.5,1.]:
                        for fl in [False,True]:
                            margins=np.concatenate([s-threshold(r,e,fl) for s,y,r in cal]);far=float((margins[~labels]>=0).mean());frr=float((margins[labels]<0).mean())
                            if far<=calfar+1e-12:candidates.append(((far+frr)/2,far,e,fl))
                    _,_,eta,floor=min(candidates,key=lambda x:(x[0],x[1],x[2],x[3]))
                name=method+'_'+kind+'_'+point;results[name]={};calm=np.concatenate([transform(s,r,kind)-(threshold(r,eta,floor) if kind=='ridge' else tau) for s,y,r in cal]);locked[name]=dict(threshold=tau,eta=eta,background_floor=floor,cal_far=float((calm[~labels]>=0).mean()),cal_tar=float((calm[labels]>=0).mean()))
                if point=='cal_far05':assert locked[name]['cal_far']<=.05+1e-12
                for cond in ['sameword','crossword']:
                    results[name][cond]={};dr=[]
                    for direction in [0,1]:
                        r=reg[f['target'],cond,direction,method];raw_tau=r['mu']+tau*(r['sigma']+1e-6) if kind=='zscore' else threshold(r,eta,floor) if kind=='ridge' else tau;dr.append(dict(mu=r['mu'],sigma=r['sigma'],equivalent_cosine_threshold=raw_tau))
                    diagnosis[name+'_'+cond]=dr
                    for scope,attacks in [('unseen_attacker',[f['attacker']]),('all_seven_sources',[u for u in range(1,9) if u!=f['target']])]:
                        raw=[];margin=[];truth=[]
                        for direction in [0,1]:
                            s,y=query(f['target'],attacks,cond,direction,method);r=reg[f['target'],cond,direction,method];t=threshold(r,eta,floor) if kind=='ridge' else tau;raw.extend(s);margin.extend(transform(s,r,kind)-t);truth.extend(y)
                        raw=np.array(raw);margin=np.array(margin);truth=np.array(truth);tar=float((margin[truth]>=0).mean());far=float((margin[~truth]>=0).mean());results[name][cond][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,raw_auc=float(roc_auc_score(truth,raw)),margin_auc=float(roc_auc_score(truth,margin)),genuine_n=int(truth.sum()),impostor_n=int((~truth).sum()))
    return results,locked,diagnosis
def main(a):
    torch.set_num_threads(1);root=Path('outputs/p1_multimodal_bundle');out=Path(a.out);out.mkdir(exist_ok=True,parents=True);m=np.load(root/'metadata_private.npz');d={k:m[k] for k in m.files};old=np.load('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');keep=old['user']!=9;manifest=json.loads(Path('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_manifest_private.json').read_text(encoding='utf-8'));exact={r['sample_id'] for r in manifest['records'] if r['pair_mode']=='exact_timestamp'};valid=np.array([str(s) in exact for s in d['sample_id']]);za=np.load(root/(a.teacher+'_fit_private.npy'))[keep].astype('float32');extra=np.load(root/(a.teacher+'_enrollment_private.npz'));lookup={str(s):i for i,s in enumerate(d['sample_id'])}
    for j,s in enumerate(extra['sample_id']):za[lookup[str(s)]]=extra['audio'][j]
    p.write(out/'protocol.json',dict(teacher=a.teacher,methods=['raw','enrollment-background-zscore','existing Ridge'],templates=['E0','E2'],calibration=['same prior CalEER','empiricalCalFAR<=5%'],background='4Basefitidentityprototypes,equalidentityweight,noCalTargetAttacker',normalization='(cosine-mean(template dot Basecenters))/(populationstd+1e-6)',no_hyperparameter_search=True,query='HMDonly;targetenrollmentonly for stats',limits='4background identities;not ASNorm;5%CalFAR notunknownFARguarantee;samegenerator ratherthansame rawthreshold acrosssame/cross'))
    results={}
    for seed in [42,43,44]:
        for target in range(1,9):
            key=f'target{target}_seed{seed}';folder=out/key;folder.mkdir(exist_ok=True);path=folder/'results_private.json'
            if path.exists():results[key]=json.loads(path.read_text());continue
            f=p.fold(target);source=Path(f'outputs/p1_enrollment_{a.teacher}_20261108')/key;prior=json.loads((source/'results_private.json').read_text());ridge=json.loads((Path(f'outputs/enrollment_threshold_{a.teacher}_20261108')/key/'results_private.json').read_text());net=p.Mapper();net.load_state_dict(torch.load(source/'mapper_private.pt',map_location='cpu',weights_only=True));net.eval();zb=np.load(root/key/'acc_private.npy').astype('float32');fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit')&valid);v,cal,diag=evaluate(net,za,zb,d,f,fit,prior['selection']['selected_lambda'],ridge['learned_threshold_model'])
            for method in ['E0','E2']:
                for cond in ['sameword','crossword']:
                    for scope in ['unseen_attacker','all_seven_sources']:
                        for metric in ['tar','far','hter']:
                            assert v[method+'_raw_eer'][cond][scope][metric]==prior['methods'][method][cond][scope][metric]
                            assert v[method+'_ridge_eer'][cond][scope][metric]==ridge['methods'][method+'_adaptive_guarded'][cond][scope][metric]
            value=dict(target=target,seed=seed,methods=v,calibration=cal,registration_statistics=diag);p.write(path,value);results[key]=value;p.write(out/'results_private.json',results);print('NORMALIZATION',a.teacher,key,flush=True)
    p.write(out/'status.json',dict(state='complete',folds=len(results)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',choices=['ecapa','eres'],default='ecapa');ap.add_argument('--out',default='outputs/background_norm_ecapa_20261108');main(ap.parse_args())