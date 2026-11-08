"""E3 enrollment-only fusion quality gating; reuse frozen P1 encoders/maps."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
import run_p1_multimodal_enrollment as p

def quality(net,za,zb,words):
    pa=p.norm(za.mean(0));pb=p.norm(zb.mean(0))
    with torch.no_grad():mapped=net(torch.tensor(pa[None])).numpy()[0];allmapped=net(torch.tensor(za)).numpy()
    cross=words[:,None]!=words[None,:]
    qa=float((za@pa).mean());qb=float((zb@pb).mean())
    qac=float((za@za.T)[cross].mean());qbc=float((zb@zb.T)[cross].mean())
    agree=float(mapped@pb);paired=float((allmapped*zb).sum(1).mean())
    features=np.array([qa,qb,qac,qbc,agree,paired],dtype='float64')
    return features,pb,mapped
def blend(pb,pa,w):return pb if w==0 else pa if w==1 else p.norm((1-w)*pb+w*pa)
def train_gate(za,zb,d,f,fit,seed,steps):
    features=[];labels=[];identity=[];auc_rows=[]
    for held in f['base']:
        users=[u for u in f['base'] if u!=held];trainix=fit[np.isin(d['user'][fit],users)]
        torch.manual_seed(seed);net=p.Mapper();p.train(net,za,zb,d,trainix,users,seed,steps);net.eval()
        rng=np.random.RandomState(seed+held*1000);pool=fit[d['user'][fit]==held];words=np.unique(d['word'][pool])
        query=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='develop'))[:2] for u in f['base'] for w in [8,9,18,19]])
        truth=d['user'][query]==held
        for episode in range(24):
            ww=rng.choice(words,3,replace=False);en=np.concatenate([rng.choice(pool[d['word'][pool]==w],4,replace=False) for w in ww]);q,pb,pa=quality(net,za[en],zb[en],d['word'][en])
            values=[float(roc_auc_score(truth,zb[query]@blend(pb,pa,w))) for w in p.LAMBDAS]
            j=max(range(len(values)),key=lambda j:(values[j],-p.LAMBDAS[j]));features.append(q);labels.append(p.LAMBDAS[j]);identity.append(held);auc_rows.append(values)
    x=np.asarray(features);y=np.asarray(labels);scaler=StandardScaler().fit(x);xx=scaler.transform(x)
    full=Ridge(alpha=10.).fit(xx,y);hmd=Ridge(alpha=10.).fit(xx[:,[1,3]],y)
    distribution=np.clip(full.predict(xx),0,1)
    return scaler,full,hmd,distribution,dict(episodes=len(y),base_users=f['base'],label_lambda_counts={str(w):int((y==w).sum()) for w in p.LAMBDAS},features=['audio_mean_consistency','hmd_mean_consistency','audio_crossword_consistency','hmd_crossword_consistency','prototype_agreement','paired_mapped_agreement'],ridge_alpha=10.,coef=full.coef_.tolist(),intercept=float(full.intercept_),limits='4Base identities;mapper-only inneridentityholdout;fixedencoder saw all4Base;oraclelambda is Base-development training target,never outertest',fit_quality_mean=x.mean(0).tolist(),fit_quality_std=x.std(0).tolist())
def evaluate(net,za,zb,d,f,weight,gate,seed):
    scaler,full,hmd,distribution,_=gate;templates={};selected={}
    for u in f['cal']+[f['target']]:
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                wa=[24,25,26] if direction==0 else [27,28,29];wb=[27,28,29] if direction==0 else [24,25,26];ew=wb if cond=='sameword' else wa
                en=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[:4] for w in ew])
                q,pb,pa=quality(net,za[en],zb[en],d['word'][en]);x=scaler.transform(q[None]);wf=float(np.clip(full.predict(x)[0],0,1));wh=float(np.clip(hmd.predict(x[:,[1,3]])[0],0,1))
                # Random control samples the same fitted Base-gate distribution.
                rng=np.random.RandomState(seed+u*1000+direction*10+(cond=='crossword'));wr=float(rng.choice(distribution))
                weights={'E0':0.,'E2':weight,'E3':wf,'HMD_quality':wh,'random_quality':wr}
                templates[u,cond,direction]={m:blend(pb,pa,w) for m,w in weights.items()};selected[f'{u}_{cond}_{direction}']=dict(weights=weights,quality=q.tolist())
    def score(client,attacks,cond,direction,method):
        ww=[27,28,29] if direction==0 else [24,25,26];ii=np.concatenate([np.flatnonzero((d['user']==u)&(d['word']==w)&(d['role']=='final'))[4:8] for u in [client]+attacks for w in ww])
        return zb[ii]@templates[client,cond,direction][method],d['user'][ii]==client
    results={};diagnosis={}
    for method in ['E0','E2','E3','HMD_quality','random_quality']:
        cs=[];cp=[]
        for cond in ['sameword','crossword']:
            for direction in [0,1]:
                for client in f['cal']:
                    s,y=score(client,[u for u in f['base']+f['cal'] if u!=client],cond,direction,method);cs.extend(s);cp.extend(y)
        cs=np.asarray(cs);cp=np.asarray(cp);tau=p.tau_eer(cs[cp],cs[~cp]);results[method]={}
        for cond in ['sameword','crossword']:
            results[method][cond]={}
            for scope,attacks in [('unseen_attacker',[f['attacker']]),('all_seven_sources',[u for u in range(1,9) if u!=f['target']])]:
                ss=[];yy=[]
                for direction in [0,1]:
                    s,y=score(f['target'],attacks,cond,direction,method);ss.extend(s);yy.extend(y)
                ss=np.asarray(ss);yy=np.asarray(yy);tar=float((ss[yy]>=tau).mean());far=float((ss[~yy]>=tau).mean());fpr,tpr,_=roc_curve(yy,ss);k=np.argmin(np.abs(fpr-(1-tpr)))
                results[method][cond][scope]=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(yy,ss)),descriptive_eer=float((fpr[k]+1-tpr[k])/2),threshold=tau,genuine_n=int(yy.sum()),impostor_n=int((~yy).sum()))
                # Retrospective threshold is diagnostic ONLY; never used by gates/scoring.
                if scope=='unseen_attacker' and cond=='crossword':
                    oracle=p.tau_eer(ss[yy],ss[~yy]);oh=float(((ss[yy]<oracle).mean()+(ss[~yy]>=oracle).mean())/2)
                    diagnosis[method]=dict(cal_threshold=tau,cal_genuine_median=float(np.median(cs[cp])),cal_impostor_median=float(np.median(cs[~cp])),target_genuine_median=float(np.median(ss[yy])),target_impostor_median=float(np.median(ss[~yy])),frozen_threshold_hter=(1-tar+far)/2,retrospective_test_threshold=oracle,retrospective_test_hter=oh,warning='test-label threshold is optimistic diagnostic,not deployable performance and never fed back')
    return results,selected,diagnosis
def main(a):
    torch.set_num_threads(1);root=Path('outputs/p1_multimodal_bundle');out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    m=np.load(root/'metadata_private.npz');d={k:m[k] for k in m.files};original=np.load('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');keep=original['user']!=9
    man=json.loads(Path('outputs/identity_kd_all9_raw_full_multisource_20261103/raw9_manifest_private.json').read_text(encoding='utf-8'));exact={r['sample_id'] for r in man['records'] if r['pair_mode']=='exact_timestamp'};valid=np.array([str(s) in exact for s in d['sample_id']])
    za=np.load(root/(a.teacher+'_fit_private.npy'))[keep].astype('float32');extra=np.load(root/(a.teacher+'_enrollment_private.npz'));lookup={str(s):i for i,s in enumerate(d['sample_id'])}
    for j,s in enumerate(extra['sample_id']):za[lookup[str(s)]]=extra['audio'][j]
    p.write(out/'protocol.json',dict(experiment='E3 enrollment quality gate',teacher=a.teacher,backbones_and_mapper='reuse frozen P1 per-fold models, no encoder changes',quality='registration embeddings only,not measuredSNR',gate='standardscaled Ridge alpha10->clip[0,1];fixed configuration before test',training='4Base mapperheldout identities x24 support episodes,12support+Base develop HMDquery',controls=['E0','E2','HMD_quality','random_quality'],threshold='method-specific independentCal threshold sharedsame/cross;no target-specific adjustment',limits='4Base identities;innermapperheldout only;inferredwordlabels;no synchronization coupling claim'))
    allresults={}
    for seed in [42,43,44]:
        for target in range(1,9):
            key=f'target{target}_seed{seed}';folder=out/key;folder.mkdir(exist_ok=True);path=folder/'results_private.json'
            if path.exists():allresults[key]=json.loads(path.read_text());continue
            f=p.fold(target);prior=Path(f'outputs/p1_enrollment_{a.teacher}_20261108')/key;old=json.loads((prior/'results_private.json').read_text());zb=np.load(root/key/'acc_private.npy').astype('float32');fit=np.flatnonzero(np.isin(d['user'],f['base'])&(d['role']=='fit')&valid)
            p.write(out/'status.json',dict(state='training_enrollment_gate',target=target,seed=seed,completed_folds=len(allresults)))
            gate=train_gate(za,zb,d,f,fit,seed,old['selection']['selected_step']);net=p.Mapper();net.load_state_dict(torch.load(prior/'mapper_private.pt',map_location='cpu',weights_only=True));net.eval()
            metrics,weights,diagnosis=evaluate(net,za,zb,d,f,old['selection']['selected_lambda'],gate,seed)
            for method in ['E0','E2']:
                for cond in ['sameword','crossword']:
                    for scope in ['unseen_attacker','all_seven_sources']:
                        for k in ['tar','far','hter','auc','threshold']:assert abs(metrics[method][cond][scope][k]-old['methods'][method][cond][scope][k])<1e-6
            result=dict(target=target,seed=seed,fold=f,methods=metrics,registration_quality=weights,gate_training=gate[-1],threshold_diagnosis=diagnosis);p.write(path,result);allresults[key]=result;p.write(out/'results_private.json',allresults);print('E3_FOLD',a.teacher,key,flush=True)
    p.write(out/'status.json',dict(state='complete',folds=len(allresults)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',choices=['ecapa','eres'],default='ecapa');ap.add_argument('--out',default='outputs/e3_enrollment_ecapa_20261108');main(ap.parse_args())
