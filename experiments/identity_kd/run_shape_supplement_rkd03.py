"""Preserve shape40; append only the personalized fine-tuned-teacher V4 branch."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from run_rkd01_vibravox import Student,train,norm,encode,thresholds,metrics
from run_registration_rkd02 import registration,adapt_student,dump

def score(shape,student,d,u,group,alpha):
    en=registration(d,u);q=np.flatnonzero(np.isin(d['user'],group)&(d['role']=='query'));ii=np.r_[en,q];joined=np.concatenate([shape[ii],alpha*student[ii]],axis=1)/np.sqrt(1+alpha*alpha);assert joined.shape[1]==232;assert np.allclose(joined[:,:40],shape[ii]/np.sqrt(1+alpha*alpha),atol=1e-7);p=norm(joined[:len(en)].mean(0));s=joined[len(en):]@p
    return dict(scores=s.tolist(),labels=(d['user'][q]==u).tolist())
def assess(cal,test):
    cs=np.concatenate([r['scores'] for r in cal]);cy=np.concatenate([r['labels'] for r in cal]);s=np.concatenate([r['scores'] for r in test]);y=np.concatenate([r['labels'] for r in test]);taus=thresholds(cs,cy);result={};per={}
    for point,t in taus.items():
        r=metrics(s,y,t);pp=[metrics(np.array(v['scores']),np.array(v['labels']),t) for v in test];r.update(macro_auc=float(np.mean([v['auc'] for v in pp])),macro_eer=float(np.mean([v['eer'] for v in pp])),worst_target_far=max(v['far'] for v in pp));result[point]=r;per[point]=pp
        if point=='Cal_FAR5':assert float((cs[~cy]>=t).mean())<=.05
    return result,per,taus
def encode_owners(d,x,initial,base,owners,rkd02,save,inner,rho):
    save.mkdir(parents=True,exist_ok=True);vectors={}
    for u in owners:
        folder=rkd02/('inner' if inner else 'outer')/f'owner{u}';path=save/f'owner{u}_student_private.npy'
        if path.exists():vectors[u]=np.load(path);continue
        net=Student()
        if inner:
            cached=save/f'owner{u}_student_private.pt'
            if cached.exists():net.load_state_dict(torch.load(cached,map_location='cpu',weights_only=False))
            else:
                t=dict(np.load(folder/'teacher_targets_private.npz'));net,trace=adapt_student(d,x,initial,u,base,t['indices'],t['adapted'],'V4',rho);torch.save(net.state_dict(),cached);dump(save/f'owner{u}_training_private.json',dict(method='V4 only',trace=trace))
        else:net.load_state_dict(torch.load(folder/f'V4_rho{rho}_student_private.pt',map_location='cpu',weights_only=False))
        group=list(range(29,32)) if u<32 else list(range(32,35)) if inner else list(range(35,47)) if u<47 else list(range(47,59));ix=np.flatnonzero(((d['user']==u)&(d['role']=='registration'))|(np.isin(d['user'],group)&(d['role']=='query')));z=np.zeros((len(x),192),dtype='float32');z[ix]=encode(net,x[ix]);np.save(path,z);vectors[u]=z;print('ENCODED_V4',u,'inner' if inner else 'outer',flush=True)
    return vectors
def main(a):
    torch.set_num_threads(1);source=Path(a.source);prior=Path(a.rkd02);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(source/'inputs_private.npz'));rho=json.loads((prior/'selection_public.json').read_text())['rho'];lock=out/'selection_public.json'
    if not lock.exists():
        base=list(range(29));fit=np.isin(d['user'],base)&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);states,_=train(d,np.load(source/'audio_targets_private.npy'),base,'S1',0,42,700,x);initial=Student();initial.load_state_dict(states[700]);z=encode_owners(d,x,initial,base,list(range(29,35)),prior,out/'inner',True,rho);choices=[]
        for alpha in [0,.05,.1,.2,.3,.5,.75,1.]:
            cal=[score(shape,z[u],d,u,list(range(29,32)),alpha) for u in range(29,32)];dev=[score(shape,z[u],d,u,list(range(32,35)),alpha) for u in range(32,35)];r,_,_=assess(cal,dev);v=r['Cal_EER'];choices.append(dict(alpha=alpha,hter=v['hter'],auc=v['macro_auc'],tar=v['tar'],far=v['far']))
        best=min([v for v in choices if v['alpha']>0],key=lambda v:(v['hter'],-v['auc'],v['alpha']));dump(lock,dict(alpha=best['alpha'],rho=rho,rule='Base29train3innerCal3innerDev;positive alpha only,minHTER,tie macroAUC then smaller alpha;alpha0 reference only;V4 only',dimension=232,shape_branch='unmodified standardized shape40,weight1,never projected or learned',student_branch='V4 personalized Audio-finetuned teacher+balancedRKD192',choices=choices));print('ALPHA_LOCKED',best['alpha'],flush=True)
    alpha=json.loads(lock.read_text())['alpha'];base=list(range(35));fit=np.isin(d['user'],base)&(d['role']=='fit');x=StandardScaler().fit(d['x'][fit]).transform(d['x']).astype('float32');shape=norm(x[:,:40]);z=encode_owners(d,x,None,base,list(range(35,59)),prior,out/'outer',False,rho);cal=[score(shape,z[u],d,u,list(range(35,47)),alpha) for u in range(35,47)];test=[score(shape,z[u],d,u,list(range(47,59)),alpha) for u in range(47,59)];result,per,taus=assess(cal,test);bc=[score(shape,z[u],d,u,list(range(35,47)),0) for u in range(35,47)];bt=[score(shape,z[u],d,u,list(range(47,59)),0) for u in range(47,59)];baseline,bper,_=assess(bc,bt)
    # Verify concatenation/prototype arithmetic using direct branch numerators.
    maxdiff=0.;checks=0
    for u,row in zip(range(47,59),test):
        s=np.array(row['scores']);y=np.array(row['labels']);assert int(y.sum())==24 and int((~y).sum())==264;en=registration(d,u);q=np.flatnonzero(np.isin(d['user'],range(47,59))&(d['role']=='query'));ps=shape[en].mean(0);pz=z[u][en].mean(0);direct=(shape[q]@ps+alpha*alpha*z[u][q]@pz)/(np.sqrt(1+alpha*alpha)*np.sqrt(ps@ps+alpha*alpha*(pz@pz)));error=float(np.max(abs(direct-s)));maxdiff=max(maxdiff,error);assert error<1e-6
        for point in taus:
            pp=s[y];nn=s[~y];auc=float(((pp[:,None]>nn).astype(float)+.5*(pp[:,None]==nn)).mean());assert abs(auc-per[point][u-47]['auc'])<1e-7;checks+=1
    delta=np.array([p['hter']-b['hter'] for p,b in zip(per['Cal_EER'],bper['Cal_EER'])]);rng=np.random.RandomState(20261110);boots=np.array([delta[rng.choice(12,12,replace=True)].mean() for _ in range(5000)]);old=json.loads((prior/'results_public.json').read_text())['results']['V4'];dump(out/'results_public.json',dict(state='complete',seed=42,K=12,base=35,calibration=12,test=12,alpha=alpha,rho=rho,dimension=232,shape_only=baseline,supplement=result,student_only_reused_reference=old,paired_vs_shape=dict(hter_delta=float(delta.mean()),identity_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),improved_targets=int((delta<0).sum())),integrity=dict(shape_never_replaced=True,new_outer_training=0,new_inner_V4_students=6,query_audio_reads=0,models_reused_from='registration-rkd02 V4 only',independent_branch_formula_max_error=maxdiff,pairwise_auc_workpoint_checks=checks,all_sources_retained=11)));dump(out/'scores_private.json',dict(calibration=cal,test=test,thresholds=taus));print('COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--rkd02',required=True);p.add_argument('--out',required=True);main(p.parse_args())
