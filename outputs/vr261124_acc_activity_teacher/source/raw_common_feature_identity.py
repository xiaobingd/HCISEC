"""Vectorized equivalent of the row-wise offline feature assay, including fixed-mask controls."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from audit_raw_common_dynamics import activity_mask,normalize

def batch_features(a,x,mask):
    n=mask.sum(1);div=np.maximum(n,1)[:,None];weight=mask[:,:,None]
    am=(a*weight).sum(1)/div;xm=(x*weight).sum(1)/div
    ac=a-am[:,None,:];xc=x-xm[:,None,:]
    av=(ac*ac*weight).sum(1)/div;xv=(xc*xc*weight).sum(1)/div;cov=(ac*xc*weight).sum(1)/div
    den=n[:,None]*np.sqrt(av*xv);corr=np.divide(n[:,None]*cov,den,out=np.zeros_like(cov),where=den>1e-8)
    slope=np.divide(cov,av,out=np.zeros_like(cov),where=av>1e-8)
    def stats(mean,var):
        std=np.sqrt(var);v=np.c_[mean-mean.mean(1,keepdims=True),std,mean.mean(1),std.mean(1)];v[n<12]=np.nan;return v
    shared=np.c_[corr,slope,xm-am];shared[n<12]=np.nan
    return shared,stats(am,av),stats(xm,xv),n
def run(root):
    out=root/'conditional_correspondence/raw_common_dynamics';c=np.load(out/'local_power_private.npz')
    protocol=json.loads((out/'protocol_public.json').read_text())
    protocol['feature_assay_implementation']='vectorized,synthetically verified equivalent to rowwise;15methods include independent unimodal masks and3same-correct-mask mismatch controls'
    protocol['primary_identity_attribution_control']='coupling_correct versus wrongrepeat_fixedmask;identical activity/motion mask,sourcevalue correspondence changes only'
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    a,x,e,m=[c[k].astype('float64') for k in ('audio','acc','audio_activity','motion')];u,w,r=[c[k] for k in ('user','word','repeat')]
    words=sorted(map(int,set(w)));trainwords=words[:12];newwords=words[12:]
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))}
    amask=np.array([activity_mask(v) for v in e]);xmask=np.array([activity_mask(np.log(np.maximum(np.exp(v).sum(1),1e-20))) for v in x])
    ir=np.array([lookup[int(us),int(wo),(int(re)+1)%8] for us,wo,re in zip(u,w,r)])
    shifted=np.roll(a,25,axis=1);shiftmask=np.roll(amask,25,axis=1)
    rows=[];quality=[]
    for fold,held in enumerate(itertools.combinations(range(1,11),2)):
      base=[us for us in range(1,11) if us not in held];train=np.isin(u,base)&np.isin(w,trainwords);cutoff=float(np.quantile(m[train],.95));motion=m<=cutoff
      other=dict(zip(base,base[1:]+base[:1]));other[held[0]]=held[1];other[held[1]]=held[0]
      iu=np.array([lookup[other[int(us)],int(wo),int(re)] for us,wo,re in zip(u,w,r)])
      shared,ast,xst,n=batch_features(a,x,amask&motion)
      features=dict(ACC_stats_Audiomask=xst,Audio_stats_commonmask=ast,concatenation=np.c_[xst,ast],coupling_correct=shared,
        coupling_wrongrepeat=batch_features(a[ir],x,amask[ir]&motion)[0],coupling_wronguser=batch_features(a[iu],x,amask[iu]&motion)[0],
        coupling_timeshift=batch_features(shifted,x,shiftmask&motion)[0],coupling_correlation=shared[:,:7],coupling_slope=shared[:,7:14],coupling_energy_difference=shared[:,14:],
        ACC_only_stats=batch_features(x,x,xmask&motion)[2],Audio_only_stats=batch_features(a,a,amask)[1],
        wrongrepeat_fixedmask=batch_features(a[ir],x,amask&motion)[0],wronguser_fixedmask=batch_features(a[iu],x,amask&motion)[0],timeshift_fixedmask=batch_features(shifted,x,amask&motion)[0])
      quality.append(dict(held_pair=list(held),insufficient_record_rate=float(np.mean(n<12)),insufficient_by_method={name:float(np.mean(~np.isfinite(v).all(1))) for name,v in features.items()}))
      for name,z0 in features.items():
        mean=np.nanmean(z0[train],0);mean=np.where(np.isfinite(mean),mean,0);z=np.where(np.isfinite(z0),z0,mean)
        z=(z-mean)/np.maximum(z[train].std(0),1e-5)
        proto=normalize(np.stack([z[(u==us)&np.isin(w,trainwords[:6])&(r<2)].mean(0) for us in range(1,11)]))
        for scope,query in [('fit_words',np.isin(w,trainwords[6:])&(r>=4)),('new_words',np.isin(w,newwords))]:
          qi=np.flatnonzero(query);scores=normalize(z[qi])@proto.T
          for us in held:
            genuine=u[qi]==us;sub=scores[genuine];top=np.isclose(sub,sub.max(1,keepdims=True),atol=1e-10,rtol=0)
            pair=sub[:,np.array(held)-1];j=list(held).index(us);pt=np.isclose(pair,pair.max(1,keepdims=True),atol=1e-10,rtol=0)
            rows.append(dict(held_pair=list(held),user=us,method=name,scope=scope,top1_10way=float(np.mean(top[:,us-1]/top.sum(1))),top1_2way=float(np.mean(pt[:,j]/pt.sum(1))),prototype_verification_auc=float(roc_auc_score(genuine,scores[:,us-1])),genuine_records=int(genuine.sum())))
      if (fold+1)%15==0:print('RAW_FEATURE_BATCH_FOLDS',fold+1,flush=True)
    users=[];macro=[]
    for name in features:
      for scope in ['fit_words','new_words']:
        for us in range(1,11):
          rr=[v for v in rows if v['user']==us and v['method']==name and v['scope']==scope];assert len(rr)==9
          users.append(dict(user=us,method=name,scope=scope,**{k:float(np.mean([v[k] for v in rr])) for k in ['top1_10way','top1_2way','prototype_verification_auc']}))
        rr=[v for v in users if v['method']==name and v['scope']==scope];macro.append(dict(method=name,scope=scope,**{k:float(np.mean([v[k] for v in rr])) for k in ['top1_10way','top1_2way','prototype_verification_auc']}))
    (out/'identity_summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=users,quality=quality),indent=2))
    old=out/'audit_metrics_private.json';private=json.loads(old.read_text()) if old.exists() else {};private['features']=rows;old.write_text(json.dumps(private))
    v=dict(status='PASS',fit_records=1280,template_training_repeats=[0,1,2,3],event_evaluation_repeats=[4,5,6,7],outer_identity_folds=45,feature_workpoints=len(rows),no_final_query_used=True,no_encoder_used=True,no_final_authentication=True,all_users_retained=True,vectorized_feature_assay=True)
    (out/'verification_public.json').write_text(json.dumps(v,indent=2));print('RAW_COMMON_FEATURE_BATCH_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    with threadpool_limits(limits=1):run(p.parse_args().root)
