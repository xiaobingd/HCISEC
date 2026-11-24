"""Content-conditional event correspondence, followed by offline coupling identity assay."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

def activity_mask(e):
    lo,hi=np.quantile(e,[.2,.9]);return e>=max(lo+.25*(hi-lo),np.log(1e-12))
def band_correlation(a,x,mask):
    if mask.sum()<12:return np.zeros(a.shape[1]),True
    aa=a[mask]-a[mask].mean(0);xx=x[mask]-x[mask].mean(0)
    denom=np.sqrt((aa*aa).sum(0)*(xx*xx).sum(0))
    return np.divide((aa*xx).sum(0),denom,out=np.zeros(a.shape[1]),where=denom>1e-8),False
def compare(a,x,ea,mx,cutoff,lag,conditioned):
    length=len(a)-abs(lag)
    ai=slice(max(lag,0),max(lag,0)+length);xi=slice(max(-lag,0),max(-lag,0)+length)
    mask=(activity_mask(ea)[ai]&(mx[xi]<=cutoff)) if conditioned else np.ones(length,bool)
    return band_correlation(a[ai],x[xi],mask)
def marginal_stats(v,mask):
    if mask.sum()<12:return np.full(16,np.nan)
    mean,std=v[mask].mean(0),v[mask].std(0)
    return np.r_[mean-mean.mean(),std,mean.mean(),std.mean()]
def identity_features(a,x,e,m,cutoff,audio_mask=None):
    mask=(activity_mask(e) if audio_mask is None else audio_mask)&(m<=cutoff);n=int(mask.sum())
    if n<12:return np.full(21,np.nan),np.full(16,np.nan),np.full(16,np.nan),n
    am,xm=a[mask].mean(0),x[mask].mean(0);ast,xst=a[mask].std(0),x[mask].std(0)
    corr,_=band_correlation(a,x,mask);cov=((a[mask]-am)*(x[mask]-xm)).mean(0)
    slope=np.divide(cov,ast*ast,out=np.zeros(7),where=ast*ast>1e-8)
    shared=np.r_[corr,slope,xm-am]
    def stats(mean,std):return np.r_[mean-mean.mean(),std,mean.mean(),std.mean()]
    return shared,stats(am,ast),stats(xm,xst),n
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def main(root,event_only=False):
    out=root/'conditional_correspondence/raw_common_dynamics';c=np.load(out/'local_power_private.npz')
    a,x,e,m=[c[k].astype('float64') for k in ('audio','acc','audio_activity','motion')]
    u,w,r=[c[k] for k in ('user','word','repeat')];words=sorted(map(int,set(w)));trainwords=words[:12];newwords=words[12:]
    assert a.shape==x.shape and a.shape[0]==1280 and a.shape[-1]==7
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))}
    protocol=dict(records=1280,users=10,words=16,train_words=trainwords,new_words=newwords,
      question1='sameuser/sameword8reps;first4estimatefixedword+identity timefrequencytemplate,last4truevswrongrepeat matching chance.25',
      template='subtract each modality first4mean withouttimewarping;evaluate last4only;removesrepeatedword/identitytemplate,not allcontentvariation',
      activity='Audio300-3500Hz energy proxy threshold max(log1e-12,Q20+.25*(Q90-Q20)),notvalidatedVAD',
      motion='ACC<20Hz powerproxy;Q95 fromothertrainingidentities/train12words only;notvalidatedmotionremoval',
      event_controls='allframes vsactivity+lowmotion,rawvsfixedtemplate residual,lags -200,-100,0,100,200ms;NO perpairlagoptimization',
      primary_event_test='template residual+activity/lowmotion+0lag;user-average true-minuswrong correlation;one-sided exact10user signflip;7band maxT correction',
      feature_identity='OFFLINE pairedAudio+ACC coupling statistics;Audio requiredforbothsupportandquery inthisassay,notACC-onlydeployment',
      coupling='7perbandPearson correlations,7regression slopes,7empirical logenergydifferences;not calibrated physicaltransferfunction',
      controls='independent ACC-only activity+motion mask16statistics,independent Audio-only activitymask16statistics;commonmask unimodal stats;concatenation32;correctcoupling;wrongrepeat/wronguser/timeshift eachwithcandidate activitymask AND fixedcorrectAudioactivitymask;correlationonly,slopeonly,energydifferenceonly',
      registration='fit first6words,reps0/1=12pairedrecords peruser;queriesfitlast6words/reps4..7=24ornew4words/all8=32;no finalquerywords',
      outer='leave2of10users out;featuremean/std,missingimputation,motionthreshold only8baseidentities/train12words;noclassifieroptimization',
      wronguser='cyclic permutation within8baseidentities separately;held2swap;heldAudio neverentersbase normalization',
      missing='under12validframes allfeaturesmissing;impute base/training mean;keep allusersandrecords;report insufficientrate',
      limits='physicalsync unverified,44100candidate timebase;3secwindowsoverlap possible;VADcenter alreadyusedtocrop;lackindependentwear/device/dayvalidation;no identity-specific physiology claim fromonesession')
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    events=[];band_user=[]
    for us in range(1,11):
      threshold=float(np.quantile(m[(u!=us)&np.isin(w,trainwords)],.95))
      for word in words:
        template=np.array([lookup[us,word,rep] for rep in range(4)]);evals=np.array([lookup[us,word,rep] for rep in range(4,8)])
        ta,tx=a[template].mean(0),x[template].mean(0)
        for residual in (False,True):
          for conditioned in (False,True):
            for lag in (-10,-5,0,5,10):
              score=np.zeros((4,4,7));bad=0
              for ai,ia in enumerate(evals):
                for xi,ix in enumerate(evals):
                  score[xi,ai],missing=compare(a[ia]-ta if residual else a[ia],x[ix]-tx if residual else x[ix],e[ia],m[ix],threshold,lag,conditioned);bad+=missing
              total=score.mean(2);tops=np.isclose(total,total.max(1,keepdims=True),atol=1e-10,rtol=0)
              top1=float(np.mean(np.diag(tops)/tops.sum(1)));margin=float(np.diag(total).mean()-total[~np.eye(4,dtype=bool)].mean())
              events.append(dict(user=us,word=word,residual=residual,conditioned=conditioned,lag_ms=lag*20,top1=top1,correlation_advantage=margin,insufficient_pairs=bad))
              if residual and conditioned and lag==0:
                adv=score[np.arange(4),np.arange(4)].mean(0)-score[~np.eye(4,dtype=bool)].mean(0)
                band_user.append(dict(user=us,word=word,advantage=adv.tolist()))
    modes=[];peruser_events=[]
    for residual in (False,True):
      for conditioned in (False,True):
        for lag in (-200,-100,0,100,200):
          means=[]
          for us in range(1,11):
            rr=[v for v in events if v['user']==us and v['residual']==residual and v['conditioned']==conditioned and v['lag_ms']==lag]
            row=dict(user=us,residual=residual,conditioned=conditioned,lag_ms=lag,top1=float(np.mean([v['top1'] for v in rr])),correlation_advantage=float(np.mean([v['correlation_advantage'] for v in rr])),insufficient_pair_rate=sum(v['insufficient_pairs'] for v in rr)/(16*16))
            peruser_events.append(row);means.append(row)
          modes.append(dict(residual=residual,conditioned=conditioned,lag_ms=lag,top1=float(np.mean([v['top1'] for v in means])),correlation_advantage=float(np.mean([v['correlation_advantage'] for v in means]))))
    primary=[v for v in peruser_events if v['residual'] and v['conditioned'] and v['lag_ms']==0]
    adv=np.array([v['correlation_advantage'] for v in primary]);signs=np.array(list(itertools.product([-1.,1.],repeat=10)))
    pval=float(np.mean(signs@adv/10>=adv.mean()-1e-12))
    bands=np.array([np.mean([v['advantage'] for v in band_user if v['user']==us],0) for us in range(1,11)])
    den=np.maximum(bands.std(0,ddof=1)/np.sqrt(10),1e-12);observed=bands.mean(0)/den;null=(signs@bands/10)/den
    bandstats=[dict(band=i,advantage=float(bands[:,i].mean()),positive_users=int((bands[:,i]>0).sum()),maxT_p=float(np.mean(null.max(1)>=observed[i]-1e-12))) for i in range(7)]
    event_result=dict(macro=modes,per_user=peruser_events,primary_exact_user_signflip_p=pval,bands=bandstats)
    (out/'event_correspondence_public.json').write_text(json.dumps(event_result,indent=2));print('RAW_EVENT_AUDIT_DONE',flush=True)
    if event_only:return
    feature_rows=[];quality=[]
    names=['ACC_stats_Audiomask','Audio_stats_commonmask','concatenation','coupling_correct','coupling_wrongrepeat','coupling_wronguser','coupling_timeshift','coupling_correlation','coupling_slope','coupling_energy_difference','ACC_only_stats','Audio_only_stats','wrongrepeat_fixedmask','wronguser_fixedmask','timeshift_fixedmask']
    audio_masks=np.array([activity_mask(v) for v in e]);acc_masks=np.array([activity_mask(np.log(np.maximum(np.exp(v).sum(1),1e-20))) for v in x])
    for held in itertools.combinations(range(1,11),2):
      base=[us for us in range(1,11) if us not in held];train=np.isin(u,base)&np.isin(w,trainwords)
      threshold=float(np.quantile(m[train],.95));others=dict(zip(base,base[1:]+base[:1]));others[held[0]]=held[1];others[held[1]]=held[0]
      features={k:[] for k in names};invalid=[]
      for i in range(1280):
        shared,as_,xs,n=identity_features(a[i],x[i],e[i],m[i],threshold,audio_masks[i])
        ir=lookup[int(u[i]),int(w[i]),(int(r[i])+1)%8];iu=lookup[others[int(u[i])],int(w[i]),int(r[i])]
        wr=identity_features(a[ir],x[i],e[ir],m[i],threshold,audio_masks[ir])[0];wu=identity_features(a[iu],x[i],e[iu],m[i],threshold,audio_masks[iu])[0]
        wt=identity_features(np.roll(a[i],25,axis=0),x[i],np.roll(e[i],25),m[i],threshold,np.roll(audio_masks[i],25))[0]
        acc_only=marginal_stats(x[i],acc_masks[i]&(m[i]<=threshold));audio_only=marginal_stats(a[i],audio_masks[i])
        wr_fixed=identity_features(a[ir],x[i],e[ir],m[i],threshold,audio_masks[i])[0]
        wu_fixed=identity_features(a[iu],x[i],e[iu],m[i],threshold,audio_masks[i])[0]
        wt_fixed=identity_features(np.roll(a[i],25,axis=0),x[i],e[i],m[i],threshold,audio_masks[i])[0]
        values=[xs,as_,np.r_[xs,as_],shared,wr,wu,wt,shared[:7],shared[7:14],shared[14:],acc_only,audio_only,wr_fixed,wu_fixed,wt_fixed]
        for key,value in zip(names,values):features[key].append(value)
        invalid.append(n<12)
      quality.append(dict(held_pair=list(held),insufficient_record_rate=float(np.mean(invalid)),insufficient_by_method={name:float(np.mean(~np.isfinite(np.array(value)).all(1))) for name,value in features.items()}))
      for name,feature in features.items():
        z=np.array(feature);mean=np.nanmean(z[train],0);mean=np.where(np.isfinite(mean),mean,0)
        z=np.where(np.isfinite(z),z,mean);scale=np.maximum(z[train].std(0),1e-5);z=(z-mean)/scale
        prototypes=[]
        for us in range(1,11):
          ii=np.flatnonzero((u==us)&np.isin(w,trainwords[:6])&(r<2));assert len(ii)==12
          prototypes.append(z[ii].mean(0))
        prototypes=normalize(np.array(prototypes))
        for scope,query in [('fit_words',np.isin(w,trainwords[6:])&(r>=4)),('new_words',np.isin(w,newwords))]:
          qi=np.flatnonzero(query);scores=normalize(z[qi])@prototypes.T
          for us in held:
            genuine=u[qi]==us;target=us-1;sub=scores[genuine];tops=np.isclose(sub,sub.max(1,keepdims=True),atol=1e-10,rtol=0)
            pair=sub[:,np.array(held)-1];j=list(held).index(us);pt=np.isclose(pair,pair.max(1,keepdims=True),atol=1e-10,rtol=0)
            feature_rows.append(dict(held_pair=list(held),user=us,method=name,scope=scope,top1_10way=float(np.mean(tops[:,target]/tops.sum(1))),top1_2way=float(np.mean(pt[:,j]/pt.sum(1))),
              prototype_verification_auc=float(roc_auc_score(genuine,scores[:,target])),genuine_records=int(genuine.sum())))
    users=[];macro=[]
    for name in names:
      for scope in ['fit_words','new_words']:
        for us in range(1,11):
          part=[v for v in feature_rows if v['user']==us and v['method']==name and v['scope']==scope];assert len(part)==9
          users.append(dict(user=us,method=name,scope=scope,**{k:float(np.mean([v[k] for v in part])) for k in ['top1_10way','top1_2way','prototype_verification_auc']}))
        rr=[v for v in users if v['method']==name and v['scope']==scope]
        macro.append(dict(method=name,scope=scope,**{k:float(np.mean([v[k] for v in rr])) for k in ['top1_10way','top1_2way','prototype_verification_auc']}))
    (out/'identity_summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=users,quality=quality),indent=2))
    (out/'audit_metrics_private.json').write_text(json.dumps(dict(events=events,features=feature_rows)))
    (out/'verification_public.json').write_text(json.dumps(dict(status='PASS',fit_records=1280,template_training_repeats=[0,1,2,3],event_evaluation_repeats=[4,5,6,7],outer_identity_folds=45,feature_workpoints=len(feature_rows),no_final_query_used=True,no_encoder_used=True,no_final_authentication=True,all_users_retained=True),indent=2))
    print('RAW_COMMON_IDENTITY_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--event-only',action='store_true');args=p.parse_args()
    with threadpool_limits(limits=1):main(args.root,args.event_only)
