"""Nested leave-two-identity-out linear common-subspace audit, fit Audio only.
No final authentication, score fusion or query-Audio access.
"""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)

def fit_mapping(a,x,method,dimension,ridge):
    """Map both centered modalities into a common low-dimensional coordinate system."""
    ma,mx=a.mean(0),x.mean(0);ac,xc=a-ma,x-mx;n=len(a)
    if method=='rcca':
        def whitener(v):
            _,s,vt=np.linalg.svd(v,full_matrices=False);keep=s>max(float(s[0])*1e-7,1e-12)
            s,vt=s[keep],vt[keep];variance=(s*s).sum()/(n-1)/len(s)
            return vt.T/np.sqrt(s*s/(n-1)+ridge*variance)
        wa,wx=whitener(ac),whitener(xc)
        u,s,vt=np.linalg.svd((ac@wa).T@(xc@wx)/(n-1),full_matrices=False)
        pa,px=wa@u[:,:dimension],wx@vt.T[:,:dimension]
    elif method=='lowrank_ridge':
        variance=float((xc*xc).sum()/(n-1)/np.linalg.matrix_rank(xc,tol=1e-7))
        b=xc.T@np.linalg.solve(xc@xc.T+(n-1)*ridge*variance*np.eye(n),ac)
        _,s,vt=np.linalg.svd(xc@b,full_matrices=False)
        pa=vt[:dimension].T;px=b@pa
    else:raise ValueError(method)
    assert pa.shape[1]==px.shape[1]==dimension and np.isfinite(pa).all() and np.isfinite(px).all()
    return dict(ma=ma,mx=mx,pa=pa,px=px)

def similarity(model,a,x):
    return unit((x-model['mx'])@model['px'])@unit((a-model['ma'])@model['pa']).T

def target_metrics(scores,target,cohort):
    own=list(cohort).index(target);s=scores[own];negative=np.delete(s,own)
    top=np.isclose(s,s.max(),rtol=0,atol=1e-10)
    delta=s[own]-negative
    return dict(top1=float(top[own]/top.sum()),auc=float(np.mean((delta>1e-10)+.5*(abs(delta)<=1e-10))),margin=float(s[own]-negative.max()))

def main(root):
    out=root/'conditional_correspondence/linear_shared_identity';out.mkdir(exist_ok=True)
    d=np.load(root/'phrase_auth/frontend_private.npz');c=np.load(root/'conditional_correspondence/audio_conditioned_verifier/fit_ecapa_private.npz');idx=c['indices']
    u,w,reps,ids=[d[k][idx] for k in ('user','word','repeat','sample_id')]
    assert len(idx)==1280 and np.array_equal(ids,c['sample_id']) and bool(c['length_aware'])
    words=sorted(map(int,set(w)));assert words==[0,1,2,4,5,6,7,10,11,13,14,15,16,17,33,34]
    a=c['embedding'].astype('float64');fitwords=words[:12];devwords=words[12:]
    users=list(range(1,11));pairs=list(itertools.combinations(users,2));grid=list(itertools.product([1,2,4],[.1,1.,10.]))
    protocol=dict(users=10,seeds=[42,43,44],outer_folds=45,outer_held_identities=2,inner='leave1of8trainingidentities out,7fit;mean developmentidentity AUC over8folds,choose dimension/ridge withoutouteridentities',
      methods=['rcca','lowrank_ridge'],dimensions=[1,2,4],ridge=[.1,1.,10.],
      rcca='center each modality usingfitidentities;SVD rankrestricted ridge covariance whitening;SVDcrosscov;ridge scaled by mean nonzero covariance eigenvalue',
      lowrank_ridge='center;ridge ACC->Audio linearregression;truncate predictedAudio SVD;common coords ACC@B@V andAudio@V',
      fit_words=fitwords,internal_development_words=devwords,
      samples='one cross-word identityprototype pertrainingidentity;48utterances permodality for6fitwords;rank<=7outer/<=6inner,not thousands ofindependent identities',
      orientations='both Audiofirst6->ACClast6 andAudiolast6->ACCfirst6;inner/newwordsscopefirst2vslast2also reversed',
      selection='mean inner-dev heldidentity AUC vs7trainingprototypes;round12decimals toremove floatingtail ties,then smaller dimension then stronger ridge;correct hyperparams lockedalso forwrongidentity trainingcontrol',
      wrong='retrain same linear method withderangedAudiotrainingidentity assignment;outer identities remainexcluded;noouterlabels in selection',
      baselines='rawACC withinmodalitycrossword cosine andrawAudio withinmodalitycrossword cosine;Audio baseline usesfitwordcache only,notdeploymentclaim',
      evaluation='heldpair2way chance.5 primary;all10prototypes including8seen and2held,heldtargets only chance.1 secondary;noqueryAudio outsidefit16',
      scopes='fit_words_identity_holdout;mapping_untrained_words_identity_holdout. latter4words usedoninnerdevelopmentidentities forhyperparameters,notblindheldwords;globalACCencoder previouslysawall16words/all10identities',
      repeat_views='all8,eve4,odd4 repeatedrecordings;reliabilityspearman across18nonselfinteridentitypairrelations for10candidates;statisticsclusterbytargetnotrecording',
      no_final_authentication=True,no_REG_CAL_Audio=True,no_fusion=True,base_encoders_updated=False,
      limitation='identityholdout appliesonlylinear mappings;pretrainedACC seesall10;historically inspectedcohort;3secondwindows mayoverlap;44100candidateAudiointerpretation')
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    def proto(z,ww,parity):
        return np.stack([z[(u==us)&np.isin(w,ww)&(np.ones(len(u),bool) if parity=='all' else reps%2==int(parity))].mean(0) for us in users])
    views={}
    with np.load(root/'conditional_correspondence/stage3/views_private.npz') as vf:
      for seed in (42,43,44):views[seed]=vf[f'{seed}_A_ACC'][idx].astype('float64')
    audio_proto={(scope,orientation,parity):proto(a,(fitwords[:6],fitwords[6:])[orientation] if scope=='fit' else (devwords[:2],devwords[2:])[orientation],parity)
      for scope in ['fit','newwords'] for orientation in (0,1) for parity in ['all','0','1']}
    rows=[];choices=[];stability=[];private_models={};exclusions=[]
    for seed in (42,43,44):
      x=views[seed]
      xp={(scope,orientation,parity):proto(x,(fitwords[:6],fitwords[6:])[1-orientation] if scope=='fit' else (devwords[:2],devwords[2:])[1-orientation],parity)
        for scope in ['fit','newwords'] for orientation in (0,1) for parity in ['all','0','1']}
      acc_support={(scope,orientation,parity):proto(x,(fitwords[:6],fitwords[6:])[orientation] if scope=='fit' else (devwords[:2],devwords[2:])[orientation],parity)
        for scope in ['fit','newwords'] for orientation in (0,1) for parity in ['all','0','1']}
      audio_query={(scope,orientation,parity):proto(a,(fitwords[:6],fitwords[6:])[1-orientation] if scope=='fit' else (devwords[:2],devwords[2:])[1-orientation],parity)
        for scope in ['fit','newwords'] for orientation in (0,1) for parity in ['all','0','1']}
      for fold,held in enumerate(pairs):
        train=[us for us in users if us not in held];ti=np.array(train)-1
        for orientation in (0,1):
          afit=audio_proto['fit',orientation,'all'];xfit=xp['fit',orientation,'all'];adev=audio_proto['newwords',orientation,'all'];xdev=xp['newwords',orientation,'all']
          for method in ['rcca','lowrank_ridge']:
            candidates=[]
            for dim,ridge in grid:
              inner=[]
              for dev in train:
                it=np.array([us for us in train if us!=dev])-1;assert not set(it+1)&set(held)
                model=fit_mapping(afit[it],xfit[it],method,dim,ridge)
                score=similarity(model,adev,xdev);s=score[dev-1,np.array(train)-1];j=train.index(dev)
                negative=np.delete(s,j);delta=s[j]-negative;inner.append(float(np.mean((delta>1e-10)+.5*(abs(delta)<=1e-10))))
              candidates.append(dict(dimension=dim,ridge=ridge,inner_auc=float(np.mean(inner))))
            best=min(candidates,key=lambda r:(-round(r['inner_auc'],12),r['dimension'],-r['ridge']))
            choices.append(dict(seed=seed,held_users=list(held),orientation=orientation,method=method,**best,candidates=candidates))
            rng=np.random.default_rng(20261124+seed*100+fold*2+orientation);perm=rng.permutation(8)
            while np.any(perm==np.arange(8)):perm=rng.permutation(8)
            for control,aa in [('correct',afit[ti]),('identity_wrong',afit[ti[perm]])]:
              model=fit_mapping(aa,xfit[ti],method,best['dimension'],best['ridge']);name=method+'_'+control
              key=f'{seed}_{fold}_{orientation}_{name}';private_models[key+'_ma']=model['ma'];private_models[key+'_mx']=model['mx'];private_models[key+'_pa']=model['pa'];private_models[key+'_px']=model['px']
              for scope in ['fit','newwords']:
                scores=[]
                for parity in ['all','0','1']:
                  score=similarity(model,audio_proto[scope,orientation,parity],xp[scope,orientation,parity]);scores.append(score)
                  for target in held:
                    pairscore=score[np.array(held)-1][:,np.array(held)-1]
                    rows.append(dict(seed=seed,held_pair=list(held),user=target,orientation=orientation,method=name,scope=scope,repeats=parity,
                      **target_metrics(pairscore,target,list(held)),top1_10way=target_metrics(score,target,users)['top1']))
                mask=np.zeros((10,10),bool);mask[np.array(held)-1,:]=True;mask[np.array(held)-1,np.array(held)-1]=False
                rho=float(spearmanr(scores[1][mask],scores[2][mask]).statistic)
                stability.append(dict(seed=seed,held_pair=list(held),orientation=orientation,method=name,scope=scope,even_odd_relation_spearman=0. if not np.isfinite(rho) else rho))
          for scope in ['fit','newwords']:
            for parity in ['all','0','1']:
              for name,score in [('raw_ACC',unit(xp[scope,orientation,parity])@unit(acc_support[scope,orientation,parity]).T),
                ('raw_Audio',unit(audio_query[scope,orientation,parity])@unit(audio_proto[scope,orientation,parity]).T)]:
                for target in held:
                  pairscore=score[np.array(held)-1][:,np.array(held)-1]
                  rows.append(dict(seed=seed,held_pair=list(held),user=target,orientation=orientation,method=name,scope=scope,repeats=parity,
                    **target_metrics(pairscore,target,list(held)),top1_10way=target_metrics(score,target,users)['top1']))
        exclusions.append(dict(seed=seed,held_pair=list(held),mapping_training_users=train,held_records_in_fit_or_selection=0))
      print('LINEAR_AUDIT_SEED_DONE',seed,flush=True)
    assert len(choices)==540 and len(rows)==19440
    names=sorted({r['method'] for r in rows});per_user=[]
    for name in names:
      for scope in ['fit','newwords']:
       for parity in ['all','0','1']:
        for target in users:
          rr=[r for r in rows if r['method']==name and r['scope']==scope and r['repeats']==parity and r['user']==target]
          assert len(rr)==54
          per_user.append(dict(method=name,scope=scope,repeats=parity,user=target,**{k:float(np.mean([r[k] for r in rr])) for k in ['top1','auc','margin','top1_10way']}))
    macro=[]
    for name in names:
     for scope in ['fit','newwords']:
      for parity in ['all','0','1']:
        rr=[r for r in per_user if r['method']==name and r['scope']==scope and r['repeats']==parity]
        macro.append(dict(method=name,scope=scope,repeats=parity,**{k:float(np.mean([r[k] for r in rr])) for k in ['top1','auc','margin','top1_10way']}))
    result=dict(macro=macro,per_user=per_user,verification=dict(status='PASS',outer_folds=135,model_orientations=270,candidate_selections=540,workpoints=len(rows),no_final_authentication=True,only_fit_audio=True,encoders_unchanged=True))
    (out/'summary_public.json').write_text(json.dumps(result,indent=2));(out/'selection_public.json').write_text(json.dumps(choices))
    (out/'stability_public.json').write_text(json.dumps(stability));(out/'exclusion_audit_public.json').write_text(json.dumps(exclusions))
    (out/'metrics_private.json').write_text(json.dumps(rows));np.savez_compressed(out/'linear_models_private.npz',**private_models)
    print('LINEAR_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    with threadpool_limits(limits=1):main(args.root)
