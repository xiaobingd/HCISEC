"""Locked Audio relationship prediction from fresh, held-out-word ACC."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from audit_audio_acc_relations import normalize,rho,ranking,rank_auc

OUT=Path('outputs/mei_relation_predictability_20261117')
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def relation(z):
    # z=user,word,8recording,dimension. Independent4+4 recording halves.
    z=normalize(z);p=normalize(np.stack([z[:,:,:4].mean(2),z[:,:,4:].mean(2)]))
    return (np.einsum('urv,kqv->qruk',p[0],p[1])+np.einsum('urv,kqv->qruk',p[1],p[0]))/2

def cross_prior(s):return s[~np.eye(len(s),dtype=bool)].mean(0)

def features(s,prior,words):
    w,n=s.shape[0],s.shape[-1];mask=~np.eye(n,dtype=bool)
    rmean=(s*mask).sum(-1)/(n-1);cmean=(s*mask).sum(-2)/(n-1)
    rstd=np.sqrt((((s-rmean[...,None])**2)*mask).sum(-1)/(n-1))
    cstd=np.sqrt((((s-cmean[...,None,:])**2)*mask).sum(-2)/(n-1))
    diag=np.diagonal(s,axis1=-2,axis2=-1)
    shape=s.shape
    def b(x):return np.broadcast_to(x,shape)
    same=np.eye(w,dtype=float)[:,:,None,None]
    cq=(np.asarray(words)>=10)[:,None,None,None];cr=(np.asarray(words)>=10)[None,:,None,None]
    return np.stack([s,b(prior),s-prior,b(rmean[...,None]),b(rstd[...,None]),
        b(cmean[...,None,:]),b(cstd[...,None,:]),b(diag[...,None]),b(diag[...,None,:]),
        s-rmean[...,None],s-cmean[...,None,:],b(same),b(cq),b(cr)],-1)

def fit_predict(x,y,test,wordmask=None,edgemask=None,columns=None):
    w,n=x.shape[0],x.shape[2];mask=np.broadcast_to(~np.eye(n,dtype=bool),x.shape[:-1]).copy()
    if wordmask is not None:mask&=wordmask[:,:,None,None]
    if edgemask is not None:mask&=edgemask[None,None]
    columns=np.arange(x.shape[-1]) if columns is None else np.asarray(columns)
    xx=x[...,columns][mask];yy=y[mask]
    # Equal total sameword/crossword weight; no final query or Audio target.
    same=np.broadcast_to(np.eye(w,dtype=bool)[:,:,None,None],mask.shape)[mask]
    weights=np.ones(len(yy));weights[same]=max(1,(~same).sum()/max(1,same.sum()))
    scaler=StandardScaler().fit(xx);model=Ridge(alpha=10).fit(scaler.transform(xx),yy,sample_weight=weights)
    v=test[...,columns];pred=model.predict(scaler.transform(v.reshape(-1,len(columns)))).reshape(v.shape[:-1])
    return pred

def rank_arrays(pred,truth):
    n=pred.shape[-1];rp=[];rt=[]
    for u in range(n):
        ix=np.arange(n)!=u
        rp.append(rankdata(pred[...,u,ix],axis=-1));rt.append(rankdata(truth[...,u,ix],axis=-1))
    rp=np.stack(rp,-2);rt=np.stack(rt,-2);rp-=rp.mean(-1,keepdims=True);rt-=rt.mean(-1,keepdims=True)
    den=np.linalg.norm(rp,axis=-1)*np.linalg.norm(rt,axis=-1)
    return np.divide((rp*rt).sum(-1),den,out=np.zeros_like(den),where=den>1e-12)

def metrics(pred,truth):
    n=pred.shape[-1];w=len(pred);cor=rank_arrays(pred,truth);out={}
    for cond,mask in [('sameword',np.eye(w,dtype=bool)),('crossword',~np.eye(w,dtype=bool))]:
        per=[]
        for u in range(n):
            ix=np.arange(n)!=u;a=pred[mask][:,u,ix];b=truth[mask][:,u,ix]
            residual=None if np.max(np.ptp(a,axis=0))<1e-12 else rho(a-a.mean(0,keepdims=True),b-b.mean(0,keepdims=True))
            per.append(dict(user=f'U{u+1}',task_spearman=float(cor[mask][:,u].mean()),
                top1=float(np.mean(a.argmax(-1)==b.argmax(-1))),
                within_identity_pair_task_residual_spearman=residual))
        usable=[x['within_identity_pair_task_residual_spearman'] for x in per if x['within_identity_pair_task_residual_spearman'] is not None]
        average=ranking(pred[mask].mean(0),truth[mask].mean(0))
        out[cond]=dict(task_spearman_macro=float(cor[mask].mean()),
            top1=float(np.mean([x['top1'] for x in per])),aggregate_identity_ranking_spearman=average['spearman_macro'],
            residual_spearman_macro=float(np.mean(usable)) if usable else None,
            per_user=per)
    return out

def identity_permutation_concordance(pred,truth,count=1999):
    n=pred.shape[-1];w=len(pred)
    def ranked(s):
        rr=np.zeros_like(s)
        for u in range(n):
            ix=np.arange(n)!=u;v=rankdata(s[...,u,ix],axis=-1)
            v-=v.mean(-1,keepdims=True);v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
            rr[...,u,ix]=v
        return rr
    p,q=ranked(pred),ranked(truth);mask=~np.eye(w,dtype=bool)
    actual=float(((p*q).sum((-2,-1))/n)[mask].mean())
    permutations=np.array([np.random.RandomState(20261117+i).permutation(n) for i in range(count)])
    null=q[:,:,permutations[:,:,None],permutations[:,None,:]]
    assert null.shape==(w,w,count,n,n)
    values=((null*p[:,:,None]).sum((-2,-1))/n)[mask].mean(0)
    return dict(task_spearman=actual,p_one_sided=float((1+(values>=actual-1e-12).sum())/(1+count)),
        null_95pct=np.quantile(values,[.025,.975]).tolist(),identity_permutations=count)

def write_concordance_controls(prediction,truth):
    tests={name:identity_permutation_concordance(prediction[name],truth) for name in ['ACC_similarity','Audio_Ridge']}
    previous=0.
    for j,name in enumerate(sorted(tests,key=lambda name:tests[name]['p_one_sided'])):
        previous=max(previous,min(1.,(2-j)*tests[name]['p_one_sided']));tests[name]['p_holm_2']=previous
    dump(OUT/'identity_permutation_public.json',dict(primary_two_methods=tests,
        scope='all validation words share paired row/column identity permutation;shared-identity dependence preserved;one-sided test',
        timing='supplemental inference,not model selection or retraining'))

def build_predictions(train_acc,val_acc,train_audio):
    prior=cross_prior(train_acc);teacherprior=cross_prior(train_audio)
    x=features(train_acc,prior,list(range(12)));v=features(val_acc,prior,list(range(12,18)))
    prediction=dict(ACC_similarity=val_acc.copy(),ACC_fixed_prior=np.broadcast_to(prior,val_acc.shape).copy(),
        Audio_fixed_identity_memory=np.broadcast_to(teacherprior,val_acc.shape).copy(),
        ACC_self_Ridge=fit_predict(x,train_acc,v),Audio_Ridge=fit_predict(x,train_audio,v),
        Audio_Ridge_no_word_type=fit_predict(x,train_audio,v,columns=list(range(11))),
        Audio_residual_Ridge=teacherprior+fit_predict(x,train_audio-teacherprior,v),
        Audio_Ridge_command_only=fit_predict(x,train_audio,v,wordmask=np.outer(np.arange(12)>=10,np.arange(12)>=10)))
    n=len(prior);dyad=np.zeros_like(val_acc);endpoint=np.zeros_like(val_acc)
    for u in range(n):
        mask=np.ones((n,n),dtype=bool);mask[u,:]=mask[:,u]=False
        p=fit_predict(x,train_audio,v,edgemask=mask);endpoint[:,:,u]=p[:,:,u]
        for j in range(u+1,n):
            mask=np.ones((n,n),dtype=bool);mask[u,j]=mask[j,u]=False
            p=fit_predict(x,train_audio,v,edgemask=mask)
            dyad[:,:,u,j]=p[:,:,u,j];dyad[:,:,j,u]=p[:,:,j,u]
    prediction['Audio_Ridge_heldout_dyad']=dyad;prediction['Audio_Ridge_heldout_endpoint']=endpoint
    rng=np.random.RandomState(20261117);null=[]
    for _ in range(99):
        p=rng.permutation(n);null.append(fit_predict(x,train_audio[:,:,p[:,None],p[None,:]],v))
    return prediction,np.stack(null)

def main():
    encoder_protocol=json.loads((OUT/'protocol_public.json').read_text());assert encoder_protocol['state']=='encoder_complete'
    assert encoder_protocol['training_words']==list(range(12))
    assert hashlib.sha256((OUT/'acc_fresh_private.pt').read_bytes()).hexdigest()==encoder_protocol['checkpoint_sha256']
    c=np.load(OUT/'acc_embeddings_private.npz');f=np.load('outputs/mei_common_frontend_private.npz')
    t=np.load('outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz')
    assert np.array_equal(f['sample_id'][c['indices']],c['sample_id'])
    assert np.array_equal(f['sample_id'],t['sample_id'])
    assert np.array_equal(f['user'][c['indices']],c['user']) and np.array_equal(f['word'][c['indices']],c['word'])
    fit=np.flatnonzero((f['word']>=0)&(f['word']<12));normalization=np.load(OUT/'normalization_private.npz')
    assert np.array_equal(fit,normalization['fit_indices'])
    spectrum=f['spectrum'][fit]
    assert np.allclose(normalization['mean'],spectrum.mean((0,3),keepdims=True))
    assert np.allclose(normalization['std'],spectrum.std((0,3),keepdims=True).clip(1e-4))
    assert np.array_equal(t['indices'],np.flatnonzero((f['word']>=0)&(f['word']<18))) and int(t['done'])==len(t['indices'])
    def select(words):return np.concatenate([np.flatnonzero((c['user']==u)&(c['word']==w))[:8] for u in range(1,10) for w in words])
    ii=select(range(12));jj=select(range(12,18));assert len(ii)==864 and len(jj)==432 and not set(ii)&set(jj)
    train_acc=relation(c['embedding'][ii].reshape(9,12,8,-1));val_acc=relation(c['embedding'][jj].reshape(9,6,8,-1))
    # Only training Audio targets are accessed during prediction fitting.
    train_audio=relation(t['embedding'][c['indices'][ii]].reshape(9,12,8,-1))
    prediction,null=build_predictions(train_acc,val_acc,train_audio)
    # Size controls: refit all combinations, no score-dependent user selection.
    cohorts=[];cohort_predictions=[]
    for size in [5,7,9]:
        for users in itertools.combinations(range(9),size):
            users=np.array(users);a=train_acc[:,:,users[:,None],users[None,:]];v=val_acc[:,:,users[:,None],users[None,:]]
            y=train_audio[:,:,users[:,None],users[None,:]];prior=cross_prior(a)
            x=features(a,prior,range(12));xx=features(v,prior,range(12,18))
            cohorts.append(users);cohort_predictions.append(fit_predict(x,y,xx))
    np.savez_compressed(OUT/'predictions_private.npz',**prediction,permutation_predictions=null)
    lock=dict(state='predictions_locked_before_validation_Audio',methods=list(prediction),ridge_alpha=10,
        no_validation_selection=True,word_ids_as_onehot=False,identity_ids_as_features=False,
        training_Audio_words=list(range(12)),null_training_identity_permutations=99,
        cohort_counts={str(n):sum(len(u)==n for u in cohorts) for n in [5,7,9]},
        endpoint_holdout='all Audio supervision involving target excluded;ACC encoder still trained all9',
        checkpoint_sha256=encoder_protocol['checkpoint_sha256'])
    dump(OUT/'prediction_lock_public.json',lock);print('PREDICTIONS_LOCKED',flush=True)
    # Validation Audio is first used here, after every prediction is fixed.
    val_audio=relation(t['embedding'][c['indices'][jj]].reshape(9,6,8,-1))
    results={name:metrics(p,val_audio) for name,p in prediction.items()}
    write_concordance_controls(prediction,val_audio)
    null_values=[metrics(p,val_audio)['crossword']['task_spearman_macro'] for p in null]
    actual=results['Audio_Ridge']['crossword']['task_spearman_macro']
    control=dict(correct_training_audio_task_spearman=actual,
        null_median=float(np.median(null_values)),null_95pct=np.quantile(null_values,[.025,.975]).tolist(),
        p_one_sided=float((1+sum(v>=actual-1e-12 for v in null_values))/(1+len(null_values))))
    rng=np.random.RandomState(117);mask=~np.eye(6,dtype=bool);bootstrap={}
    base=rank_arrays(prediction['ACC_similarity'],val_audio).mean(-1)
    self_ridge=rank_arrays(prediction['ACC_self_Ridge'],val_audio).mean(-1)
    memory=rank_arrays(prediction['Audio_fixed_identity_memory'],val_audio).mean(-1)
    weights_list=[]
    while len(weights_list)<500:
        count=np.bincount(rng.randint(6,size=6),minlength=6);weights=np.outer(count,count)*mask
        if weights.sum()>0:weights_list.append(weights)
    for name,p in prediction.items():
        values=rank_arrays(p,val_audio).mean(-1);samples=[];difference=[];vs_memory=[];vs_self=[]
        for weights in weights_list:
            samples.append(float((weights*values).sum()/weights.sum()))
            difference.append(float((weights*(values-base)).sum()/weights.sum()))
            vs_memory.append(float((weights*(values-memory)).sum()/weights.sum()))
            vs_self.append(float((weights*(values-self_ridge)).sum()/weights.sum()))
        bootstrap[name]=dict(word_block_95pct=np.quantile(samples,[.025,.975]).tolist(),
            difference_from_ACC_95pct=np.quantile(difference,[.025,.975]).tolist(),
            difference_from_ACC_self_Ridge_95pct=np.quantile(vs_self,[.025,.975]).tolist(),
            difference_from_fixed_Audio_memory_95pct=np.quantile(vs_memory,[.025,.975]).tolist())
    cohort_rows=[]
    for users,p in zip(cohorts,cohort_predictions):
        aa=val_audio[:,:,users[:,None],users[None,:]];baseline=val_acc[:,:,users[:,None],users[None,:]]
        b=metrics(baseline,aa)['crossword'];v=metrics(p,aa)['crossword']
        cohort_rows.append(dict(size=len(users),baseline=b['task_spearman_macro'],supervised=v['task_spearman_macro'],
            baseline_top1=b['top1'],supervised_top1=v['top1']))
    cohort_summary=[]
    for size in [5,7,9]:
        rr=[v for v in cohort_rows if v['size']==size]
        cohort_summary.append(dict(users=size,cohorts=len(rr),
            baseline=float(np.mean([v['baseline'] for v in rr])),supervised=float(np.mean([v['supervised'] for v in rr])),
            improvement_range=[float(min(v['supervised']-v['baseline'] for v in rr)),float(max(v['supervised']-v['baseline'] for v in rr))],
            fraction_improved=float(np.mean([v['supervised']>v['baseline'] for v in rr])),
            baseline_top1=float(np.mean([v['baseline_top1'] for v in rr])),supervised_top1=float(np.mean([v['supervised_top1'] for v in rr])),random_top1=1/(size-1)))
    # Independent authentication diagnostic, with exactly the same records.
    authentication=[]
    for name,z in [('Audio',t['embedding'][c['indices'][jj]].reshape(9,6,8,-1)),('ACC_fresh',c['embedding'][jj].reshape(9,6,8,-1))]:
        z=normalize(z);proto=normalize(np.stack([z[:,:,:4].mean(2),z[:,:,4:].mean(2)]));row=[]
        for u in range(9):
            values={'sameword':[],'crossword':[]}
            for fold in [0,1]:
                q=z[:,:,4:] if fold==0 else z[:,:,:4]
                labels=np.repeat(np.arange(9),4)==u
                for qw in range(6):
                    for rw in range(6):values['sameword' if qw==rw else 'crossword'].append(rank_auc(labels,q[:,qw].reshape(36,-1)@proto[fold,u,rw]))
            row.append(dict(user=f'U{u+1}',**{k:float(np.mean(v)) for k,v in values.items()}))
        authentication.append(dict(modality=name,per_user=row,macro={k:float(np.mean([v[k] for v in row])) for k in ['sameword','crossword']}))
    dump(OUT/'results_public.json',dict(methods=results,word_block_intervals=bootstrap,training_identity_permutation_control=control,
        all_combination_user_count_control=cohort_summary,independent_word_authentication=authentication,
        validation_words=list(range(12,18)),all_validation_words_are_commands=True,
        limits='known-user unseen-word;recording halves are not independent users;CI conditional on9users;single encoder seed'))
    dump(OUT/'verification_public.json',dict(state='PASS',sample_ids_and_labels_aligned=True,
        encoder_training_words=list(range(12)),normalization_fit_words=list(range(12)),validation_words=list(range(12,18)),
        old_E0_not_loaded=True,fresh_random_initialization=True,fixed_step_no_validation_checkpoint_selection=True,
        prediction_lock_before_validation_audio=True,all9_users_included=True,
        no_new_Audio_extraction=True,no_Audio_words18_29_used=True,no_audio_in_ACC_training=True,
        not_an_unseen_encoder_user_experiment=True,source_encoder_hash=encoder_protocol['checkpoint_sha256']))
    dump(OUT/'status.json',dict(stage='complete'));print('PREDICTABILITY_COMPLETE',flush=True)

if __name__=='__main__':main()
