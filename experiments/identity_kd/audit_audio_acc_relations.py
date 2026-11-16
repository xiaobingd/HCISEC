"""Training-word-only, model-free Audio/ACC relation audit. No waveform reads."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def validate_alignment(front,teacher,encoded):
    ids=front['sample_id'];u=front['user'];w=front['word']
    fit=np.flatnonzero((w>=0)&(w<18))
    assert len(set(ids))==len(ids),'duplicate sample IDs'
    assert np.array_equal(ids,teacher['sample_id']) and np.array_equal(ids,encoded['sample_id']),'sample_id mismatch'
    assert np.array_equal(u,encoded['user']) and np.array_equal(w,encoded['word']),'identity/word mismatch'
    assert np.array_equal(fit,teacher['indices']) and int(teacher['done'])==len(fit),'Audio scope/incomplete extraction'
    return fit

def rho(a,b):
    a,b=np.asarray(a).ravel(),np.asarray(b).ravel()
    a=rankdata(a);b=rankdata(b);a-=a.mean();b-=b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den<1e-12 else float(a@b/den)

def rank_auc(labels,scores):
    difference=scores[labels][:,None]-scores[~labels][None,:]
    return float((difference>0).mean()+.5*(difference==0).mean())

def pair_residual(m):
    # Remove additive speaker-node effects from the 36 unordered dyads.
    n=len(m);i,j=np.triu_indices(n,1);design=np.zeros((len(i),n))
    design[np.arange(len(i)),i]=1;design[np.arange(len(i)),j]=1
    value=((m+m.T)/2)[i,j]
    residual=value-design@np.linalg.lstsq(design,value,rcond=None)[0]
    out=np.zeros((n,n));out[i,j]=out[j,i]=residual
    return out

def pair_vector(m,kind):
    return np.diag(m) if kind=='same_user_cross_word' else ((m+m.T)/2)[np.triu_indices(len(m),1)]

def collapse(score,condition,weights=None):
    # score: queryword,registrationword,target,queryidentity; words are blocks.
    w=len(score);weights=np.ones(w) if weights is None else weights
    mask=np.eye(w,dtype=bool)
    ww=np.diag(weights) if condition=='different_user_same_word' else np.outer(weights,weights)*(~mask)
    assert ww.sum()>0
    return np.einsum('ij,ijuv->uv',ww,score)/ww.sum()

def hardest(m,k):
    m=m.copy();np.fill_diagonal(m,-np.inf)
    return np.argsort(-m,axis=1,kind='stable')[:,:k]

def ranking(a,b):
    n=len(a);per=[]
    for u in range(n):
        ix=np.arange(n)!=u
        per.append(dict(user=f'U{u+1}',spearman=rho(a[u,ix],b[u,ix]),
            topk_overlap={str(k):len(set(hardest(a,k)[u])&set(hardest(b,k)[u]))/k for k in [1,2,3]}))
    return dict(spearman_macro=float(np.mean([r['spearman'] for r in per])),per_user=per,
        topk_overlap={str(k):float(np.mean([r['topk_overlap'][str(k)] for r in per])) for k in [1,2,3]},
        random_overlap={str(k):k/(n-1) for k in [1,2,3]})

def permutation_test(a,b,kind,permutations):
    actual=rho(pair_vector(a,kind),pair_vector(b,kind));null=[]
    for p in permutations:null.append(rho(pair_vector(a[np.ix_(p,p)],kind),pair_vector(b,kind)))
    null=np.array(null,dtype=float)
    return dict(spearman=actual,identity_permutation_p_one_sided=float((1+(null>=actual).sum())/(1+len(null))),
        null_95pct=np.quantile(null,[.025,.975]).tolist(),units=len(pair_vector(a,kind)))

def analyze(audio,acc,bootstraps=500,permutations=1999,seed=20261116):
    # Shape: user,word,8 distinct recordings,dimension. Every task uses 4+4.
    assert audio.shape[:3]==acc.shape[:3] and audio.shape[2]==8
    n,w=audio.shape[:2];rng=np.random.RandomState(seed)
    embeddings=[normalize(audio),normalize(acc)]
    proto=[normalize(np.stack([z[:,:,:4].mean(2),z[:,:,4:].mean(2)])) for z in embeddings]
    # fold,qword,rword,target,queryidentity
    allscores=[np.einsum('urv,kqv->qruk',p[0],p[1])[None] for p in proto]
    allscores=[np.concatenate([s,np.einsum('urv,kqv->qruk',p[1],p[0])[None]]) for s,p in zip(allscores,proto)]
    assert all(s.shape==(2,w,w,n,n) for s in allscores)
    scores=[s.mean(0) for s in allscores]
    conditions=['same_user_cross_word','different_user_same_word','different_user_cross_word']
    perms=np.array([rng.permutation(n) for _ in range(permutations)])
    relation={};matrices={}
    for condition in conditions:
        a,b=[collapse(s,condition) for s in scores];matrices[condition]=(a,b)
        result=permutation_test(a,b,condition,perms)
        ci=[]
        for _ in range(bootstraps):
            weights=np.bincount(rng.randint(w,size=w),minlength=w)
            aa,bb=[collapse(s,condition,weights) for s in scores]
            ci.append(rho(pair_vector(aa,condition),pair_vector(bb,condition)))
        result['word_block_bootstrap_95pct_conditional_on_users']=np.quantile(ci,[.025,.975]).tolist()
        loo=[]
        for u in range(n):
            ix=np.arange(n)!=u
            loo.append(dict(removed_user=f'U{u+1}',spearman=rho(pair_vector(a[np.ix_(ix,ix)],condition),pair_vector(b[np.ix_(ix,ix)],condition))))
        result['leave_one_user_out_sensitivity']=loo
        # Word-pair centered sample correlations are descriptive, never used as independent tests.
        mask=np.eye(n,dtype=bool) if condition.startswith('same_user') else ~np.eye(n,dtype=bool)
        wa=np.eye(w,dtype=bool) if condition=='different_user_same_word' else ~np.eye(w,dtype=bool)
        aa,bb=[s[wa][:,mask] for s in scores]
        result['wordpair_centered_descriptive_spearman']=rho(aa-aa.mean(1,keepdims=True),bb-bb.mean(1,keepdims=True))
        repeat=[];wordhalf=[]
        for m,p in zip(['Audio','ACC'],proto):
            within=[np.einsum('urv,kqv->qruk',p[f],p[f]) for f in [0,1]]
            mm=[collapse(s,condition) for s in within]
            repeat.append(dict(modality=m,spearman=rho(pair_vector(mm[0],condition),pair_vector(mm[1],condition))))
        for ss in scores:
            half=[collapse(ss[np.ix_(ix,ix)],condition) for ix in [np.arange(w//2),np.arange(w//2,w)]]
            wordhalf.append(rho(pair_vector(half[0],condition),pair_vector(half[1],condition)))
        result['repeat_half_relation_reliability']=repeat
        result['disjoint_wordgroup_relation_reliability']=dict(Audio=wordhalf[0],ACC=wordhalf[1])
        relation[condition]=result
        print('CONDITION_COMPLETE',condition,flush=True)
    # Authentication: query recording scores, macro over users and word tasks.
    auth=[];margins=[]
    for label,z,p,s in zip(['Audio','ACC'],embeddings,proto,allscores):
        aucs={'sameword':np.zeros(n),'crossword':np.zeros(n)};counts={'sameword':0,'crossword':0}
        margin=np.empty((2,w,w,n))
        for f in [0,1]:
            q=z[:,:,4:8] if f==0 else z[:,:,:4]
            for qw in range(w):
                labels=np.repeat(np.arange(n),4)
                for rw in range(w):
                    kind='sameword' if qw==rw else 'crossword';counts[kind]+=1
                    sim=p[f,:,rw]@q[:,qw].reshape(n*4,-1).T
                    for u in range(n):aucs[kind][u]+=rank_auc(labels==u,sim[u])
                    cell=s[f,qw,rw];off=cell.copy();np.fill_diagonal(off,-np.inf)
                    margin[f,qw,rw]=np.diag(cell)-off.max(1)
        auth.append(dict(modality=label,per_user=[dict(user=f'U{u+1}',**{k:float(v[u]/counts[k]) for k,v in aucs.items()}) for u in range(n)],
                         macro_auc={k:float(v.mean()/counts[k]) for k,v in aucs.items()}))
        margins.append(margin.mean(0)[~np.eye(w,dtype=bool)])
    a,b=matrices['different_user_cross_word'];ranks=ranking(a,b)
    def rowranks(m):
        r=np.zeros_like(m)
        for u in range(n):
            ix=np.arange(n)!=u;v=rankdata(m[u,ix]);v-=v.mean();r[u,ix]=v/np.linalg.norm(v)
        return r
    ra,rb=rowranks(a),rowranks(b)
    nullrank=(ra[perms[:,:,None],perms[:,None,:]]*rb).sum((1,2))/n
    nulltop={}
    for k in [1,2,3]:
        ta,tb=np.zeros_like(a),np.zeros_like(b)
        ta[np.arange(n)[:,None],hardest(a,k)]=1;tb[np.arange(n)[:,None],hardest(b,k)]=1
        nulltop[str(k)]=((ta[perms[:,:,None],perms[:,None,:]]*tb).sum((1,2))/(n*k)).tolist()
    ranks['identity_permutation_p_one_sided']=(1+sum(x>=ranks['spearman_macro'] for x in nullrank))/(1+len(perms))
    ranks['topk_permutation_p_one_sided']={k:(1+sum(x>=ranks['topk_overlap'][k] for x in v))/(1+len(v)) for k,v in nulltop.items()}
    ma,mb=margins;owner_corr=rho(ma.mean(0),mb.mean(0))
    margin=dict(per_user=[dict(user=f'U{u+1}',audio_mean=float(ma[:,u].mean()),acc_mean=float(mb[:,u].mean()),
                  within_user_task_spearman=rho(ma[:,u],mb[:,u])) for u in range(n)],
                owner_mean_spearman=owner_corr,
                owner_mean_identity_permutation_p_one_sided=(1+sum(rho(ma.mean(0)[p],mb.mean(0))>=owner_corr for p in perms))/(1+len(perms)),
                wordpair_centered_descriptive_spearman=rho(ma-ma.mean(1,keepdims=True),mb-mb.mean(1,keepdims=True)))
    for r in relation.values():r['identity_permutation_p_holm_3']=None
    ordered=sorted(conditions,key=lambda c:relation[c]['identity_permutation_p_one_sided']);previous=0.
    for j,c in enumerate(ordered):
        previous=max(previous,min(1.,(3-j)*relation[c]['identity_permutation_p_one_sided']));relation[c]['identity_permutation_p_holm_3']=previous
    # All directed relations reported; none selected from final data or discarded.
    ar=np.zeros((n,n));br=ar.copy()
    for u in range(n):
        ix=np.arange(n)!=u;ar[u,ix]=rankdata(-a[u,ix]);br[u,ix]=rankdata(-b[u,ix])
    stability=[]
    for p,ss in zip(proto,scores):
        stability.append([collapse(np.einsum('urv,kqv->qruk',p[f],p[f]),'different_user_cross_word') for f in [0,1]]+
            [collapse(ss[np.ix_(ix,ix)],'different_user_cross_word') for ix in [np.arange(w//2),np.arange(w//2,w)]])
    pairs=[]
    for u in range(n):
        for v in range(n):
            if u!=v:pairs.append(dict(target=f'U{u+1}',impostor=f'U{v+1}',audio_difficulty_rank=float(ar[u,v]),acc_difficulty_rank=float(br[u,v]),
                                      shared_top2=bool(ar[u,v]<=2 and br[u,v]<=2),
                                      audio_top2_stable_across_repeat_and_word_splits=bool(all(v in hardest(s,2)[u] for s in stability[0])),
                                      acc_top2_stable_across_repeat_and_word_splits=bool(all(v in hardest(s,2)[u] for s in stability[1]))))
    nuisance=permutation_test(pair_residual(a),pair_residual(b),'different_user_cross_word',perms)
    return dict(authentication=auth,relation_conditions=relation,hard_impostor_ranking=ranks,margin=margin,all_directed_relations=pairs,
        cross_word_dyadic_relation_after_additive_user_effect_removal=nuisance,
        pooled_unconditioned_spearman_descriptive=rho(scores[0],scores[1]),
        settings=dict(users=n,words=w,repeats_used=8,registration=4,query=4,folds=2,bootstrap_word_blocks=bootstraps,identity_permutations=permutations,seed=seed),
        inference_scope='conditional training-set representation audit;not unseen-recording or unseen-user performance;no new training')

def main(args):
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    paths=[Path(args.frontend),Path(args.audio),Path(args.acc)]
    if not all(p.exists() for p in paths):
        (out/'unavailable_public.json').write_text(json.dumps(dict(state='unavailable',reason='private cache missing;no numeric results')),encoding='utf-8');return
    f,t,e=[np.load(p,allow_pickle=False) for p in paths]
    ids=f['sample_id'];u=f['user'];w=f['word'];fit=np.flatnonzero((w>=0)&(w<18))
    assert np.array_equal(fit,validate_alignment(f,t,e))
    assert set(u[fit])==set(range(1,10)) and set(w[fit])==set(range(18))
    # The only stored Audio values accessed outside fit are a zero-placeholder
    # validation. This cache contains no out-of-fit Audio embeddings.
    audio=t['embedding'];assert np.count_nonzero(audio[np.setdiff1d(np.arange(len(u)),fit)])==0
    audio=audio[fit];acc=e['E0'][fit];uu=u[fit];ww=w[fit]
    assert np.isfinite(audio).all() and np.isfinite(acc).all()
    assert np.all(np.linalg.norm(audio,axis=1)>.99) and np.all(np.linalg.norm(acc,axis=1)>.99)
    counts=[];selected=[]
    for user in range(1,10):
        row=[]
        for word in range(18):
            ix=np.flatnonzero((uu==user)&(ww==word));assert len(ix)>=8
            row.append(len(ix));selected.extend(ix[:8])
        counts.append(dict(user=f'U{user}',word_counts=row))
    selected=np.array(selected)
    duplicates={}
    for name,z in [('Audio',audio),('ACC_E0',acc)]:
        duplicates[name]=int(len(z)-len(np.unique(z,axis=0)))
        assert duplicates[name]==0,'exact duplicate embeddings require investigation'
    check=dict(state='PASS',available_fit_records=len(fit),balanced_used_records=len(selected),
        excluded_fit_records=len(fit)-len(selected),sample_ids_unique_and_exactly_aligned=True,
        user_word_labels_match_frontend_and_E0=True,audio_indices_exactly_fit_0_17=True,nonfit_audio_is_zero_placeholder=True,
        no_waveform_or_final_audio_reads=True,all_nine_users_included=True,per_user_word_counts=counts,
        exact_duplicate_embedding_rows=duplicates,record_order='within each user-word,first8 cached occurrence order;not independently verified repetition metadata',
        word_semantics='nominal historical mapping;not transcript-verified',
        E0_training_exposure='all audit recordings used to train E0;selected checkpoint used Dev18..20;no Audio distillation',
        source_sha256=[dict(kind=k,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in zip(['frontend','train_audio_cache','E0_cache'],paths)])
    result=analyze(audio[selected].reshape(9,18,8,-1),acc[selected].reshape(9,18,8,-1),args.bootstraps,args.permutations)
    for name,value in [('data_checks_public.json',check),('results_public.json',result)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(state='complete',authentication=result['authentication'],ranking=result['hard_impostor_ranking']['spearman_macro'],relations={k:v['spearman'] for k,v in result['relation_conditions'].items()}),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frontend',default='outputs/mei_common_frontend_private.npz')
    p.add_argument('--audio',default='outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz')
    p.add_argument('--acc',default='outputs/mei_crossword_representation_20261116/vectors_private.npz')
    p.add_argument('--out',default='outputs/mei_audio_acc_relation_audit_20261116')
    p.add_argument('--bootstraps',type=int,default=500);p.add_argument('--permutations',type=int,default=1999)
    main(p.parse_args())
