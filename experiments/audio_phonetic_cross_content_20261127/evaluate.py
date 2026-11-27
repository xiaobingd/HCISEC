"""Two-phase threshold lock then ACC-only cross-content evaluation. Private scores stay local."""
import argparse,json,hashlib,time,itertools
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve,roc_auc_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from common import Student
from train import ARMS,digest,protocol
from prepare_acc import PRIVATE,OUT,OPT,DEV,CAL,SHORT,LONG,CHINESE

SEEDS=[42,43,44];FAMILIES=dict(english_short=SHORT,english_long=LONG,chinese=CHINESE)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def metrics(labels,scores,threshold):
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores);assert labels.any() and (~labels).any()
    fpr,tpr,_=roc_curve(labels,scores,drop_intermediate=False);fnr=1-tpr;d=fpr-fnr;j=np.flatnonzero(d>=0)[0]
    if j==0:eer=float((fpr[0]+fnr[0])/2)
    else:
        a=-d[j-1]/(d[j]-d[j-1]);eer=float(fpr[j-1]+a*(fpr[j]-fpr[j-1]))
    far=float(np.mean(scores[~labels]>=threshold));frr=float(np.mean(scores[labels]<threshold))
    return dict(EER=eer,AUC=float(roc_auc_score(labels,scores)),FAR=far,FRR=frr,TAR=1-frr,HTER=(far+frr)/2,
      genuine_count=int(labels.sum()),impostor_count=int((~labels).sum()))

def load_inputs():
    data=np.load(PRIVATE/'acc_private.npz');norm=np.load(PRIVATE/'normalization_final_private.npz')
    assert np.array_equal(norm['fit_indices'],np.flatnonzero(np.isin(data['word'],OPT+DEV)))
    x=torch.tensor((data['acc']-norm['mean'][None,:,None])/norm['std'][None,:,None]);valid=torch.tensor(data['valid'])
    return data,norm,x,valid

def load_model(seed,arm):
    path=PRIVATE/f'final_{seed}_{arm}_private.pt';saved=torch.load(path,weights_only=True)
    assert saved['steps']==1000 and saved['protocol_sha']==digest(protocol())
    model=Student(392,16);model.load_state_dict(saved['model']);model.eval()
    return model,saved,path

def embed(model,x,valid,indices):
    result=[]
    with torch.inference_mode():
        for batch in np.array_split(indices,max(1,int(np.ceil(len(indices)/64)))):
            result.append(model(x[batch],valid[batch],auxiliary=False)[0].numpy())
    z=np.concatenate(result);assert np.isfinite(z).all() and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
    return z

def calibrate():
    data,norm,x,valid=load_inputs();w,u=data['word'],data['user'];reg=data['registration_indices'];cal=np.flatnonzero(np.isin(w,CAL))
    assert reg.shape==(10,12) and not np.isin(cal,reg).any();thresholds=[];templates=[];checkpoint_hashes=[]
    for seed in SEEDS:
      for arm in ARMS:
        model,saved,path=load_model(seed,arm);zreg=embed(model,x,valid,reg.ravel()).reshape(10,12,192)
        proto=normalize(zreg.mean(1));zcal=embed(model,x,valid,cal);score=zcal@proto.T;tau=[]
        for user in range(1,11):
            genuine=score[u[cal]==user,user-1];impostor=score[u[cal]!=user,user-1]
            tau.append((np.quantile(genuine,.1)+np.quantile(impostor,.95))/2)
        thresholds.append(tau);templates.append(proto);checkpoint_hashes.append(sha(path))
        # Deployment contains no phoneme, word, or identity-classifier training heads.
        encoder={key:value for key,value in saved['model'].items() if key.startswith(('backbone.','attention.','identity.'))}
        torch.save(dict(encoder=encoder,normalization_mean=torch.tensor(norm['mean']),normalization_std=torch.tensor(norm['std']),
          templates=torch.tensor(proto),thresholds=torch.tensor(tau),query_input='rawACC only;noAudio/noWordLabel'),PRIVATE/f'deploy_{seed}_{arm}_private.pt')
    path=PRIVATE/'threshold_lock_private.npz';meta=OUT/'threshold_lock_public.json'
    values=np.asarray(thresholds);proto=np.asarray(templates)
    if meta.exists():
        old=np.load(path);public=json.loads(meta.read_text(encoding='utf-8'))
        assert np.allclose(old['thresholds'],values,atol=1e-12) and public['checkpoint_sha256']==checkpoint_hashes
        print('EXISTING_THRESHOLD_LOCK_REPLAYED',flush=True);return
    np.savez_compressed(path,thresholds=values,templates=proto,calibration_indices=cal,registration_indices=reg)
    meta.write_text(json.dumps(dict(status='LOCKED',locked_at_unix=time.time(),count=int(values.size),models=15,users=10,
      checkpoint_sha256=checkpoint_hashes,lock_sha256=sha(path),development_lock_sha256=sha(OUT/'development_lock_public.json'),
      formula='(genuineCalQ10+all9identityImpostorCalQ95)/2',quantile_interpolation='numpy linear',
      query_scores_computed_before_lock=0,query_Audio_read=0),indent=2),encoding='utf-8')
    print('ALL_150_THRESHOLDS_LOCKED_BEFORE_QUERIES',flush=True)

def comparison_stats(users):
    rng=np.random.default_rng(271127);rows=[];draw=rng.integers(0,10,size=(10000,10));signs=np.array(list(itertools.product([-1,1],repeat=10)))
    for control in ['T0','T2','T4']:
        d=[]
        for user in range(1,11):
            a=np.mean([r['EER'] for r in users if r['user']==user and r['arm']=='T3' and r['family']!='chinese'])
            b=np.mean([r['EER'] for r in users if r['user']==user and r['arm']==control and r['family']!='chinese'])
            d.append(a-b)
        d=np.array(d);mean=float(d.mean());ci=np.quantile(d[draw].mean(1),[.025,.975]);p=float(np.mean(abs((signs*d).mean(1))>=abs(mean)-1e-12))
        seed_differences={str(seed):float(np.mean([r['EER'] for r in users if r['seed']==seed and r['arm']=='T3' and r['family']!='chinese'])-np.mean([r['EER'] for r in users if r['seed']==seed and r['arm']==control and r['family']!='chinese'])) for seed in SEEDS}
        rows.append(dict(comparison='T3-'+control,metric='English equal-family seed-averaged user-macro EER',
          mean_difference=mean,user_cluster_bootstrap_95CI=ci.tolist(),exact_user_signflip_p=p,users=10,bootstrap_replicates=10000,
          per_seed_difference=seed_differences,seeds_improved=int(sum(v< -1e-8 for v in seed_differences.values())),users_improved=int(np.sum(d< -1e-8)),
          interpretation='negative favors T3'))
    order=np.argsort([r['exact_user_signflip_p'] for r in rows]);last=0.
    for rank,idx in enumerate(order):
        last=max(last,(3-rank)*rows[int(idx)]['exact_user_signflip_p']);rows[int(idx)]['Holm_corrected_p']=min(1.,last)
    return rows

def probe(z,users,words):
    rows=[]
    for parity in [0,1]:
        tr=users%2==parity;te=~tr;model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500,solver='lbfgs',random_state=42))
        model.fit(z[tr],words[tr]);pred=model.predict(z[te]);rows.append(float(balanced_accuracy_score(words[te],pred)))
    return dict(train_word_balanced_accuracy=float(np.mean(rows)),two_fold_accuracy=rows,chance=1/16,
      description='16 fit-word classifier;odd/even user-held-out probe;encoder itself closed-set trained on all10users;explanation only,not model selection;no unseen-word-class accuracy')

def query():
    assert (OUT/'threshold_lock_public.json').exists();lockpublic=json.loads((OUT/'threshold_lock_public.json').read_text(encoding='utf-8'))
    assert sha(PRIVATE/'threshold_lock_private.npz')==lockpublic['lock_sha256']
    first=OUT/'query_phase_public.json'
    if not first.exists():first.write_text(json.dumps(dict(query_scoring_started_at_unix=time.time(),threshold_lock_sha256=lockpublic['lock_sha256'],all_thresholds_locked=150,query_Audio_read=0),indent=2))
    data,norm,x,valid=load_inputs();u,w=data['user'],data['word'];lock=np.load(PRIVATE/'threshold_lock_private.npz')
    queryidx=np.flatnonzero(np.isin(w,SHORT+LONG+CHINESE));fit=np.flatnonzero(np.isin(w,OPT+DEV));rows=[];wordrows=[];probes=[];modelindex=0
    for seed in SEEDS:
      for arm in ARMS:
        model,_,path=load_model(seed,arm);assert sha(path)==lockpublic['checkpoint_sha256'][modelindex]
        z=embed(model,x,valid,queryidx);scores=z@lock['templates'][modelindex].T;tau=lock['thresholds'][modelindex]
        np.savez_compressed(PRIVATE/f'query_scores_{seed}_{arm}_private.npz',scores=scores,indices=queryidx)
        for family,words in FAMILIES.items():
          keep=np.isin(w[queryidx],words)
          for user in range(1,11):
            rows.append(dict(seed=seed,arm=arm,family=family,user=user,**metrics(u[queryidx][keep]==user,scores[keep,user-1],tau[user-1])))
          for word in words:
            keepword=w[queryidx]==word
            for user in range(1,11):
                wordrows.append(dict(seed=seed,arm=arm,family=family,word=word,user=user,**metrics(u[queryidx][keepword]==user,scores[keepword,user-1],tau[user-1])))
        zfit=embed(model,x,valid,fit);probes.append(dict(seed=seed,arm=arm,**probe(zfit,u[fit],w[fit])))
        print('ACC_ONLY_QUERY_COMPLETE',seed,arm,flush=True);modelindex+=1
    names=['EER','AUC','FAR','FRR','TAR','HTER'];summary=[]
    for arm in ARMS:
      for family in list(FAMILIES)+['english_primary']:
        rr=[r for r in rows if r['arm']==arm and (r['family']==family or (family=='english_primary' and r['family']!='chinese'))]
        summary.append(dict(arm=arm,family=family,**{k:float(np.mean([r[k] for r in rr])) for k in names}))
    result=dict(status='COMPLETE',protocol_sha256=digest(protocol()),models=15,seeds=SEEDS,users=10,primary='equal English short/long family user macro then seed mean',
      metrics_units='fractions,not percentages',summary=summary,per_seed_user=rows,per_seed_word_user=wordrows,content_probes=probes,comparisons=comparison_stats(rows),
      sameword_matched_diagnostic=dict(status='NOT_COMPARABLE',K=12,G_content=None,reason='Per-person-per-word continuous windows insufficient for12 independent same-word registration plus an independent query;no window reuse and no K adjustment after queries.'),
      query_Audio_read=0,query_Audio_clock_read=0,all_other_9_identities_retained=True,all_predeclared_query_words_retained=True)
    (OUT/'results_public.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL_SUMMARY',json.dumps([r for r in summary if r['family']=='english_primary']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['calibrate','query'],required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        calibrate() if args.phase=='calibrate' else query()
