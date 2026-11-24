"""Development-only observability/identity screening; never reads query Audio."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.stats import rankdata
import torch
from train_crossword_mechanism import Network,DEV

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'conditional_correspondence/band_screen';OUT.mkdir(exist_ok=True)

def auc_scores(scores,permutation):
    # pooled word-pair trials; permutation moves identities consistently across words.
    ranks=rankdata(scores.reshape(26,-1),axis=1).reshape(scores.shape)
    npos=scores.shape[1]*10;nneg=npos*9
    chosen=ranks[:,:,np.arange(10),permutation].sum((1,2))
    return (chosen-npos*(npos+1)/2)/(npos*nneg)

def identity_matrices(values,indices,u,w,r):
    result=[]
    for a,b in itertools.permutations(DEV,2):
        p=[];q=[]
        for us in range(1,11):
            ea=indices[(u[indices]==us)&(w[indices]==a)&(r[indices]<4)]
            qb=indices[(u[indices]==us)&(w[indices]==b)&(r[indices]>=4)]
            assert len(ea)==len(qb)==4
            p.append(values[ea].mean(0));q.append(values[qb].mean(0))
        p=np.array(p);q=np.array(q)
        p=p/np.maximum(np.linalg.norm(p,axis=2,keepdims=True),1e-8)
        q=q/np.maximum(np.linalg.norm(q,axis=2,keepdims=True),1e-8)
        result.append(np.einsum('ift,jft->fij',p,q))
    return np.stack(result,axis=1)

def main():
    torch.set_num_threads(2)
    d=np.load(ROOT/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz')
    u,w,r,fit=d['user'],d['word'],d['repeat'],d['fit'];ids=d['sample_id'];dev=np.isin(w,DEV);di=np.flatnonzero(dev)
    c=np.load(ROOT/'conditional_correspondence/stage2/audio_targets_private.npz')
    assert np.array_equal(c['sample_id'],ids) and np.array_equal(c['allowed'],fit|dev)
    target=c['target'];am=target[fit].mean((0,2),keepdims=True);asd=target[fit].std((0,2),keepdims=True)
    target=(target-am)/np.maximum(asd,1e-5)
    raw=d['acc'];spec=[]
    for i in range(0,3200,80):
        f,t,z=signal.stft(raw[i:i+80],1000,nperseg=100,noverlap=80,nfft=100,boundary=None,padded=False)
        spec.append(np.log(np.maximum(abs(z[:,:,1:50])**2,1e-20)).reshape(-1,147,len(t)))
    spec=np.concatenate(spec).astype('float32');mean=spec[fit].mean((0,2),keepdims=True);std=spec[fit].std((0,2),keepdims=True)
    x=torch.tensor((spec[di]-mean)/np.maximum(std,1e-5));del raw,spec
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))}
    wrong=np.array([lookup[int(u[i]),int(w[i]),(int(r[i])+1)%8] for i in di])
    predictions={};allpred={};loss={};paired=[];seedadv=[]
    for seed in (42,43,44):
        for method in ('B_correct','C_wrong_repeat','D_wrong_user'):
            net=Network();net.load_state_dict(torch.load(ROOT/f'conditional_correspondence/stage3/{method}_{seed}_private.pt',map_location='cpu',weights_only=True))
            with torch.inference_mode():pred=np.concatenate([net.eval()(x[i:i+80])[1].numpy() for i in range(0,len(di),80)])
            allpred[seed,method]=pred
            loss[seed,method]=((pred-target[di])**2).mean(2)
            if method=='B_correct':
                predictions[seed]=pred
                adv=((pred-target[wrong])**2).mean(2)-loss[seed,method]
                paired.append(np.array([adv[u[di]==us].mean(0) for us in range(1,11)]))
                seedadv.append(adv.mean(0))
            print('BAND_PRED',seed,method,flush=True)
    useradv=np.mean(paired,axis=0);seedadv=np.array(seedadv)
    signs=np.array(list(itertools.product([-1.,1.],repeat=10)))
    denom=np.maximum(useradv.std(0,ddof=1)/np.sqrt(10),1e-12)
    observed=useradv.mean(0)/denom
    # max-T across26correlated frequency bins, enumeration at user level.
    null=(signs@useradv/10)/denom
    p_fwer=np.mean(null.max(1)[:,None]>=observed[None,:]-1e-12,axis=0)
    fullpred=np.zeros_like(target)
    fullpred[di]=np.mean(list(predictions.values()),axis=0)
    audio_scores=identity_matrices(target,di,u,w,r)
    pred_scores=identity_matrices(fullpred,di,u,w,r)
    control_scores={}
    for method in ('C_wrong_repeat','D_wrong_user'):
        control=np.zeros_like(target);control[di]=np.mean([allpred[seed,method] for seed in (42,43,44)],axis=0)
        control_scores[method]=identity_matrices(control,di,u,w,r)
    # Additional attribution audit, separate from the original two-gate candidate selection.
    def per_user_auc(scores):
        result=[]
        for us in range(10):
            ranks=rankdata(scores[:,:,us,:].reshape(26,-1),axis=1).reshape(26,12,10)
            result.append((ranks[:,:,us].sum(1)-12*13/2)/(12*108))
        return np.array(result)
    body=per_user_auc(pred_scores);specificity=[]
    for method,scores in control_scores.items():
        delta=body-per_user_auc(scores);den=np.maximum(delta.std(0,ddof=1)/np.sqrt(10),1e-12)
        observed_delta=delta.mean(0)/den;null_delta=(signs@delta/10)/den
        adjusted=np.mean(null_delta.max(1)[:,None]>=observed_delta[None,:]-1e-12,axis=0)
        specificity.append(dict(control=method,delta=delta.mean(0),p=adjusted,auc=auc_scores(scores,np.arange(10))))
    identity={};rng=np.random.default_rng(4242);perms=np.array([rng.permutation(10) for _ in range(1000)])
    for name,scores in [('audio',audio_scores),('acc_prediction',pred_scores)]:
        true=auc_scores(scores,np.arange(10));nullauc=np.stack([auc_scores(scores,p) for p in perms])
        pval=(1+(nullauc.max(1)[:,None]>=true[None,:]-1e-12).sum(0))/1001
        identity[name]=(true,pval)
    bins=[];users=[]
    for band,freq in enumerate(range(50,301,10)):
        ctrl={method:float(np.mean([loss[s,method][:,band].mean()-loss[s,'B_correct'][:,band].mean() for s in (42,43,44)])) for method in ('C_wrong_repeat','D_wrong_user')}
        consistent=bool((seedadv[:,band]>0).all())
        obs=bool(p_fwer[band]<=.05 and consistent and min(ctrl.values())>0)
        ids_pass=bool(identity['audio'][1][band]<=.05 and identity['acc_prediction'][1][band]<=.05 and min(identity['audio'][0][band],identity['acc_prediction'][0][band])>.5)
        bins.append(dict(hz=freq,paired_mse_advantage=float(useradv[:,band].mean()),paired_user_maxT_p=float(p_fwer[band]),positive_all_seeds=consistent,
            correct_vs_repeat_model_mse_advantage=ctrl['C_wrong_repeat'],correct_vs_other_user_model_mse_advantage=ctrl['D_wrong_user'],
            audio_crossword_auc=float(identity['audio'][0][band]),audio_identity_fwer_p=float(identity['audio'][1][band]),
            acc_prediction_crossword_auc=float(identity['acc_prediction'][0][band]),acc_prediction_identity_fwer_p=float(identity['acc_prediction'][1][band]),
            observability_pass=obs,identity_pass=ids_pass,selected=obs and ids_pass))
        for audit in specificity:
            bins[-1][audit['control']+'_identity_auc']=float(audit['auc'][band])
            bins[-1][audit['control']+'_paired_user_auc_gain']=float(audit['delta'][band])
            bins[-1][audit['control']+'_gain_maxT_p']=float(audit['p'][band])
        bins[-1]['audio_specific_identity_support']=bool(obs and ids_pass and all(a['delta'][band]>0 and a['p'][band]<=.05 for a in specificity))
        for us in range(1,11):users.append(dict(user=us,hz=freq,paired_mse_advantage=float(useradv[us-1,band])))
    selected=[a['hz'] for a in bins if a['selected']]
    summary=dict(train_words=sorted(set(w[fit].tolist())),development_words=DEV,development_records=320,users=10,
        candidates=26,observability_pass_hz=[a['hz'] for a in bins if a['observability_pass']],identity_pass_hz=[a['hz'] for a in bins if a['identity_pass']],selected_hz=selected,
        additional_control_audit_supported_hz=[a['hz'] for a in bins if a['audio_specific_identity_support']],
        rules='dev paired-advantage user maxT p<=.05,positive all3seeds,correct model lowerMSEthanC/D;Audio andACCpred cross-word identity AUC>.5 with identity permutation FWERp<=.05',
        identity_evaluation='12ordered cross-word pairs;first4enrollment repeats,last4query repeats;all10users+9attackers;node permutation consistent acrosswordpairs',
        selection_status='exploratory on historically inspected development words,not independent confirmation;not authentication improvement',
        no_final_audio=True,no_authentication_query_metrics_read=True,models_frozen=True,physical_sync_verified=False,
        limitations='Known users;3second windows may overlap withinword/include neighbors;VAD centers persist;frequency bins correlated;no coherence or speech transferfunction measurement')
    (OUT/'summary_public.json').write_text(json.dumps(summary,indent=2));(OUT/'frequency_public.json').write_text(json.dumps(bins,indent=2));(OUT/'per_user_public.json').write_text(json.dumps(users,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
