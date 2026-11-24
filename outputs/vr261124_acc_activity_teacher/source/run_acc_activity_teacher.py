"""Audio activity proxy -> ACC-only local mask; held identity and held-word assay."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,balanced_accuracy_score
from threadpoolctl import threadpool_limits
from audit_raw_common_dynamics import activity_mask,normalize
from raw_common_feature_identity import batch_features

class ActivityNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(15,16,5,padding=2),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,1,1))
    def forward(self,x):return self.net(x).squeeze(1)
def acc_input(x,m):return np.concatenate([x,x-np.median(x,axis=1,keepdims=True),m[:,:,None]],axis=2)
def infer_acc(net,inputs,mean,std):
    """Inference interface has no Audio or identity label argument."""
    xx=((inputs-mean)/std).astype('float32');result=[]
    with torch.inference_mode():
        for start in range(0,len(xx),64):result.append(net(torch.tensor(xx[start:start+64].transpose(0,2,1))).sigmoid().numpy())
    return np.concatenate(result)
def choose_threshold(prob,truth):
    rows=[]
    for threshold in [.3,.4,.5,.6,.7]:
        pred=prob>=threshold;rows.append(dict(threshold=threshold,f1=float(f1_score(truth.ravel(),pred.ravel(),zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(truth.ravel(),pred.ravel()))))
    return min(rows,key=lambda v:(-round(v['f1'],12),-round(v['balanced_accuracy'],12),abs(v['threshold']-.5),v['threshold'])),rows
def cutoff(score,label):return float((np.quantile(score[label],.1)+np.quantile(score[~label],.95))/2)
def measure(score,label,tau):
    accept=score>=tau;fa=int(accept[~label].sum());fr=int((~accept[label]).sum());ng=int(label.sum());ni=int((~label).sum())
    fp,tp,_=roc_curve(label,score,drop_intermediate=False);i=np.argmin(abs(fp-(1-tp)))
    return dict(eer=float((fp[i]+1-tp[i])/2),auc=float(roc_auc_score(label,score)),far=fa/ni,frr=fr/ng,hter=(fa/ni+fr/ng)/2,genuine_trials=ng,attack_trials=ni,false_accepts=fa,false_rejects=fr)
def main(root):
    torch.set_num_threads(2);out=root/'conditional_correspondence/acc_activity_teacher';out.mkdir(exist_ok=True)
    c=np.load(root/'conditional_correspondence/raw_common_dynamics/local_power_private.npz')
    x,m,e=[c[k].astype('float64') for k in ('acc','motion','audio_activity')];u,w,r=[c[k] for k in ('user','word','repeat')]
    assert x.shape==(1280,146,7);words=sorted(map(int,set(w)));trainwords=words[:12];newwords=words[12:]
    inputs=acc_input(x,m);truth=np.array([activity_mask(v) for v in e]);selfmask=np.array([activity_mask(np.log(np.maximum(np.exp(v).sum(1),1e-20))) for v in x])
    lookup={(int(us),int(wo),int(re)):i for i,(us,wo,re) in enumerate(zip(u,w,r))};wrong=np.array([lookup[int(us),int(wo),(int(re)+4)%8] for us,wo,re in zip(u,w,r)])
    names=['ACC_self','Audio_proxy_upperbound','Predicted_correct','Predicted_wrongrepeat','Time_prior']
    protocol=dict(users=10,seeds=[42,43,44],fit_words=trainwords,new_words=newwords,
      split='heldtarget andnextcyclicdevidentity excludedfrompredictor training;8baseidentities,12fitwords;devidentity4newwordsonlyselectprobthreshold;targetneverused',
      network='15channels:7logpowers+7perrecordmedian-centeredlogpowers+1motion;Conv15to16k5 GN SiLU,Conv16to16k3 GN SiLU,Conv16to1k1;noidentityclassifier',
      training='400steps AdamW.001 wd.01,clip1;8usersx4randomrecords,matchedinit/batches;train-labelposweight only;no checkpoint search',
      wrong='sameuser sameword Audio activityproxy fromrepeat(r+4)%8;preserveword/identity andcropcontext;cannotenterheldtarget/dev',
      probability_threshold='grid.3,.4,.5,.6,.7 maximizeAudio-proxy F1 oninnerdevidentity new4words;balancedaccuracy thennear.5tie;notidentitymetrics',
      proxy='previousAudio300-3500Hz relativeenergy proxy,notgroundtruthVAD;physicalsync remainsunverified;no lagsearch',
      controls=names,time_prior='meantrainingAudio activityprobability vsframeposition,nosignalinput;samedevthresholdrule',
      inference='predicted interface consumesACC only;Audio targetlabels usedonly training/dev orofflineevaluation ofmaskquality;upperbound explicitlyqueryAudio assisted',
      identity='same16ACCstatistics andcosineprototypes;12regrecords(first6words/reps0,1);cal fitlast6words/reps0..3;queryfitlast6/reps4..7 ORnew4/all8',
      normalization='inputchannels,featurescaler/imputation,motionQ95only8baseidentities/train12words',
      authentication_threshold='CALgenuineQ10+impostorQ95 midpoint;all150thresholdssavedbeforequeryscores',
      scope='maskpredictor identityholdout,closed10personbackground;notfullopen-setunseenattackers;historicallyexamined16wordcohort;rawstatistics,noexistingidentityencodermodified',
      input_crop='historical3secondACC windows centeredusingAudio-VAD metadata;mask/identityinference hasnoAudioargument,butthisisNOTend-to-endcontinuousACC-onlycropping',
      limits='new4wordsseenondevidentity forprobabilitythreshold,not blindwordconfirmation;overlapping3secondwindows;noindependentwear/daydata;notgroundtruthspeechlabels')
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    context={};locks=[];training=[];selection=[];quality=[];saved={};audits=[]
    for seed in (42,43,44):
      for held in range(1,11):
        dev=held%10+1;base=[us for us in range(1,11) if us not in (held,dev)];fit=np.isin(u,base)&np.isin(w,trainwords);ti=np.flatnonzero(fit)
        di=np.flatnonzero((u==dev)&np.isin(w,newwords));mean=inputs[fit].mean((0,1));std=np.maximum(inputs[fit].std((0,1)),1e-5)
        xx=torch.tensor(((inputs-mean)/std).astype('float32').transpose(0,2,1));rng=np.random.default_rng(seed*100+held)
        batches=np.array([np.concatenate([rng.choice(np.flatnonzero(fit&(u==us)),4,replace=False) for us in base]) for _ in range(400)])
        assert not np.isin(u[batches],[held,dev]).any() and np.isin(w[batches],trainwords).all()
        assert not np.isin(u[wrong[batches]],[held,dev]).any() and np.array_equal(u[wrong[batches]],u[batches]) and np.array_equal(w[wrong[batches]],w[batches])
        digest=hashlib.sha256(batches.tobytes()).hexdigest();np.savez_compressed(out/f'fold_seed{seed}_held{held}_private.npz',mean=mean,std=std,batches=batches,training_indices=ti,development_indices=di)
        audits.append(dict(seed=seed,target=held,development_user=dev,training_users=base,training_records=len(ti),episode_sha256=digest,target_training_records=0,dev_training_records=0))
        motion=m<=float(np.quantile(m[fit],.95));masks={'ACC_self':selfmask&motion,'Audio_proxy_upperbound':truth&motion}
        probs={}
        for method in ['Predicted_correct','Predicted_wrongrepeat']:
          torch.manual_seed(seed*100+held);net=ActivityNet();target=truth if method=='Predicted_correct' else truth[wrong]
          pos=float(target[fit].mean());weight=torch.tensor((1-pos)/max(pos,1e-6),dtype=torch.float32);labels=torch.tensor(target,dtype=torch.float32)
          optimizer=torch.optim.AdamW(net.parameters(),lr=.001);trace=[];path=out/f'{method}_seed{seed}_held{held}_private.pt'
          for step,batch in enumerate(batches):
            logit=net(xx[batch]);loss=F.binary_cross_entropy_with_logits(logit,labels[batch],pos_weight=weight)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
            if (step+1)%100==0:trace.append(dict(step=step+1,loss=float(loss.detach())))
          net.eval();torch.save(net.state_dict(),path);prob=infer_acc(net,inputs,mean,std);best,candidates=choose_threshold(prob[di],truth[di])
          masks[method]=(prob>=best['threshold'])&motion;probs[method]=prob
          selection.append(dict(seed=seed,user=held,development_user=dev,method=method,probability_threshold=best['threshold'],candidates=candidates))
          training.append(dict(seed=seed,user=held,method=method,steps=400,trace=trace))
        prior=np.broadcast_to(truth[fit].mean(0),(1280,146));best,candidates=choose_threshold(prior[di],truth[di]);masks['Time_prior']=(prior>=best['threshold'])&motion;probs['Time_prior']=prior
        selection.append(dict(seed=seed,user=held,development_user=dev,method='Time_prior',probability_threshold=best['threshold'],candidates=candidates))
        for method,prob in probs.items():
          for scope,mask in [('fit_words',(u==held)&np.isin(w,trainwords[6:])&(r>=4)),('new_words',(u==held)&np.isin(w,newwords))]:
            tau=next(v['probability_threshold'] for v in selection if v['seed']==seed and v['user']==held and v['method']==method);pred=prob[mask]>=tau
            quality.append(dict(seed=seed,user=held,method=method,scope=scope,f1=float(f1_score(truth[mask].ravel(),pred.ravel(),zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(truth[mask].ravel(),pred.ravel())),proxy_auc=float(roc_auc_score(truth[mask].ravel(),prob[mask].ravel())),positive_fraction=float(pred.mean())))
        for name,mask in masks.items():
          saved[f'{seed}_{held}_{name}']=mask
          z0=batch_features(x,x,mask)[2];meanf=np.nanmean(z0[fit],0);meanf=np.where(np.isfinite(meanf),meanf,0);z=np.where(np.isfinite(z0),z0,meanf);z=(z-meanf)/np.maximum(z[fit].std(0),1e-5)
          proto=normalize(np.stack([z[(u==us)&np.isin(w,trainwords[:6])&(r<2)].mean(0) for us in range(1,11)]))
          ci=np.flatnonzero(np.isin(w,trainwords[6:])&(r<4));score=normalize(z[ci])@proto[held-1]
          locks.append(dict(seed=seed,user=held,method=name,threshold=cutoff(score,u[ci]==held)))
          context[seed,held,name]=dict(z=z,proto=proto,insufficient=float(np.mean(mask.sum(1)<12)))
        (out/'training_public.json').write_text(json.dumps(training));print('ACC_ACTIVITY_FOLD_DONE',seed,held,flush=True)
    np.savez_compressed(out/'masks_private.npz',**saved);(out/'thresholds_private.json').write_text(json.dumps(locks));(out/'mask_threshold_selection_private.json').write_text(json.dumps(selection))
    (out/'exclusion_audit_public.json').write_text(json.dumps(audits));(out/'mask_quality_public.json').write_text(json.dumps(quality));print('ACC_ACTIVITY_ALL_THRESHOLDS_LOCKED',flush=True)
    rows=[]
    for seed in (42,43,44):
      for held in range(1,11):
       for name in names:
        ctx=context[seed,held,name];z,proto=ctx['z'],ctx['proto'];tau=next(v['threshold'] for v in locks if (v['seed'],v['user'],v['method'])==(seed,held,name))
        for scope,query in [('fit_words',np.isin(w,trainwords[6:])&(r>=4)),('new_words',np.isin(w,newwords))]:
          qi=np.flatnonzero(query);scores=normalize(z[qi])@proto.T;genuine=u[qi]==held;sub=scores[genuine];top=np.isclose(sub,sub.max(1,keepdims=True),rtol=0,atol=1e-10)
          rows.append(dict(seed=seed,user=held,method=name,scope=scope,top1_10way=float(np.mean(top[:,held-1]/top.sum(1))),insufficient_record_fraction=ctx['insufficient'],**measure(scores[:,held-1],genuine,tau)))
    macro=[];users=[]
    for name in names:
     for scope in ['fit_words','new_words']:
      rr=[v for v in rows if v['method']==name and v['scope']==scope]
      macro.append(dict(method=name,scope=scope,**{k:float(np.mean([v[k] for v in rr])) for k in ['eer','hter','far','frr','auc','top1_10way','insufficient_record_fraction']}))
      for us in range(1,11):
        pp=[v for v in rr if v['user']==us];users.append(dict(method=name,scope=scope,user=us,**{k:float(np.mean([v[k] for v in pp])) for k in ['eer','hter','far','frr','auc','top1_10way']}))
    (out/'per_seed_user_public.json').write_text(json.dumps(rows));(out/'summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=users),indent=2))
    (out/'verification_public.json').write_text(json.dumps(dict(status='PASS',models=60,folds=30,identity_thresholds=150,probability_selections=90,workpoints=len(rows),inference_only_ACC_for_predicted=True,all_thresholds_before_query_scores=True,global_identity_encoder_changed=False,parameters=sum(p.numel() for p in ActivityNet().parameters())),indent=2))
    print('ACC_ACTIVITY_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    with threadpool_limits(limits=2):main(p.parse_args().root)
