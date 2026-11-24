"""Independent word split, genuinely IMU-anchored windows, no test-word Audio."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,balanced_accuracy_score
from threadpoolctl import threadpool_limits

class ActivityNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(15,16,5,padding=2),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,1,1))
    def forward(self,x):return self.net(x).squeeze(1)
def own_activity(x):
    e=np.log(np.maximum(np.exp(x).sum(2),1e-20));lo=np.quantile(e,.2,axis=1);hi=np.quantile(e,.9,axis=1)
    return e>=np.maximum(lo+.25*(hi-lo),np.log(1e-12))[:,None]
def acc_input(x,m):return np.concatenate([x,x-np.median(x,axis=1,keepdims=True),m[:,:,None]],2)
def infer(net,inputs,mean,std):
    net.eval();out=[]
    with torch.inference_mode():
        for i in range(0,len(inputs),64):out.append(net(torch.tensor(((inputs[i:i+64]-mean)/std).astype('float32').transpose(0,2,1))).sigmoid().numpy())
    return np.concatenate(out)
def choose_probability(prob,truth,valid):
    p,y=prob[valid],truth[valid].astype(bool);rows=[]
    for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        pred=p>=t;rows.append(dict(threshold=t,f1=float(f1_score(y,pred,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,pred))))
    best=min(rows,key=lambda r:(-round(r['f1'],12),-round(r['balanced_accuracy'],12),abs(r['threshold']-.5),r['threshold']))
    return best,rows
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def stats(x,mask):
    count=mask.sum(1);div=np.maximum(count,1)[:,None];weight=mask[:,:,None];mean=(x*weight).sum(1)/div
    std=np.sqrt(((x-mean[:,None,:])**2*weight).sum(1)/div)
    z=np.c_[mean-mean.mean(1,keepdims=True),std,mean.mean(1),std.mean(1)];z[count<12]=np.nan
    return z,count<12
def choose_auth(score,y):
    # Low-activity windows always reject; they cannot be rescued by lowering tau.
    values=np.unique(score[score>=-1]);candidates=np.r_[-1.,(values[:-1]+values[1:])/2,1.+1e-6] if len(values) else np.array([1.+1e-6])
    accept=score[None,:]>=candidates[:,None];far=accept[:,~y].mean(1);frr=(~accept[:,y]).mean(1)
    j=min(range(len(candidates)),key=lambda i:(round(float((far[i]+frr[i])/2),12),round(float(abs(far[i]-frr[i])),12),-candidates[i]))
    return float(candidates[j])
def measure(score,y,tau):
    accept=score>=tau;ng,ni=int(y.sum()),int((~y).sum());fa=int(accept[~y].sum());fr=int((~accept[y]).sum());fp,tp,_=roc_curve(y,score,drop_intermediate=False);j=np.argmin(abs(fp-(1-tp)))
    return dict(eer=float((fp[j]+1-tp[j])/2),hter=(fa/ni+fr/ng)/2,far=fa/ni,frr=fr/ng,auc=float(roc_auc_score(y,score)),genuine_trials=ng,attack_trials=ni,false_accepts=fa,false_rejects=fr)
def main(root):
    torch.set_num_threads(2);base=root/'continuous_acc_activity';out=base/'results';out.mkdir(exist_ok=True)
    split=json.loads((base/'word_split_public.json').read_text());c=np.load(base/'continuous_private.npz')
    x,m,y=c['acc'].astype('float64'),c['motion'].astype('float64'),c['audio_proxy'].astype('float64');u,w=c['user'],c['word'];valid=c['imu_valid'];ordinal=c['ordinal']
    trainwords,devwords,testwords=split['train'],split['development'],split['test'];test=np.isin(w,testwords);assert np.isnan(y[test]).all()
    cats=[[5,7,8,15,18,25,26,29,30,33],[0,1,2,4,6,10,11,13,14,16,17,20,21,22,23,24,27,28,31,34],[3,9,12,19,32],[35,36,37,38,39]]
    regwords=sorted(sum([sorted(set(cat)&set(trainwords))[:n] for cat,n in zip(cats,[2,6,2,2])],[]));assert len(regwords)==12
    names=['ACC_direct','ACC_self_trained','Audio_correct','Audio_wrongwindow','Fixed_time_prior'];inputs=acc_input(x,m);selflabels=own_activity(x)
    wrong=np.arange(len(u))
    for us in range(1,11):
      for word in range(40):
        ix=np.flatnonzero((u==us)&(w==word));ix=ix[np.argsort(ordinal[ix])];assert len(ix)>=2
        wrong[ix]=np.roll(ix,-(len(ix)//2))
    assert np.array_equal(u,u[wrong]) and np.array_equal(w,w[wrong]) and (wrong!=np.arange(len(u))).all()
    dataset_hash=hashlib.sha256((base/'continuous_private.npz').read_bytes()).hexdigest()
    protocol=dict(seeds=[42,43,44],users=10,word_split=split,registration_words=regwords,windows=len(u),dataset_sha256=dataset_hash,
      crop='firstvalidIMU+.5seconds,nonoverlap3secondgrids;noVADcentering;testAudioanditsclockfilesneveropened',
      network='same2081parameterConv15to16k5 GN SiLU Conv16to16k3 GN SiLU Conv16to1k1;7ACCpowers,7median-centered,1motion',
      models=['ACC_self_trained','Audio_correct','Audio_wrongwindow'],budget='400steps AdamW.001 wd.01 clip1;8baseidentitiesx4randomwindows,matchedinit/batches for3neuralarms;posweight sameformula',
      label_sources='ACCself perwindowenergy;correctAudio recording-level300-3500Hz energyproxy;wrongAudio sameuserwords nonoverlapwindowpermutation halfwayroundrecording,notverifiedutteranceindex',
      time_prior='nonparammeanAudio-proxy occupancybyframeposition onbase/trainwindows;noACCsignalinput;nooptimizerbudget,thisisreference',
      probability_selection='grid.1..9 F1 thenbalancedaccuracy thennear.5 ononeinnerdevidentity/dev8words;ACCselfarmusesACCselflabels only;Audio/wrong/priorusecorrectAudio proxy;targetexcluded',
      identity_split='heldtarget plusnextcyclicdevidentity excludedfrompredictor training;8baseidentities;input/feature normalizersandmotionQ95 onlybase/train24',
      feature='16rawACCbandshape,std,levelstatistics;samecosineprototype;noidentityencodertraining or embeddingalignment',
      enrollment='12fixedwindows,firstwindowof12stratifiedtrainwords,allmethodssameindices;noAudioqualityselection',
      calibration='allIMUwindows fromdev8words,targetvsother9users;minHTER,thenbalanceFARFRR,thenhigherthreshold;no-test tuning',
      query='allfixed3secondwindows fromtest8words,possibly0or multipleutterances;noAudiofilter;under12selectedvalidframes alwaysrejectscore-1.1,reportedcoverage;noquerywindowremoval',
      lock='all150identitythresholds serializedbeforetestscorecomputation;noEER/HTERselectionofactivitymodel',
      scope='newstratified24/8/8wordprotocol withinhistoricallystudied10people;maskidentityholdout,closedbackground,notfullyunseenattackeropenset;physicaldelayunverified',
      Audio_clock='teacheronly:hostcallbackvsPCMframeendaffinemap,44100candidateforactivitySTFT;notperutterancelagsearch',
      no_test_Audio=True,no_existing_model_modified=True,parameters=sum(p.numel() for p in ActivityNet().parameters()))
    text=json.dumps(protocol,sort_keys=True);protocol_hash=hashlib.sha256(text.encode()).hexdigest();(out/'protocol_public.json').write_text(text)
    trainlog=[];selection=[];audit=[];quality=[];locks=[];context={};savedmasks={}
    for seed in [42,43,44]:
      for held in range(1,11):
        dev=held%10+1;baseusers=[us for us in range(1,11) if us not in (held,dev)];fit=np.isin(u,baseusers)&np.isin(w,trainwords);ti=np.flatnonzero(fit);di=np.flatnonzero((u==dev)&np.isin(w,devwords))
        assert not test[ti].any() and np.isfinite(y[ti]).any()
        mean,std=inputs[fit].mean((0,1)),np.maximum(inputs[fit].std((0,1)),1e-5);xx=torch.tensor(((inputs-mean)/std).astype('float32').transpose(0,2,1))
        rng=np.random.default_rng(seed*100+held);batches=np.array([np.concatenate([rng.choice(np.flatnonzero(fit&(u==us)),4,replace=False) for us in baseusers]) for _ in range(400)])
        assert not np.isin(u[batches],[held,dev]).any() and np.isin(w[batches],trainwords).all() and np.isin(w[wrong[batches]],trainwords).all()
        motion=m<=np.quantile(m[fit],.95);usable=motion&valid;masks={'ACC_direct':selflabels&usable};probs={}
        np.savez_compressed(out/f'fold_seed{seed}_held{held}_private.npz',mean=mean,std=std,batches=batches,training_indices=ti,development_indices=di)
        audit.append(dict(seed=seed,user=held,development_user=dev,training_users=baseusers,training_windows=len(ti),development_windows=len(di),batch_sha256=hashlib.sha256(batches.tobytes()).hexdigest()))
        for name in ['ACC_self_trained','Audio_correct','Audio_wrongwindow']:
          torch.manual_seed(seed*100+held);net=ActivityNet();target=selflabels.astype('float64') if name=='ACC_self_trained' else y if name=='Audio_correct' else y[wrong]
          labelvalid=np.isfinite(target)&valid;filled=np.nan_to_num(target,nan=0.).astype('float32');pos=float(filled[fit][labelvalid[fit]].mean());weight=torch.tensor((1-pos)/max(pos,1e-6),dtype=torch.float32)
          path=out/f'{name}_seed{seed}_held{held}_private.pt';trace=[]
          if path.exists():
            ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['protocol_sha256']==protocol_hash;net.load_state_dict(ck['model'])
          else:
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001);targettensor=torch.tensor(filled);v=torch.tensor(labelvalid,dtype=torch.float32)
            for step,batch in enumerate(batches):
              logit=net(xx[batch]);loss=(F.binary_cross_entropy_with_logits(logit,targettensor[batch],pos_weight=weight,reduction='none')*v[batch]).sum()/v[batch].sum().clamp_min(1)
              optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
              if (step+1)%100==0:trace.append(dict(step=step+1,loss=float(loss.detach())))
            torch.save(dict(model=net.state_dict(),protocol_sha256=protocol_hash,steps=400),path)
          prob=infer(net,inputs,mean,std);devtruth=selflabels[di] if name=='ACC_self_trained' else y[di];devvalid=np.isfinite(devtruth)&valid[di]
          best,candidates=choose_probability(prob[di],devtruth,devvalid);masks[name]=(prob>=best['threshold'])&usable;probs[name]=prob
          selection.append(dict(seed=seed,user=held,method=name,threshold=best['threshold'],candidates=candidates))
          trainlog.append(dict(seed=seed,user=held,method=name,steps=400,trace=trace,valid_training_label_fraction=float(labelvalid[fit].mean())))
        prior=np.broadcast_to(np.nanmean(y[fit],0),y.shape);best,candidates=choose_probability(prior[di],y[di],np.isfinite(y[di])&valid[di]);masks['Fixed_time_prior']=(prior>=best['threshold'])&usable;probs['Fixed_time_prior']=prior
        selection.append(dict(seed=seed,user=held,method='Fixed_time_prior',threshold=best['threshold'],candidates=candidates))
        # Proxy quality is evaluated only on target's development words, never test Audio.
        qi=np.flatnonzero((u==held)&np.isin(w,devwords));qvalid=np.isfinite(y[qi])&valid[qi]
        for name,prob in probs.items():
          tau=next(v['threshold'] for v in selection if v['seed']==seed and v['user']==held and v['method']==name);truth=y[qi][qvalid].astype(bool);p=prob[qi][qvalid]
          quality.append(dict(seed=seed,user=held,method=name,scope='held_identity_development_words_only',f1=float(f1_score(truth,p>=tau,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(truth,p>=tau)),proxy_auc=float(roc_auc_score(truth,p))))
        for name,mask in masks.items():
          savedmasks[f'{seed}_{held}_{name}']=mask;z0,missing=stats(x,mask);mf=np.nanmean(z0[fit],0);mf=np.where(np.isfinite(mf),mf,0);z=np.where(np.isfinite(z0),z0,mf);z=(z-mf)/np.maximum(z[fit].std(0),1e-5)
          enrollment=[np.flatnonzero((u==us)&(w==word)&(ordinal==0))[0] for us in range(1,11) for word in regwords];enrollment=np.array(enrollment).reshape(10,12)
          proto=unit(np.stack([z[ix].mean(0) for ix in enrollment]));ci=np.flatnonzero(np.isin(w,devwords));score=unit(z[ci])@proto[held-1];score[missing[ci]]=-1.1
          tau=choose_auth(score,u[ci]==held);locks.append(dict(seed=seed,user=held,method=name,threshold=tau));context[seed,held,name]=dict(z=z,proto=proto,missing=missing)
        (out/'training_public.json').write_text(json.dumps(trainlog));print('CONTINUOUS_FOLD_DONE',seed,held,flush=True)
    np.savez_compressed(out/'masks_private.npz',**savedmasks);(out/'identity_thresholds_private.json').write_text(json.dumps(locks));(out/'mask_thresholds_private.json').write_text(json.dumps(selection));(out/'exclusion_audit_public.json').write_text(json.dumps(audit));(out/'mask_quality_public.json').write_text(json.dumps(quality))
    print('CONTINUOUS_ALL_THRESHOLDS_LOCKED',flush=True)
    rows=[];qi=np.flatnonzero(test)
    for seed in [42,43,44]:
     for held in range(1,11):
      for name in names:
        ctx=context[seed,held,name];score=unit(ctx['z'][qi])@ctx['proto'].T;score[ctx['missing'][qi]]=-1.1;genuine=u[qi]==held;own=score[genuine];top=np.isclose(own,own.max(1,keepdims=True),atol=1e-10,rtol=0);covered=~ctx['missing'][qi][genuine]
        tau=next(v['threshold'] for v in locks if v['seed']==seed and v['user']==held and v['method']==name)
        rows.append(dict(seed=seed,user=held,method=name,query_coverage=float(covered.mean()),registration_valid_windows=int((~ctx['missing'][np.flatnonzero((u==held)&np.isin(w,regwords)&(ordinal==0))]).sum()),top1_10way=float(np.mean((top[:,held-1]/top.sum(1))*covered)),**measure(score[:,held-1],genuine,tau)))
    macro=[];peruser=[]
    for name in names:
      rr=[v for v in rows if v['method']==name];keys=['eer','hter','far','frr','auc','query_coverage','top1_10way']
      macro.append(dict(method=name,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
      for us in range(1,11):
        pp=[v for v in rr if v['user']==us];peruser.append(dict(method=name,user=us,**{k:float(np.mean([v[k] for v in pp])) for k in keys}))
    (out/'per_seed_user_public.json').write_text(json.dumps(rows));(out/'summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=peruser),indent=2))
    (out/'verification_public.json').write_text(json.dumps(dict(status='PASS',models=90,target_folds=30,mask_threshold_selections=120,identity_thresholds=150,workpoints=150,test_Audio_labels_absent=bool(np.isnan(y[test]).all()),Audio_VAD_centering=False,all_thresholds_before_test_scores=True,no_test_window_dropped=True,protocol_sha256=protocol_hash),indent=2))
    print('CONTINUOUS_ACTIVITY_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    with threadpool_limits(limits=2):main(p.parse_args().root)
