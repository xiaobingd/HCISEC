"""Joint cross-word identity and Audio activity learning, no hard activity masks."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from run_continuous import acc_input,stats,unit,choose_auth,measure

class JointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(15,16,5,padding=2),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU())
        self.time=nn.Conv1d(16,1,1);nn.init.zeros_(self.time.weight);nn.init.zeros_(self.time.bias)
        self.project=nn.Linear(48,32);self.classifier=nn.Linear(32,8,bias=False)
    def forward(self,x,raw,valid):
        h=self.body(x);logit=self.time(h).squeeze(1)
        weight=F.softmax(logit.masked_fill(~valid,-1e4),dim=1)*valid.float();weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-8)
        mean=(h*weight[:,None,:]).sum(2);variance=((h-mean[:,:,None]).square()*weight[:,None,:]).sum(2)
        z=F.normalize(self.project(torch.cat([mean,(variance+1e-6).sqrt(),raw],1)),dim=1)
        return z,logit,weight
def crossword_supcon(z,y,word):
    logits=z@z.T/.1;pos=(y[:,None]==y[None,:])&(word[:,None]!=word[None,:]);valid=(y[:,None]!=y[None,:])|pos
    assert pos.any(1).all()
    logp=logits-torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1,keepdim=True)
    return -((logp*pos).sum(1)/pos.sum(1)).mean()
def embed(net,inputs,raw,valid,indices,normalizers,with_logits=False):
    mi,si,mr,sr=normalizers;result=[];activity=[];net.eval()
    with torch.inference_mode():
      for start in range(0,len(indices),64):
        ii=indices[start:start+64];x=torch.tensor(((inputs[ii]-mi)/si).astype('float32').transpose(0,2,1));r=torch.tensor(((raw[ii]-mr)/sr).astype('float32'));v=torch.tensor(valid[ii])
        z,logit,_=net(x,r,v);result.append(z.numpy());activity.append(logit.sigmoid().numpy())
    return (np.concatenate(result),np.concatenate(activity)) if with_logits else np.concatenate(result)
def gcos(a,b):
    a=torch.cat([v.reshape(-1) for v in a]);b=torch.cat([v.reshape(-1) for v in b]);return float(F.cosine_similarity(a[None],b[None])),float(b.norm()/a.norm().clamp_min(1e-10))
def save_checkpoint(payload,path):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(payload,temporary);os.replace(temporary,path)
def main(root,train_only=False,folds=None,worker_id='worker'):
    torch.set_num_threads(2);base=root/'continuous_acc_activity';out=base/'joint_identity';out.mkdir(exist_ok=True)
    split=json.loads((base/'word_split_public.json').read_text());old=json.loads((base/'results/protocol_public.json').read_text());c=np.load(base/'continuous_private.npz')
    x,m,y=c['acc'].astype('float64'),c['motion'].astype('float64'),c['audio_proxy'].astype('float64');u,w,ordinal=c['user'],c['word'],c['ordinal'];valid=c['imu_valid'];test=np.isin(w,split['test']);visible=~test
    assert np.isnan(y[test]).all();inputs=acc_input(x,m);raw,missing=stats(x,valid);assert not missing.any()
    regwords=old['registration_words'];enroll=np.array([[np.flatnonzero((u==us)&(w==word)&(ordinal==0))[0] for word in regwords] for us in range(1,11)])
    wrong=np.arange(len(u))
    for us in range(1,11):
      for word in range(40):
        ix=np.flatnonzero((u==us)&(w==word));ix=ix[np.argsort(ordinal[ix])];wrong[ix]=np.roll(ix,-(len(ix)//2))
    grid=[.03,.1];arms=['Identity_only','Joint_correct','Joint_wrongwindow']
    protocol=dict(word_split=split,data_sha256=hashlib.sha256((base/'continuous_private.npz').read_bytes()).hexdigest(),seeds=[42,43,44],
      encoder='same15->16k5 GN SiLU->16k3 GN SiLU;learnedtime1x1softmax;weightedmean/std32 plusfullwindowraw16skip->linear32L2;temporary8classhead',
      identity='CE(scale16,normalized8classweights)+.1crosswordSupCon(temperature.1);sameuserdifferentwordpositives,sameuser_samewordignored',
      joint='Lidentity+lambda*balancedAudio_activityBCE;sharedtimehead usedforsoftpoolandBCE,bothlossesbackpropbodyandtime;NO hardactivitythreshold',
      raw_skip='16fullwindowstatistics,standardizedonlybase/train;notsoleidentityrepresentation;trainableprojectionmixesrawskipandpooledCNNfeatures',
      auxiliary_grid=grid,selection='train bothJoint arms400steps perlambda;sharedlambda maximizesmean correct+wrong innerdevidentity crosswordAUC,thenmeanEER,thensmallerlambda;nooutertarget/test selection',
      budget='400steps everycandidate,AdamW.001 wd.01 clip1;8usersx2sharedwordsx2windows=32,allarms/candidatesidenticalbatchindices/init',
      computation='150candidatefits(30baseline+60correct+60wrong);90selectedmodels;correct/wrongequal800-stepsearchbudget;baselinehasnolambdatosearch;eachfinalmodel400steps',
      identity_exclusion='heldtarget andnextcyclicdevidentity excludedfromallneuraloptimization;8baseusers/train24words;lambda selector9usercohort excludesoutertarget',
      auxiliary_wrong='sameuser/sameword othernonoverlap3swindow,half-recordingrotation;notverifiedutterancealignment',
      hyperparameter_development='innerdevidentity enrolledon12trainwords,queriedondev8words,targetvs8baseuserdevqueries;noidentitytrainingondevidentity',
      registration_words=regwords,registration='same12IMUwindows aspreviouscontinuousphase;noAudioqualityselection',
      identity_calibration='same dev8words,personal minHTER/FARFRR balance/higherthreshold;all90thresholdssavedbeforetestembedding/scoring',
      query='all433independenttestwordIMUwindows,noAudio,testlabelsallmissing;softattentionusesallvalidframes;nofavorablewindow/userfilter',
      historical_test_exposure='same8testwordsalready evaluatedpreviousactivity-onlyexperiment;thisisexploratoryreplication,notnewblindconfirmation',
      no_embedding_alignment=True,no_existing_model_changes=True,parameters=sum(p.numel() for p in JointNet().parameters()))
    text=json.dumps(protocol,sort_keys=True);ph=hashlib.sha256(text.encode()).hexdigest();pp=out/'protocol_public.json'
    if pp.exists():assert pp.read_text()==text,'Use fresh output directory if protocol changed'
    else:pp.write_text(text)
    if not train_only:np.savez_compressed(out/'enrollment_indices_private.npz',indices=enroll)
    selected=[];logs=[];audits=[];locks=[];quality=[];context={}
    for seed in [42,43,44]:
      for held in range(1,11):
        if folds is not None and (seed,held) not in folds:continue
        dev=held%10+1;baseusers=[us for us in range(1,11) if us not in (held,dev)];fit=np.isin(u,baseusers)&np.isin(w,split['train']);ti=np.flatnonzero(fit)
        norm=(inputs[fit].mean((0,1)),np.maximum(inputs[fit].std((0,1)),1e-5),raw[fit].mean(0),np.maximum(raw[fit].std(0),1e-5));mi,si,mr,sr=norm
        xx=torch.tensor(((inputs-mi)/si).astype('float32').transpose(0,2,1));rr=torch.tensor(((raw-mr)/sr).astype('float32'));vv=torch.tensor(valid)
        cls=np.array([baseusers.index(int(us)) if us in baseusers else -1 for us in u]);yy=torch.tensor(cls,dtype=torch.long);ww=torch.tensor(w,dtype=torch.long)
        rng=np.random.default_rng(seed*100+held);batches=[]
        for step in range(400):
          words=rng.choice(split['train'],2,replace=False);batches.append(np.concatenate([np.concatenate([rng.choice(np.flatnonzero((u==us)&(w==word)),2,replace=False) for word in words]) for us in baseusers]))
        batches=np.array(batches);assert np.isin(w[batches],split['train']).all() and not np.isin(u[batches],[held,dev]).any()
        assert np.array_equal(u[wrong[batches]],u[batches]) and np.array_equal(w[wrong[batches]],w[batches])
        np.savez_compressed(out/f'fold_seed{seed}_held{held}_private.npz',batches=batches,training_indices=ti,mi=mi,si=si,mr=mr,sr=sr)
        audits.append(dict(seed=seed,user=held,development_user=dev,training_users=baseusers,batch_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),training_windows=len(ti)))
        cache={};candidates=[]
        for arm in arms:
          for lam in ([0.] if arm=='Identity_only' else grid):
            torch.manual_seed(seed*100+held);net=JointNet();path=out/f'{arm}_lambda{lam}_seed{seed}_held{held}_private.pt';trace=[]
            if path.exists():
              ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['protocol_sha256']==ph;net.load_state_dict(ck['model']);trace=ck['trace']
            else:
              optimizer=torch.optim.AdamW(net.parameters(),lr=.001);target=y[wrong] if arm=='Joint_wrongwindow' else y
              filled=torch.tensor(np.nan_to_num(target,nan=0.).astype('float32'));labelvalid=torch.tensor(np.isfinite(target)&valid,dtype=torch.float32)
              pos=float(np.nanmean(target[fit]));posweight=torch.tensor((1-pos)/max(pos,1e-6),dtype=torch.float32)
              for step,batch in enumerate(batches):
                z,logit,weight=net(xx[batch],rr[batch],vv[batch]);ce=F.cross_entropy(16*F.linear(z,F.normalize(net.classifier.weight,dim=1)),yy[batch]);con=crossword_supcon(z,yy[batch],ww[batch]);identity=ce+.1*con
                aux=identity.new_zeros(())
                if lam:aux=(F.binary_cross_entropy_with_logits(logit,filled[batch],pos_weight=posweight,reduction='none')*labelvalid[batch]).sum()/labelvalid[batch].sum().clamp_min(1)
                diagnostic={}
                if (step+1)%100==0 and lam:
                  gi=torch.autograd.grad(identity,net.body.parameters(),retain_graph=True);ga=torch.autograd.grad(lam*aux,net.body.parameters(),retain_graph=True);cos,ratio=gcos(gi,ga);diagnostic=dict(body_gradient_cosine=cos,aux_to_identity_gradient_ratio=ratio)
                loss=identity+lam*aux;optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
                if (step+1)%100==0:trace.append(dict(step=step+1,identity_ce=float(ce.detach()),crossword_supcon=float(con.detach()),activity_bce=float(aux.detach()),**diagnostic))
              save_checkpoint(dict(model=net.state_dict(),protocol_sha256=ph,lambda_value=lam,steps=400,trace=trace),path)
            net.eval();cache[arm,lam]=net;logs.append(dict(seed=seed,user=held,arm=arm,lambda_value=lam,steps=400,trace=trace))
            if arm!='Identity_only' and not train_only:
              ix=np.flatnonzero(visible&(u!=held));zv=embed(net,inputs,raw,valid,ix,norm);mapping=np.full(len(u),-1,int);mapping[ix]=np.arange(len(ix))
              devreg=enroll[dev-1];proto=unit(zv[mapping[devreg]].mean(0));qi=np.flatnonzero((u!=held)&np.isin(w,split['development']));score=zv[mapping[qi]]@proto
              met=measure(score,u[qi]==dev,0.)
              candidates.append(dict(arm=arm,lambda_value=lam,development_auc=met['auc'],development_eer=met['eer']))
        if train_only:
          (out/f'worker_{worker_id}_training_private.json').write_text(json.dumps(logs))
          print('JOINT_WORKER_FOLD_TRAINED',worker_id,seed,held,flush=True)
          continue
        shared=[]
        for lam in grid:
          rows=[v for v in candidates if v['lambda_value']==lam];assert len(rows)==2
          shared.append(dict(lambda_value=lam,mean_development_auc=float(np.mean([v['development_auc'] for v in rows])),mean_development_eer=float(np.mean([v['development_eer'] for v in rows]))))
        best=min(shared,key=lambda v:(-round(v['mean_development_auc'],12),round(v['mean_development_eer'],12),v['lambda_value']));chosen=best['lambda_value']
        selected.append(dict(seed=seed,user=held,development_user=dev,shared_lambda=chosen,candidates=candidates,shared_scores=shared))
        for arm in arms:
          lam=0. if arm=='Identity_only' else chosen;net=cache[arm,lam];ix=np.flatnonzero(visible);zv,prob=embed(net,inputs,raw,valid,ix,norm,with_logits=True);mapping=np.full(len(u),-1,int);mapping[ix]=np.arange(len(ix))
          proto=unit(np.stack([zv[mapping[reg]].mean(0) for reg in enroll]));ci=np.flatnonzero(np.isin(w,split['development']));score=zv[mapping[ci]]@proto[held-1];tau=choose_auth(score,u[ci]==held)
          locks.append(dict(seed=seed,user=held,arm=arm,threshold=tau));context[seed,held,arm]=dict(net=net,norm=norm,proto=proto)
          qi=np.flatnonzero((u==held)&np.isin(w,split['development']));available=np.isfinite(y[qi])&valid[qi]
          quality.append(dict(seed=seed,user=held,arm=arm,activity_proxy_auc=float(roc_auc_score(y[qi][available],prob[mapping[qi]][available]))))
        (out/'training_public.json').write_text(json.dumps(logs));(out/'selection_public.json').write_text(json.dumps(selected));print('JOINT_IDENTITY_FOLD_DONE',seed,held,chosen,flush=True)
    if train_only:
      print('JOINT_WORKER_COMPLETE',worker_id,flush=True);return
    (out/'identity_thresholds_private.json').write_text(json.dumps(locks));(out/'exclusion_audit_public.json').write_text(json.dumps(audits));(out/'activity_quality_public.json').write_text(json.dumps(quality));print('JOINT_IDENTITY_ALL_THRESHOLDS_LOCKED',flush=True)
    qi=np.flatnonzero(test);rows=[]
    for seed in [42,43,44]:
     for held in range(1,11):
      for arm in arms:
        ctx=context[seed,held,arm];zq=embed(ctx['net'],inputs,raw,valid,qi,ctx['norm']);scores=zq@ctx['proto'].T;genuine=u[qi]==held;own=scores[genuine];top=np.isclose(own,own.max(1,keepdims=True),atol=1e-7,rtol=0)
        tau=next(v['threshold'] for v in locks if (v['seed'],v['user'],v['arm'])==(seed,held,arm))
        rows.append(dict(seed=seed,user=held,arm=arm,top1_10way=float(np.mean(top[:,held-1]/top.sum(1))),**measure(scores[:,held-1],genuine,tau)))
    macro=[];peruser=[]
    for arm in arms:
      rr=[v for v in rows if v['arm']==arm];keys=['eer','hter','far','frr','auc','top1_10way'];macro.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
      for us in range(1,11):
        pp=[v for v in rr if v['user']==us];peruser.append(dict(arm=arm,user=us,**{k:float(np.mean([v[k] for v in pp])) for k in keys}))
    (out/'per_seed_user_public.json').write_text(json.dumps(rows));(out/'summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=peruser),indent=2))
    (out/'verification_public.json').write_text(json.dumps(dict(status='PASS',candidate_models=150,selected_models=90,folds=30,thresholds=90,workpoints=len(rows),test_Audio_labels_missing=True,hard_activity_mask=False,identity_gradient_reaches_shared_encoder=True,all_thresholds_before_test_inference=True,protocol_sha256=ph),indent=2));print('JOINT_IDENTITY_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--train-only',action='store_true');p.add_argument('--folds');p.add_argument('--worker-id',default='worker');args=p.parse_args()
    folds={tuple(map(int,v.split(':'))) for v in args.folds.split(',')} if args.folds else None
    assert folds is None or args.train_only,'Partial folds are only supported for checkpoint pretraining'
    with threadpool_limits(limits=2):main(args.root,args.train_only,folds,args.worker_id)
