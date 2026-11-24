"""Cross-word cross-modal identity learning; frozen encoders, registration Audio only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
class Projection(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,32))
    def forward(self,x):return F.normalize(self.layers(x),dim=-1)
class Pair(nn.Module):
    def __init__(self):
        super().__init__();self.audio=Projection();self.acc=Projection()
def bidirectional(a,x,y):
    logits=a@x.T/.1;positive=(y[:,None]==y[None,:]).float()
    return -.5*((F.log_softmax(logits,1)*positive).sum(1)/positive.sum(1)+(F.log_softmax(logits.T,1)*positive.T).sum(1)/positive.T.sum(1)).mean()
def main(root):
    sys.path.insert(0,str(root));from prepare_phrase_auth import FIT,REG,CAL,SHORT,ENGLISH,CHINESE
    from calibrate_threeword import cutoff,measure
    torch.set_num_threads(2)
    out=root/'conditional_correspondence/crossmodal_identity_projection';out.mkdir(exist_ok=True)
    d=np.load(root/'phrase_auth/frontend_private.npz');u,w,r,ids=[d[k] for k in ('user','word','repeat','sample_id')]
    fit=np.isin(w,FIT);reg=np.isin(w,REG);visible=np.isin(w,FIT+REG+CAL);vi=np.flatnonzero(visible)
    index=np.full(3200,-1,dtype=int);index[vi]=np.arange(len(vi));assert fit.sum()==1280 and reg.sum()==320
    af=np.load(root/'conditional_correspondence/audio_conditioned_verifier/fit_ecapa_private.npz')
    ar=np.load(root/'conditional_correspondence/enrollment_identity_quality/registered_ecapa_private.npz')
    for cache,mask in [(af,fit),(ar,reg)]:
        assert np.array_equal(cache['indices'],np.flatnonzero(mask)) and np.array_equal(cache['sample_id'],ids[mask]) and bool(cache['length_aware'])
    audio=np.zeros((3200,192),dtype='float32');audio[fit]=af['embedding'];audio[reg]=ar['embedding']
    trainwords=sorted(FIT)[:12];inner=sorted(FIT)[12:];grid=[0.,.1,.25,.5,1.]
    protocol=dict(users=10,seeds=[42,43,44],projector='two independent192->64GELU->32L2 heads;frozen ACC192 and official ECAPA192',
      training='bidirectional supervised contrastive,temperature.1;8users x2reps;all Audio one word,all ACC different word;all same-identity pairs positive',
      steps=400,optimizer='AdamW.001 wd.01,clip1',train_words=trainwords,inner_words=inner,
      exclusion='per fold held target and next cyclic inner-development identity excluded from projection optimization;8 projection-training users;held target also excluded from alpha selection',
      capacity_control='same two heads and episode indices,replace Audio inputs with same-word ACC;ACC-ACC cross-word matching',
      alpha_grid=grid,alpha_selection='inner first2words 12support records,thirdword calibration,fourthword development;only innerdev identity genuine and8trainingidentity attacks;minimum HTER then EER then smaller alpha',
      registration='exact legacyACCcrossword12,3per4REGwords;all controls same12ACC;Audio same12;no new enrollment ranking',
      score='originalACCcosine+alpha*cos(P_ACC(queryACC),P_Audio(enrollmentAudio));all projected prototypes normalized after sample averaging',
      wrong_controls='C nextotherregistereduser Audio;D fixed192feature-coordinate permutation before Audio projection;same trained model and alpha;both separately CAL calibrated and with correctAudio threshold diagnostic',
      projection_control='AudioModel_ACCprototype uses same jointly trained ACC head and same B alpha,changes prototype to projected enrollmentACC,isolates Audio reference contribution',
      calibration=CAL,queries=dict(english_word=SHORT,english_long=ENGLISH,chinese_long=CHINESE),query='singleACCrecord,noqueryAudio,no3wordfusion',
      threshold='(CALgenuineQ10+CALimpostorQ95)/2;all locks andalphas saved before finalquery scoring',
      retrieval='10-way enrollmentAudio prototypes vs heldtarget CAL ACC queries;target not used to trainprojection;CAL allidentities allows target verificationAUC;no model selection onCAL',
      scope='global ACC encoder already trainedall10fit identities includinginnerwords;not fully unseen-identity or genuinely unseen-innerword encoder;historically inspected finalcohort',
      limitations='44100candidate Audio interpretation,physicalsync not independently verified;3secondACCwindows maycontain adjacentrepeats')
    (out/'protocol_public.json').write_text(json.dumps(protocol,indent=2))
    names=['raw_ACC','ACC_ACC_fusion','Audio_correct','Audio_other','Audio_shuffled','AudioModel_ACCprototype','Audio_other_Bthreshold','Audio_shuffled_Bthreshold']
    legacy=json.loads((root/'conditional_correspondence/enrollment_identity_quality_fixed_background/thresholds_private.json').read_text())
    context={};locks=[];training=[];selection=[];retrieval=[];episode_audit=[]
    with np.load(root/'conditional_correspondence/stage3/views_private.npz') as vf:known={s:vf[f'{s}_A_ACC'][vi].copy() for s in (42,43,44)}
    by={(int(us),int(wo)):np.flatnonzero((u==us)&(w==wo)) for us in range(1,11) for wo in FIT}
    for seed in (42,43,44):
      z=known[seed]
      def zz(ix):
        assert np.all(index[ix]>=0);return z[index[ix]]
      chosen={}
      for us in range(1,11):
        parts=[]
        for wo in REG:
          ix=np.flatnonzero((u==us)&(w==wo));ix=ix[np.argsort(r[ix])]
          other=np.flatnonzero((u==us)&np.isin(w,REG)&(w!=wo));score=rankdata(zz(ix)@unit(zz(other).mean(0)))/8
          parts.extend(ix[np.lexsort((np.arange(8),-score))[:3]])
        chosen[us]=np.array(parts)
      rawp=unit(np.stack([zz(chosen[us]).mean(0) for us in range(1,11)]))
      context[seed]=dict(rawp=rawp)
      for held in range(1,11):
        dev=held%10+1;trainusers=[us for us in range(1,11) if us not in (held,dev)];rng=np.random.default_rng(seed*100+held)
        episodes=[]
        for step in range(400):
          wa,wx=rng.choice(trainwords,2,replace=False)
          ia=np.concatenate([rng.choice(by[us,int(wa)],2,replace=False) for us in trainusers])
          ix=np.concatenate([rng.choice(by[us,int(wx)],2,replace=False) for us in trainusers])
          assert not np.isin(u[ia],[held,dev]).any() and not np.isin(u[ix],[held,dev]).any()
          assert np.array_equal(u[ia],u[ix]) and not np.intersect1d(w[ia],w[ix]).size
          episodes.append((ia,ix))
        eis=np.array(episodes);np.savez_compressed(out/f'episodes_seed{seed}_held{held}_private.npz',indices=eis)
        episode_audit.append(dict(seed=seed,held_target=held,inner_development_user=dev,train_users=8,steps=400,
          episode_sha256=hashlib.sha256(eis.tobytes()).hexdigest(),target_records=0,development_identity_optimization_records=0))
        ys=torch.tensor(np.repeat(np.arange(8),2));models={};foldalpha={};foldproto={}
        for variant in ['crossmodal','acc_only']:
          torch.manual_seed(seed*100+held);net=Pair();path=out/f'{variant}_seed{seed}_held{held}_private.pt';trace=[]
          if path.exists():net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
          else:
            opt=torch.optim.AdamW(net.parameters(),lr=.001)
            for step,(ia,ix) in enumerate(episodes):
              aa=torch.tensor(audio[ia] if variant=='crossmodal' else zz(ia));xx=torch.tensor(zz(ix))
              loss=bidirectional(net.audio(aa),net.acc(xx),ys);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
              if (step+1)%100==0:trace.append(dict(step=step+1,loss=float(loss.detach())))
            torch.save(net.state_dict(),path)
          net.eval();models[variant]=net
          # Inner role separation: alpha sees no target and no final REG/CAL/query.
          en=np.concatenate([np.flatnonzero((u==dev)&(w==wo))[np.argsort(r[(u==dev)&(w==wo)])[:6]] for wo in inner[:2]])
          assert len(en)==12 and not np.isin(w[en],trainwords).any()
          ci=np.flatnonzero((u!=held)&(w==inner[2]));qi=np.flatnonzero((u!=held)&(w==inner[3]))
          rawproto=unit(zz(en).mean(0));cb=zz(ci)@rawproto;qb=zz(qi)@rawproto
          with torch.inference_mode():
            enroll=torch.tensor(audio[en] if variant=='crossmodal' else zz(en))
            proto=F.normalize(net.audio(enroll).mean(0),dim=0)
            cs=(net.acc(torch.tensor(zz(ci)))@proto).numpy();qs=(net.acc(torch.tensor(zz(qi)))@proto).numpy()
          candidates=[]
          for alpha in grid:
            tau=cutoff(cb+alpha*cs,u[ci]==dev,'quantile_gap');m=measure(qb+alpha*qs,u[qi]==dev,tau)
            candidates.append(dict(alpha=alpha,**m))
          best=min(candidates,key=lambda a:(a['hter'],a['eer'],a['alpha']));foldalpha[variant]=best['alpha']
          selection.append(dict(seed=seed,held_target=held,inner_development_user=dev,variant=variant,alpha=best['alpha'],candidates=candidates))
          training.append(dict(seed=seed,held_target=held,variant=variant,steps=400,trace=trace))
        # Build common registration prototypes; no final-query features scored.
        cm,ac=models['crossmodal'],models['acc_only'];perm=np.random.default_rng(70000+seed*100+held).permutation(192)
        with torch.inference_mode():
          ap=unit(np.stack([cm.audio(torch.tensor(audio[chosen[us]])).mean(0).numpy() for us in range(1,11)]))
          xp=unit(np.stack([cm.acc(torch.tensor(zz(chosen[us]))).mean(0).numpy() for us in range(1,11)]))
          wrong=ap[held%10];correct=ap[held-1]
          destroyed=unit(cm.audio(torch.tensor(audio[chosen[held]][:,perm])).mean(0).numpy())
          accproto=unit(ac.audio(torch.tensor(zz(chosen[held]))).mean(0).numpy())
        c=dict(models=models,alpha=foldalpha,rawp=rawp[held-1],ap=correct,wrong=wrong,shuffled=destroyed,xp=xp[held-1],accproto=accproto)
        context[seed,held]=c
        def score_methods(x,b):
          with torch.inference_mode():
            xc=cm.acc(torch.tensor(x)).numpy();xa=ac.acc(torch.tensor(x)).numpy()
          a=foldalpha['crossmodal'];res={'raw_ACC':b,'ACC_ACC_fusion':b+foldalpha['acc_only']*(xa@accproto),
            'Audio_correct':b+a*(xc@correct),'Audio_other':b+a*(xc@wrong),'Audio_shuffled':b+a*(xc@destroyed),
            'AudioModel_ACCprototype':b+a*(xc@xp[held-1])}
          res['Audio_other_Bthreshold']=res['Audio_other'];res['Audio_shuffled_Bthreshold']=res['Audio_shuffled'];return res
        ci=np.flatnonzero(np.isin(w,CAL));cal=score_methods(zz(ci),(z@rawp.T)[index[ci],held-1]);foldlocks={}
        for name in names:
          tau=cutoff(cal[name],u[ci]==held,'quantile_gap')
          if name.endswith('_Bthreshold'):tau=foldlocks['Audio_correct']
          if name=='raw_ACC':
            ref=next(a['threshold'] for a in legacy if a['seed']==seed and a['user']==held and a['method']=='acc_crossword12' and a['backend']=='cosine' and a['words']==1)
            assert abs(tau-ref)<1e-6;tau=ref
          foldlocks[name]=tau;locks.append(dict(seed=seed,user=held,method=name,threshold=tau))
        # Cross-modal retrieval on held target, CAL only, no selection.
        with torch.inference_mode():scores=cm.acc(torch.tensor(zz(ci))).numpy()@ap.T
        own=u[ci]==held;g=scores[own];positive=g[:,held-1];negative=np.delete(g,held-1,axis=1).ravel()
        retrieval.append(dict(seed=seed,user=held,top1=float(np.mean(g.argmax(1)==held-1)),genuine_queries=int(own.sum()),
          crossmodal_target_auc=float(roc_auc_score(u[ci]==held,scores[:,held-1])),
          same_query_retrieval_auc=float(roc_auc_score(np.r_[np.ones(len(positive)),np.zeros(len(negative))],np.r_[positive,negative])),
          genuine_minus_hard_margin=float(np.mean(positive-np.delete(g,held-1,axis=1).max(1)))))
        (out/'training_public.json').write_text(json.dumps(training));(out/'episode_audit_public.json').write_text(json.dumps(episode_audit))
        print('CROSSMODAL_FOLD_DONE',seed,held,foldalpha,flush=True)
    (out/'thresholds_private.json').write_text(json.dumps(locks));(out/'alpha_selection_public.json').write_text(json.dumps(selection))
    (out/'retrieval_public.json').write_text(json.dumps(retrieval));print('CROSSMODAL_ALL_LOCKED',flush=True)
    rows=[]
    for seed in (42,43,44):
      with np.load(root/'conditional_correspondence/stage3/views_private.npz') as vf:z=vf[f'{seed}_A_ACC']
      baseline=z@context[seed]['rawp'].T
      for held in range(1,11):
        c=context[seed,held];cm,ac=c['models']['crossmodal'],c['models']['acc_only'];alpha=c['alpha']
        for family,words in dict(english_word=SHORT,english_long=ENGLISH,chinese_long=CHINESE).items():
          qi=np.flatnonzero(np.isin(w,words));b=baseline[qi,held-1]
          with torch.inference_mode():xc=cm.acc(torch.tensor(z[qi])).numpy();xa=ac.acc(torch.tensor(z[qi])).numpy()
          scores={'raw_ACC':b,'ACC_ACC_fusion':b+alpha['acc_only']*(xa@c['accproto']),
            'Audio_correct':b+alpha['crossmodal']*(xc@c['ap']),'Audio_other':b+alpha['crossmodal']*(xc@c['wrong']),
            'Audio_shuffled':b+alpha['crossmodal']*(xc@c['shuffled']),'AudioModel_ACCprototype':b+alpha['crossmodal']*(xc@c['xp'])}
          scores['Audio_other_Bthreshold']=scores['Audio_other'];scores['Audio_shuffled_Bthreshold']=scores['Audio_shuffled']
          for name in names:
            tau=next(a['threshold'] for a in locks if a['seed']==seed and a['user']==held and a['method']==name)
            rows.append(dict(seed=seed,user=held,method=name,family=family,**measure(scores[name],u[qi]==held,tau)))
    macro=[];users=[]
    for name in names:
      for family in ('english_word','english_long','chinese_long'):
        part=[a for a in rows if a['method']==name and a['family']==family]
        macro.append(dict(method=name,family=family,**{k:float(np.mean([a[k] for a in part])) for k in ('eer','hter','far','frr','auc')}))
        for us in range(1,11):
          pp=[a for a in part if a['user']==us];users.append(dict(method=name,family=family,user=us,**{k:float(np.mean([a[k] for a in pp])) for k in ('eer','hter','far','frr','auc')}))
    (out/'per_seed_user_public.json').write_text(json.dumps(rows));(out/'summary_public.json').write_text(json.dumps(dict(macro=macro,per_user=users),indent=2))
    (out/'verification_public.json').write_text(json.dumps(dict(status='PASS',models=60,folds=30,episode_tasks=12000,thresholds=len(locks),workpoints=len(rows),target_in_training=False,innerdev_in_training=False,query_audio_used=False,all_locks_before_query_scoring=True,global_encoders_updated=False),indent=2))
    print('CROSSMODAL_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
