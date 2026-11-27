"""Five frozen ablations, development lock, full16word retraining; CPU execution."""
import argparse,json,hashlib,time,gc
import numpy as np
import torch
from scipy.special import xlogy
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from common import Student,crossword_supcon,phone_kl,shifted_posteriors,template
from prepare_acc import BASE,PRIVATE,OUT,OPT,DEV

ARMS=['T0','T1','T2','T3','T4']
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def protocol():
    data=json.loads((OUT/'data_audit_public.json').read_text(encoding='utf-8'))
    return dict(source_plan_sha='65f31238c0e3eeed7e19bac79fd6f1bbe3d58ce9',data_sha256=data['data_sha256'],
      model='147->64conv5 GN8 GELU->64conv3dilation2GELU->64conv3dilation3GELU;phoneLinear64x392;attentionstats128->192L2;normalizedidentity10scale16;GRL1wordMLP192->64->Nfit',
      vocabulary=392,optimizer='AdamW lr.0003 wd.01 gradclip1',lambda_phone=.1,lambda_content=.05,supcon_weight=.1,supcon_temperature=.1,
      development=dict(words=OPT,evaluation_words=DEV,seed=42,steps=200,selection='no authentication query/cal scores;finite/convergence/gradient and phonetic validation only;default losses retainedunless documented before queries'),
      final=dict(words=OPT+DEV,seeds=[42,43,44],steps=1000,checkpoints='fixedlaststep,noearlystop,noqueryhyperselection'),
      batch='80=10usersx4commontrainwordsx2distinctwindows;identicalindices/initperseed across5arms',
      arms=dict(T0='identityCE+.1crosswordSupCon',T1='T0+.05globalwordCEthroughGRL1',T2='T1+.1dictionaryuniformfixed.5..2.5secphoneKL',T3='T1+.1realAudioCTCsoftposteriorKL',T4='T1+.1temporalshuffleofsameAudioCTCposteriorKL'),
      teacher='frozenfacebook/wav2vec2-lv-60-espeak-cv-ft revisionae45363bf3413b374fecd9dc8bc1df0e24c3b7f4;fullvocabincludingblank,notVAD;Gaussian100ms fromCTCRFclocks',
      roles=data['roles'],registration_K=data['registration_K'],
      thresholds='(genuineCalQ10+impostorCalQ95)/2;all150tauslockedbeforeanyqueryscores',
      scope='closedset10knownidentities,historicalprompts exploratory;independentIMUorigin3sec noqueryAudio;actual2179windowcount not3200nominalutterances',
      no_query_Audio=True,no_user_word_filter=True,device='cpu',precision='float32')

def inputs(stage):
    data=np.load(PRIVATE/'acc_private.npz');acc=data['acc'];valid=data['valid'];u,w=data['user'],data['word'];fitwords=OPT if stage=='development' else OPT+DEV
    fit=np.flatnonzero(np.isin(w,fitwords));values=acc[fit].transpose(0,2,1)[valid[fit]]
    mean=values.mean(0);std=np.maximum(values.std(0),1e-6);x=torch.tensor((acc-mean[None,:,None])/std[None,:,None]);del acc,values
    np.savez_compressed(PRIVATE/f'normalization_{stage}_private.npz',mean=mean,std=std,fit_indices=fit)
    return data,x,torch.tensor(valid),u,w,fitwords,fit

def dictionary():
    lex=json.loads((OUT/'lexicon_public.json').read_text(encoding='utf-8'))['lexicon'];targets=[];valid=[]
    for word in OPT+DEV:
        q,m=template(lex[str(word)]['ids'],392);targets.append(q);valid.append(m)
    return np.array(targets),np.array(valid),{wo:i for i,wo in enumerate(OPT+DEV)}

def batch_indices(u,w,words,seed,steps):
    rng=np.random.default_rng(seed);lookup={(us,wo):np.flatnonzero((u==us)&(w==wo)) for us in range(1,11) for wo in words}
    assert min(map(len,lookup.values()))>=2
    batches=[]
    for _ in range(steps):
        chosen=rng.choice(words,4,replace=False);batches.append([int(i) for us in range(1,11) for wo in chosen for i in rng.choice(lookup[us,int(wo)],2,replace=False)])
    return np.array(batches)

def gradient_stats(identity,phone,content,params):
    def flatten(loss):
        gg=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gg,params)])
    a,b,c=flatten(identity),flatten(phone),flatten(content)
    def cosine(x,y):return float(F.cosine_similarity(x[None],y[None])) if x.norm()>0 and y.norm()>0 else 0.
    return dict(identity_grad=float(a.norm()),phone_grad=float(b.norm()),content_reverse_grad=float(c.norm()),
      phone_to_id_ratio=float(b.norm()/a.norm().clamp_min(1e-12)),content_to_id_ratio=float(c.norm()/a.norm().clamp_min(1e-12)),
      phone_id_cosine=cosine(a,b),content_id_cosine=cosine(a,c))

def train(stage,arms,threads):
    torch.set_num_threads(threads);OUT.mkdir(exist_ok=True);spec=protocol();specpath=OUT/'protocol.json'
    if specpath.exists():assert digest(json.loads(specpath.read_text(encoding='utf-8')))==digest(spec)
    else:specpath.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    if stage=='final':
        assert (OUT/'development_lock_public.json').exists(),'Complete development audit before finalretrain'
        lock=json.loads((OUT/'development_lock_public.json').read_text(encoding='utf-8'))
        assert lock['teacher_cache_sha256']==hashlib.sha256((PRIVATE/'teacher_private.npz').read_bytes()).hexdigest(),'Teacher changed after development lock'
    data,x,valid,u,w,words,fit=inputs(stage);qdict,mdict,dictindex=dictionary();dicindices=np.array([dictindex.get(int(wo),-1) for wo in w]);dict_logsum=xlogy(qdict,qdict).sum(-1)
    real=None;realvalid=None;inverse=np.full(len(u),-1,dtype=int)
    if any(arm in ['T3','T4'] for arm in arms):
        cache=np.load(PRIVATE/'teacher_private.npz');real=cache['posterior'];realvalid=cache['valid'];inverse[cache['indices']]=np.arange(len(real));assert (inverse[fit]>=0).all();real_logsum=xlogy(real,real).sum(-1)
    seeds=[42] if stage=='development' else [42,43,44];steps=200 if stage=='development' else 1000;ph=digest(spec)
    yc=torch.tensor(u-1,dtype=torch.long);wc=torch.tensor(w,dtype=torch.long);wordmap={wo:i for i,wo in enumerate(words)};wordclass=torch.tensor([wordmap.get(int(wo),-1) for wo in w],dtype=torch.long)
    for seed in seeds:
      batches=batch_indices(u,w,words,seed,steps);np.savez_compressed(PRIVATE/f'{stage}_seed{seed}_batches_private.npz',batches=batches)
      for arm in arms:
        path=PRIVATE/f'{stage}_{seed}_{arm}_private.pt'
        if path.exists():
          saved=torch.load(path,weights_only=True);assert saved['protocol_sha']==ph and saved['steps']==steps;print('CHECKPOINT_REUSED',stage,seed,arm,flush=True);continue
        torch.manual_seed(seed);model=Student(392,len(words));model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);traces=[];diagnostics=[]
        initial_sha=hashlib.sha256(b''.join(value.detach().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
        batch_sha=hashlib.sha256(batches.tobytes()).hexdigest()
        shifted=None
        if arm=='T4':
          shifted,perm,changed=shifted_posteriors(real,realvalid,seed)
          shifted_logsum=np.take_along_axis(real_logsum,perm,axis=1)
          assert np.allclose(shifted.sum(2),1,atol=1e-5)
          np.savez_compressed(PRIVATE/f'{stage}_seed{seed}_temporal_permutation_private.npz',permutation=perm)
          (OUT/f'{stage}_seed{seed}_shuffle_public.json').write_text(json.dumps(dict(mean_frame_L1_change=float(np.mean(changed)),nonconstant_window_fraction=float(np.mean(np.array(changed)>1e-3)),mask_preserved=True,per_window_distribution_preserved=True),indent=2))
        began=time.perf_counter()
        for step,ii in enumerate(batches):
          optimizer.zero_grad();z,phone_logits,word_logits=model(x[ii],valid[ii]);ce=F.cross_entropy(16*z@F.normalize(model.classifier.weight,dim=1).T,yc[ii]);sup=crossword_supcon(z,yc[ii],wc[ii]);identity=ce+.1*sup
          lp=z.sum()*0;lc=z.sum()*0;nf=0
          if arm!='T0':lc=F.cross_entropy(word_logits,wordclass[ii])
          if arm=='T2':
            qi=torch.tensor(qdict[dicindices[ii]]);mask=torch.tensor(mdict[dicindices[ii]])&valid[ii];lp=phone_kl(phone_logits,qi,mask,torch.tensor(dict_logsum[dicindices[ii]]));nf=int(mask.sum())
          elif arm in ['T3','T4']:
            jj=inverse[ii];q=torch.tensor((real if arm=='T3' else shifted)[jj]);mask=torch.tensor(realvalid[jj])&valid[ii];ls=torch.tensor((real_logsum if arm=='T3' else shifted_logsum)[jj]);lp=phone_kl(phone_logits,q,mask,ls);nf=int(mask.sum())
          loss=identity+.1*lp+.05*lc
          assert torch.isfinite(loss)
          if step in [0,99,steps-1]:diagnostics.append(dict(step=step+1,**gradient_stats(identity,.1*lp,.05*lc,list(model.backbone.parameters()))))
          loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
          if (step+1)%100==0:
            row=dict(step=step+1,identity_ce=float(ce.detach()),crossword_supcon=float(sup.detach()),phone_kl=float(lp.detach()),word_ce=float(lc.detach()),valid_phone_frames=nf,preclip_gradnorm=norm,elapsed_s=time.perf_counter()-began)
            traces.append(row);print('STUDENT',stage,seed,arm,step+1,'seconds',round(row['elapsed_s'],1),flush=True)
        torch.save(dict(model=model.state_dict(),stage=stage,seed=seed,arm=arm,steps=steps,protocol_sha=ph,trace=traces,gradient=diagnostics,threads=threads,initial_sha256=initial_sha,batch_sha256=batch_sha),path)
        (OUT/f'train_{stage}_{seed}_{arm}_public.json').write_text(json.dumps(dict(stage=stage,seed=seed,arm=arm,steps=steps,parameters=sum(p.numel() for p in model.parameters()),initial_sha256=initial_sha,batch_sha256=batch_sha,trace=traces,gradient=diagnostics),indent=2))
        del model,optimizer,shifted;gc.collect()
    print('STAGE_COMPLETE',stage,arms,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['development','final'],required=True);parser.add_argument('--arms',default=','.join(ARMS));parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    chosen=args.arms.split(',');assert set(chosen)<=set(ARMS)
    with threadpool_limits(limits=args.threads):train(args.stage,chosen,args.threads)
