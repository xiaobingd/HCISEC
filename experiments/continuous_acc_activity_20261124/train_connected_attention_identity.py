"""Direct local-identity gradient into the attention used for ACC-only queries."""
import json,hashlib,itertools
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from train_phoneme_conditioned_identity import ROOT,BASE,Net as OldNet,contrast,embed,TRAIN,QUERY
from train_acc_phoneme_locator import grid_descriptors
from run_continuous import choose_auth,measure

OLD=BASE/'phoneme/identity_learning'
PRIVATE=BASE/'phoneme/connected_attention'
OUT=ROOT/'outputs/vr261127_connected_phoneme_attention'
ARMS=['C0_identity','C1_detached_Audio','C2_connected_ACC','C3_connected_Audio','C4_connected_wrongrepeat','C5_connected_shuffledphone']
EXTRA=[23,24,25]

class Net(OldNet):
    def __init__(self,connected=False):super().__init__();self.connected=connected
    def forward(self,x,local_weight=None):
        h=self.encoder(x);logit=self.attention(h).squeeze(-1);attention=logit.softmax(1)
        z=self.pool(h,x,attention)
        if local_weight is None:return z,None
        weight=(logit+local_weight.clamp_min(1e-20).log()).softmax(1) if self.connected else local_weight
        return z,self.pool(h,x,weight)

def build_net(arm):return Net(connected=arm not in ['C0_identity','C1_detached_Audio'])

def gradient_norm(loss,params):
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return float(sum(v.square().sum() for v in grads if v is not None).sqrt()) if any(v is not None for v in grads) else 0.

def load_inputs():
    PRIVATE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    p=np.load(OLD/'prepared_private.npz');raw=np.load(OLD/'inputs_private.npz');assert np.array_equal(p['indices'],raw['indices'])
    u,w=p['u'],p['w'];fit,reg,cal,query=[p[k] for k in ['fit','registration','calibration','query']]
    assert [int(v.sum()) for v in [fit,reg,cal,query]]==[240,120,120,240]
    assert np.stack([fit,reg,cal,query]).sum(0).min()==1 and np.stack([fit,reg,cal,query]).sum(0).max()==1
    a=p['Audio_teacher'].copy();e=p['ACC_teacher'].astype('float32');assert np.isnan(a[~fit]).all() and np.isfinite(a[fit]).all()
    donor=np.arange(len(u))
    for us in range(1,11):
      for wo in TRAIN:
        ii=np.flatnonzero(fit&(u==us)&(w==wo));assert len(ii)==4;donor[ii]=np.roll(ii,-2)
    assert fit[donor[fit]].all() and np.array_equal(u[donor[fit]],u[fit]) and np.array_equal(w[donor[fit]],w[fit]) and np.all(donor[fit]!=np.flatnonzero(fit))
    wrong=np.full_like(a,np.nan);wrong[fit]=a[donor[fit]]
    source_protocol=json.loads((ROOT/'outputs/vr261126_phoneme_conditioned_identity/protocol_public.json').read_text(encoding='utf-8'))
    protocol=dict(scope=source_protocol['scope'],data_protocol='samevr261126_phoneme_conditioned_identity train/reg/cal/query roles,features,normalization,registrationandcalibration',
      train_words=TRAIN,query_words=QUERY,additional_query_words=EXTRA,extra_scope='samefrozenadditionalEHwordset aspreviousstudy,alreadyviewed;noquerywordselection',
      counts=dict(fit=240,registration=120,calibration=120,query=240,additional_query=240),
      architecture=source_protocol['encoder'],parameters=5057,seeds=[42,43,44],steps=400,batch=source_protocol['batch'],optimizer=source_protocol['optimizer'],
      global_loss=source_protocol['global_loss'],local_loss=source_protocol['local_loss'],
      connection='local_weight=softmax(ACC_attention_logits+log(training_prior));sameattentionparameterused forglobalquerypooling;NO extraalignmentKL/VADloss/head',
      C0_identity='sameglobalidentitybaseline',C1_detached_Audio='exactpreviousB2;teacherlocalpooldoesnotuseattentionparameter',
      C2_connected_ACC='connectedlocalidentityloss withACCenergyprior/dictionaryphone',C3_connected_Audio='connectedlocalidentityloss withcorrectAudio prior/dictionaryphone',
      C4_connected_wrongrepeat='sameuser/sameword anotherFITrepeat prior;only4fitreps used,rotate2;noREG/CAL/queryAudio',
      C5_connected_shuffledphone='samecorrectAudio prior;balancedper-userwordwisephoneclassshuffle exactlysame previousnegativecontrol',
      query='globalsoftmaxACCattentiononly;noAudio,phonelabel,priororwordIDinput;legacycachealreadysegmented,notcontinuousendtoendAudiofree',
      budget='18models400steps each,sameinitializationandfrozenbatchesacrossarmsperseed;noearlystopping/weightsearch/userfilter',
      calibration='same12REG/12CAL peruser;all180tau locked beforebothquerysets scored;originalcalruleunchanged',
      primary='initial3words MacroEER paireduser seedmean C3vsC1(connectioneffect),C3vsC2(Audioindependent),Holm2;wrongprior/shuffledphone andadditional3wordset reported',
      teacher_sigma_s=.1,teacher_offset_s=-.1,prior_log_floor=1e-20,source_minimum_repeat_center_gap_s=source_protocol['minimum_repeat_center_gap_s'],
      all_historical_words_exploratory=True,Audio_teacher_only_fit=True,private_export=False,
      data_sha256=hashlib.sha256((OLD/'inputs_private.npz').read_bytes()).hexdigest(),roles_sha256=hashlib.sha256((OLD/'prepared_private.npz').read_bytes()).hexdigest())
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    np.savez_compressed(PRIVATE/'roles_private.npz',u=u,w=w,fit=fit,registration=reg,calibration=cal,query=query,donor=donor)
    return torch.tensor((raw['x']-p['mean'])/p['scale']),p,u,w,fit,reg,cal,query,a,e,wrong,ph

def main():
    torch.set_num_threads(2);x,p,u,w,fit,reg,cal,query,a,e,wrong,ph=load_inputs()
    y=torch.tensor(u-1,dtype=torch.long);words=torch.tensor(w,dtype=torch.long);phone=torch.tensor(p['phone'],dtype=torch.long)
    ri,ci,qi=[np.flatnonzero(v) for v in [reg,cal,query]];locks=[];trainrows=[];diags=[]
    for seed in [42,43,44]:
      source=np.load(OLD/f'seed{seed}_batches_private.npz');batches=source['batches'];shuffle=torch.tensor(source['pseudo'],dtype=torch.long)
      assert batches.shape==(400,60) and fit[batches].all()
      for arm in ARMS:
        torch.manual_seed(seed);net=build_net(arm);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);path=PRIVATE/f'{seed}_{arm}_private.pt'
        if path.exists():
          save=torch.load(path,weights_only=False);assert save['protocol_sha256']==ph;net.load_state_dict(save['model']);trace=save['trace'];diag=save['gradient']
        else:
          trace=[];diag=[]
          for step,ii in enumerate(batches):
            net.train();opt.zero_grad();prior=None
            if arm!='C0_identity':prior=torch.tensor(e[ii] if arm=='C2_connected_ACC' else wrong[ii] if arm=='C4_connected_wrongrepeat' else a[ii],dtype=torch.float32)
            z,zlocal=net(x[ii],prior);ce=F.cross_entropy(16*z@F.normalize(net.classifier.weight,dim=1).T,y[ii]);glob=contrast(z,y[ii],words[ii])
            local=z.sum()*0 if zlocal is None else contrast(zlocal,y[ii],words[ii],shuffle[ii] if arm=='C5_connected_shuffledphone' else phone[ii])
            global_loss=ce+.1*glob;loss=global_loss+.1*local
            if step in [0,99,399]:
              pp=list(net.attention.parameters());lg=gradient_norm(.1*local,pp);gg=gradient_norm(global_loss,pp)
              diag.append(dict(step=step+1,local_attention_gradient_norm=lg,global_attention_gradient_norm=gg,local_to_global_ratio=lg/max(gg,1e-12)))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
            if (step+1)%100==0:
              trace.append(dict(step=step+1,ce=float(ce.detach()),global_supcon=float(glob.detach()),local_supcon=float(local.detach())));print('TRAIN',seed,arm,step+1,flush=True)
          torch.save(dict(model=net.state_dict(),protocol_sha256=ph,trace=trace,gradient=diag,steps=400),path)
        trainrows.append(dict(seed=seed,arm=arm,trace=trace));diags.append(dict(seed=seed,arm=arm,gradient=diag))
        rz=embed(net,x,ri);cz=embed(net,x,ci);proto=np.stack([rz[u[ri]==us].mean(0) for us in range(1,11)]);proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
        for us in range(1,11):locks.append(dict(seed=seed,arm=arm,user=us,tau=choose_auth(cz@proto[us-1],u[ci]==us)))
    assert len(locks)==180;(PRIVATE/'thresholds_private.json').write_text(json.dumps(locks));print('ALL180_THRESHOLDS_LOCKED',flush=True)
    (OUT/'training_public.json').write_text(json.dumps(trainrows,indent=2));(OUT/'attention_gradient_public.json').write_text(json.dumps(diags,indent=2))
    # ExtraACC-only queries: no Audio boundaries or teacher values are accessed.
    raw=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');uu,ww=raw['user'],raw['word'];ei=np.flatnonzero(np.isin(ww,EXTRA));wave=raw['acc'];centers=np.arange(1.1,1.901,.02)
    xx=np.stack([grid_descriptors(wave[int(i)],centers) for i in ei]).astype('float32');ex=torch.tensor((xx-p['mean'])/p['scale'])
    allrows=[];wordrows=[];attention=[]
    for seed in [42,43,44]:
      for arm in ARMS:
        net=build_net(arm);net.load_state_dict(torch.load(PRIVATE/f'{seed}_{arm}_private.pt',weights_only=False)['model']);rz=embed(net,x,ri)
        proto=np.stack([rz[u[ri]==us].mean(0) for us in range(1,11)]);proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
        for phase,inputs,indices,labels,lex in [('initial',x,qi,u[qi],w[qi]),('additional_EH',ex,np.arange(len(ei)),uu[ei],ww[ei])]:
          qz=embed(net,inputs,indices)
          with torch.inference_mode():
            dist=net.attention(net.encoder(inputs[indices])).squeeze(-1).softmax(1);entropy=-(dist*dist.clamp_min(1e-20).log()).sum(1).numpy();peaks=centers[dist.argmax(1).numpy()]
          attention.append(dict(seed=seed,arm=arm,phase=phase,entropy_mean=float(entropy.mean()),peak_center_mean_s=float(peaks.mean()),peak_center_std_s=float(peaks.std())))
          for us in range(1,11):
            tau=next(v['tau'] for v in locks if v['seed']==seed and v['arm']==arm and v['user']==us);score=qz@proto[us-1]
            allrows.append(dict(seed=seed,arm=arm,phase=phase,user=us,**measure(score,labels==us,tau)))
            for wo in sorted(set(lex)):
              mask=lex==wo;wordrows.append(dict(seed=seed,arm=arm,phase=phase,user=us,word=int(wo),**measure(score[mask],labels[mask]==us,tau)))
    users=[];summary=[];keys=['eer','hter','far','frr','auc']
    for phase in ['initial','additional_EH']:
      for arm in ARMS:
        for us in range(1,11):
          rr=[v for v in allrows if v['phase']==phase and v['arm']==arm and v['user']==us];users.append(dict(phase=phase,arm=arm,user=us,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
        rr=[v for v in allrows if v['phase']==phase and v['arm']==arm];summary.append(dict(phase=phase,arm=arm,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
    for name,obj in [('summary_public.json',summary),('per_user_public.json',users),('per_seed_user_public.json',allrows),('per_word_user_public.json',wordrows),('query_attention_public.json',attention),('verification_public.json',dict(status='TRAIN_COMPLETE',models=18,thresholds=180,all180thresholds_before_both_querysets=True,query_ACC_only=True,query_Audio_unused=True,protocol_sha256=ph))]:
      (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CONNECTED_ATTENTION_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
