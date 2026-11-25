"""Matched-budget closed-set cross-word identity learning with phoneme controls."""
import argparse,json,hashlib,itertools
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from train_acc_phoneme_locator import grid_descriptors
from audit_phoneme_identity import ROOT,BASE,GROUPS
from run_continuous import choose_auth,measure

OUT=ROOT/'outputs/vr261126_phoneme_conditioned_identity'
PRIVATE=BASE/'phoneme/identity_learning'
TRAIN=[0,1,6,7,11,13];QUERY=[10,14,15]
ARMS=['B0_identity','B1_dictionary_ACC','B2_Audio_phoneme','B3_shuffled_phoneme']
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(24,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.attention=nn.Linear(32,1);nn.init.zeros_(self.attention.weight);nn.init.zeros_(self.attention.bias)
        self.project=nn.Linear(88,32);self.classifier=nn.Linear(32,10,bias=False)
    def pool(self,h,x,weight):
        mean=(h*weight[:,:,None]).sum(1);std=(((h-mean[:,None,:]).square()*weight[:,:,None]).sum(1)+1e-6).sqrt()
        raw=(x*weight[:,:,None]).sum(1)
        return F.normalize(self.project(torch.cat([mean,std,raw],1)),dim=1)
    def forward(self,x,local_weight=None):
        h=self.encoder(x);a=self.attention(h).squeeze(-1).softmax(1);global_z=self.pool(h,x,a)
        local_z=self.pool(h,x,local_weight) if local_weight is not None else None
        return global_z,local_z

def contrast(z,y,word,phone=None):
    same=y[:,None]==y[None,:];pos=same&(word[:,None]!=word[None,:])
    allowed=(~same)|pos
    if phone is not None:
        matches=phone[:,None]==phone[None,:];pos&=matches;allowed&=matches
    good=pos.any(1)
    if not good.any():return z.sum()*0
    logits=z@z.T/.1;logp=logits-torch.logsumexp(logits.masked_fill(~allowed,-1e4),1,keepdim=True)
    return -((logp*pos).sum(1)/pos.sum(1).clamp_min(1))[good].mean()

def embed(net,x,indices):
    net.eval();out=[]
    with torch.inference_mode():
        for k in range(0,len(indices),64):out.append(net(x[indices[k:k+64]])[0].numpy())
    return np.concatenate(out)

def prepare():
    PRIVATE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    c=np.load(BASE/'phoneme/features_private.npz');d=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz')
    indices=c['indices'];u,w=c['user'],c['word'];r=d['repeat'][indices];raw=d['acc'];centers=np.arange(1.1,1.901,.02)
    fit=np.isin(w,TRAIN)&(r<4);reg=np.isin(w,TRAIN)&np.isin(r,[4,5]);cal=np.isin(w,TRAIN)&(r>=6);query=np.isin(w,QUERY)
    assert [int(a.sum()) for a in [fit,reg,cal,query]]==[240,120,120,240]
    assert np.all(np.sum(np.stack([fit,reg,cal,query]),axis=0)==1)
    manifest=json.loads((BASE/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
    times=np.array([np.mean(manifest[int(i)]['imu_interval']) for i in indices]);min_gap=10.
    for us in range(1,11):
      for wo in sorted(set(w)):
        tt=np.sort(times[(u==us)&(w==wo)]);min_gap=min(min_gap,float(np.diff(tt).min()))
    assert min_gap>.92,'Central-frame input support overlaps repeated recordings'
    cache=PRIVATE/'inputs_private.npz'
    if cache.exists():
        v=np.load(cache);assert np.array_equal(v['indices'],indices);x=v['x'].copy()
    else:
        x=np.stack([grid_descriptors(raw[int(i)],centers) for i in indices]).astype('float32')
        np.savez_compressed(cache,x=x,indices=indices)
    assert x.shape==(720,41,24) and np.isfinite(x).all()
    # Read teacher boundaries ONLY for the new fit-role samples. Other roles are NaN.
    teacher=np.full((720,41),np.nan,dtype='float32');bounds=c['phonebounds'];phone=np.full(720,-1,dtype=int)
    for pos in np.flatnonzero(fit):
        rec=manifest[int(indices[pos])];lo,hi=rec['audio_start_s'],rec['audio_end_s'];il,ih=rec['imu_interval']
        center=1.5+(bounds[pos].mean()-(lo+hi)/2)*(ih-il)/(hi-lo)-.1
        weight=np.exp(-.5*((centers-center)/.1)**2);teacher[pos]=weight/weight.sum()
        phone[pos]=next(j for j,words in enumerate(GROUPS.values()) if int(w[pos]) in words)
    assert np.isnan(teacher[~fit]).all() and np.isfinite(teacher[fit]).all()
    # ACC-only local reference uses independent FFT windows, not a filter over overlapping3s.
    logband=x[:,:,:21].reshape(720,41,3,7)
    power=np.exp(np.clip(logband[:,:,:,2:],-80,80)).sum((2,3));peak=centers[np.argmax(power,axis=1)]
    accweight=np.exp(-.5*((centers[None,:]-peak[:,None])/.1)**2);accweight/=accweight.sum(1,keepdims=True)
    mean=x[fit].mean((0,1));scale=np.maximum(x[fit].std((0,1)),1e-6)
    protocol=dict(scope='closed-set cross-word;all10 identities train representations;NOT unseen-user study;historicalviewedwords exploratorynotblind',
      train_words=TRAIN,query_words=QUERY,fit_repeats=[0,1,2,3],registration_repeats=[4,5],calibration_repeats=[6,7],
      counts=dict(fit=240,registration=120,calibration=120,query=240),register_per_user=12,calibrate_per_user=12,query_per_user=24,
      roles='no recording index overlap;used rawACCframe unioncenter+/-460ms;minimumrepeatcentergap exceeds920ms;raw3seccacheitself overlapsbutunusedpartsnotmodelinput',
      minimum_repeat_center_gap_s=min_gap,input='41x24 fixed120msACC FFT/axisstd windows,20msstep;closed legacyVAD-anchored3s cache: notfullyAudiofreecontinuouspipeline',
      encoder='shared frameMLP24->32SiLU->32SiLU;learnedattention;weightedmean/std64+rawmean24->32L2;temporary10classhead;5057params',
      global_loss='normalizedCE scale16 +.1crosswordSupCon temperature.1',
      local_loss='.1SupCon onteacher-localembedding: sameidentity/samephone/differentword positives;differentidentity/samephone negatives;singletonphoneanchors skipped',
      arms=dict(B0_identity='global identity only',B1_dictionary_ACC='local loss with dictionaryphone groups andACCenergyGaussian',B2_Audio_phoneme='dictionaryphone groups plus correct fitAudioGaussian region',B3_shuffled_phoneme='sameAudioGaussian region,wordwisephoneclassshuffle separatelyforeachuser preserving2/3/1counts'),
      teacher='cachedautomaticfitAudio phonecenter -100ms,sigma100ms fixedfrompreviousexploration;reg/cal/query teacherarraysNaN;noqueryAudio inmodel orcrop',
      lexicon_limitation='phone categories originate dictionary,notAudio;Audioindependentbenefit requires B2>B1;trainingAYonlyfive,noAYcrosswordlocalpositive;querynine isgeneralizationcontrol',
      seeds=[42,43,44],steps=400,batch='10usersx6trainwordsx1fitrepeat=60,identicalbatch/init perseedacrossarms',optimizer='AdamW.001 wd.01 clip1',
      normalization='fit samples only,sharedacrossarms;nohyperparameterselection/earlystopping',
      authentication='cosine to12registration embeddings;threshold mincalHTER thenFARFRRbalance thenhigherthreshold;all120thresholdslockedbeforequeryscoring;classifierdiscarded',
      primary='pooledquery MacroEER paired byuser,seedmean, B2vsB0 andB2vsB1 exactuser signflip Holm2;B3control andHTER/FAR/FRR/AUC allreported',
      no_private_export=True,no_raw_final_Audio_opened=True)
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    np.savez_compressed(PRIVATE/'prepared_private.npz',u=u,w=w,repeat=r,indices=indices,fit=fit,registration=reg,calibration=cal,query=query,
        mean=mean,scale=scale,Audio_teacher=teacher,ACC_teacher=accweight,phone=phone)
    return torch.tensor((x-mean)/scale),u,w,r,fit,reg,cal,query,teacher,accweight,phone,ph

def main(steps):
    assert steps==400,'Fixed matched training budget'
    torch.set_num_threads(2);x,u,w,r,fit,reg,cal,query,aweight,eweight,phone,ph=prepare();trainrows=[];gradrows=[]
    y=torch.tensor(u-1,dtype=torch.long);word=torch.tensor(w,dtype=torch.long);truephone=torch.tensor(phone,dtype=torch.long)
    regidx=np.flatnonzero(reg);calidx=np.flatnonzero(cal);locks=[]
    for seed in [42,43,44]:
      rng=np.random.default_rng(seed)
      batches=np.array([[int(rng.choice(np.flatnonzero(fit&(u==us)&(w==wo)))) for us in range(1,11) for wo in TRAIN] for _ in range(400)])
      pseudo=phone.copy();rngp=np.random.default_rng(seed+9000)
      group_by_word={wo:next(j for j,words in enumerate(GROUPS.values()) if wo in words) for wo in TRAIN}
      true=np.array([group_by_word[wo] for wo in TRAIN])
      for us in range(1,11):
        pp=true.copy()
        while np.array_equal(pp,true):pp=rngp.permutation(true)
        for wo,pc in zip(TRAIN,pp):pseudo[fit&(u==us)&(w==wo)]=pc
      shuffled=torch.tensor(pseudo,dtype=torch.long)
      np.savez_compressed(PRIVATE/f'seed{seed}_batches_private.npz',batches=batches,pseudo=pseudo)
      for arm in ARMS:
        path=PRIVATE/f'{seed}_{arm}_private.pt';torch.manual_seed(seed);net=Net();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        if path.exists():
          saved=torch.load(path,weights_only=False);assert saved['protocol_sha256']==ph;net.load_state_dict(saved['model']);trace=saved['trace']
        else:
          trace=[]
          for step,ii in enumerate(batches):
            net.train();opt.zero_grad()
            local=None if arm=='B0_identity' else torch.tensor(eweight[ii] if arm=='B1_dictionary_ACC' else aweight[ii],dtype=torch.float32)
            z,zlocal=net(x[ii],local);ce=F.cross_entropy(16*z@F.normalize(net.classifier.weight,dim=1).T,y[ii]);glob=contrast(z,y[ii],word[ii])
            loc=z.sum()*0 if zlocal is None else contrast(zlocal,y[ii],word[ii],shuffled[ii] if arm=='B3_shuffled_phoneme' else truephone[ii])
            loss=ce+.1*glob+.1*loc;loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
            if (step+1)%100==0:
              trace.append(dict(step=step+1,ce=float(ce.detach()),global_supcon=float(glob.detach()),local_supcon=float(loc.detach())));print('TRAIN',seed,arm,step+1,flush=True)
          torch.save(dict(model=net.state_dict(),trace=trace,protocol_sha256=ph,steps=400),path)
        trainrows.append(dict(seed=seed,arm=arm,trace=trace))
        rz=embed(net,x,regidx);cz=embed(net,x,calidx);prototypes=np.stack([rz[u[regidx]==us].mean(0) for us in range(1,11)]);prototypes/=np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-12)
        for us in range(1,11):locks.append(dict(seed=seed,arm=arm,user=us,tau=choose_auth(cz@prototypes[us-1],u[calidx]==us)))
    assert len(locks)==120
    (PRIVATE/'thresholds_private.json').write_text(json.dumps(locks));print('ALL_THRESHOLDS_LOCKED',flush=True)
    (OUT/'training_public.json').write_text(json.dumps(trainrows,indent=2));rows=[];wordrows=[];qi=np.flatnonzero(query)
    for seed in [42,43,44]:
      for arm in ARMS:
        net=Net();net.load_state_dict(torch.load(PRIVATE/f'{seed}_{arm}_private.pt',weights_only=False)['model'])
        rz=embed(net,x,regidx);qz=embed(net,x,qi);p=np.stack([rz[u[regidx]==us].mean(0) for us in range(1,11)]);p/=np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-12)
        for us in range(1,11):
          tau=next(v['tau'] for v in locks if v['seed']==seed and v['arm']==arm and v['user']==us);score=qz@p[us-1];lab=u[qi]==us
          rows.append(dict(seed=seed,arm=arm,user=us,**measure(score,lab,tau)))
          for wo in QUERY:
            mask=w[qi]==wo;wordrows.append(dict(seed=seed,arm=arm,user=us,word=wo,**measure(score[mask],lab[mask],tau)))
    peruser=[];summary=[]
    keys=['eer','hter','far','frr','auc']
    for arm in ARMS:
      for us in range(1,11):
        rr=[v for v in rows if v['arm']==arm and v['user']==us];peruser.append(dict(arm=arm,user=us,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
      rr=[v for v in rows if v['arm']==arm];summary.append(dict(arm=arm,**{k:float(np.mean([v[k] for v in rr])) for k in keys}))
    for name,obj in [('summary_public.json',summary),('per_user_public.json',peruser),('per_seed_user_public.json',rows),('per_word_user_public.json',wordrows),('verification_public.json',dict(status='TRAIN_COMPLETE',models=12,thresholds=120,all_thresholds_locked_before_query_scoring=True,reg_cal_query_Audio_teacher_missing=True,protocol_sha256=ph))]:
      (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('IDENTITY_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=400);args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.steps)
