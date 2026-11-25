"""Audio-guided phoneme diagnostic; historical fit Audio only; NOT deployment."""
import json,sys,itertools
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/vr261119_10x40_raw_acc';OUT=ROOT/'outputs/vr261126_phoneme_identity'
sys.path.insert(0,str(ROOT/'.runtime/phoneme'))
from pocketsphinx import Decoder

GROUPS={'AE':[0,1,10],'EH':[6,11,13,14],'AY':[7,15]}
ARMS=['shared_phone','word_center','wrong_phone','offset_minus100','offset_plus100','whole_word']
def unit(z):return z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-12)
def descriptor(wave):
    assert wave.shape[0]==3
    wave=wave-wave.mean(1,keepdims=True)
    spectrum=abs(np.fft.rfft(wave*np.hanning(wave.shape[1]),n=1024,axis=1))**2
    freq=np.fft.rfftfreq(1024,1/1000)
    bands=[(20,50),(50,80),(80,120),(120,180),(180,250),(250,350),(350,450)]
    power=np.stack([np.log(np.maximum(spectrum[:,(freq>=lo)&(freq<hi)].mean(1),1e-20)) for lo,hi in bands],1)
    return np.r_[power.ravel(),wave.std(1)]
def window(wave,center,width=.12):
    lo=int(round((center-width/2)*1000));n=int(round(width*1000));hi=lo+n
    if lo<0 or hi>wave.shape[1] or n<30:return None
    return descriptor(wave[:,lo:hi])
def align(decoder,wave,text):
    pcm=np.clip(wave*32768,-32768,32767).astype('<i2').tobytes()
    decoder.set_align_text(text)
    for phonepass in [False,True]:
        if phonepass:decoder.set_alignment()
        decoder.start_utt();decoder.process_raw(pcm,full_utt=True);decoder.end_utt()
    alignment=decoder.get_alignment();result=[]
    if alignment is None:return []
    for word in alignment:
      for phone in word:
        if phone.name!='SIL':result.append((phone.name,phone.start/100,(phone.start+phone.duration)/100))
    return result
def main():
    OUT.mkdir(exist_ok=True);private=BASE/'phoneme';private.mkdir(exist_ok=True)
    raw=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');fit=raw['fit'];acc=raw['acc'];u,w,ids=[raw[k] for k in ['user','word','sample_id']]
    manifest=json.loads((BASE/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
    vocab=json.loads((ROOT/'outputs/vr261124_continuous_acc_activity/vocabulary_public.json').read_text(encoding='utf-8'))
    assert list(ids)==[v['sample_id'] for v in manifest]
    selected=sorted(set(itertools.chain.from_iterable(GROUPS.values())));indices=np.flatnonzero(fit&np.isin(w,selected));assert len(indices)==720
    allowed={manifest[int(i)]['record_group'] for i in indices};excluded={v['record_group'] for i,v in enumerate(manifest) if not fit[i]};assert allowed.isdisjoint(excluded)
    decoder=Decoder(loglevel='FATAL',samprate=16000)
    lexicon={str(wo):dict(word=vocab[str(wo)],phones=decoder.lookup_word(vocab[str(wo)].lower())) for wo in selected}
    protocol=dict(groups=GROUPS,lexicon=lexicon,users=10,fit_words_only=selected,samples=720,
      aligner='PocketSphinx5.0.4 bundled US-English acoustic model,known single-word text,word alignment then phone-state alignment;automatic estimated labels,notmanualtruth',
      dependency_sha256='3dbbca377ca081c1afeb87dac8d17e41e17dcb89ff04041edb03a73b42e53bbd',
      plans='threelexicalvowelgroups,leaveonewordout;AYonly2words andexploratory;time not in vocabulary',
      crop='120ms at aligned vowel center,wordcenter,samewordwrongphonefarthestcenter,andfixed+-100ms offsets;wholeword unequalduration contextcontrol',
      front='raw3axisACC1000Hz;DCremove;21logbandpowers+3axisstd;identical24Dallarms;noencodertraining',
      timing='AudioPCM interpreted44100;alignafter16kresample;mapphonewithinwordfractiontoexistingIMUinterval,notphysicallyvalidated;mappingmaycontain175ms residual',
      failures='anyalignment/cropfailure useswordcenter forallarms;all10users retained;coverage reported;no favorableuserfilter',
      eval='leavewordout query,othergroupwords enrollment;targetheldoutfromscaler fitting,other9users andsupportwords only;unitprototypecosine;all8queryrepsincluded;nohyperparameterselection',
      primary='EH andAEsharedphone-vswordcenterAUC,userclusterpairedsignflip,Holm2;AYexploratory;offsetsnotselected',
      scope='diagnostic oracle Audio locates historicalfit-word query phones;NOT ACC-onlydeployment;nofinaltestAudio;noEERorKDclaim;rawoldcutwindows mayoverlap butdistinctwordtrials',
      no_neural_training=True,no_final_test_audio=True)
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    features={a:np.zeros((len(indices),24)) for a in ARMS};success=np.zeros(len(indices),bool);phonebounds=np.zeros((len(indices),2));failure_reasons={}
    groups={}
    for pos,i in enumerate(indices):groups.setdefault(manifest[int(i)]['record_group'],[]).append((pos,int(i)))
    for groupno,(stem,records) in enumerate(groups.items()):
      path=Path(stem);meta=json.loads(path.with_name(path.name+'_meta.json').read_text(encoding='utf-8-sig'));rate,pcm=wavfile.read(path.parent/meta['full_audio']);assert rate==48000
      pcm=pcm.astype(float)/32768
      if pcm.ndim==2:pcm=pcm[:,0]
      for pos,i in records:
        rec=manifest[i];lo,hi=rec['audio_start_s'],rec['audio_end_s'];il,ih=rec['imu_interval'];duration=ih-il
        assert duration>0
        fallback=window(acc[i],1.5)
        for arm in ARMS:features[arm][pos]=fallback
        try:
          pad=.08;start=max(0,lo-pad);end=min(len(pcm)/44100,hi+pad)
          clip=pcm[int(round(start*44100)):int(round(end*44100))];wave=signal.resample_poly(clip,160,441)
          phones=align(decoder,wave,vocab[str(int(w[i]))].lower())
          target=next(k for k,words in GROUPS.items() if int(w[i]) in words)
          candidates=[p for p in phones if p[0]==target and .03<=p[2]-p[1]<=.8]
          if not candidates:raise ValueError('no_valid_target_phone')
          chosen=max(candidates,key=lambda p:p[2]-p[1]);ac=start+(chosen[1]+chosen[2])/2
          if not lo<=ac<=hi:raise ValueError('phone_outside_utterance')
          center=1.5+(ac-(lo+hi)/2)*duration/(hi-lo)
          others=[p for p in phones if p[0]!=target and p[2]-p[1]>=.03]
          if not others:raise ValueError('no_wrongphone_control')
          other=max(others,key=lambda p:abs((p[1]+p[2])/2-(chosen[1]+chosen[2])/2))
          oc=start+(other[1]+other[2])/2;wrongcenter=1.5+(oc-(lo+hi)/2)*duration/(hi-lo)
          vectors={'shared_phone':window(acc[i],center),'word_center':fallback,'wrong_phone':window(acc[i],wrongcenter),
            'offset_minus100':window(acc[i],center-.1),'offset_plus100':window(acc[i],center+.1),'whole_word':window(acc[i],1.5,max(duration,.03))}
          if any(v is None for v in vectors.values()):raise ValueError('crop_outside_cache')
          for arm,v in vectors.items():features[arm][pos]=v
          success[pos]=True;phonebounds[pos]=[start+chosen[1],start+chosen[2]]
        except (ValueError,RuntimeError) as e:
          key=str(e);failure_reasons[key]=failure_reasons.get(key,0)+1
      if (groupno+1)%10==0:print('PHONE_RECORDINGS',groupno+1,flush=True)
    np.savez_compressed(private/'features_private.npz',**features,success=success,phonebounds=phonebounds,indices=indices,user=u[indices],word=w[indices],sample_id=ids[indices])
    rows=[];coverage=[]
    uu,ww=u[indices],w[indices]
    for targetgroup,words in GROUPS.items():
      for us in range(1,11):
        mask=(uu==us)&np.isin(ww,words);coverage.append(dict(group=targetgroup,user=us,valid=int(success[mask].sum()),total=int(mask.sum())))
      for heldword in words:
       support=np.isin(ww,[v for v in words if v!=heldword]);query=ww==heldword
       for us in range(1,11):
        fitmask=support&(uu!=us)
        for arm,z in features.items():
          sc=StandardScaler().fit(z[fitmask]);zz=sc.transform(z)
          proto=unit(np.stack([zz[support&(uu==v)].mean(0) for v in range(1,11)]));q=unit(zz[query]);score=q@proto[us-1]
          labels=uu[query]==us
          margin=(q[labels]@proto.T)[:,us-1]-np.max(np.delete(q[labels]@proto.T,us-1,axis=1),1)
          rows.append(dict(group=targetgroup,held_word=int(heldword),user=us,arm=arm,auc=float(roc_auc_score(labels,score)),margin=float(margin.mean())))
    peruser=[];summary=[]
    for g in GROUPS:
      for us in range(1,11):
       for arm in ARMS:
        rr=[v for v in rows if v['group']==g and v['user']==us and v['arm']==arm]
        peruser.append(dict(group=g,user=us,arm=arm,auc=float(np.mean([v['auc'] for v in rr])),margin=float(np.mean([v['margin'] for v in rr]))))
      for arm in ARMS:
        rr=[v for v in peruser if v['group']==g and v['arm']==arm];summary.append(dict(group=g,arm=arm,auc=float(np.mean([v['auc'] for v in rr])),margin=float(np.mean([v['margin'] for v in rr]))))
    comparisons=[];signs=np.array(list(itertools.product([-1,1],repeat=10)))
    for g in GROUPS:
      for control in ['word_center','wrong_phone']:
        delta=np.array([next(v['auc'] for v in peruser if v['group']==g and v['user']==us and v['arm']=='shared_phone')-next(v['auc'] for v in peruser if v['group']==g and v['user']==us and v['arm']==control) for us in range(1,11)])
        comparisons.append(dict(group=g,control=control,delta_auc=float(delta.mean()),user_signflip_p=float(np.mean((signs*delta).mean(1)>=delta.mean()-1e-12)),improved_users=int((delta>0).sum())))
    from audit_local_identity_residual import holm
    primary=[v for v in comparisons if v['group'] in ['AE','EH'] and v['control']=='word_center']
    for v,p in zip(primary,holm([v['user_signflip_p'] for v in primary])):v['holm_primary_p']=float(p)
    for name,obj in [('summary_public.json',summary),('per_user_public.json',peruser),('per_word_user_public.json',rows),('coverage_public.json',coverage),('comparison_public.json',comparisons),('alignment_audit_public.json',dict(records=len(indices),recordings=len(groups),successful=int(success.sum()),failures=failure_reasons,all_users_retained=True,final_Audio_opened=False,manual_phone_validation=False))]:
      (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(coverage=int(success.sum()),comparisons=comparisons,summary=summary)))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
