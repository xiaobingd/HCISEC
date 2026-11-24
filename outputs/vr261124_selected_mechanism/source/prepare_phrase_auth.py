"""Trace prompt durations, retain full utterances, and reserve all long prompts."""
import json
from pathlib import Path
import wave
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'phrase_auth'
SOURCE=Path(r'D:\alldata\261119双路数据_merged')
REG=[20,21,22,23]
CAL=[8,18,24,30,31]
SHORT=[25,26,27,28,29]
ENGLISH=[3,9,12,19,32]
CHINESE=[35,36,37,38,39]
FIT=sorted(set(range(40))-set(REG+CAL+SHORT+ENGLISH+CHINESE))

def front(t,acc,lo,hi,length):
    center=(lo+hi)/2
    grid=center+np.arange(length)/1000-length/2000
    z=np.stack([np.interp(grid,t,c) for c in acc.T])
    valid=(grid>=lo)&(grid<hi)
    z-=z[:,valid].mean(1,keepdims=True);z[:,~valid]=0
    f,_,st=signal.stft(z,fs=1000,nperseg=256,noverlap=192,nfft=1024,boundary=None,padded=False)
    power=abs(st)**2;lp=np.log(power+1e-20)
    spec=np.stack([[np.interp(np.geomspace(10,300,40),f,col) for col in axis.T] for axis in lp]).transpose(0,2,1)
    amp=np.log(np.c_[np.sqrt(np.mean(z*z,axis=1)),np.max(abs(z),axis=1),np.quantile(abs(z),[.5,.9,.99],axis=1).T]+1e-20).ravel()
    edges=[10,20,40,80,100,150,200,300,400]
    bands=np.stack([np.log(power[:,(f>=a)&(f<b)].mean((1,2))+1e-20) for a,b in zip(edges[:-1],edges[1:])],axis=1).ravel()
    stats=np.r_[bands,np.log(z.std(axis=1)+1e-20)]
    a=spec.reshape(120,-1)
    return np.concatenate([a.mean(-1),a.std(-1),*(q.mean(-1) for q in np.array_split(a,4,axis=-1)),amp,stats]).astype('float32')

def main():
    OUT.mkdir(exist_ok=True)
    old=np.load(ROOT/'paired_raw_acc_private.npz');stats=np.load(ROOT/'stats27_private.npz')
    ids=old['sample_id'];assert np.array_equal(ids,stats['sample_id'])
    u,w,r=np.array([s.split('_') for s in ids],dtype=int).T
    lookup={str(s):i for i,s in enumerate(ids)}
    vocabulary=json.loads((ROOT/'paired_scope.json').read_text())['words']
    records=[v for v in json.loads((SOURCE/'merge_summary.json').read_text(encoding='utf8'))['records'] if v['tracking_velocity_file']]
    people=sorted(p for p in {v['person'] for v in records} if len({v['prompt_text'].casefold() for v in records if v['person']==p})==40)
    assert len(people)==10
    groups={(v['person'],v['prompt_text'].casefold()):v for v in records if v['person'] in people}
    spec=old['spectrum'].reshape(3200,120,-1)
    crop=np.column_stack([spec.mean(-1),spec.std(-1),*(a.mean(-1) for a in np.array_split(spec,4,axis=-1)),old['amplitude'],stats['features']])
    full=crop.copy();duration=np.zeros(3200);hostduration=np.zeros(3200)
    manifest=[];expanded=[];legacy_errors=[]
    fit=np.isin(w,FIT);fitix=np.flatnonzero(fit)
    previous_wave=old['wave'][fitix]
    fitpos={int(i):j for j,i in enumerate(fitix)}
    wave_matches=0;longfit=[]
    for us,person in enumerate(people,1):
        for word,prompt in enumerate(vocabulary):
            rec=groups[person,prompt]
            stem=SOURCE/person/rec['record_group']
            meta=json.loads(stem.with_name(stem.name+'_meta.json').read_text(encoding='utf-8-sig'))
            stamps=pd.read_csv(stem.with_name(stem.name+'_audio_timestamps.csv'),
                               usecols=['host_callback_perf_counter_s','frame_offset','frames'])
            offset=(stamps.frame_offset+stamps.frames).to_numpy()/48000
            callback=stamps.host_callback_perf_counter_s.to_numpy()
            accepted=[]
            for s in meta['segments']:
                lo,hi=np.interp([s['start_audio_s'],s['end_audio_s']],offset,callback)
                grid=(lo+hi)/2+np.arange(2000)/1000-1
                if ((grid>=lo)&(grid<hi)).sum()<256:continue
                accepted.append((s,float(lo),float(hi)))
            assert len(accepted)>=8
            previous_end=-1
            longgroup=any(hi-lo>2 for s,lo,hi in accepted[:8])
            if longgroup:
                raw=pd.read_csv(stem.with_name(stem.name+'_full_imu.csv'),
                    usecols=['host_read_perf_counter_s','imu_sample_time','offscale_flags','accel_x','accel_y','accel_z'])
                latest=raw.groupby('host_read_perf_counter_s').imu_sample_time.max()
                delta=np.median(latest.index.to_numpy()-latest.to_numpy())
                raw=raw[raw.offscale_flags==0].drop_duplicates('imu_sample_time').sort_values('imu_sample_time')
                t=raw.imu_sample_time.to_numpy()+delta;acc=raw[['accel_x','accel_y','accel_z']].to_numpy()
            for rep,(s,lo,hi) in enumerate(accepted[:8]):
                key=f'{us:02}_{word:02}_{rep:02}';i=lookup[key]
                assert s['start_audio_s']>=previous_end
                previous_end=s['end_audio_s']
                ap=stem.parent/s['audio_file']
                with wave.open(str(ap),'rb') as af:
                    assert af.getframerate()==48000
                    seconds=af.getnframes()/af.getframerate()
                duration[i]=seconds;hostduration[i]=hi-lo
                if hi-lo>2:
                    assert lo>=t[0] and hi<=t[-1]
                    assert np.max(np.diff(t[max(0,np.searchsorted(t,lo)-1):np.searchsorted(t,hi)+1]))<=.12
                    legacy_errors.append(float(np.abs(front(t,acc,lo,hi,2000)-crop[i]).max()))
                    full[i]=front(t,acc,lo,hi,max(2000,int(np.ceil((hi-lo)*1000))))
                    expanded.append(int(i))
                if fit[i]:
                    fs,pcm=wavfile.read(ap)
                    if pcm.ndim==2:pcm=pcm[:,0]
                    assert fs==48000
                    wav=signal.resample_poly(pcm.astype('float32')/32768,1,3)
                    if len(wav)>32000:
                        longfit.append(int(i))
                    else:
                        padded=np.zeros(32000,dtype='float32');start=(32000-len(wav))//2
                        padded[start:start+len(wav)]=wav
                        assert np.array_equal(padded.astype('float16'),previous_wave[fitpos[i]])
                        wave_matches+=1
                manifest.append(dict(sample_id=key,person=person,prompt=prompt,repeat_id=s['repeat_id'],
                    audio_file=str(ap),record_group=str(stem),imu_interval=[lo,hi],audio_duration_s=seconds))
        print('SOURCE_USER_DONE',us,flush=True)
    assert len(manifest)==3200 and np.isfinite(full).all()
    assert not longfit,'New fit Audio longer than cache: requires fresh full-utterance teacher inference'
    assert wave_matches==1280 and len(FIT)==16
    assert not legacy_errors or max(legacy_errors)<1e-3
    assert (duration>0).all() and (hostduration>0).all()
    np.savez_compressed(OUT/'frontend_private.npz',features=full,crop_features=crop,
        sample_id=ids,user=u,word=w,repeat=r,fit=fit,duration_audio=duration,duration_acc=hostduration)
    (OUT/'source_manifest_private.json').write_text(json.dumps(manifest,ensure_ascii=False))
    perword=[]
    for word in range(40):
        m=w==word
        perword.append(dict(word=word,prompt=vocabulary[word],records=int(m.sum()),
            audio_median_s=float(np.median(duration[m])),audio_max_s=float(duration[m].max()),
            host_median_s=float(np.median(hostduration[m])),host_max_s=float(hostduration[m].max()),
            audio_over2=int((duration[m]>2).sum()),host_over2=int((hostduration[m]>2).sum())))
    audit=dict(status='PASS',users=10,records=3200,train_words=FIT,train_records=1280,
        registration_words=REG,calibration_words=CAL,query_single=SHORT,
        query_english_long=ENGLISH,query_chinese_long=CHINESE,
        full_window_extended_records=len(expanded),legacy_feature_probe_max_error=max(legacy_errors,default=0),
        newfit_original_audio_cache_exact_matches=wave_matches,newfit_long_audio=len(longfit),
        word_durations=perword,query_audio_use='headers only; no query PCM or teacher forward',
        scope='host callback mapping and metadata VAD; physical latency not independently verified')
    (OUT/'preparation_public.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PHRASE_PREPARATION_COMPLETE',len(expanded),'extended',flush=True)

if __name__=='__main__':main()
