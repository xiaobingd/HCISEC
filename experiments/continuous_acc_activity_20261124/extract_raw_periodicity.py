"""Fixed raw-wave periodicity extraction. Only historical fit-word Audio files."""
import json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fft import rfft,irfft,rfftfreq
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/vr261119_10x40_raw_acc'
PRIVATE=BASE/'periodicity';OUT=ROOT/'outputs/vr261125_raw_periodicity'

def track(wave,rate):
    wave=np.atleast_2d(wave).astype(float)
    wave=signal.resample_poly(wave,4000,rate,axis=-1)
    wave=signal.sosfiltfilt(signal.butter(4,[40,450],fs=4000,btype='bandpass',output='sos'),wave,axis=-1)
    frames=np.lib.stride_tricks.sliding_window_view(wave,240,axis=-1)[:,::80,:].copy()
    frames-=frames.mean(-1,keepdims=True)
    lags=np.arange(10,67)
    numerator=irfft(abs(rfft(frames,n=512,axis=-1))**2,n=512,axis=-1).sum(0)[:,lags]
    sums=np.pad(np.cumsum(frames*frames,axis=-1),((0,0),(0,0),(1,0))).sum(0)
    denominator=np.sqrt(sums[:,240-lags]*(sums[:,240,None]-sums[:,lags]))
    corr=numerator/np.maximum(denominator,1e-20);peak=np.argmax(corr,1)
    # Earliest local maximum passing fixed .6, otherwise global max.
    local=(corr[:,1:-1]>=corr[:,:-2])&(corr[:,1:-1]>=corr[:,2:])&(corr[:,1:-1]>=.6)
    found=local.any(1);peak[found]=np.argmax(local[found],1)+1
    confidence=corr[np.arange(len(corr)),peak];lag=lags[peak].astype(float)
    mid=(peak>0)&(peak<len(lags)-1);ii=np.flatnonzero(mid);pp=peak[ii]
    den=corr[ii,pp-1]-2*corr[ii,pp]+corr[ii,pp+1]
    lag[ii]+=np.clip(.5*(corr[ii,pp-1]-corr[ii,pp+1])/np.minimum(den,-1e-10),-.5,.5)
    f0=4000/lag;energy=(frames*frames).mean((0,2))
    voiced=found&(confidence>=.6)&(energy>max(energy.max()*1e-4,1e-20))
    power=(abs(rfft(frames*np.hanning(240),n=2048,axis=-1))**2).sum(0)
    freq=rfftfreq(2048,1/4000);harm=[];h1=[]
    for t,fund in enumerate(f0):
        mask=np.zeros(len(freq),bool);first=abs(freq-fund)<=15
        for h in range(1,int(450/fund)+1):mask|=abs(freq-h*fund)<=15
        total=power[t,(freq>=40)&(freq<=450)].sum()+1e-20
        harm.append(power[t,mask].sum()/total);h1.append(power[t,first].sum()/total)
    return np.c_[f0,confidence,voiced.astype(float),harm,h1]

def summarize(t):
    v=t[:,2]>0
    if v.any():
        pitch=np.log2(t[v,0]);q=np.quantile(pitch,[.1,.5,.9]);p=np.r_[q,pitch.std(),t[v,1].mean(),v.mean()]
        h=np.r_[np.median(t[v,3]),np.std(t[v,3]),np.median(t[v,4]),np.std(t[v,4])]
    else:p=np.r_[np.zeros(5),0.];h=np.zeros(4)
    return p,h

def main():
    PRIVATE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    raw=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');idx=np.flatnonzero(raw['fit']);assert len(idx)==1280
    rawacc=raw['acc'];rawfit=raw['fit'];rawids=raw['sample_id']
    manifest=json.loads((BASE/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
    assert list(rawids)==[v['sample_id'] for v in manifest]
    allowed={manifest[int(i)]['record_group'] for i in idx}
    excluded={v['record_group'] for i,v in enumerate(manifest) if not rawfit[i]};assert allowed.isdisjoint(excluded)
    protocol=dict(samples=1280,Audio_recordings=160,band_hz=[40,450],f0_search_hz=[60,400],frame_ms=60,hop_ms=20,
      method='summed-axis normalized lagged autocorrelation; earliest local maximum >=.6, else global max; parabola lag refinement; bandlimited periodicity estimate,not gold F0/YIN',
      voicing='local autocorrelation maximum in search interior AND confidence>=.6 AND framepower>maxpower*1e-4;fixed,notlearnedVAD;reject boundary fallback to avoid50Hz->400Hz alias',
      harmonic='H1 and up-to450Hz harmonic-neighborhood(+/-15Hz) power fractions; limited number ofharmonics at highF0',
      descriptor='periodicity6:log2F0 Q10/Q50/Q90/std,voicedconfidence mean,voicedfraction;harmonic4:median/std ofharmonicfraction andH1fraction',
      audio_clock='legacy PCM48000 header interpreted44100,resampled4k;unverified physical samplerate;F0 scale conditional onthisassumption',
      inference='noAudio-basedACCcrop;3sec legacy alignedwindows mayoverlap;identity analysis descriptor-level,notphase alignment',
      fit_only=True,final_Audio_opened=False,physical_sync_verified=False)
    (OUT/'extraction_public.json').write_text(json.dumps(protocol,indent=2))
    groups={}
    for pos,i in enumerate(idx):groups.setdefault(manifest[int(i)]['record_group'],[]).append((pos,int(i)))
    ap=np.zeros((1280,6));ah=np.zeros((1280,4));xp=ap.copy();xh=ah.copy();tracksA=[];tracksX=[]
    for pos,i in enumerate(idx):
        t=track(rawacc[i],1000);xp[pos],xh[pos]=summarize(t);tracksX.append(t)
        if (pos+1)%320==0:print('ACC_WINDOWS',pos+1,flush=True)
    tracksA=[None]*1280
    for count,(stem,records) in enumerate(groups.items()):
        assert stem in allowed and stem not in excluded
        path=Path(stem);meta=json.loads(path.with_name(path.name+'_meta.json').read_text(encoding='utf-8-sig'))
        rate,pcm=wavfile.read(path.parent/meta['full_audio']);assert rate==48000 and pcm.dtype==np.int16
        pcm=pcm.astype(float)/32768
        if pcm.ndim==2:pcm=pcm[:,0]
        for pos,i in records:
            rec=manifest[i];assert rec['audio_timebase_hz']==44100
            center=(rec['audio_start_s']+rec['audio_end_s'])/2
            frames=round(center*44100)+np.arange(132300)-66150
            valid=(frames>=0)&(frames<len(pcm));wave=np.zeros(132300);wave[valid]=pcm[frames[valid]]
            t=track(wave,44100);ap[pos],ah[pos]=summarize(t);tracksA[pos]=t
        if (count+1)%40==0:print('AUDIO_GROUPS',count+1,flush=True)
    np.savez_compressed(PRIVATE/'features_private.npz',audio_periodicity=ap,audio_harmonics=ah,acc_periodicity=xp,acc_harmonics=xh,
      audio_track=np.array(tracksA),acc_track=np.array(tracksX),indices=idx,sample_id=raw['sample_id'][idx],user=raw['user'][idx],word=raw['word'][idx],repeat=raw['repeat'][idx])
    print('PERIODICITY_EXTRACTION_COMPLETE',flush=True)
if __name__=='__main__':main()
