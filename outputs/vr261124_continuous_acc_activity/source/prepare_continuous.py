"""IMU-origin non-overlapping windows. Test-word Audio and timestamps never opened."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

BANDS=[(20,50),(50,80),(80,120),(120,180),(180,250),(250,350),(350,450)]
def fixed_starts(first,last):return first+.5+3*np.arange(max(0,int(np.floor((last-first-1.)/3))))
def spectrum(wave):
    wave=wave-wave.mean(1,keepdims=True);f,t,z=signal.stft(wave,1000,nperseg=100,noverlap=80,nfft=100,boundary=None,padded=False)
    power=(abs(z)**2).sum(0)
    bands=np.stack([np.log(np.maximum(power[(f>=lo)&(f<hi)].mean(0),1e-20)) for lo,hi in BANDS],1)
    motion=np.log(np.maximum(power[f<20].mean(0),1e-20));return bands,motion,t
def main(root):
    out=root/'continuous_acc_activity';records=json.loads((out/'recordings_private.json').read_text(encoding='utf-8'));split=json.loads((out/'word_split_public.json').read_text())
    split_hash=hashlib.sha256((out/'word_split_public.json').read_bytes()).hexdigest()
    xs=[];ms=[];labels=[];valid=[];users=[];words=[];ordinals=[];ids=[];origins=[];audit=[];pcm_read=[]
    for count,rec in enumerate(records):
      us,word=rec['user'],rec['word'];p=Path(rec['stem'])
      # Only raw IMU controls origin, count, and boundaries, before any Audio read.
      imu=pd.read_csv(p.with_name(p.name+'_full_imu.csv'),usecols=['imu_sample_time','offscale_flags','accel_x','accel_y','accel_z','host_read_perf_counter_s'])
      latest=imu.groupby('host_read_perf_counter_s').imu_sample_time.max();shift=float(np.median(latest.index.to_numpy()-latest.to_numpy()))
      imu=imu[imu.offscale_flags==0].drop_duplicates('imu_sample_time').sort_values('imu_sample_time')
      times=imu.imu_sample_time.to_numpy()+shift;acc=imu[['accel_x','accel_y','accel_z']].to_numpy();starts=fixed_starts(times[0],times[-1]);assert len(starts)>=2
      assert np.all(np.diff(starts)>=3.-1e-9)
      pos=[];centers=[]
      for j,start in enumerate(starts):
        grid=start+np.arange(3000)/1000;wave=np.stack([np.interp(grid,times,col) for col in acc.T]);x,m,t=spectrum(wave)
        near=np.clip(np.searchsorted(times,start+t),1,len(times)-1);frame_valid=(times[near]-times[near-1])<=.05
        xs.append(x.astype('float32'));ms.append(m.astype('float32'));valid.append(frame_valid);labels.append(np.full(146,np.nan,dtype='float32'))
        users.append(us);words.append(word);ordinals.append(j);ids.append(f'{us:02}_{word:02}_w{j:02}');origins.append(float(start));pos.append(len(xs)-1);centers.append(start+t)
      role=next(k for k in ['train','development','test'] if word in split[k]);row=dict(user=us,word=word,role=role,windows=len(starts),source='IMUoriginfixedgrid',Audio_opened=False,Audio_timestamp_opened=False)
      if role!='test':
        meta=json.loads(p.with_name(p.name+'_meta.json').read_text(encoding='utf-8-sig'))
        clock=pd.read_csv(p.with_name(p.name+'_audio_timestamps.csv'));frame_end=(clock.frame_offset+clock.frames).to_numpy(dtype='float64');host=clock.host_callback_perf_counter_s.to_numpy()
        slope,offset=np.polyfit(frame_end-frame_end[0],host-host[0],1);intercept=host[0]+offset-slope*frame_end[0];assert slope>0
        residual=host-(intercept+slope*frame_end)
        header,pcm=wavfile.read(p.parent/meta['full_audio']);assert header==48000 and pcm.dtype==np.int16
        pcm=pcm.astype('float32')/32768
        if pcm.ndim==2:pcm=pcm[:,0]
        resampled=signal.resample_poly(pcm,160,441)
        f,tt,z=signal.stft(resampled,16000,nperseg=1600,noverlap=1280,nfft=1600,boundary=None,padded=False)
        energy=np.log(np.maximum((abs(z[(f>=300)&(f<3500)])**2).mean(0),1e-20));audio_host=intercept+slope*(tt*44100)
        threshold=max(float(np.quantile(energy,.2)+.25*(np.quantile(energy,.9)-np.quantile(energy,.2))),float(np.log(1e-12)))
        for ix,center in zip(pos,centers):
          measured=np.interp(center,audio_host,energy,left=np.nan,right=np.nan);truth=(measured>=threshold).astype('float32');truth[~np.isfinite(measured)]=np.nan;labels[ix]=truth
        row.update(Audio_opened=True,Audio_timestamp_opened=True,estimated_audio_hz=float(1/slope),callback_residual_p95_ms=float(np.quantile(abs(residual),.95)*1000))
        pcm_read.append((us,word))
      audit.append(row)
      if (count+1)%50==0:print('CONTINUOUS_RECORDINGS',count+1,flush=True)
    u=np.array(users);w=np.array(words);y=np.array(labels);test=np.isin(w,split['test'])
    assert np.isnan(y[test]).all() and len(pcm_read)==320 and not any(word in split['test'] for _,word in pcm_read)
    np.savez_compressed(out/'continuous_private.npz',acc=np.array(xs),motion=np.array(ms),audio_proxy=y,imu_valid=np.array(valid),user=u,word=w,ordinal=np.array(ordinals),sample_id=np.array(ids),window_start_host=np.array(origins))
    (out/'preparation_audit_public.json').write_text(json.dumps(dict(word_split_sha256=split_hash,recordings=audit,windows=len(xs),test_windows=int(test.sum()),Audio_recordings_read=320,test_Audio_recordings_read=0,test_Audio_timestamp_files_read=0,window_seconds=3,hop_seconds=3,Audio_VAD_used_for_windowing=False,physical_delay_calibrated=False),indent=2))
    print('CONTINUOUS_PREPARATION_COMPLETE',len(xs),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
