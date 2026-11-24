import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mapping_44100/rebuilt_audio';OUT.mkdir(exist_ok=True)
m=json.loads((ROOT/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf8'))['records']
d=np.load(ROOT/'phrase_auth/frontend_private.npz');fit=d['fit'];assert [r['sample_id'] for r in m]==list(d['sample_id'])
indices=np.flatnonzero(fit);waves=np.zeros((1280,48000),dtype='float32');targets=np.zeros((1280,26,12),dtype='float32');dur=[]
for j,i in enumerate(indices):
 fs,pcm=wavfile.read(m[i]['audio_file']);assert fs==48000 and pcm.dtype==np.int16
 pcm=pcm.astype('float32')/32768
 if pcm.ndim==2:pcm=pcm[:,0]
 corrected=signal.resample_poly(pcm,160,441).astype('float32')
 assert len(corrected)<=48000
 start=(48000-len(corrected))//2;waves[j,start:start+len(corrected)]=corrected;dur.append(len(pcm)/44100)
 f,t,z=signal.stft(waves[j],16000,nperseg=1600,noverlap=1280,nfft=1600,boundary=None,padded=False)
 bands=np.stack([np.interp(np.arange(50,301,10),f,col) for col in (abs(z)**2).T],1)
 lp=np.log(np.maximum(bands,1e-12))
 for k in range(12):targets[j,:,k]=lp[:,(t>=k*.25)&(t<(k+1)*.25)].mean(1)
 if (j+1)%320==0:print('REBUILD',j+1,flush=True)
np.savez_compressed(OUT/'training_audio_44100_private.npz',wave=waves,local_target=targets,indices=indices,sample_id=d['sample_id'][fit],sample_rate=np.array(16000),interpreted_source_rate=np.array(44100),duration_s=np.array(dur))
# Predetermined first training recording; no visual/performance selection.
i=int(indices[0]);rec=m[i];fs,pcm=wavfile.read(rec['audio_file']);pcm=pcm.astype('float64')/32768
if pcm.ndim==2:pcm=pcm[:,0]
old=signal.resample_poly(pcm,1,3);new=signal.resample_poly(pcm,160,441)
stem=Path(rec['record_group']);raw=pd.read_csv(stem.with_name(stem.name+'_full_imu.csv'),usecols=['host_read_perf_counter_s','imu_sample_time','offscale_flags','accel_x','accel_y','accel_z'])
latest=raw.groupby('host_read_perf_counter_s').imu_sample_time.max();offset=np.median(latest.index.to_numpy()-latest.to_numpy())
raw=raw[raw.offscale_flags==0].drop_duplicates('imu_sample_time').sort_values('imu_sample_time');ht=raw.imu_sample_time.to_numpy()+offset
lo,hi=rec['imu_interval'];grid=lo+np.arange(int(round((hi-lo)*1000)))/1000
acc=np.stack([np.interp(grid,ht,raw[c]) for c in ['accel_x','accel_y','accel_z']]);acc-=acc.mean(1,keepdims=True)
fig,axes=plt.subplots(3,2,figsize=(13,9),constrained_layout=True)
for row,(name,z,sr,window,hop) in enumerate([('Audio interpreted at48kHz (old)',old,16000,2048,256),('Audio interpreted at44.1kHz (rebuilt)',new,16000,2048,256),('Raw ACC:sum of3axis powers',acc,1000,128,16)]):
 f,t,st=signal.stft(z,sr,nperseg=window,noverlap=window-hop,boundary='zeros',padded=True)
 power=abs(st)**2
 if power.ndim==3:power=power.sum(0)
 db=10*np.log10(np.maximum(power,1e-30)/max(power.max(),1e-30))
 for col,limit in enumerate([500,2000 if row<2 else 500]):
  use=f<=limit;im=axes[row,col].pcolormesh(t,f[use],db[use],shading='auto',vmin=-60,vmax=0,cmap='magma')
  axes[row,col].set(title=name+(' / low band' if col==0 else ' / wider band'),xlabel='Seconds from segment start',ylabel='Hz',xlim=(0,max(len(new)/16000,hi-lo)))
  fig.colorbar(im,ax=axes[row,col],label='dB relative to panel peak')
fig.suptitle('Fixed training sample '+rec['sample_id']+' | same PCM samples; nominal host alignment, physical latency unverified')
fig.savefig(OUT/'spectrogram_comparison_private.png',dpi=150);plt.close(fig)
summary=dict(training_records=1280,nontraining_audio_read=0,source_interpretation_hz=44100,source_wav_header_hz=48000,resampled_hz=16000,
 duration_scale=48000/44100,frequency_scale=44100/48000,plot_sample_id=rec['sample_id'],old_duration_s=len(pcm)/48000,new_duration_s=len(pcm)/44100,acc_duration_s=hi-lo,
 original_files_modified=False,physical_sync_verified=False,local_target='50to300Hz/10Hz interpolated bins,100msSTFT/20ms hop,12coarse250msbins;3s centerpadding retains historical segmentation cues')
(OUT/'summary_public.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
