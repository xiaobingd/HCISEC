"""Prepare fixed3s physical ACC windows and FIT-ONLY continuous Audio targets.
Nominal host-clock mapping remains UNVERIFIED. No per-record lag optimization.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'conditional_correspondence/coarse_pilot'

def main():
    OUT.mkdir(exist_ok=True)
    d=np.load(ROOT/'phrase_auth/frontend_private.npz');ids=d['sample_id'];fit=d['fit']
    manifest=json.loads((ROOT/'phrase_auth/source_manifest_private.json').read_text(encoding='gb18030'))
    assert [v['sample_id'] for v in manifest]==list(ids)
    groups={}
    for i,v in enumerate(manifest):groups.setdefault(v['record_group'],[]).append(i)
    acc=np.zeros((3200,3,3000),dtype='float32');target=np.zeros((3200,26,12),dtype='float32');edges=[]
    bands=np.arange(50,301,10)
    for count,(stem,indices) in enumerate(groups.items()):
        p=Path(stem);meta=json.loads(p.with_name(p.name+'_meta.json').read_text(encoding='utf-8-sig'))
        raw=pd.read_csv(p.with_name(p.name+'_full_imu.csv'),usecols=['host_read_perf_counter_s','imu_sample_time','offscale_flags','accel_x','accel_y','accel_z'])
        latest=raw.groupby('host_read_perf_counter_s').imu_sample_time.max()
        shift=np.median(latest.index.to_numpy()-latest.to_numpy())
        raw=raw[raw.offscale_flags==0].drop_duplicates('imu_sample_time').sort_values('imu_sample_time')
        time=raw.imu_sample_time.to_numpy()+shift;xyz=raw[['accel_x','accel_y','accel_z']].to_numpy()
        hasfit=bool(fit[indices].any());assert hasfit==bool(fit[indices].all())
        if hasfit:
            fs,pcm=wavfile.read(p.parent/meta['full_audio'])
            assert pcm.dtype==np.int16
            pcm=pcm.astype('float32')/32768
            if pcm.ndim==2:pcm=pcm[:,0]
            assert fs==48000
        segments={Path(s['audio_file']).name:s for s in meta['segments']}
        for i in indices:
            v=manifest[i];center=np.mean(v['imu_interval']);grid=center+np.arange(3000)/1000-1.5
            z=np.stack([np.interp(grid,time,c,left=0,right=0) for c in xyz.T]);z-=z.mean(1,keepdims=True)
            acc[i]=z
            if grid[0]<time[0] or grid[-1]>time[-1]:edges.append(int(i))
            if hasfit:
                s=segments[Path(v['audio_file']).name];ac=(s['start_audio_s']+s['end_audio_s'])/2
                offsets=np.rint((ac+np.arange(144000)/48000-1.5)*48000).astype(int)
                valid=(offsets>=0)&(offsets<len(pcm));wave=np.zeros(144000,dtype='float32');wave[valid]=pcm[offsets[valid]]
                wave=signal.resample_poly(wave,1,3)
                f,t,st=signal.stft(wave,16000,nperseg=1600,noverlap=1280,nfft=1600,boundary=None,padded=False)
                power=abs(st)**2
                vals=np.stack([np.interp(bands,f,col) for col in power.T],1)
                logp=np.log(np.maximum(vals,1e-12))
                #12fixed250ms bins; no per-record stretching or VAD mask.
                for j in range(12):
                    use=(t>=j*.25)&(t<(j+1)*.25)
                    target[i,:,j]=logp[:,use].mean(1)
        if (count+1)%40==0:print('CONDITIONAL_RAW',count+1,flush=True)
    assert len(groups)==400 and fit.sum()==1280 and np.isfinite(acc).all() and np.isfinite(target).all()
    np.savez_compressed(OUT/'raw_inputs_private.npz',acc=acc,audio_target=target,user=d['user'],word=d['word'],repeat=d['repeat'],fit=fit,sample_id=ids)
    (OUT/'preparation_public.json').write_text(json.dumps(dict(status='PASS',users=10,records=3200,training_audio_records=1280,
        nontraining_audio_pcm_read=0,acc_window_seconds=3,audio_frequency_bins_hz=bands.tolist(),audio_time_bins=12,
        physical_sync_verified=False,nominal_alignment='existing host callback mapping plus median IMU packet offset;no lag search',
        vad_mask_applied=False,center='historical VAD midpoint;center timing may still carry content/duration information',
        windows_intersect_recording_edge=len(edges),query_audio_teacher_features=False),indent=2))
    print('CONDITIONAL_RAW_PREPARED',flush=True)

if __name__=='__main__':main()
