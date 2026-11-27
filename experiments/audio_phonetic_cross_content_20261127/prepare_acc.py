"""147-channel nonoverlap IMU-origin frontend. Never opens Audio or its clocks."""
import json,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/vr261119_10x40_raw_acc'
PRIVATE=ROOT/Path(os.environ.get('PHONETIC_PRIVATE_OUT','outputs/vr261119_10x40_raw_acc/hierarchical_phonetic'))
OUT=ROOT/Path(os.environ.get('PHONETIC_PUBLIC_OUT','outputs/vr261127_audio_phonetic_cross_content'))
OPT=[0,1,2,4,5,6,7,10,11,13,14,15];DEV=[16,17,33,34]
REG=[20,21,22,23];CAL=[8,18,24,30,31]
SHORT=[25,26,27,28,29];LONG=[3,9,12,19,32];CHINESE=[35,36,37,38,39]
ROLES=dict(optimize=OPT,development=DEV,registration=REG,calibration=CAL,english_short=SHORT,english_long=LONG,chinese=CHINESE)
def fixed_starts(first,last):return first+.5+3*np.arange(max(0,int(np.floor((last-first-1.)/3))))
def spectrum(wave):
    wave=wave-wave.mean(1,keepdims=True)
    f,t,z=signal.stft(wave,1000,nperseg=100,noverlap=80,nfft=100,boundary=None,padded=False)
    assert np.array_equal(f[1:50],np.arange(10,491,10))
    return np.log(np.maximum(abs(z[:,1:50,:])**2,1e-20)).reshape(147,146),t

def main():
    PRIVATE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (PRIVATE/'acc_private.npz').exists() and (OUT/'data_audit_public.json').exists():
        audit=json.loads((OUT/'data_audit_public.json').read_text(encoding='utf-8'))
        assert audit['status']=='PASS' and hashlib.sha256((PRIVATE/'acc_private.npz').read_bytes()).hexdigest()==audit['data_sha256']
        assert (PRIVATE/'windows_private.json').exists() and (PRIVATE/'recordings_private.json').exists()
        print('IMMUTABLE_ACC_CACHE_REUSED;newrawdatarequiresaseparatestudydirectory',flush=True);return
    records=json.loads((BASE/'continuous_acc_activity/recordings_private.json').read_text(encoding='utf-8'))
    vocab=json.loads((ROOT/'outputs/vr261124_continuous_acc_activity/vocabulary_public.json').read_text(encoding='utf-8'))
    assert len(records)==400 and sorted(sum(ROLES.values(),[]))==list(range(40))
    xs=[];masks=[];u=[];w=[];ordinal=[];ids=[];starts_all=[];frame_times=[];manifest=[];audit=[];hashes=[]
    for counter,rec in enumerate(records):
      user,word=int(rec['user']),int(rec['word']);stem=Path(rec['stem']);role=next(k for k,v in ROLES.items() if word in v)
      assert rec['prompt'].strip().lower()==vocab[str(word)].strip().lower()
      rawpath=stem.with_name(stem.name+'_full_imu.csv')
      d=pd.read_csv(rawpath,usecols=['imu_sample_time','offscale_flags','accel_x','accel_y','accel_z','host_read_perf_counter_s'])
      latest=d.groupby('host_read_perf_counter_s').imu_sample_time.max();shift=float(np.median(latest.index.to_numpy()-latest.to_numpy()))
      d=d[d.offscale_flags==0].drop_duplicates('imu_sample_time').sort_values('imu_sample_time')
      tt=d.imu_sample_time.to_numpy()+shift;acc=d[['accel_x','accel_y','accel_z']].to_numpy()
      starts=fixed_starts(tt[0],tt[-1]);assert len(starts)>=2 and (np.diff(starts)>=3-1e-9).all()
      hashes.append(hashlib.sha256(rawpath.read_bytes()).hexdigest())
      for j,start in enumerate(starts):
        grid=start+np.arange(3000)/1000;wave=np.stack([np.interp(grid,tt,axis) for axis in acc.T]);xx,t=spectrum(wave)
        near=np.clip(np.searchsorted(tt,start+t),1,len(tt)-1);valid=(tt[near]-tt[near-1])<=.05;assert valid.any()
        xs.append(xx.astype('float32'));masks.append(valid);u.append(user);w.append(word);ordinal.append(j);ids.append(f'{user:02}_{word:02}_w{j:02}')
        starts_all.append(float(start));frame_times.append(start+t)
        manifest.append(dict(sample_id=ids[-1],user=user,word=word,role=role,record_group=str(stem),window_start=float(start),window_end=float(start+3),sample_grid_points=3000,valid_frames=int(valid.sum())))
      audit.append(dict(user=user,word=word,role=role,windows=len(starts),valid_acc=True,Audio_opened=False,Audio_timestamps_opened=False))
      if (counter+1)%50==0:print('RAW_IMU_RECORDINGS',counter+1,flush=True)
    x=np.array(xs);mask=np.array(masks);u=np.array(u);w=np.array(w);ordinal=np.array(ordinal)
    assert x.shape[1:]==(147,146) and len(set(ids))==len(ids) and len(set(hashes))==400 and np.isfinite(x).all()
    # Select exact same feasible K before any model results, first3 fixed windows perREGword.
    minimum=min(int(np.sum((u==us)&(w==word))) for us in range(1,11) for word in REG)
    per_word=min(3,minimum);k=4*per_word;regidx=np.array([[int(np.flatnonzero((u==us)&(w==word)&(ordinal==j))[0]) for word in REG for j in range(per_word)] for us in range(1,11)])
    teacher_allowed=np.isin(w,OPT+DEV)
    np.savez_compressed(PRIVATE/'acc_private.npz',acc=x,valid=mask,user=u,word=w,ordinal=ordinal,sample_id=np.array(ids),starts=np.array(starts_all),frame_times=np.array(frame_times),registration_indices=regidx)
    (PRIVATE/'windows_private.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
    (PRIVATE/'recordings_private.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    sameword_possible=all(int(np.sum((u==us)&(w==word)))>=k+1 for us in range(1,11) for word in SHORT+LONG+CHINESE)
    report=dict(status='PASS',users=10,recordings=400,actual_windows=len(x),nominal_utterances=3200,roles=ROLES,
      actual_role_windows={role:int(np.isin(w,words).sum()) for role,words in ROLES.items()},records=audit,
      input_channels=147,frequency_hz=list(range(10,491,10)),frames=146,stft_window_ms=100,stft_hop_ms=20,processing_hz=1000,
      source='rawhardwareaccelxyz;no trackingvelocity',window='firstvalidIMU+.5s thennonoverlap3sec,origins independentofAudio',
      unique_source_hashes=len(set(hashes)),registration_per_word=per_word,registration_K=k,registration_indices_before_model_results=True,
      role_record_group_overlap=0,window_overlap_same_record=0,Audio_files_opened=0,Audio_timestamp_files_opened=0,
      same_content_diagnostic_same_K_feasible=sameword_possible,same_content_reason='needsKseparate samewordREGwindows plusatleast1query peruser/word;notforce-reuseorpool',
      physical_sync_verified=False,data_sha256=hashlib.sha256((PRIVATE/'acc_private.npz').read_bytes()).hexdigest())
    (OUT/'data_audit_public.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'role_split_public.json').write_text(json.dumps(dict(roles=ROLES,vocabulary=vocab,source_plan_sha='65f31238c0e3eeed7e19bac79fd6f1bbe3d58ce9'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('ACC_PREPARATION_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['records','frequency_hz']}),flush=True)
if __name__=='__main__':main()
