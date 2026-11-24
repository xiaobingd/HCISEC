import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'conditional_correspondence';OUT.mkdir(exist_ok=True)
d=np.load(ROOT/'paired_raw_acc_private.npz');f=np.load(ROOT/'phrase_auth/frontend_private.npz')
manifest=json.loads((ROOT/'phrase_auth/source_manifest_private.json').read_text(encoding='gb18030'))
assert np.array_equal(d['sample_id'],f['sample_id'])
fit=f['fit'];rows=[];groups={};waves=d['wave'];spectra=d['spectrum']
for i in np.flatnonzero(fit):groups.setdefault(manifest[i]['record_group'],[]).append(int(i))
retr=[]
for count,(stem,indices) in enumerate(groups.items()):
 p=Path(stem);a=pd.read_csv(p.with_name(p.name+'_audio_timestamps.csv'))
 t=a.host_callback_perf_counter_s.to_numpy();x=(a.frame_offset+a.frames).to_numpy()/48000
 coef=np.polyfit(x-x[0],t-t[0],1);err=(t-t[0])-np.polyval(coef,x-x[0])
 imu=pd.read_csv(p.with_name(p.name+'_full_imu.csv'),usecols=['host_read_perf_counter_s','imu_sample_time'])
 latest=imu.groupby('host_read_perf_counter_s').imu_sample_time.max()
 packet=latest.index.to_numpy()-latest.to_numpy();packet-=np.median(packet)
 frames=a.frame_offset.to_numpy();lens=a.frames.to_numpy()
 row=dict(user=int(f['user'][indices[0]]),word=int(f['word'][indices[0]]),audio_slope=float(coef[0]),audio_residual_p95_ms=float(np.quantile(abs(err),.95)*1000),imu_packet_offset_p95_ms=float(np.quantile(abs(packet),.95)*1000),audio_offsets_contiguous=bool(np.all(frames[1:]==frames[:-1]+lens[:-1])))
 rows.append(row)
 # Nonlearned energy-profile retrieval: same identity/word, eight candidate recordings.
 audio=[];acc=[]
 for i in indices:
  wav=waves[i].astype('float64');freq,tt,z=signal.stft(wav,16000,nperseg=4096,noverlap=3072,boundary=None,padded=False)
  env=np.log(np.maximum((abs(z[freq<=300])**2).sum(0),1e-20))
  q=np.exp(spectra[i].astype('float64')).sum((0,1));q=np.log(np.maximum(q,1e-20))
  assert len(env)==len(q)==28
  audio.append(env);acc.append(q)
 def normalize(z):
  z=np.array(z);z-=z.mean(1,keepdims=True);return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
 aa=normalize(audio);xx=normalize(acc);sim=xx@aa.T
 # All candidates are evaluated identically; no per-pair lag optimization.
 ranks=(sim>np.diag(sim)[:,None]).sum(1)+1
 retr.append(dict(user=row['user'],word=row['word'],top1=float(np.mean(ranks==1)),mean_rank=float(ranks.mean()),diagonal_cosine=float(np.diag(sim).mean()),offdiagonal_cosine=float(sim[~np.eye(8,dtype=bool)].mean())))
 if (count+1)%20==0:print('SYNC_AUDIT',count+1,flush=True)
(OUT/'recording_sync_public.json').write_text(json.dumps(rows,indent=2))
(OUT/'conditional_energy_retrieval_public.json').write_text(json.dumps(retr,indent=2))
summary=dict(groups=len(rows),records=int(fit.sum()),scope='16training words only;no development/finalAudio;untrained energy diagnostic',physical_sync_verified=False,audio_residual_p95_ms={k:float(v) for k,v in zip(['median','max'],[np.median([r['audio_residual_p95_ms'] for r in rows]),max(r['audio_residual_p95_ms'] for r in rows)])},imu_packet_p95_ms={k:float(v) for k,v in zip(['median','max'],[np.median([r['imu_packet_offset_p95_ms'] for r in rows]),max(r['imu_packet_offset_p95_ms'] for r in rows)])},conditional_top1=float(np.mean([r['top1'] for r in retr])),chance_top1=.125,conditional_margin=float(np.mean([r['diagonal_cosine']-r['offdiagonal_cosine'] for r in retr])),note='Host-clock residual is not physical latency; metadata VAD and duration can aid energy retrieval;not evidence of learned correspondence or identity gain')
(OUT/'sync_summary_public.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


