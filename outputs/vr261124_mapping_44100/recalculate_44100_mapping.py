import json,wave
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mapping_44100';OUT.mkdir(exist_ok=True)
m=json.loads((ROOT/'phrase_auth/source_manifest_private.json').read_text(encoding='gb18030'))
groups={}
for i,v in enumerate(m):groups.setdefault(v['record_group'],[]).append(i)
rows=[];maps=[];rates={}
for count,(stem,ix) in enumerate(groups.items()):
 p=Path(stem);meta=json.loads(p.with_name(p.name+'_meta.json').read_text(encoding='utf-8-sig'))
 with wave.open(str(p.parent/meta['full_audio'])) as af:fs=af.getframerate();nf=af.getnframes()
 rates[str(fs)]=rates.get(str(fs),0)+1
 a=pd.read_csv(p.with_name(p.name+'_audio_timestamps.csv'));frames=(a.frame_offset+a.frames).to_numpy();t=a.host_callback_perf_counter_s.to_numpy()
 segments={Path(s['audio_file']).name:s for s in meta['segments']}
 row=dict(reported_rate=meta.get('audio',{}).get('reported_rate_hz'),requested_rate=meta.get('audio',{}).get('requested_rate_hz'),timestamp_end_frame=int(frames[-1]),timestamp_frames_match_wav=bool(frames[-1]==nf),user=int(m[ix[0]]['sample_id'].split('_')[0]),word=int(m[ix[0]]['sample_id'].split('_')[1]),wav_rate=fs,wav_frames=nf)
 for rate in (44100,48000):
  x=frames/rate;co=np.polyfit(x-x[0],t-t[0],1);err=t-t[0]-np.polyval(co,x-x[0]);row[str(rate)]=dict(slope=float(co[0]),effective_frame_rate=float(rate/co[0]),residual_p95_ms=float(np.quantile(abs(err),.95)*1000),reference_audio_s=float(x[0]),reference_host_s=float(t[0]),intercept_relative_s=float(co[1]))
 for i in ix:
  s=segments[Path(m[i]['audio_file']).name];bounds=np.array([s['start_audio_s'],s['end_audio_s']]);old=np.interp(bounds,frames/48000,t);new=np.interp(bounds,frames/44100,t);coherent=np.interp(bounds*48000/44100,frames/44100,t)
  maps.append(dict(sample_id=m[i]['sample_id'],audio_bounds_s=bounds.tolist(),audio_bounds_44100_consistent_s=(bounds*48000/44100).tolist(),host_bounds_44100_consistent=coherent.tolist(),consistent_max_error_s=float(abs(coherent-old).max()),host_bounds_44100=new.tolist(),host_bounds_48000=old.tolist(),center_shift_ms=float((new.mean()-old.mean())*1000),outside_44100=bool(bounds[0]<frames[0]/44100 or bounds[1]>frames[-1]/44100)))
 rows.append(row)
 if (count+1)%80==0:print('REMAP',count+1,flush=True)
summary=dict(recording_groups=len(rows),segments=len(maps),wav_rate_counts=rates,requested_mapping_rate=44100,
 note='44100 is a sensitivity recomputation,not proof of hardware clock;metadata segment seconds left as recorded;affine residual invariant under rate rescaling',
 by_rate={str(rate):dict(slope_median=float(np.median([r[str(rate)]['slope'] for r in rows])),effective_frame_rate_median=float(np.median([r[str(rate)]['effective_frame_rate'] for r in rows])),residual_p95_median_ms=float(np.median([r[str(rate)]['residual_p95_ms'] for r in rows])),residual_p95_max_ms=max(r[str(rate)]['residual_p95_ms'] for r in rows)) for rate in (44100,48000)},
 candidate_segment_center_shift_ms=dict(median=float(np.median([r['center_shift_ms'] for r in maps])),min=min(r['center_shift_ms'] for r in maps),max=max(r['center_shift_ms'] for r in maps)),outside_candidate_range=sum(r['outside_44100'] for r in maps),consistent_mapping_max_difference_ms=max(r['consistent_max_error_s'] for r in maps)*1000,timestamp_frames_match_wav_groups=sum(r['timestamp_frames_match_wav'] for r in rows),reported_rate_counts={str(rate):sum(r['reported_rate']==rate for r in rows) for rate in set(r['reported_rate'] for r in rows)},historical_caches_modified=False)
(OUT/'summary_public.json').write_text(json.dumps(summary,indent=2));(OUT/'recording_public.json').write_text(json.dumps(rows));(OUT/'segment_maps_private.json').write_text(json.dumps(maps));print(json.dumps(summary),flush=True)
