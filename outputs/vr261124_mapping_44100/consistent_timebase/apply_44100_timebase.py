"""Materialize a consistent44100 timebase without rewriting source WAV/metadata."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'mapping_44100'
OUT=BASE/'consistent_timebase';OUT.mkdir(exist_ok=True)

def main():
    source=ROOT/'phrase_auth/source_manifest_private.json'
    old=json.loads(source.read_text(encoding='gb18030'))
    maps=json.loads((BASE/'segment_maps_private.json').read_text())
    recordings=json.loads((BASE/'recording_public.json').read_text())
    lookup={m['sample_id']:m for m in maps}
    recs={(r['user'],r['word']):r for r in recordings}
    assert len(old)==len(lookup)==3200
    records=[];oldframes=[];newframes=[];hosterrors=[];durations=[]
    for r in old:
        m=lookup[r['sample_id']];oldtime=np.array(m['audio_bounds_s'])
        newtime=oldtime*48000/44100
        before=np.rint(oldtime*48000).astype('int64');after=np.rint(newtime*44100).astype('int64')
        assert np.array_equal(before,after) and after[1]>after[0]>=0
        us,word,_=map(int,r['sample_id'].split('_'));rec=recs[us,word]
        assert after[1]<=rec['wav_frames']
        assert np.allclose(newtime,m['audio_bounds_44100_consistent_s'],atol=1e-12,rtol=0)
        hb=np.array(m['host_bounds_44100_consistent']);prior=np.array(m['host_bounds_48000'])
        hosterrors.append(float(abs(hb-prior).max()))
        item=dict(r)
        # Replace the legacy duration so it cannot be silently interpreted as44100 time.
        item['audio_duration_original_48000_s']=item.pop('audio_duration_s')
        item.update(audio_timebase_hz=44100,audio_start_s=float(newtime[0]),audio_end_s=float(newtime[1]),
            audio_duration_s=float(item['audio_duration_original_48000_s']*48000/44100),
            full_audio_frame_bounds=after.tolist(),imu_interval=hb.tolist(),
            historical_audio_bounds_48000_s=oldtime.tolist(),physical_sync_verified=False)
        records.append(item);oldframes.append(before);newframes.append(after);durations.append(item['audio_duration_s'])
    assert max(hosterrors)<1e-9
    manifest=dict(schema='hc-isec-audio-acc-timebase-v1',interpreted_audio_rate_hz=44100,
        source_wav_header_rate_hz=48000,conversion_factor=48000/44100,
        rule='seconds44100 = seconds48000 *48000/44100;frame offsets invariant',
        physical_sync_verified=False,source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=records)
    (OUT/'aligned_manifest_private.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf8')
    np.savez_compressed(OUT/'aligned_intervals_private.npz',sample_id=np.array([r['sample_id'] for r in records]),
        audio_frame_bounds=np.array(newframes),audio_time_bounds_s=np.array([[r['audio_start_s'],r['audio_end_s']] for r in records]),
        imu_host_bounds_s=np.array([r['imu_interval'] for r in records]),interpreted_audio_rate_hz=np.array(44100))
    verification=dict(status='PASS',records=3200,recording_groups=400,audio_timebase_hz=44100,seconds_multiplier=48000/44100,
        pcm_frame_boundaries_identical=True,max_host_boundary_change_ms=max(hosterrors)*1000,
        duration_median_s=float(np.median(durations)),source_wav_or_metadata_modified=False,
        historical_features_or_models_modified=False,physical_sync_verified=False,
        usage='new offline inputs must read this manifest44100 timebase explicitly;old loaders still use48000 and are not automatically changed')
    (OUT/'verification_public.json').write_text(json.dumps(verification,indent=2))
    users=[dict(user=us,records=sum(int(r['sample_id'].split('_')[0])==us for r in records),
        median_duration_44100_s=float(np.median([r['audio_duration_s'] for r in records if int(r['sample_id'].split('_')[0])==us]))) for us in range(1,11)]
    (OUT/'per_user_public.json').write_text(json.dumps(users,indent=2))
    (OUT/'report.md').write_text('# 44.1kHz一致时间基落盘\n\n全部400组、3200段已同步换算。旧秒数乘48000/44100，帧偏移统一除44100。已生成私有aligned_manifest和aligned_intervals，明确记录44100时间基、原48000秒数、原始帧边界及主机IMU边界。\n\n逐段检查：3200段PCM帧边界全部一致，主机边界最大变化0ms。时长统一增加8.8435%。原始WAV头、采集日志、历史模型及特征均未修改；下游需明确读取新manifest，不得将44100秒数再与旧48000帧轴混用。\n\n这落实的是一致单位换算，不代表已校准物理延迟。采样率解释仍需独立采集核验。逐用户时长见per_user_public.json。\n',encoding='utf8')
    print(json.dumps(verification),flush=True)

if __name__=='__main__':main()
