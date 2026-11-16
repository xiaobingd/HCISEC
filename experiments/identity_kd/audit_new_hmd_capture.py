"""Read-only local raw capture audit. Audio/IMU traces and source maps stay private."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse
import csv
import hashlib
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter,sosfiltfilt,welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORDS={'0':'zero','1':'one','2':'two','3':'three','4':'four'}
def normalized_prompt(value):
    text=str(value).strip().lower()
    return {**WORDS,'5':'five','6':'six','7':'seven','8':'eight','9':'nine'}.get(text,text)
def readjson(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def payload_hash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def normalized(w):
    if np.issubdtype(w.dtype,np.integer):
        if w.dtype==np.uint8:return (w.astype(float)-128)/128
        return w.astype(float)/max(abs(np.iinfo(w.dtype).min),np.iinfo(w.dtype).max)
    return w.astype(float)
def blocks(x,n):
    usable=(len(x)//n)*n
    return np.sqrt(np.mean(x[:usable].reshape(-1,n,*x.shape[1:])**2,axis=1))

def metadata_only(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    private_path=out/'corrected_source_manifest_private.json'
    prior=readjson(private_path) if private_path.exists() else []
    aliases={str(Path(p)):r['recording'] for r in prior for p in r.get('source_file_sha256',{}) if p.endswith('_meta.json')}
    rows=[];sources=[];subjects=set();new=0
    for path in sorted(root.glob('*/*_meta.json')):
        m=readjson(path);alias=aliases.get(str(path))
        if alias is None:new+=1;alias=f'F{new:02d}'
        subjects.add(str(m.get('subject_id')))
        rows.append(dict(recording=alias,payload_previously_audited=str(path) in aliases,
            declared_word=normalized_prompt(m.get('prompt_text')),status=m.get('status'),
            capture_wall_duration_s=m.get('capture_wall_duration_s'),
            audio_duration_s=m.get('audio',{}).get('audio_duration_s'),
            imu_span_s=m.get('imu',{}).get('sample_span_s'),
            overflow_callbacks=m.get('audio',{}).get('overflow_callbacks'),
            vad_detected=m.get('vad',{}).get('detected'),
            integrity=m.get('capture_integrity_pass')))
        sources.append(dict(recording=alias,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(scope='metadata-only followup;do not add counts to earlier payload snapshot',
        completed_at_bj=datetime.now(timezone(timedelta(hours=8))).isoformat(),records=rows,
        metadata_identity_count=len(subjects),unmatched_path_metadata_records=new,
        path_match_limit='unmatched paths may be new recordings or renamed folders;not independent new sample count')
    dump(out/'metadata_followup_public.json',result);dump(out/'metadata_followup_sources_private.json',sources)
    print('METADATA_SNAPSHOT',len(rows),'new',new,flush=True)

def main(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    folders=sorted([p for p in root.iterdir() if p.is_dir()])
    rows=[];private=[];hashes={};visual=[];identity=set();labels=set()
    for folder in folders:
        protocol_path=folder/'protocol.json'
        if not protocol_path.exists():continue
        protocol=readjson(protocol_path)
        identity.add(str(protocol.get('subject')));labels.add((str(protocol.get('session')),str(protocol.get('wear'))))
        metas=sorted(folder.glob('*_meta.json'))
        for meta_path in metas:
            m=readjson(meta_path);alias=f'C{len(rows)+1:02d}';word=WORDS.get(folder.name)
            audio_path=folder/m['full_audio'];imu_path=folder/m['full_imu'];ts_path=folder/m['audio_timestamps']
            sr,wav=wavfile.read(audio_path);audio=normalized(wav);channels=1 if wav.ndim==1 else wav.shape[1]
            mono=audio if audio.ndim==1 else audio.mean(1)
            imu=np.genfromtxt(imu_path,delimiter=',',names=True,encoding='utf-8-sig',dtype=float)
            names=imu.dtype.names;time=imu['imu_sample_time'];axes=np.stack([imu[f'accel_{k}'] for k in 'xyz'],1)
            dt=np.diff(time);positive=dt[dt>0];median=float(np.median(positive));fs=1/median;duration=float(time[-1]-time[0])
            with ts_path.open(encoding='utf-8-sig',newline='') as handle:ts=list(csv.DictReader(handle))
            offsets=np.array([float(x['frame_offset']) for x in ts]);frames=np.array([float(x['frames']) for x in ts])
            adc=np.array([float(x['adc_time_raw']) for x in ts]);host=np.array([float(x['host_callback_perf_counter_s']) for x in ts])
            bad_frame=int(np.sum(offsets[1:]!=offsets[:-1]+frames[:-1]))
            main_hash=payload_hash(wav);hashes.setdefault(('full_audio',main_hash),[]).append(alias)
            ih=payload_hash(axes);hashes.setdefault(('full_imu_axes',ih),[]).append(alias)
            segment_rows=[];segment_mismatch=0;missing=[]
            for segment in m.get('segments',[]):
                sp=folder/segment['audio_file']
                if not sp.exists():missing.append(segment['audio_file']);continue
                rr,sw=wavfile.read(sp);start=int(round(segment['start_audio_s']*sr));end=int(round(segment['end_audio_s']*sr))
                expected=wav[start:end]
                exact=rr==sr and sw.shape==expected.shape and np.array_equal(sw,expected)
                segment_mismatch+=int(not exact)
                sh=payload_hash(sw);hashes.setdefault(('audio_segment',sh),[]).append(f"{alias}/R{segment['repeat_id']:02d}")
                segment_rows.append(dict(repeat_id=segment['repeat_id'],word_override=word,
                    audio_file=str(sp),start_audio_s=segment['start_audio_s'],end_audio_s=segment['end_audio_s'],
                    duration_s=len(sw)/rr,hash=sh,exact_full_slice_match=exact,
                    imu_file=segment.get('imu_file'),imu_alignment_verified=segment.get('imu_alignment_verified')))
            dc=axes.mean(0);centered=axes-dc
            gap_count=int((dt>1.5*median).sum());nonmono=int((dt<=0).sum())
            # Spectra are descriptive; no speech/non-speech SNR or coherence
            # is inferred without verified cross-clock synchronization.
            boundaries=np.r_[0,np.flatnonzero((dt>1.5*median)|(dt<=0))+1,len(time)]
            runs=list(zip(boundaries[:-1],boundaries[1:]));longest=max(runs,key=lambda p:p[1]-p[0])
            continuous=centered[longest[0]:longest[1]]
            freq,psd=welch(continuous,fs=fs,nperseg=min(2048,len(continuous)),axis=0)
            powers={}
            for low,high in [(1,10),(10,50),(50,100),(100,300),(300,450)]:
                mask=(freq>=low)&(freq<high)
                powers[f'{low}_{high}_Hz']=np.trapz(psd[mask],freq[mask],axis=0).tolist()
            callback_dt=np.diff(host)
            ar=blocks(mono,round(sr*.02));rms_db=20*np.log10(np.maximum(ar,1e-12))
            row=dict(recording=alias,confirmed_word=word,metadata_prompt=m.get('prompt_text'),
                metadata_word_id=m.get('word_id'),normalized_metadata_prompt=normalized_prompt(m.get('prompt_text')),
                word_label_conflict=word is not None and word!=normalized_prompt(m.get('prompt_text')),
                expected_repeats=m.get('expected_repeats'),segments=len(segment_rows),segment_ids=[s['repeat_id'] for s in segment_rows],
                segment_duration_range_s=[float(min([s['duration_s'] for s in segment_rows],default=0)),float(max([s['duration_s'] for s in segment_rows],default=0))],
                sample_rate=sr,audio_channels=channels,audio_duration_s=len(wav)/sr,audio_peak=float(np.max(abs(audio))),
                audio_rms_dbfs=float(20*np.log10(max(float(np.sqrt(np.mean(audio**2))),1e-12))),
                audio_clipping_fraction=float(np.mean(abs(audio)>=.9999)),
                audio_20ms_rms_dbfs_q10_q50_q90=np.quantile(rms_db,[.1,.5,.9]).tolist(),
                imu_samples=len(axes),imu_span_s=duration,imu_median_hz=fs,imu_mean_hz=(len(axes)-1)/duration,
                imu_dt_ms_q01_q50_q99=np.quantile(dt,[.01,.5,.99]).__mul__(1000).tolist(),
                imu_gap_count_over_1_5_dt=gap_count,imu_max_gap_ms=float(dt.max()*1000),nonmonotonic_imu=nonmono,
                analysis_gap_threshold_ms=1.5*median*1000,
                metadata_imu_gap_count=m.get('imu',{}).get('gap_count'),
                spectrum_longest_continuous_samples=len(continuous),
                spectrum_longest_continuous_span_s=float(time[longest[1]-1]-time[longest[0]]),
                nonfinite_axis_values=int((~np.isfinite(axes)).sum()),
                offscale_rows=int(np.count_nonzero(imu['offscale_flags'])),
                accel_mean_xyz=dc.tolist(),accel_mean_vector_norm=float(np.linalg.norm(dc)),
                accel_detrended_rms_xyz=np.sqrt(np.mean(centered**2,axis=0)).tolist(),
                imu_psd_bandpower_xyz=powers,
                callback_rows=len(ts),audio_frame_contiguity_errors=bad_frame,
                audio_timestamp_frame_total=int(offsets[-1]+frames[-1]) if len(ts) else 0,
                timestamp_frames_match_wav=bool(len(ts) and offsets[-1]+frames[-1]==len(wav)),
                callback_host_span_s=float(host[-1]-host[0]),
                callback_host_dt_ms_q01_q50_q99=(np.quantile(callback_dt,[.01,.5,.99])*1000).tolist(),
                callback_intervals_under_1ms=int(np.sum(callback_dt<.001)),
                adc_backsteps=int(np.sum(np.diff(adc)<0)),adc_unique_values=int(len(np.unique(adc))),
                adc_range_s=[float(adc.min()),float(adc.max())],
                adc_minus_audio_index_variation_s=float(np.ptp(adc-offsets/sr)),
                metadata_adc_timestamps_plausible=m.get('audio',{}).get('adc_timestamps_plausible'),
                alignment=m.get('imu_audio_alignment'),audio_segments_imu_aligned=m.get('audio_segments_are_imu_aligned'),
                all_segments_have_imu_files=bool(segment_rows) and all(s['imu_file'] is not None for s in segment_rows),
                capture_integrity_pass=m.get('capture_integrity_pass'),capture_quality_issues=m.get('capture_quality_issues',[]),
                capture_wall_duration_s=m.get('capture_wall_duration_s'),
                audio_overflow_callbacks=m.get('audio',{}).get('overflow_callbacks'),
                audio_queue_dropped=m.get('audio',{}).get('queue_dropped'),
                segment_full_slice_mismatches=segment_mismatch,missing_segment_files=len(missing),
                segment_overlap_pairs=sum(b['start_audio_s']<a['end_audio_s'] for a,b in zip(segment_rows[:-1],segment_rows[1:])),
                audio_imu_duration_difference_s=len(wav)/sr-duration)
            rows.append(row)
            private.append(dict(recording=alias,source_folder=str(folder),metadata_subject=protocol.get('subject'),
                metadata_session=protocol.get('session'),metadata_wear=protocol.get('wear'),
                corrected_word=word,word_provenance='user confirmed directory0-4 areEnglish digits;not ASR verified' if word else 'unconfirmed extra trial',
                original_metadata_prompt=m.get('prompt_text'),full_audio=str(audio_path),full_imu=str(imu_path),
                full_audio_payload_hash=main_hash,imu_axes_hash=ih,segments=segment_rows,
                source_file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [protocol_path,meta_path,audio_path,imu_path,ts_path]}))
            if word is not None:
                sos=butter(4,[10,min(300,fs*.45)],btype='bandpass',fs=fs,output='sos')
                tx=[];bx=[];width=round(fs*.02)
                for start,end in runs:
                    if end-start<max(50,width):continue
                    filtered=sosfiltfilt(sos,centered[start:end],axis=0);br=blocks(filtered,width)
                    usable=len(br)*width
                    times=time[start:start+usable].reshape(-1,width).mean(1)-time[0]
                    tx.extend(times.tolist()+[float('nan')]);bx.extend(br.tolist()+[[float('nan')]*3])
                visual.append((row,ar,np.array(tx),np.array(bx),freq,psd,m.get('segments',[])))
            print('RECORD_AUDITED',alias,'segments',len(segment_rows),'integrity',m.get('capture_integrity_pass'),flush=True)
    duplicates=[dict(kind=k[0],recordings=v) for k,v in hashes.items() if len(v)>1]
    result=dict(state='complete',recordings=len(rows),metadata_identity_count=len(identity),metadata_session_wear_count=len(labels),
        confirmed_words=list(WORDS.values()),confirmed_recordings=sum(r['confirmed_word'] is not None for r in rows),
        total_audio_segments=sum(r['segments'] for r in rows),confirmed_word_audio_segments=sum(r['segments'] for r in rows if r['confirmed_word'] is not None),
        duplicate_payload_groups=duplicates,records=rows,
        limitations=['identity is metadata-only;not independent speaker verification','word overrides user-confirmed,not transcribed',
        'cross-clock alignment unverified;no coherence,voice-SNR or paired IMU slicing claimed','full recordings and segments are not independent samples'])
    dump(out/'summary_public.json',result);dump(out/'corrected_source_manifest_private.json',private)
    fig,axs=plt.subplots(len(visual),3,figsize=(15,2.45*len(visual)),squeeze=False)
    colors=['#2674bb','#d67d22','#39834e']
    for k,(r,ar,tx,br,freq,psd,segments) in enumerate(visual):
        axs[k,0].plot((np.arange(len(ar))+.5)*.02,20*np.log10(np.maximum(ar,1e-8)),color='#2674bb',lw=.8)
        for seg in segments:axs[k,0].axvspan(seg['start_audio_s'],seg['end_audio_s'],alpha=.12,color='green')
        axs[k,0].set(title=f"{r['recording']} / {r['confirmed_word']} : Audio RMS",xlabel='Audio relative time (s)',ylabel='dBFS')
        for axis in range(3):
            axs[k,1].plot(tx,br[:,axis],lw=.65,color=colors[axis],label='XYZ'[axis])
            mask=(freq>=10)&(freq<=450)
            axs[k,2].plot(freq[mask],10*np.log10(np.maximum(psd[mask,axis],1e-16)),lw=.75,color=colors[axis])
        axs[k,1].set(title='ACC 10-300Hz RMS / independent clock',xlabel='IMU relative time (s)',ylabel='Acceleration units');axs[k,1].legend(loc='upper right',fontsize=7)
        axs[k,2].set(title='ACC PSD: longest gap-free run',xlabel='Frequency (Hz)',ylabel='PSD (dB)')
        for ax in axs[k]:ax.grid(alpha=.2)
    fig.suptitle('New raw capture audit: Audio and IMU clocks NOT verified aligned',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.975]);fig.savefig(out/'raw_overview_private.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',default='outputs/new_hmd_capture_audit_20261117');p.add_argument('--metadata-only',action='store_true')
    args=p.parse_args();metadata_only(args) if args.metadata_only else main(args)
