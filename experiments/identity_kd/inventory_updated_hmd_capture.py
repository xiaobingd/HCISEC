"""Recursive two-participant inventory;keep human utterance counts separate from VAD."""
import argparse
import csv
import hashlib
import json
import math
import wave
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from audit_new_hmd_capture import normalized_prompt

WORDS=['zero','one','two','three','four','five','six','seven','eight','nine']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def wav_info(path):
    try:
        with wave.open(str(path),'rb') as h:return dict(frames=h.getnframes(),sample_rate=h.getframerate(),channels=h.getnchannels(),duration_s=h.getnframes()/h.getframerate())
    except wave.Error:
        from scipy.io import wavfile
        sr,data=wavfile.read(path,mmap=True)
        return dict(frames=len(data),sample_rate=sr,channels=1 if data.ndim==1 else data.shape[1],duration_s=len(data)/sr)

def main(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    prior=Path('outputs/new_hmd_capture_audit_20261117/identity_overrides_private.json')
    rules=load(prior).get('by_metadata_sha256',{}) if prior.exists() else {}
    word_override=out/'word_overrides_private.json'
    word_rules=load(word_override).get('by_metadata_sha256',{}) if word_override.exists() else {}
    roots=sorted(p for p in root.iterdir() if p.is_dir());assert len(roots)==2,'Expected user-confirmed two participant folders'
    started=datetime.now(timezone(timedelta(hours=8))).isoformat();rows=[];sources=[];participants=[]
    for ordinal,participant_root in enumerate(roots,1):
        paths=sorted(participant_root.rglob('*_meta.json'))
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        known={rules[h] for h in hashes.values() if h in rules};assert len(known)<=1,'Conflicting identities in one folder'
        identity=next(iter(known)) if known else f'NEW{ordinal:02d}'
        participants.append(dict(participant_id=identity,prior_identity_hash_match=bool(known),word_records=len(paths)))
        for path in paths:
            m=load(path);folder=path.parent;label=normalized_prompt(m.get('prompt_text'))
            override=identity=='N01' and folder.name in ['0','1','2','3','4']
            if override:label=WORDS[int(folder.name)]
            corrected_by_user=hashes[str(path)] in word_rules
            if corrected_by_user:label=word_rules[hashes[str(path)]]
            alias=f'R{len(rows)+1:03d}';audio_path=folder/m['full_audio'];imu_path=folder/m['full_imu']
            exists=audio_path.exists() and imu_path.exists()
            info=wav_info(audio_path) if audio_path.exists() else None
            segments=m.get('segments',[]);missing=0;duration_total=0.;duplicate_repeat_ids=len(segments)-len({s['repeat_id'] for s in segments})
            for segment in segments:
                sp=folder/segment['audio_file']
                if not sp.exists():missing+=1
                else:duration_total+=wav_info(sp)['duration_s']
            times=[];offscale=0;invalid=0
            if imu_path.exists():
                with imu_path.open(encoding='utf-8-sig',newline='') as h:
                    reader=csv.DictReader(h)
                    for sample in reader:
                        try:
                            t=float(sample['imu_sample_time']);flag=float(sample['offscale_flags'])
                            if not math.isfinite(t):invalid+=1;continue
                            times.append(t);offscale+=int(flag!=0)
                        except (ValueError,TypeError,KeyError):invalid+=1
            tt=np.asarray(times,dtype=float);dt=np.diff(tt);pos=dt[dt>0]
            fs=1/np.median(pos) if len(pos) else None;span=float(tt[-1]-tt[0]) if len(tt)>1 else None
            overflow=m.get('audio',{}).get('overflow_callbacks');qd=m.get('audio',{}).get('queue_dropped')
            actual_segments=len(segments)-missing
            row=dict(recording=alias,participant_id=identity,word=label,
                word_provenance='user-confirmed correction' if corrected_by_user else ('user-confirmed original directory digit' if override else 'metadata prompt,not independently transcribed'),
                user_stated_min_actual_utterances=10,utterance_count_independently_verified=False,
                metadata_expected_repeats=m.get('expected_repeats'),vad_detected=m.get('vad',{}).get('detected'),
                saved_audio_segments=actual_segments,missing_audio_segment_files=missing,
                duplicate_repeat_ids=duplicate_repeat_ids,segment_total_duration_s=duration_total,
                audio=info,audio_overflow_callbacks=overflow,audio_queue_dropped=qd,
                capture_wall_duration_s=m.get('capture_wall_duration_s'),
                actual_imu_samples=len(tt),actual_imu_span_s=span,imu_median_hz=float(fs) if fs else None,
                imu_max_gap_ms=float(dt.max()*1000) if len(dt) else None,
                imu_nonmonotonic=int((dt<=0).sum()),imu_invalid_rows=invalid,imu_offscale_rows=offscale,
                metadata_capture_integrity_pass=m.get('capture_integrity_pass'),
                metadata_quality_issues=m.get('capture_quality_issues',[]),
                alignment=m.get('imu_audio_alignment'),
                all_segments_imu_aligned=bool(segments) and all(s.get('imu_alignment_verified') is True and s.get('imu_file') for s in segments),
                metadata_wav_frames_match=bool(info and info['frames']==m.get('audio',{}).get('frames')),
                full_payload_files_present=exists,metadata_word_id=m.get('word_id'))
            rows.append(row)
            sources.append(dict(recording=alias,participant_id=identity,participant_root=str(participant_root),
                metadata_path=str(path),metadata_sha256=hashes[str(path)],full_audio=str(audio_path),full_imu=str(imu_path),
                actual_metadata_subject=m.get('subject_id'),canonical_word=label,
                segments=[dict(repeat_id=s['repeat_id'],audio_path=str(folder/s['audio_file']),imu_path=s.get('imu_file'),
                    start_audio_s=s['start_audio_s'],end_audio_s=s['end_audio_s']) for s in segments]))
            dump(out/'status.json',dict(stage='inventory',participant_id=identity,completed=len(rows)))
            print('RECORD_CHECKED',alias,identity,label,'segments',actual_segments,'overflow',overflow,flush=True)
    assert len({p['participant_id'] for p in participants})==2
    write_inventory(out,rows,sources,participants,started)


def write_inventory(out,rows,sources,participants,started):
    allwords=sorted({r['word'] for r in rows});matrix=[]
    for word in allwords:
        item=dict(word=word,participants={})
        for p in participants:
            rr=[r for r in rows if r['participant_id']==p['participant_id'] and r['word']==word]
            item['participants'][p['participant_id']]=dict(recordings=len(rr),saved_segments=sum(r['saved_audio_segments'] for r in rr),
                vad_detected=sum(r['vad_detected'] or 0 for r in rr),integrity_pass_recordings=sum(bool(r['metadata_capture_integrity_pass']) for r in rr))
        matrix.append(item)
    shared=[v['word'] for v in matrix if all(x['recordings']>0 for x in v['participants'].values())]
    result=dict(started_at_bj=started,completed_at_bj=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        participant_count=2,participants=participants,total_word_recordings=len(rows),
        total_saved_audio_segments=sum(r['saved_audio_segments'] for r in rows),word_matrix=matrix,shared_words=shared,records=rows,
        human_statement='each word spoken at least10 times;VAD/saved segments are distinct observations,not ground-truth utterance count',
        words_not_ASR_verified=True,raw_files_unchanged_by_inventory=True,
        private_subject_names_paths_and_signals_not_published=True)
    dump(out/'inventory_public.json',result);dump(out/'source_manifest_private.json',sources)
    summary=[]
    for p in participants:
        rr=[r for r in rows if r['participant_id']==p['participant_id']]
        summary.append(dict(participant_id=p['participant_id'],words=len({r['word'] for r in rr}),recordings=len(rr),
            saved_segments=sum(r['saved_audio_segments'] for r in rr),
            recordings_with_overflow=sum((r['audio_overflow_callbacks'] or 0)>0 for r in rr),
            integrity_pass_recordings=sum(bool(r['metadata_capture_integrity_pass']) for r in rr),
            aligned_word_recordings=sum(bool(r['all_segments_imu_aligned']) for r in rr)))
    dump(out/'summary_public.json',dict(participants=summary,shared_words=shared))
    lines=['# 更新后的两人原始数据清单与质量核查','',
      f"完成时间：{result['completed_at_bj']}。按用户确认的两个主目录划分参与者，旧匿名身份通过元数据哈希锚定，原始P001标签不作为合并依据。", '',
      '**用户确认每词实际发声至少10次。下表的切片数量来自VAD与实际保存文件，不是反推的发声次数。少于10可能是漏检、多次发声被合并为一段或保存丢失；多于10可能是实际多说或过分割；本次不否定或改写用户的实际发声说明。**','',
      '|匿名参与者|词数|连续词录制数|已保存Audio切片|有音频溢出的录制|采集完整性通过|已验证同步词录制|','|---|---:|---:|---:|---:|---:|---:|']
    for p in summary:lines.append(f"|{p['participant_id']}|{p['words']}|{p['recordings']}|{p['saved_segments']}|{p['recordings_with_overflow']}|{p['integrity_pass_recordings']}|{p['aligned_word_recordings']}|")
    lines+=['','## 逐词对应清单','', '|词语|'+'|'.join(p['participant_id']+' 保存切片/完整性通过录制' for p in participants)+'|','|---|'+'|'.join('---:' for _ in participants)+'|']
    for item in matrix:
        values=[item['participants'][p['participant_id']] for p in participants]
        lines.append('|'+item['word']+'|'+'|'.join('—' if v['recordings']==0 else f"{v['saved_segments']} / {v['integrity_pass_recordings']}" for v in values)+'|')
    lines+=['',f"双方声明的共同词有{len(shared)}个："+', '.join(shared)+'。未共同覆盖的词不会混入同内容对照。',
      '', '## 逐录制核查','',
      '|记录|参与者|词语|VAD/保存切片|WAV秒|墙钟秒|溢出回调|IMU最大间隔ms|完整性|','|---|---|---|---:|---:|---:|---:|---:|---|']
    def show(v):return '—' if v is None else f'{v:.3f}'
    for r in rows:lines.append(f"|{r['recording']}|{r['participant_id']}|{r['word']}|{r['vad_detected']} / {r['saved_audio_segments']}|{show(r['audio']['duration_s'] if r['audio'] else None)}|{show(r['capture_wall_duration_s'])}|{r['audio_overflow_callbacks']}|{show(r['imu_max_gap_ms'])}|{'通过' if r['metadata_capture_integrity_pass'] else '不通过/待核查'}|")
    lines+=['','## 当前可用范围','',
      '元数据提示中有seven.eight、ni等疑似不规范词标签时，先保留原值并等待用户确认，不按文件夹批次名称强行推断。独立私有词修正表以元数据SHA256绑定，防止改名/排序变化串标签。',
      '两个人的身份目录与逐词对应已经可以整理。能否形成至少10个完整配对样本，仍需依据原始信号连续性、切分和同步确定，不能用人为补齐VAD数量替代。',
      '先检查完整WAV，区分VAD漏检/合并与实际信号缺失；只对确认丢失的内容安排重录。完成连续性和同步核验后，再将Audio片段对应到IMU。原始完整录音和切片不重复计为独立样本；暂不进行新模型训练或认证指标计算。',
      '', '本轮只读WAV头/元数据及原生IMU时间，核查实际保存数量、时长与采样连续性；没有做逐条内容转写，没有计算语音发声真实次数，也没有重复执行全部负载哈希去重。完整路径、姓名和原始信号只在私有清单/源目录保留。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(out/'status.json',dict(stage='complete'));print('INVENTORY_COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',default='outputs/new_hmd_two_participants_inventory_20261117');main(p.parse_args())
