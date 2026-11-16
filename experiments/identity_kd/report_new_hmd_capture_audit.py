"""Freeze the completed raw audit snapshot and report anonymous quality metrics."""
import hashlib
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
from audit_new_hmd_capture import normalized_prompt,dump

OUT=Path('outputs/new_hmd_capture_audit_20261117')
def main():
    path=OUT/'summary_public.json';r=json.loads(path.read_text(encoding='utf-8'))
    snapshot=r.get('snapshot_completed_at_bj',datetime.fromtimestamp(path.stat().st_mtime,timezone(timedelta(hours=8))).isoformat())
    r['snapshot_completed_at_bj']=snapshot
    private=json.loads((OUT/'corrected_source_manifest_private.json').read_text(encoding='utf-8'))
    for row,source in zip(r['records'],private):
        row['normalized_metadata_prompt']=normalized_prompt(row['metadata_prompt'])
        row['word_label_conflict']=row['confirmed_word'] is not None and row['confirmed_word']!=row['normalized_metadata_prompt']
        clips=source['segments'];row['segment_overlap_pairs']=sum(b['start_audio_s']<a['end_audio_s'] for a,b in zip(clips[:-1],clips[1:]))
        source['paired_training_ready']=False
        source['corrected_word_numeric_id']=None if row['confirmed_word'] is None else ['zero','one','two','three','four'].index(row['confirmed_word'])
        for clip in clips:clip['unique_record_key']=f"{row['recording']}_R{clip['repeat_id']:02d}"
    dump(path,r);dump(OUT/'corrected_source_manifest_private.json',private)
    mainrows=[v for v in r['records'] if v['confirmed_word'] is not None]
    extra=[v for v in r['records'] if v['confirmed_word'] is None]
    lines=['# 新采集Audio／原始IMU审计（文件快照）','',
      f"原始负载核查快照：{snapshot}（北京时间）。本次原始负载扫描包含{r['recordings']}段完整录制、{r['total_audio_segments']}个Audio切片；后续新增文件另列元数据快照，不将两批统计相加。", '',
      '## 结论','',
      '新文件具有直接三轴加速度、原生IMU采样时间及完整录音，具备做物理信号分析的基础；目前采集完整性和同步仍有明确故障，不能直接作为已同步Audio–ACC配对训练集。',
      '', '已确认目录0–4是zero、one、two、three、four。五个主录制中，只有zero通过现有采集完整性检查，仍未验证跨时钟同步；其余四个都出现音频溢出。one另有约665ms IMU断档；two的VAD切出12段，需要人工核验，不能直接当作多采了两次。',
      '', '## 五词主录制','',
      '|目录/词|VAD切片数|WAV秒|采集墙钟秒|音频溢出回调|IMU最大间隔ms|采集完整性|','|---|---:|---:|---:|---:|---:|---|']
    for v in mainrows:
        digit=['zero','one','two','three','four'].index(v['confirmed_word'])
        lines.append(f"|{digit} / {v['confirmed_word']}|{v['segments']}|{v['audio_duration_s']:.3f}|{v['capture_wall_duration_s']:.3f}|{v['audio_overflow_callbacks']}|{v['imu_max_gap_ms']:.3f}|{'通过' if v['capture_integrity_pass'] else '不通过'}|")
    lines+=['','每词原计划10次，五词应为50次；当前VAD生成52个主录制切片，不是已核验的52次有效发声。音频均为16kHz单声道；IMU中位采样率约999.0–999.1Hz，但中位采样率正常并不消除中途断档。五段主录音未见数字削波，主要问题是时间连续性。',
      '', '## 额外负载核查记录','', '|匿名记录|文件声明提示|Audio秒|IMU跨度秒|切片数|音频溢出回调|完整性|','|---|---|---:|---:|---:|---:|---|']
    for v in extra:lines.append(f"|{v['recording']}|{v['normalized_metadata_prompt']}|{v['audio_duration_s']:.3f}|{v['imu_span_s']:.3f}|{v['segments']}|{v['audio_overflow_callbacks']}|{'通过' if v['capture_integrity_pass'] else '不通过'}|")
    lines+=['','额外记录仅按文件提示列出词，不把提示当作已做语义转写。真实参与者身份需结合采集者说明确认；元数据只有一个身份标记，不能据文件夹数量推断人数，也不能据此计算多人认证FAR。',
      '', '## 已定位的问题','',
      '1. **标签模板复用。** 主五词均复用W001；提示文本混有one、one-zero、2、3、4。数字2/3/4可规范化成two/three/four，但zero的one提示及one的one-zero提示与用户确认不一致。按用户说明写了独立私有修正清单，没有修改原始protocol/meta。不同词下同名W001_R01文件不能平铺覆盖。',
      '2. **采集丢块。** one的WAV17.152秒，墙钟约20.061秒，差2.909秒，同时12次音频溢出；IMU最大间隔665.417ms。two/three/four也分别有2/2/4次溢出。文件内frame_offset连续只表示已保存帧被连续编号，不证明原始音频没有丢失。',
      '3. **ADC时间戳异常。** 已核查负载的ADC原始时间反复只有0与0.016两个值并多次回退，不能用来建立可信的跨设备映射。host_callback和host_read记录的是回调/批量读取到达事件，不能直接当成声音和振动的物理采样时刻。',
      '4. **没有配对IMU切片。** 当前IMU段路径为空，所有录制均标UNVERIFIED。Audio切片完整不等于Audio–IMU已对齐；不能按相同起点秒数直接裁IMU。',
      '', '## 重复及切片核查','',
      f"- 本快照完整Audio、完整IMU轴负载和Audio切片的精确哈希重复组：{len(r['duplicate_payload_groups'])}。只排除精确重复，不声称没有近重复。",
      f"- 切片与full_audio相应样本范围不一致：{sum(v['segment_full_slice_mismatches'] for v in r['records'])}；切片文件缺失：{sum(v['missing_segment_files'] for v in r['records'])}；相邻切片时间重叠对：{sum(v['segment_overlap_pairs'] for v in r['records'])}。",
      '- full_audio与其segments是父子数据，不得作为两套独立样本重复入训练/测试。完整录制内部切片还共享佩戴与会话条件。',
      '', '## 如何使用这批数据','',
      '- 保留所有原始文件。目录0优先用于检查声学/振动结构及设计同步验证；它尚不能直接被认定为同步配对数据。',
      '- 目录1–4及严重溢出的额外试录建议在采集链路修复后重录。连续IMU片段仍可用于单模态信号诊断；跨断档不得强行当作均匀连续样本做谱分析。',
      '- 先验证音频输入设备/驱动和采集线程：连续录制时溢出/丢队列为0、WAV时长与实际采集时长匹配。再用独立同步事件验证时钟偏移与漂移，之后才生成IMU配对切片。不能靠改WAV采样率或整体拉伸填补未知丢块。',
      '- 明确记录真实subject/session/wear/word身份。新字段为accel_x/y/z，旧九人使用vVelocity；后续应建立匹配的新前端，避免将不同物理量直接当作同分布输入。',
      '- 本次未训练认证模型、未算发声SNR/跨模态相干性，也未筛选身份表现。先解决采集完整性和标签/同步问题，再扩大采集。',
      '', '## 图与保存方式','',
      'raw_overview_private.png仅在本地保存：Audio RMS、IMU带通RMS各自使用独立相对时间，不作为对齐证据。IMU带通在连续片段内计算，不跨断档滤波；PSD只使用最长无断档区间。10–300Hz只是诊断图频带，原始CSV没有裁剪。图不上传GitHub。',
      '', '公开代码和匿名统计；原始波形、身份/路径映射、修正清单及信号图均保持私有。原始目录不改名、不删数据。']
    follow=OUT/'metadata_followup_public.json'
    if follow.exists():
        f=json.loads(follow.read_text(encoding='utf-8'));lines+=['','## 扫描后的元数据跟进','',
          f"跟进时间：{f['completed_at_bj']}。共读取{len(f['records'])}份元数据，其中{f.get('unmatched_path_metadata_records',f.get('new_metadata_records',0))}份无法按原路径匹配此前快照，可能为新增录制或目录改名，不能当作新增独立样本数。跟进阶段未重做这些记录的波形/切片哈希验证。", '',
          '|记录|声明词|墙钟秒|Audio秒|溢出回调|切片数|完整性|','|---|---|---:|---:|---:|---:|---|']
        for v in f['records']:
            if v['payload_previously_audited']:continue
            lines.append(f"|{v['recording']}|{v['declared_word']}|{v['capture_wall_duration_s']}|{v['audio_duration_s']}|{v['overflow_callbacks']}|{v['vad_detected']}|{'通过' if v['integrity'] else '不通过/待确认'}|")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification=dict(state='PASS',scope='stored snapshot consistency;capture quality is a separate failing condition',
        raw_recordings=len(r['records']),main_confirmed_recordings=len(mainrows),
        all_confirmed_word_overrides_match_user_statement=all(v['confirmed_word'] in ['zero','one','two','three','four'] for v in mainrows),
        segment_slice_mismatches=sum(v['segment_full_slice_mismatches'] for v in r['records']),
        missing_segment_files=sum(v['missing_segment_files'] for v in r['records']),
        new_data_ready_for_paired_training=False,original_files_only_read_by_audit=True,
        private_paths_identity_maps_and_signal_plot_not_published=True)
    assert verification['segment_slice_mismatches']==0 and verification['missing_segment_files']==0
    dump(OUT/'verification_public.json',verification);print('REPORT_COMPLETE',snapshot)

if __name__=='__main__':main()
