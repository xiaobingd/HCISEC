import hashlib,json
from pathlib import Path
import numpy as np
root=Path('outputs/vr261119_10x40_raw_acc');out=root/'continuous_acc_activity';out.mkdir(exist_ok=True)
old=json.loads((root/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records'];groups={}
for rec in old:
    user,word,_=map(int,rec['sample_id'].split('_'));groups.setdefault((user,word),rec['record_group'])
assert len(groups)==400
records=[];spans=[];cue_fields=set()
for (user,word),stem in sorted(groups.items()):
    path=Path(stem);meta=json.loads(path.with_name(path.name+'_meta.json').read_text(encoding='utf-8-sig'))
    span=float(meta['imu']['sample_span_s']);spans.append(span)
    cue_fields.update(k for k in meta if any(t in k.lower() for t in ['cue','prompt_time','trial_time']))
    records.append(dict(user=user,word=word,stem=stem,prompt=meta['prompt_text'],imu_span_s=span))
(out/'recordings_private.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
categories=dict(digits=[5,7,8,15,18,25,26,29,30,33],english_commands=[0,1,2,4,6,10,11,13,14,16,17,20,21,22,23,24,27,28,31,34],english_phrases=[3,9,12,19,32],chinese=[35,36,37,38,39])
rng=np.random.default_rng(20261124);split=dict(train=[],development=[],test=[])
for values in categories.values():
    z=rng.permutation(values);n=len(z);split['train']+=z[:int(n*.6)].tolist();split['development']+=z[int(n*.6):int(n*.8)].tolist();split['test']+=z[int(n*.8):].tolist()
split={k:sorted(v) for k,v in split.items()};assert list(map(len,split.values()))==[24,8,8]
split.update(seed=20261124,window_seconds=3,hop_seconds=3,recording_edge_guard_seconds=.5,window_anchor='first valid IMU timestamp + .5sec;noAudioorVAD boundaries',locked_before_waveform_extraction=True)
text=json.dumps(split,sort_keys=True);(out/'word_split_public.json').write_text(text)
result=dict(recordings=400,users=10,words=40,prompt_timestamp_fields=sorted(cue_fields),imu_span_s_min=min(spans),imu_span_s_median=float(np.median(spans)),imu_span_s_max=max(spans),min_expected_full_windows=int((min(spans)-1)//3),word_split_sha256=hashlib.sha256(text.encode()).hexdigest(),test_Audio_read=False)
(out/'source_audit_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
