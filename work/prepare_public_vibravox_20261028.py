"""Prepare fresh paired inputs from raw VibraVox WAVs, no old clipped arrays."""
import sys,json,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'work/public_signal_runtime_20261028'));sys.path.insert(0,str(R/'work'))
import numpy as np
from public_dataset_frontend_20261028 import wav,resample,pair,WIN
P=R/'outputs/vibravox_pilot_20261020';O=R/'outputs/public_baseline_comparison_20261028';O.mkdir(exist_ok=True)
records=json.loads((P/'indexed_manifest_owner80.json').read_text(encoding='utf-8'))['records'];meta={}
for p in (P/'global_text_scan_cache').glob('*.json'):
 for row in json.loads(p.read_text(encoding='utf-8')):meta[(row['speaker_id'],row['sentence_id'])]=row
owners=sorted([u for u,n in collections.Counter(r['speaker_id'] for r in records).items() if n>=60]);unknown=sorted(set(r['speaker_id'] for r in records)-set(owners));assert len(owners)==len(unknown)==2
roles={};used=[]
for j,u in enumerate(owners):
 ids=np.flatnonzero([r['speaker_id']==u for r in records]);ids=np.random.RandomState(20261028+j).permutation(ids)[:60]
 roles[u]={};offset=0
 for name,n in [('gan_train',30),('gan_val',6),('auth_train',10),('calibration',6),('final',8)]:roles[u][name]=ids[offset:offset+n].tolist();used.extend(ids[offset:offset+n].tolist());offset+=n
for u in unknown:
 ids=[i for i,r in enumerate(records) if r['speaker_id']==u];roles[u]={'final':ids};used.extend(ids)
byrole=collections.defaultdict(set)
for u,groups in roles.items():
 for role,ids in groups.items():
  for i in ids:byrole[role].add(meta[(records[i]['speaker_id'],records[i]['sentence_id'])]['normalized_text'])
for a in byrole:
 for b in byrole:
  if a!=b:assert not byrole[a]&byrole[b],('cross-role exact text overlap',a,b)
xx=[];yy=[];units=[];speakers=[];source=[]
for i in sorted(set(used)):
 r=records[i];signals={};info={}
 for key in ['acc','mic']:
  fs,w,inf=wav(P/'raw'/r[key]['file']);assert w.shape[1]==1;assert inf['sha256']==r[key]['sha256'];signals[key]=resample(w[:,0],fs);info[key]=inf
 total=min(len(signals['acc']),len(signals['mic']));assert total>=2*WIN;start=(total-2*WIN)//2
 for b in [start,start+WIN]:
  x,y=pair(signals['acc'][b:b+WIN],signals['mic'][b:b+WIN]);xx.append(x);yy.append(y);units.append(i);speakers.append(r['speaker_id'])
 source.append({'unit':i,'speaker':r['speaker_id'],'sentence_id':r['sentence_id'],'text':meta[(r['speaker_id'],r['sentence_id'])]['normalized_text'],'waveform':info})
np.savez_compressed(O/'vibravox_raw_db.npz',x_db=np.stack(xx),y_db=np.stack(yy),unit_id=np.asarray(units),speaker_id=np.asarray(speakers))
protocol={'dataset':'VibraVox','owners':owners,'unknown':unknown,'roles':roles,'source':source,'normalization':'Not yet fitted; pooled GAN TRAIN of both owners only, per-dataset shared reference, no per-user normalization',
 'sampling':'two adjacent central0.8s windows per utterance; role split by whole utterance; score once per utterance after averaging window features',
 'content':'normalized text audited from full official train metadata; no exact text crosses roles; no shared prompt equality across different people claimed',
 'selection':'Cached pilot cohort, original owner had previously been chosen after outcome inspection. New split exploratory; no blind or population estimate.',
 'frontend':'Shared16k; DCremove; FFT512 hop160; BC linear STFT dB80x80, microphone HTK80Mel power dB80x80. Proper PCM bit-depth conversion; ACC float physical waveform not divided by32768.'}
(O/'vibravox_protocol_private.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print('VIBRAVOX_READY',len(source),len(units),owners,unknown,flush=True)
