"""Prepare ESMB raw data and shared TRAIN-only scales for the matched pilot."""
import sys,json
from pathlib import Path
import numpy as np
from public_dataset_frontend_20261028 import wav,pair,WIN,FS
ROOT=Path('/autodl-fs/data/public_baseline_compare_20261028');ROOT.mkdir(parents=True,exist_ok=True)
V=ROOT/'vibravox';V.mkdir(exist_ok=True)
E=ROOT/'esmb';E.mkdir(exist_ok=True)
roles={};sources=[];xx=[];yy=[];unit=[];speaker=[];uid=0;filehash={}
files={'0001':'two_speakers/0001_4.wav','0002':'two_speakers/0002_4.wav','0003':'attack_speakers/0003_12.wav','0004':'attack_speakers/0004_4.wav'}
for j,(u,rel) in enumerate(files.items()):
 fs,w,meta=wav(Path('/autodl-fs/data/esmb_baseline_20261020')/rel);assert fs==FS and w.shape[1]==2;filehash[u]=meta
 starts=np.random.RandomState(20261028+j).permutation(np.arange(2,600,2))
 specs=[('gan_train',30),('gan_val',6),('auth_train',10),('calibration',6),('final',8)] if j<2 else [('final',24)]
 roles[u]={};position=0
 for role,n in specs:
  roles[u][role]=[]
  for seconds in starts[position:position+n]:
   b=int(seconds*FS);assert b+2*WIN<=len(w)
   for start in [b,b+WIN]:
    a,m=pair(w[start:start+WIN,1],w[start:start+WIN,0]);xx.append(a);yy.append(m);unit.append(uid);speaker.append(u)
   roles[u][role].append(uid);sources.append({'unit':uid,'speaker':u,'start_seconds':int(seconds),'duration':1.6,'file':rel});uid+=1
  position+=n
protocol={'dataset':'ESMB','owners':['0001','0002'],'unknown':['0003','0004'],'roles':roles,'source_units':sources,'files':files,
 'source_files':filehash,'content':'Official release lacks prompt transcripts in this cache; time-isolated source blocks are NOT sentence-held-out or content-matched.',
 'waveform_frontend':'Channel1 BC, channel0 microphone per user correction; same16k FFT512hop160 BC-STFT/Mic-Mel frontend as VibraVox; source units1.6s, two adjacent0.8s windows.',
 'limitations':'Historical speakers/files previously explored, same recording per identity; no cross-session or physiological identity causality claim.'}
np.savez_compressed(str(E/'raw_db.npz'),x_db=np.stack(xx),y_db=np.stack(yy),unit_id=np.array(unit),speaker_id=np.array(speaker))
(E/'protocol_private.json').write_text(json.dumps(protocol,indent=2))
for dataset in ['esmb','vibravox']:
 p=ROOT/dataset;d=dict(np.load(str(p/'raw_db.npz')));protocol=json.loads((p/'protocol_private.json').read_text());upstream=sum((protocol['roles'][u]['gan_train'] for u in protocol['owners']),[]);mask=np.isin(d['unit_id'],upstream)
 scales=[];clip={}
 for k,out in [('x_db','acc'),('y_db','audio')]:
  lo,hi=np.percentile(d[k][mask],[1,99]);assert hi>lo;scales.append([float(lo),float(hi)]);d[out]=np.clip(2*(d[k]-lo)/(hi-lo)-1,-1,1).astype('float32')
  clip[k]={u:float(np.mean((d[k][d['speaker_id']==u]<lo)|(d[k][d['speaker_id']==u]>hi))) for u in set(d['speaker_id'])}
 np.savez_compressed(str(p/'inputs.npz'),acc=d['acc'],audio=d['audio'],unit_id=d['unit_id'],speaker_id=d['speaker_id'])
 protocol['scales_bc_mel']=scales;protocol['normalization_fit_units']=upstream;protocol['clip_fractions']=clip
 protocol['shared_normalization']='Pooled GAN TRAIN of both owners only, one scale/domain/dataset applied to every user and role, never per-user.'
 (p/'protocol_private.json').write_text(json.dumps(protocol,indent=2));print('DATA_READY',dataset,d['acc'].shape,scales,flush=True)
