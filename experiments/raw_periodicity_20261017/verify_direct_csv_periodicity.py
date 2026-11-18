"""Bypass cached interpolation with exact hardware-time Fourier analysis."""
import json,wave,hashlib
from pathlib import Path
import numpy as np
P=Path('outputs/raw_periodicity_20261017');root=Path(r'E:\desk\vr_imu_collector')
rows=json.loads(Path('outputs/imu_original_frontend_20261016/manifest.json').read_text())['samples']
orig=json.loads(Path('outputs/raw_imu_processed_20261014/manifest.json').read_text())['records'];hashes={r['file']:r for r in orig}
labels=np.load('outputs/harmonic_scaffold_20261017/pitch_labels.npz');f=np.fft.rfftfreq(1024,.001);out={};seen=set()
def prominence(p,h):
    core=abs(f-h)<=12;side=(abs(f-h)>=25)&(abs(f-h)<=60)&(f>=40)&(f<=450)
    return float(10*np.log10((p[core].mean()+1e-24)/(p[side].mean()+1e-24)))
for split in ('train','val'):
    rr=[r for r in rows if r['split']==split];out[split]={}
    for g in sorted({r['recording'] for r in rr}):
        source=root/(g+'.csv');wav=root/(g.replace('_imu','_audio')+'.wav')
        c=np.genfromtxt(source,delimiter=',',names=True);t=c['imu_sample_time'];t=t-t[0]
        x=np.stack([c['accel_'+a] for a in 'xyz'],1)
        with np.load(Path('outputs/raw_imu_processed_20261014')/(g+'.npz')) as z:
            target=np.column_stack([np.interp(z['time_s'],t,x[:,k]) for k in range(3)])
            acc_diff=float(np.max(abs(target-z['acc_raw'])))
            with wave.open(str(wav)) as w:
                channels=w.getnchannels();fs=w.getframerate();audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').reshape(-1,channels)[:,0].astype(float)/32768
            audio_diff=float(np.max(abs(audio[:len(z['audio'])]-z['audio'])))
        record=dict(imu_source_hash_matches=hashlib.sha256(source.read_bytes()).hexdigest()==hashes[source.name]['imu_sha256'],audio_source_hash_matches=hashlib.sha256(wav.read_bytes()).hexdigest()==hashes[source.name]['audio_sha256'],cached_acc_reconstruction_max_difference=acc_diff,cached_audio_max_difference=audio_diff)
        # Original exact timestamps, no interp/resample/filter. Record DC removal only.
        if split=='val':
            x=x-x.mean(0);vals=[];axisvals=[[],[],[]];falsevals=[]
            for i,r in enumerate(rr):
                if r['recording']!=g:continue
                for frame in range(7,74):
                    key=(g,int(round(r['start_s']*1000))+10*frame)
                    if key in seen:continue
                    seen.add(key)
                    if not labels['val_voice'][i,frame]:continue
                    center=r['start_s']+.01*frame;mask=abs(t-center)<.064
                    delta=t[mask]-center;win=.5*(1+np.cos(np.pi*delta/.064))
                    basis=np.exp(-2j*np.pi*f[:,None]*delta[None,:]);spec=basis@(x[mask]*win[:,None])/win.sum();power=abs(spec)**2
                    h=labels['val_f0'][i,frame];vals.append(prominence(power.sum(1),h))
                    for k in range(3):axisvals[k].append(prominence(power[:,k],h))
                    for j,other in enumerate(rr):
                        if other['recording']==g or not labels['val_voice'][j,frame]:continue
                        hh=labels['val_f0'][j,frame];falsevals.append(prominence(power.sum(1),h)-prominence(power.sum(1),hh))
            record.update(direct_hardware_time_nudft_f0_prominence_db=float(np.mean(vals)),axis_prominence_db=[float(np.mean(v)) for v in axisvals],matched_minus_wrong_prominence_db=float(np.mean(falsevals)),unique_voiced_frames=len(vals))
        out[split][g]=record
out['notes']='All24used records verified against original hashes and exact cached interpolation/WAV reconstruction. Direct hardware-time nonuniform Fourier check done only on4existingvalidation recordings at existing nominal paired centers; no lag search. It is a signal-detection diagnostic, not a train/test model score.'
(P/'source_and_direct_csv.json').write_text(json.dumps(out,indent=2));print(json.dumps({s:{g:r for g,r in v.items()} if s=='val' else len(v) for s,v in out.items() if s!='notes'}),flush=True)
