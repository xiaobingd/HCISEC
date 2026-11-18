"""Separate mathematical sanity check from evidence in the measured IMU."""
import json,numpy as np
from pathlib import Path
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('outputs/raw_periodicity_20261017');p.mkdir(exist_ok=True)
a=np.load('outputs/harmonic_scaffold_20261017/axes_val.npz');lab=np.load('outputs/harmonic_scaffold_20261017/pitch_labels.npz');x=a['axes'];ids=a['ids'];f0=lab['val_f0'];v=lab['val_voice']
rows=[r for r in json.loads(Path('outputs/imu_original_frontend_20261016/manifest.json').read_text())['samples'] if r['split']=='val']
freq=np.fft.rfftfreq(1024,.001)
def stft(z):return signal.stft(z,fs=1000,nperseg=128,noverlap=118,nfft=1024,axis=-1)[2]
def prominence(P,ff):
    out=[]
    for t,h in enumerate(ff):
        if not np.isfinite(h) or not 60<=h<=480:out.append(np.nan);continue
        b=abs(freq-h)<=12;s=(abs(freq-h)>=25)&(abs(freq-h)<=60)&(freq>=40)&(freq<=490)
        out.append(10*np.log10((P[b,t].mean()+1e-24)/(P[s,t].mean()+1e-24)))
    return np.array(out)
def fold(h):return abs((h+500)%1000-500)
raw=np.mean(abs(stft(x))**2,axis=1);mag=abs(stft(np.linalg.norm(x,axis=1)))**2
out={};seen=set();unique=np.zeros_like(v)
for i,r in enumerate(rows):
    for t in range(7,74):
        key=(r['recording'],int(round(r['start_s']*1000))+t*10)
        if key not in seen:unique[i,t]=True;seen.add(key)
for name,P in [('signed_filtered_xyz_power',raw),('time_vector_magnitude',mag)]:
    out[name]={}
    for tag,ff in [('microphone_F0',f0),('half_F0',f0/2),('folded_twice_F0',fold(2*f0))]:
        values=np.array([prominence(pp,hh) for pp,hh in zip(P,ff)]);mask=v&unique&np.isfinite(values)
        per={str(g):dict(prominence_db=float(np.mean(values[mask&(ids==g)[:,None]])),frames=int(np.sum(mask&(ids==g)[:,None]))) for g in np.unique(ids)}
        out[name][tag]=dict(macro_prominence_db=float(np.mean([r['prominence_db'] for r in per.values()])),per_record=per)

# Positive mathematical control: proportional axes, no sensor noise.
tt=np.arange(800)/1000;synthetic_f0=220+110*tt/.8;phase=2*np.pi*np.cumsum(synthetic_f0)/1000
z=np.array([1.,.6,.3])[:,None]*np.sin(phase)[None]
known=220+110*np.arange(81)*.01/.8
sp=np.mean(abs(stft(z))**2,axis=0);mp=abs(stft(np.linalg.norm(z,axis=0)))**2
for name,P in [('synthetic_signed',sp),('synthetic_time_magnitude',mp)]:
    out[name]={tag:float(np.nanmean(prominence(P,ff)[7:74])) for tag,ff in [('true_F0',known),('folded_twice_F0',fold(2*known))]}
out['notes']='Measured inputs use existing signed-axis export: original DC/FIR/Tukey, no normalization. Same measured frames per eligible mapping, deduplicated; mappings near Nyquist can have different eligible counts. Broad spectral prominence is not actual SNR. Synthetic no-noise chirp demonstrates mathematical distortion only, not its causal importance in measured recordings.'
(p/'frequency_mapping.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
fig,axes=plt.subplots(2,2,figsize=(12,7))
chosen=[('Known chirp, signed XYZ',sp,known),('Known chirp, time-vector magnitude',mp,known),('Measured first validation window, signed XYZ',raw[0],f0[0]),('Measured same window, time-vector magnitude',mag[0],f0[0])]
for ax,(name,P,ff) in zip(axes.flat,chosen):
    # Synthetic and measured have different units; common scale within each pair.
    lo,hi=(-70,0) if 'Known' in name else (-100,-40)
    ax.imshow(10*np.log10(P+1e-24),origin='lower',aspect='auto',extent=[0,.8,0,500],vmin=lo,vmax=hi,cmap='magma')
    ax.plot(np.arange(81)*.01,ff,'c',label='F0 reference',linewidth=1);ax.plot(np.arange(81)*.01,fold(2*ff),'lime',label='folded 2F0',linewidth=1)
    ax.set_ylim(40,500);ax.set_title(name,fontsize=9);ax.set_ylabel('Hz');ax.set_xlabel('seconds')
axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(p/'frequency_mapping.png',dpi=120)
