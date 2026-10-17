"""No model training: audit raw IMU/microphone periodic coupling and controls."""
import json,csv
from pathlib import Path
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('outputs/raw_periodicity_20261017');OUT.mkdir(exist_ok=True)
BASE=Path('outputs/imu_original_frontend_20261016')
manifest=json.loads((BASE/'manifest.json').read_text())
labels=dict(np.load('outputs/harmonic_scaffold_20261017/pitch_labels.npz'))
grid=np.arange(60,401,2.);rng=np.random.default_rng(42)
fir=signal.firwin(101,[40,499],pass_zero=False,fs=1000)
recordings={};groups={s:[] for s in ('train','val')};seen=set();quality={}

def stft(x):
    return signal.stft(x,fs=1000,nperseg=128,noverlap=118,nfft=1024,axis=-1)[2]

for split in ('train','val'):
    rows=[r for r in manifest['samples'] if r['split']==split]
    for i,row in enumerate(rows):
        g=row['recording']
        if g not in recordings:
            with np.load(Path('outputs/raw_imu_processed_20261014')/(g+'.npz')) as d:
                x=d['acc_raw'].copy();audio=d['audio'].copy();fs=int(d['audio_fs'])
            # Preserve nominal existing pairing. No lag fitting, stretching, or gain normalization.
            if fs!=16000:audio=signal.resample(audio,int(len(audio)*16000/fs))
            audio1k=signal.resample_poly(audio,1,16)
            x=x-x.mean(0)
            xf=signal.filtfilt(fir,[1.],x,axis=0,padlen=min(303,len(x)-1))
            recordings[g]=dict(raw=x.T,filtered=xf.T,audio=audio1k,split=split)
            src=Path(r'E:\desk\vr_imu_collector')/(g+'.csv')
            c=np.genfromtxt(src,delimiter=',',names=True)
            times=c['imu_sample_time'];aq=np.stack([c['accel_'+axis] for axis in 'xyz'])
            qs=[]
            for ax in aq:
                steps=np.round(np.diff(np.unique(ax)),6);positive=steps[steps>1e-6]
                values,counts=np.unique(positive,return_counts=True)
                top=np.argsort(counts)[-3:][::-1]
                qs.append(dict(unique_values=int(len(np.unique(ax))),identical_consecutive_fraction=float(np.mean(np.diff(ax)==0)),common_positive_steps=[float(v) for v in values[top]],step_counts=[int(v) for v in counts[top]]))
            quality[g]=dict(split=split,raw_samples=len(times),median_hardware_dt_ms=float(np.median(np.diff(times))*1000),max_hardware_gap_ms=float(np.max(np.diff(times))*1000),nonincreasing_hardware_timestamps=int(np.sum(np.diff(times)<=0)),offscale_rows=int(np.sum(c['offscale_flags']!=0)),observed_csv_axis_quantization=qs)
        r=recordings[g];start=int(round(row['start_s']*1000));x=r['raw'][:,start:start+800];xf=r['filtered'][:,start:start+800];a=r['audio'][start:start+800]
        assert x.shape==(3,800) and len(a)==800
        f=np.fft.rfftfreq(1024,1/1000)
        raw=abs(stft(x))**2;filtered=abs(stft(xf))**2;mag=abs(stft(np.sqrt(np.sum(xf*xf,axis=0))))**2;ap=abs(stft(a))**2
        # Absolute sample indices deduplicate overlapping windows, not by estimated quality.
        unique=np.zeros(81,dtype=bool)
        for t in range(7,74):
            key=(g,start+10*t)
            if key not in seen:unique[t]=True;seen.add(key)
        groups[split].append(dict(id=g,start_s=row['start_s'],raw=raw,filtered=filtered,mag=mag,audio=ap,f0=labels[split+'_f0'][i],voice=labels[split+'_voice'][i],prob=labels[split+'_prob'][i],unique=unique,x=x,xf=xf,a=a))

def interp_curve(p,hz):
    return np.array([np.interp(v,f,p[:,t]) if np.isfinite(v) and 40<=v<=450 else np.nan for t,v in enumerate(hz)])
_cached_p=None;_cached_integral=None
def prominence(p,hz):
    # Exact band averages via cumulative sums; cache one spectrum across wrong-pair queries.
    global _cached_p,_cached_integral
    if p is not _cached_p:
        _cached_p=p;_cached_integral=np.pad(np.cumsum(p,axis=0),((1,0),(0,0)))
    h=np.nan_to_num(hz,nan=230.);t=np.arange(len(h))
    def band(lo,hi):
        left=np.searchsorted(f,lo,side='left');right=np.searchsorted(f,hi,side='right');count=np.maximum(right-left,0)
        return _cached_integral[right,t]-_cached_integral[left,t],count
    core,ncore=band(h-12,h+12)
    a,na=band(np.maximum(h-60,40),np.minimum(h-25,450))
    b,nb=band(np.maximum(h+25,40),np.minimum(h+60,450))
    val=10*np.log10((core/np.maximum(ncore,1)+1e-24)/((a+b)/np.maximum(na+nb,1)+1e-24))
    val[~(np.isfinite(hz)&(hz>=60)&(hz<=400))]=np.nan
    return val
def peak_f0(p,whiten=False):
    q=np.log(p+1e-24)
    if whiten:q=q-signal.savgol_filter(q,101,2,axis=0)
    band=(f>=60)&(f<=400)
    return f[band][q[band].argmax(0)]
def rpa(pred,row):
    mask=row['voice']&row['unique'];err=abs(1200*np.log2(pred/np.nan_to_num(row['f0'],nan=234.5184)))
    return float(np.mean(err[mask]<50)) if mask.any() else None
def condition_stats(rows,source):
    per={}
    for g in sorted({r['id'] for r in rows}):
        rr=[r for r in rows if r['id']==g];speech=[];quiet=[];matched=[];wrong=[];shifted=[];matched_shift=[];pe=[];pw=[];conf=[]
        for row in rr:
            p=source(row);mask=row['voice']&row['unique'];pr=prominence(p,row['f0']);speech.extend(pr[mask].tolist())
            # For quiet controls use this recording's median voiced reference F0.
            ref=np.full(81,np.nanmedian(np.concatenate([z['f0'][z['voice']] for z in rr])))
            quiet.extend(prominence(p,ref)[~row['voice']&row['unique']].tolist())
            conf.extend(pr[mask&(row['prob']>=.8)].tolist())
            for other in rows:
                if other['id']==g:continue
                both=mask&other['voice'];a=pr[both];b=prominence(p,other['f0'])[both];matched.extend(a.tolist());wrong.extend(b.tolist())
            # Circular 200ms time-shift is a negative pairing control, not an alignment correction.
            shf=np.roll(row['f0'],20);shv=np.roll(row['voice'],20);both=mask&shv
            matched_shift.extend(pr[both].tolist());shifted.extend(prominence(p,shf)[both].tolist())
            for target,white in ((pe,False),(pw,True)):
                pred=peak_f0(p,white);err=abs(1200*np.log2(pred/np.nan_to_num(row['f0'],nan=234.5184)));target.extend((err[mask]<50).tolist())
        mean=lambda z:float(np.nanmean(z)) if len(z) else None
        per[g]=dict(unique_voiced_frames=len(speech),unique_quiet_frames=len(quiet),prominence_db=mean(speech),quiet_median_pitch_prominence_db=mean(quiet),high_confidence_prominence_db=mean(conf),matched_common_prominence_db=mean(matched),wrong_record_prominence_db=mean(wrong),matched_minus_wrong_db=mean(np.array(matched)-np.array(wrong)),matched_shift_common_db=mean(matched_shift),shift200ms_prominence_db=mean(shifted),matched_minus_shift_db=mean(np.array(matched_shift)-np.array(shifted)),raw_peak_rpa=mean(pe),flattened_peak_rpa=mean(pw))
    keys=['prominence_db','quiet_median_pitch_prominence_db','matched_minus_wrong_db','matched_minus_shift_db','raw_peak_rpa','flattened_peak_rpa']
    macro={k:float(np.mean([v[k] for v in per.values() if v[k] is not None])) for k in keys}
    return dict(per_record=per,macro=macro)

sources={**{axis:(lambda row,i=i:row['raw'][i]) for i,axis in enumerate('xyz')},'raw_xyz_power':lambda row:row['raw'].sum(0),'filtered_xyz_power':lambda row:row['filtered'].sum(0),'original_time_magnitude':lambda row:row['mag'],'microphone_positive_reference':lambda row:row['audio']}
results={'source':'same personal raw vAccel recordings;20train4val;test/28reserve excluded','protocol':dict(f0_reference='existing pYIN60-400Hz,64ms/10ms; automatic not manual',acc_spectral_frame_ms=128,acc_fft=1024,actual_frequency_resolution_hz=7.8125,fft_grid_hz=.9765625,main_prominence='mean power F0+-12Hz / local sidebands25-60Hz,40-450Hz, dB; not physical SNR',edges_ms=70,overlap='deduplicated absolute sample index within each recording',statistics='macro mean of recordings; frames not independent; no inferential p-values',coherence='256ms no-overlap segments, phase-randomized-block null; no lag fitting; wrong-record control matched to equal lengths'), 'quality':quality,'spectral':{}}
for split,rows in groups.items():
    results['spectral'][split]={k:condition_stats(rows,fn) for k,fn in sources.items()}
    print(split,{k:v['macro'] for k,v in results['spectral'][split].items()},flush=True)

# Known oracle positive control, NEVER training data or a claimed restoration result.
positive={}
for db in (0,10,20):
    injected=[]
    for row in groups['val']:
        xx=row['x'].copy();af=signal.filtfilt(fir,[1.],row['a'],padlen=303)
        amp=np.sqrt(np.mean(row['xf'][0]**2))/max(np.sqrt(np.mean(af**2)),1e-12)*10**(db/20)
        xx[0]+=amp*af
        injected.append(dict(row,raw=abs(stft(xx))**2))
    positive[str(db)+'dB_oracle_mic_injection']=condition_stats(injected,lambda r:r['raw'].sum(0))
results['positive_controls']=positive

def coh(x,a):
    n=min(x.shape[-1],len(a));x=x[...,:n];a=a[:n]
    _,_,X=signal.stft(x,fs=1000,nperseg=256,noverlap=0,nfft=256,detrend='constant',boundary=None,padded=False,axis=-1)
    _,_,A=signal.stft(a,fs=1000,nperseg=256,noverlap=0,nfft=256,detrend='constant',boundary=None,padded=False)
    den=np.mean(abs(X)**2,-1)*np.mean(abs(A)**2,-1)[None,:]
    C=abs(np.mean(X*A.conj()[None],-1))**2/np.maximum(den,1e-30)
    return C,X,A,den
cf=np.fft.rfftfreq(256,.001);cb=(cf>=60)&(cf<=400);coherence={};coh_curves={}
for g,r in recordings.items():
    if r['split']!='val':continue
    C,X,A,den=coh(r['raw'],r['audio']);stat=float(C[:,cb].mean());null=[]
    for _ in range(200):
        phase=np.exp(1j*rng.uniform(-np.pi,np.pi,A.shape));nC=abs(np.mean(X*(A*phase).conj()[None],-1))**2/np.maximum(den,1e-30);null.append(float(nC[:,cb].mean()))
    wrong=[];own_same_length=[]
    for gg,other in recordings.items():
        if gg==g or other['split']!='val':continue
        n=min(r['raw'].shape[-1],len(other['audio']),len(r['audio']))
        wrong.append(float(coh(r['raw'][...,:n],other['audio'][:n])[0][:,cb].mean()));own_same_length.append(float(coh(r['raw'][...,:n],r['audio'][:n])[0][:,cb].mean()))
    coherence[g]=dict(nonoverlap_segments=int(X.shape[-1]),mean_msc_60_400hz=stat,axis_mean_msc=[float(c[cb].mean()) for c in C],phase_null_median=float(np.median(null)),phase_null_95pct=float(np.percentile(null,95)),matched_equal_length_msc=float(np.mean(own_same_length)),wrong_record_equal_length_msc=float(np.mean(wrong)),matched_minus_wrong_msc=float(np.mean(own_same_length)-np.mean(wrong)))
    coh_curves[g]=C
results['coherence_validation']=coherence
(OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

# Spectral plots use TRAIN-only shared scales. No per-panel normalization.
train_axes=np.concatenate([10*np.log10(r['raw']+1e-24).reshape(-1) for r in groups['train']]);clo,chi=np.percentile(train_axes,[5,99.5])
train_audio=np.concatenate([10*np.log10(r['audio']+1e-24).reshape(-1) for r in groups['train']]);alo,ahi=np.percentile(train_audio,[5,99.5])
results['plot_scales']={'acc_db':[float(clo),float(chi)],'audio_db':[float(alo),float(ahi)],'fitted':'training windows only, same ACC scale every axis/window'}
(OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
chosen=[next(r for r in groups['val'] if r['id']==g) for g in sorted({r['id'] for r in groups['val']})]
fig,axes=plt.subplots(5,4,figsize=(13,12),squeeze=False)
freqmask=(f>=40)&(f<=450)
for col,row in enumerate(chosen):
    data=[row['audio']]+list(row['raw'])+[row['raw'].sum(0)]
    for rr,(name,p) in enumerate(zip(['Microphone low band','Raw ACC X','Raw ACC Y','Raw ACC Z','Raw XYZ power sum'],data)):
        ax=axes[rr,col];lims=(alo,ahi) if rr==0 else (clo,chi)
        ax.imshow(10*np.log10(p[freqmask]+1e-24),origin='lower',aspect='auto',extent=[0,.8,f[freqmask][0],f[freqmask][-1]],vmin=lims[0],vmax=lims[1],cmap='magma')
        ax.plot(np.arange(81)*.01,np.where(row['voice'],row['f0'],np.nan),color='cyan',linewidth=1)
        ax.set_title(name+' / '+row['id'].replace('recording_',''),fontsize=8);ax.tick_params(labelsize=7)
fig.suptitle('Raw IMU periodicity audit: cyan = microphone-estimated F0, not an ACC prediction\nTRAIN-only shared ACC colour scale across axes/recordings; first existing window per validation recording',fontsize=10)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'raw_acc_f0_overlay.png',dpi=115)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
keys=['raw_xyz_power','filtered_xyz_power','original_time_magnitude','microphone_positive_reference'];names=['Raw XYZ','Filtered XYZ','Time magnitude','Microphone reference']
for offset,(k,name) in enumerate(zip(keys,names)):
    vals=[v['matched_minus_wrong_db'] for v in results['spectral']['val'][k]['per_record'].values()]
    axes[0].scatter(np.full(len(vals),offset),vals,label=name);axes[0].plot([offset-.1,offset+.1],[np.mean(vals)]*2,color='black')
axes[0].axhline(0,color='gray',linestyle='--');axes[0].set_xticks(range(4),names,rotation=15);axes[0].set_ylabel('F0 prominence: matched minus wrong recording / dB');axes[0].set_title('Four recordings, not independent-frame significance')
for g,C in coh_curves.items():axes[1].plot(cf,C.mean(0),label=g.replace('recording_',''))
axes[1].set_xlim(40,450);axes[1].set_ylim(0,1);axes[1].set_xlabel('Frequency / Hz');axes[1].set_ylabel('Magnitude-squared coherence, mean XYZ');axes[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'coupling_summary.png',dpi=120)
print('DONE',OUT,flush=True)
