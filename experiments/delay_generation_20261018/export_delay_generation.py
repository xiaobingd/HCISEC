"""Shared full-XYZ context; original20train4val membership and58/16 targets."""
import os,json,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy import signal,ndimage
OUT=Path('outputs/delay_generation_20261018');OUT.mkdir(exist_ok=True)
pa=argparse.ArgumentParser();pa.add_argument('--validation-offset',type=float,default=0.);args=pa.parse_args();offset=args.validation_offset
old=Path('outputs/activity_alignment_20261018');proto=json.loads((old/'data_protocol.json').read_text());sc=proto['scales'];lo,hi=sc['spec_lo_hi'];mu=np.array(sc['energy_mean'])[:,None];sd=np.array(sc['energy_std'])[:,None]
threshold=proto['audio_activity_threshold'];manifest=json.loads(Path('outputs/imu_original_frontend_20261016/manifest.json').read_text());cache={};data={}
bands=[(2,10),(10,40),(40,100),(100,200),(200,450),(450,490)]
def trailing_rms(x,w):
 z=np.r_[0.,np.cumsum(x*x)];end=np.arange(1,len(x)+1);start=np.maximum(end-w,0);return np.sqrt(np.maximum((z[end]-z[start])/(end-start),0))
for split in (['val'] if offset else ['train','val']):
 xs=[];cs=[];aa=[];ref=[];ids=[]
 rows=[r for r in manifest['samples'] if r['split']==split]
 for row in rows:
  g=row['recording']
  if g not in cache:
   with np.load(Path('outputs/raw_imu_processed_20261014')/(g+'.npz')) as d:x=d['acc_raw'].copy();audio=d['audio'];fs=int(d['audio_fs'])
   x-=x.mean(0);energies=[]
   for xx in [x]+[signal.sosfiltfilt(signal.butter(4,b,btype='bandpass',fs=1000,output='sos'),x,axis=0) for b in bands]:
    for ms in [50,250]:
     p=np.maximum(ndimage.uniform_filter1d(xx*xx,size=ms,axis=0,mode='reflect'),0)
     energies.extend([np.log10(np.sqrt(p[:,j])+1e-12) for j in range(3)]);energies.append(np.log10(np.sqrt(p.sum(1))+1e-12))
   t=np.arange(int(len(audio)/fs*100))*.01;ar=trailing_rms(audio,round(fs*.02))[np.rint(t*fs).astype(int)];active=(np.log10(ar+1e-9)>threshold).astype(float)
   release=np.zeros(len(active))
   for i in range(len(active)):release[i]=max(active[i],release[i-1]*np.exp(-.01/.25) if i else 0.)
   response=np.interp(t-.2,t,release,left=release[0],right=release[-1]);cache[g]=(x,np.array(energies),t,active,response)
  x,e,t,active,response=cache[g];start=row['start_s']+offset;b=round(start*1000);assert b>=0;w=x[b:b+1280].T;assert w.shape==(3,1280),'No padding or exclusions allowed'
  _,_,z=signal.stft(w,fs=1000,nperseg=128,noverlap=118,nfft=128,axis=-1);assert z.shape==(3,65,129)
  spec=(2*(20*np.log10(abs(z)+1e-12)-lo)/(hi-lo)-1).reshape(195,129);ix=b+np.arange(129)*10;assert ix[-1]<e.shape[1]
  xs.append(np.concatenate([spec,(e[:,ix]-mu)/sd]));cs.append(np.concatenate([z.real.reshape(195,129),z.imag.reshape(195,129)]))
  times=start+np.arange(129)*.01;aa.append(np.interp(times,t,response));ref.append(np.interp(times[:81],t,active));ids.append(g)
 with np.load(old/('data_'+split+'.npz')) as d:y=d['y'].copy();expected=d['ids'].copy()
 assert np.array_equal(ids,expected);data[split]=dict(base=np.array(xs),complex=np.array(cs),y=y,activity=np.array(aa),audio_activity=np.array(ref),ids=np.array(ids))
scale=json.loads((OUT/'data_protocol.json').read_text())['complex_scale'] if offset else float(np.sqrt(np.mean(data['train']['complex']**2)));assert scale>0
for split,d in data.items():
 x=np.concatenate([d['base'],d['complex']/scale],axis=1).astype('float32');assert np.isfinite(x).all()
 filename='validation_offset_'+str(round(offset*1000))+'ms.npz' if offset else 'data_'+split+'.npz'
 np.savez_compressed(OUT/filename,x=x,y=d['y'],activity=d['activity'].astype('float32'),audio_activity=d['audio_activity'].astype('float32'),ids=d['ids'])
 print(split,x.shape,flush=True)
if offset:raise SystemExit(0)
protocol=dict(source='same personal raw vAccel20train4reusedval;58/16 existing.8s targets;4test+28reserve untouched',context='1.28s ACC from original target start, to generate0.8s Mel;480ms future context; no padding/exclusion or source timestamp edits; offline window task',features='641x129: separateXYZ195 logmagnitudes +56multiscale energies +390real/imag STFT channels; preserve axis phase/amplitude; magnitude/energy scales frozen priorTRAIN, complex one sharedTRAIN scalar, no per-user/record gain normalization',complex_scale=scale,activity='original TRAIN audio threshold,20ms trailingRMS, instantaneousattack/exponentialrelease tau250ms, delayed200ms; computed fullrecord then sliced, bothTRAIN andVAL identical',alignment='learned ACC-only localattention from audio-time t to ACC latent at t+80..300ms; TRAIN-motivated fixed bounds; encoder receptivefield +/-70ms extends effectivecontext; no microphone at inference',limits='singleperson, no identity accuracy. Complex phase inclusion/newcontext changes frontend for ALL newroutes, so not an isolated comparison to earlier experiment',test='not read')
(OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
