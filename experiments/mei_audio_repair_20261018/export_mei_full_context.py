"""Same clean sources/scales, full2s context; verify shared legacy-window interiors."""
import os,sys,json,argparse,zipfile,io
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy import signal,ndimage

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.source.parent))
from data0723_frontend_helpers import create_mel_filterbank
dp=json.loads((args.source/'data_protocol.json').read_text());norm=dp['normalization'];lo,hi=norm['spec_lo_hi'];mu=np.array(norm['energy_mean'])[:,None];sd=np.array(norm['energy_std'])[:,None];alo,ahi=norm['audio_lo_hi']
fb=create_mel_filterbank(16000,512,80,20,7500);out=args.study/'full_context_data';out.mkdir(exist_ok=True);bands=[(2,10),(10,40),(40,100),(100,200),(200,450),(450,490)]
rows=[r for r in dp['source_audit'] if r['domain']=='mei'];assert len(rows)==240
all_data={s:dict(x=[],y=[],ids=[]) for s in ['train','validation']}
with zipfile.ZipFile(args.archive) as zf:
    for k,row in enumerate(rows):
        task,key=row['recording'].split('__',1);prefix=task+'/wangjingmei/'
        a=np.load(io.BytesIO(zf.read(prefix+key+'.npy'))).astype('float64');mic=np.load(io.BytesIO(zf.read(prefix+key.replace('accel_data','audio_data',1)+'.npy'))).astype('float64')
        order=np.argsort(a[:,0],kind='stable');t=a[order,0];x=a[order,1:4];keep=np.r_[True,np.diff(t)>0];t=t[keep];x=x[keep];t-=t[0]
        duration=min(t[-1],len(mic)/48000);grid=np.arange(0,duration,.001);x=np.stack([np.interp(grid,t,x[:,j]) for j in range(3)],1);x-=x.mean(0)
        # Existing source audit spans1.997-2.000s. Keep the actual waveform length;
        # the unchanged STFT boundary/padded-frame convention yields201 frames.
        assert 1997<=len(x)<=2000
        audio=signal.resample_poly(mic.mean(1),1,3);assert len(audio)==32000
        energy=[]
        for xx in [x]+[signal.sosfiltfilt(signal.butter(4,b,btype='bandpass',fs=1000,output='sos'),x,axis=0) for b in bands]:
            for ms in (50,250):
                power=np.maximum(ndimage.uniform_filter1d(xx*xx,size=ms,axis=0,mode='reflect'),0)
                energy.extend([np.log10(np.sqrt(power[:,j])+1e-12) for j in range(3)]);energy.append(np.log10(np.sqrt(power.sum(1))+1e-12))
        energy=np.array(energy);_,_,z=signal.stft(x.T,fs=1000,nperseg=128,noverlap=118,nfft=128,axis=-1);assert z.shape==(3,65,201)
        spec=(2*(20*np.log10(abs(z)+1e-12)-lo)/(hi-lo)-1).reshape(195,201);ix=np.minimum(np.arange(201)*10,len(x)-1)
        xx=np.concatenate([spec,(energy[:,ix]-mu)/sd]).astype('float32');_,_,az=signal.stft(audio,fs=16000,nperseg=512,noverlap=352);yy=np.clip(2*(10*np.log10(fb@abs(az)**2+1e-10)-alo)/(ahi-alo)-1,-1,1).astype('float32');assert yy.shape==(80,201)
        d=all_data[row['split']];d['x'].append(xx);d['y'].append(yy);d['ids'].append(row['recording'])
        if (k+1)%60==0:print('FULL_CONTEXT',k+1,flush=True)
audit={}
for split,d in all_data.items():
    d={k:np.array(v) for k,v in d.items()};np.savez_compressed(out/('full_'+split+'.npz'),**d)
    legacy=dict(np.load(args.source/'data'/('mei180_train.npz' if split=='train' else 'mei_val.npz')));lookup={str(n):i for i,n in enumerate(d['ids'])};xe=ye=0.
    for name in np.unique(legacy['ids']):
        li=np.flatnonzero(legacy['ids']==name);assert len(li)==4;idx=lookup[str(name)]
        for j in range(3):
            # Only starts0/.4/.8; final source-specific1.199/1.2 is excluded from common scoring.
            xe=max(xe,float(np.max(abs(d['x'][idx,:,j*40+8:j*40+73]-legacy['x'][li[j],:,8:-8]))))
            ye=max(ye,float(np.max(abs(d['y'][idx,:,j*40+8:j*40+73]-legacy['y'][li[j],:,8:-8]))))
    assert xe<1e-4 and ye<1e-4,(split,xe,ye)
    audit[split]=dict(recordings=len(d['ids']),ACC_max_shared_interior_error=xe,Mel_max_shared_interior_error=ye)
protocol=dict(source='EXACT same180cleanTRAIN/60REUSEDval source pairs and frozen shared calibration; noTEST/auth read',context='Full~2s ACC input and2s output201frames; offline task can use later same-recordingACC. No lag-search/time-shift/resampling change, no microphone inference. Not a causal real-time model.',preprocessing='Same signedXYZ0-500HzSTFT+56energy and80Mel; window length only changes. STFT boundary padding unchanged; raw waveforms not padded. Allsourcefilters/scales unchanged.',common_score='First3 original.8s windows at0/.4/.8s, discard8edgeframes each (65interiorframes), SAME60source records. Exactparity checked against legacyX/Y. Last~1.2s starts not used for common score to avoid1ms source-specific rounding.',parity=audit,scope='Full context is an exploratory follow-up after12localrepairfits failed. Common score not directly comparable to older all240-window/all-frameMAE.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol),flush=True)
