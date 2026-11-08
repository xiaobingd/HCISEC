"""Base-only physical targets and held-out Cal predictability, no final targets."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT=Path('/autodl-fs/data/esmb_observable_audio_20261109');ROOT.mkdir(exist_ok=True)
RAW=Path('/autodl-fs/data/esmb_shared_identity_20261020/raw')
BASE=['0005','0006','0007','0021'];CAL=['0022','0023','0026','0027']
FS=16000;WIN=1024;HOP=256;EDGES=np.geomspace(50,8000,33)

def feature(x):
    x=x.astype('float64')/32768.;x-=x.mean()
    frames=np.lib.stride_tricks.sliding_window_view(x,WIN)[::HOP].copy();frames-=frames.mean(1,keepdims=True)
    rms=np.sqrt(np.mean(frames**2,1)+1e-12)
    ac=np.fft.irfft(np.abs(np.fft.rfft(frames,n=2048,axis=1))**2,n=2048,axis=1)[:,:WIN]
    # Lag-dependent energy normalization avoids penalizing long lags purely for overlap.
    ss=np.pad(np.cumsum(frames**2,axis=1),((0,0),(1,0)))
    lags=np.arange(32,321);den=np.sqrt(np.maximum(ss[:,WIN-lags]*(ss[:,WIN,None]-ss[:,lags]),1e-24));rho=ac[:,lags]/den
    peaks=signal.find_peaks
    pitch=[];period=[]
    for row in rho:
        pp=peaks(row)[0]
        k=int(pp[np.argmax(row[pp])]) if len(pp) else int(np.argmax(row));pitch.append(FS/lags[k]);period.append(float(row[k]))
    power=np.abs(np.fft.rfft(frames*np.hanning(WIN),n=2048,axis=1))**2
    f=np.fft.rfftfreq(2048,1/FS);band=np.stack([np.log(np.mean(power[:,(f>=a)&(f<b)],1)+1e-12) for a,b in zip(EDGES[:-1],EDGES[1:])],1)
    band-=band.mean(1,keepdims=True)
    return np.column_stack([band,np.log(rms),np.log(pitch),period]),rms

start=time.time();records=[];cohs=[];nullcohs=[];lags=[];xs=[];ys=[];meta=[];previous={}
for role,names,ids in [('fit',BASE,np.linspace(0,79,24,dtype=int)),('dev',BASE,np.linspace(80,99,8,dtype=int)),('cal',CAL,np.linspace(160,199,16,dtype=int))]:
    for uid,name in enumerate(names):
        sr,w=wavfile.read(RAW/(name+'_4.wav'),mmap=True);assert sr==FS and w.shape[1]==2
        for j in ids:
            pair=np.array(w[j*3*FS:j*3*FS+2*FS]);assert len(pair)==32000
            af,ar=feature(pair[:,0]);bf,br=feature(pair[:,1]);assert len(af)==len(bf)
            if role=='fit':
                f,c=signal.coherence(pair[:,0].astype(float),pair[:,1].astype(float),fs=FS,nperseg=2048,noverlap=1024)
                cohs.append(c)
                other=previous.get(int(j))
                if other is not None:
                    _,nc=signal.coherence(other,pair[:,1].astype(float),fs=FS,nperseg=2048,noverlap=1024);nullcohs.append(nc)
                previous[int(j)]=pair[:,0].astype(float)
                a=np.log(ar+1e-12);b=np.log(br+1e-12);a-=a.mean();b-=b.mean();corr=signal.correlate(b,a,mode='full');ll=signal.correlation_lags(len(b),len(a));mask=np.abs(ll)<=6;lags.append(float(ll[mask][np.argmax(corr[mask])]*HOP/FS))
            xs.append(bf);ys.append(af);meta.append(dict(user=name,role=role,window=int(j)));print('FEATURE',name,role,int(j),flush=True)
X=np.concatenate(xs);Y=np.concatenate(ys);sizes=np.array([len(x) for x in xs]);offset=np.r_[0,np.cumsum(sizes)];roles=np.repeat([x['role'] for x in meta],sizes);users=np.repeat([x['user'] for x in meta],sizes)
train=roles=='fit';dev=roles=='dev';cal=roles=='cal'
mean=np.mean(cohs,axis=0);null=np.mean(nullcohs,axis=0);eligible=(f>=50)&(f<=4000);selected=eligible&(mean>np.quantile(np.asarray(nullcohs),.95,axis=0))
band_observable=np.array([np.any(selected&(f>=a)&(f<b)) for a,b in zip(EDGES[:-1],EDGES[1:])])
# Same-person wrong-time and same-index other-person controls only alter training targets.
shifted=np.concatenate([np.roll(y,21,axis=0) for y in ys]);wrong=Y.copy()
lookup={(m['user'],m['role'],m['window']):i for i,m in enumerate(meta)}
for i,m in enumerate(meta):
    if m['role']!='fit':continue
    u=BASE[(BASE.index(m['user'])+1)%len(BASE)];k=lookup[u,m['role'],m['window']];wrong[offset[i]:offset[i+1]]=ys[k]
targets={'periodicity':34,'log_envelope':32}
results={}
for label,target in [('paired',Y),('same_user_wrong_time',shifted),('different_user_same_index',wrong)]:
    results[label]={}
    for key,col in targets.items():
        model=make_pipeline(StandardScaler(),Ridge(alpha=10));model.fit(X[train],target[train,col]);pr=model.predict(X[cal]);truth=Y[cal,col];corr=float(np.corrcoef(pr,truth)[0,1]);results[label][key]=dict(cal_r2=float(r2_score(truth,pr)),cal_corr=corr)
    voiced=(Y[:,34]>=.6)
    model=make_pipeline(StandardScaler(),Ridge(alpha=10));mask=train&voiced;model.fit(X[mask],target[mask,33]);pred=model.predict(X[cal&voiced]);truth=Y[cal&voiced,33]
    results[label]['f0']=dict(cal_voiced_frames=int((cal&voiced).sum()),cents_mae=float(np.mean(np.abs(pred-truth))*1200/np.log(2)),cal_r2=float(r2_score(truth,pred)))
    cols=np.flatnonzero(band_observable)
    model=make_pipeline(StandardScaler(),Ridge(alpha=10));model.fit(X[train],target[train][:,cols]);pr=model.predict(X[cal]);results[label]['shared_spectrum']=dict(bands=int(len(cols)),cal_r2=float(r2_score(Y[cal][:,cols],pr)))
bands={}
for a,b in [(50,100),(100,300),(300,800),(800,2000),(2000,4000),(4000,8000)]:
    mask=(f>=a)&(f<b);bands[f'{a}-{b}']=dict(paired_coherence=float(mean[mask].mean()),wrong_user_coherence=float(null[mask].mean()),excess=float((mean[mask]-null[mask]).mean()))
summary=dict(status='complete_diagnostic_not_authentication',base=BASE,cal=CAL,final_targets_accessed=False,window_counts={r:sum(m['role']==r for m in meta) for r in ['fit','dev','cal']},audio_features='64ms1024sample autocorrelation;16ms hop;50..500Hz candidateF0; normalized correlation>=.6 is proxy NOT calibrated voicing probability',regression='StandardScaler+Ridge10;Basefit only;unseen Cal evaluated once;no targetAudio/no authentication results',coherence=bands,lag_seconds=dict(median=float(np.median(lags)),q10=float(np.quantile(lags,.1)),q90=float(np.quantile(lags,.9)),search_range=[-.096,.096],interpretation='empirical envelope lag,not proof of propagation delay'),observable_band_centers=np.sqrt(EDGES[:-1]*EDGES[1:])[band_observable].tolist(),results=results,elapsed_seconds=time.time()-start,limits=['one recording/user;no verified text','shared excitation does not prove bone-only propagation','F0 autocorrelation can make octave errors','framewise regression is only predictability diagnostic','observability does not establish identity gain','no final target identities or Audio accessed'])
(ROOT/'summary_public.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(ROOT/'features_private.npz',X=X,Y=Y,roles=roles,users=users,offset=offset);(ROOT/'records_private.json').write_text(json.dumps(meta,indent=2));print('DONE',json.dumps(summary),flush=True)