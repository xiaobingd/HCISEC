"""ESMB preprocessing diagnostic; content is UNVERIFIED, not a T3 replacement."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from types import SimpleNamespace
BASE=['0005','0006','0007','0021','0022','0023','0003'];CAL=['0026','0027'];TARGET=['0001','0002','0004','0024','0025']
def source(name):
    if name in ['0001','0002']:return Path('/autodl-fs/data/esmb_baseline_20261020/two_speakers')/(name+'_4.wav')
    if name in ['0003','0004']:return Path('/autodl-fs/data/esmb_baseline_20261020/attack_speakers')/(name+('_12.wav' if name=='0003' else '_4.wav'))
    return Path('/autodl-fs/data/esmb_shared_identity_20261020/raw')/(name+'_4.wav')
old=SimpleNamespace(BASE=BASE,CAL=CAL,USERS=BASE+CAL+TARGET,source=source)

ROOT=Path('/autodl-fs/data/esmb_preprocessing_rbf_20261109')
def main():
    ROOT.mkdir(exist_ok=True);start=time.time();features=[];labels=[];indices=[];seen={};dups=[];quality=[];channels=[]
    for u,name in enumerate(old.USERS):
        fs,w=wavfile.read(old.source(name),mmap=True);assert fs==16000 and w.ndim==2 and w.shape[1]==2
        for j in range(240):
            pair=np.asarray(w[j*48000:j*48000+32000]);assert pair.shape==(32000,2)
            h=hashlib.sha256(pair.tobytes()).hexdigest()
            if h in seen:dups.append([seen[h],[u,j]])
            seen[h]=[u,j]
            x=pair[:,1].astype(float)/32768.;clip=float((np.abs(x)>=.999).mean());x-=x.mean()
            f,t,z=signal.stft(x,fs=fs,nperseg=1024,noverlap=768,nfft=2048,boundary=None,padded=False)
            p=np.abs(z)**2;edges=np.geomspace(50,1200,41)
            centers=np.sqrt(edges[:-1]*edges[1:])
            b=np.stack([np.interp(centers,f,np.log(p[:,it]+1e-12)) for it in range(p.shape[1])],axis=1)
            assert np.isfinite(b).all()
            # One scalar mean/std per original window removes absolute level but retains relative spectrum.
            n=(b-b.mean())/max(b.std(),1e-4)
            def pool(v):return np.concatenate([v.mean(1),v.std(1)]+[v[:,part].mean(1) for part in np.array_split(np.arange(v.shape[1]),4)])
            rms=np.sqrt(np.mean(x*x)+1e-12);energy=np.r_[np.log(rms+1e-12),np.log(np.quantile(np.abs(x),[.1,.5,.9,.99])+1e-12)]
            features.append((pool(b),pool(n),np.r_[pool(n),energy]));labels.append(u);indices.append(j)
            quality.append([clip,float(rms)])
            if j<10:
                ratios=[]
                for ch in range(2):
                    xx=pair[:,ch].astype(float);ff,pp=signal.welch(xx,fs=fs,nperseg=2048)
                    ratios.append(float(pp[(ff>=2000)&(ff<=7000)].sum()/(pp[(ff>=50)&(ff<2000)].sum()+1e-12)))
                channels.append(ratios)
        print('PREPARED',name,flush=True)
    assert not dups,'Exact duplicate recordings require resolving before evaluation'
    y=np.array(labels);idx=np.array(indices);nb=len(old.BASE);nc=len(old.CAL);base=y<nb;fit=base&(idx<120);cal=list(range(nb,nb+nc));targets=list(range(nb+nc,len(old.USERS)))
    result={};private={}
    for k,method in enumerate(['global_scale','instance_norm','instance_norm_plus_energy']):
        x=np.stack([v[k] for v in features]);sc=StandardScaler().fit(x[fit]);x=sc.transform(x)
        def head(u):
            mask=((y==u)|base)&(idx<120)
            model=SVC(C=1.,gamma='scale',kernel='rbf',class_weight='balanced').fit(x[mask],y[mask]==u)
            return model
        cs=[];cy=[]
        for u in cal:
            m=head(u);q=np.isin(y,cal)&(idx>=200);cs.extend(m.decision_function(x[q]));cy.extend(y[q]==u)
        cs=np.array(cs);cy=np.array(cy);thresholds=np.r_[np.nextafter(cs.min(),-np.inf),np.unique(cs),np.nextafter(cs.max(),np.inf)]
        # Cal identities alone determine the rule. Tie-break prioritizes lower Cal FAR.
        tau=min(thresholds,key=lambda t:(abs(float((cs[~cy]>=t).mean())-float((cs[cy]<t).mean())),float((cs[~cy]>=t).mean())))
        rows=[]
        for u in targets:
            m=head(u);q=np.isin(y,targets)&(idx>=200);s=m.decision_function(x[q]);truth=y[q]==u
            tar=float((s[truth]>=tau).mean());far=float((s[~truth]>=tau).mean());rows.append(dict(user=old.USERS[u],tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(truth,s))))
        private[method]={'threshold':float(tau),'targets':rows};result[method]={key:float(np.mean([r[key] for r in rows])) for key in ['tar','far','hter','auc']};print('RESULT',method,result[method],flush=True)
    public=dict(status='complete_preprocessing_diagnostic',base_identities=nb,cal_identities=nc,unseen_target_identities=len(targets),registration_per_user=120,query_per_user=40,unseen_attack_samples_per_target=160,methods=result,elapsed_seconds=time.time()-start,duplicate_pairs=len(dups),clipped_sample_fraction_max=float(np.max(np.array(quality)[:,0])),channel_high_low_ratio_median=np.median(channels,axis=0).tolist(),protocol='BC channel1,Audio channel0 follows previously corrected ESMB mapping;16kHz;2sec every3sec;first120 registration/background;200..239 query;Base-only scaler;fixed balanced SVC RBF C1 gamma scale;Cal-only EER threshold;all5 targets and all4 unseen attackers retained',features='50..1200Hz logpower40bands;mean/std plus4temporalblocks=240dims;energy branch5dims;no hard100..500Hz bandpass;mean removal;no per-user filter;no quality exclusions',limits=['text and channel semantics not independently reverified this run','same-session windows,not independent recording sessions','not strict same-content or cross-content','fixed handcrafted frontend,not original772-dimensional full baseline or T3','two Cal identities only;previously explored identities;single deterministic diagnostic'])
    (ROOT/'summary_public.json').write_text(json.dumps(public,indent=2));(ROOT/'results_private.json').write_text(json.dumps(private,indent=2));(ROOT/'status.json').write_text(json.dumps({'state':'complete'}));print('COMPLETE',json.dumps(public),flush=True)
if __name__=='__main__':main()
