"""Independent HMD velocity/derived acceleration axes and raw microphone channels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

def digest(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def features(x,fs,low,high):
    x=x.astype(float);x-=x.mean();win=256 if fs==1000 else 1024;hop=64 if fs==1000 else 256;nfft=1024 if fs==1000 else 2048
    f,t,z=signal.stft(x,fs=fs,nperseg=win,noverlap=win-hop,nfft=nfft,boundary=None,padded=False)
    centers=np.geomspace(low,high,40);p=np.log(np.abs(z)**2+1e-20);b=np.stack([np.interp(centers,f,p[:,j]) for j in range(p.shape[1])],axis=1)
    return np.concatenate([b.mean(1),b.std(1)]+[b[:,q].mean(1) for q in np.array_split(np.arange(b.shape[1]),4)]).astype('float32')

def prepare(a,out):
    cache=out/'features_private.npz'
    if cache.exists():return {k:v for k,v in np.load(cache).items()}
    manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'));rows=manifest['records'];users=manifest['user_order'];bank=defaultdict(list);meta=[];seen_a=set();seen_m=set();excluded=Counter();path_repairs=0
    # Old manifest contains mojibake in one Chinese directory component.
    # Only repair missing paths; exact original array hashes remain mandatory.
    resolved={}
    for r in rows:
        for key,sub in [('acc_file','acc'),('audio_file','audio')]:
            p=Path(r[key])
            if not p.exists():
                p=Path(a.raw_root)/r['user']/sub/p.name;assert p.exists(),str(p);path_repairs+=1
            resolved[r['sample_id'],key]=p
    for r in rows:
        if r['pair_mode']!='exact_timestamp':excluded['unverified_pairing']+=1;continue
        acc=np.load(resolved[r['sample_id'],'acc_file'],allow_pickle=False);mic=np.load(resolved[r['sample_id'],'audio_file'],allow_pickle=False);ah=digest(acc);mh=digest(mic)
        assert ah==r['acc_array_sha256'] and mh==r['audio_array_sha256'],'Source changed since manifest audit'
        assert ah not in seen_a and mh not in seen_m,'Duplicate recording payload';seen_a.add(ah);seen_m.add(mh)
        assert acc.ndim==2 and acc.shape[1]==4 and mic.shape==(96000,2) and np.isfinite(acc).all() and np.isfinite(mic).all()
        t=acc[:,0].astype(float);v=acc[:,1:].astype(float);order=np.argsort(t,kind='stable');t=t[order];v=v[order];keep=np.r_[True,np.diff(t)>0];t=t[keep];v=v[keep]
        assert len(t)>=100 and t[0]<=.05 and t[-1]>=1.8
        q=np.clip(np.arange(2000)/1000.,t[0],t[-1]);deriv=np.gradient(v,t,axis=0,edge_order=2)
        for source,values in [('velocity',v),('derived_acceleration',deriv)]:
            xyz=[features(np.interp(q,t,values[:,k]),1000,10,300) for k in range(3)]
            for k,axis in enumerate('XYZ'):bank[source+'_'+axis].append(xyz[k])
            bank[source+'_XYZ'].append(np.concatenate(xyz))
        wav=signal.resample_poly(mic.astype(float),1,3,axis=0)
        for k,side in enumerate(['left','right']):
            bank['audio_'+side+'_full'].append(features(wav[:,k],16000,50,7600))
            bank['audio_'+side+'_low'].append(features(wav[:,k],16000,10,300))
        meta.append(r)
        if len(meta)%100==0:print('PREPARED',len(meta),flush=True)
    d={k:np.stack(v) for k,v in bank.items()};d.update(user=np.array([r['user_id'] for r in meta]),word=np.array([r['word_id'] for r in meta]),row=np.array([r['row'] for r in meta]),sample_id=np.array([r['sample_id'] for r in meta]))
    np.savez_compressed(cache,**d);(out/'source_audit_public.json').write_text(json.dumps(dict(users=len(users),source_records=len(rows),included_records=len(meta),excluded=dict(excluded),repaired_source_paths=path_repairs,payload_duplicates=0,source_hash_mismatches=0,user_count_limitation='nine confirmed identities,not ten',word_mapping='inherited raw reconstruction recording-order hypothesis;not independently verified semantic transcript',source_channels='HMD originalXYZ velocity;dv/dt-derived acceleration;microphone two raw channels;not bone-conduction stereo'),indent=2))
    return d

def main(a):
    start=time.time();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);d=prepare(a,out);y=d['user'];words=d['word'];users=np.unique(y);assert len(users)==9
    split_same=np.full(len(y),'',dtype='<U5')
    for u in users:
        for w in range(30):
            ix=np.flatnonzero((y==u)&(words==w));ix=ix[np.argsort(d['row'][ix],kind='stable')];assert 7<=len(ix)<=10,(u,w,len(ix))
            split_same[ix[:5]]='fit';split_same[ix[5:7]]='cal';split_same[ix[7:]]='query'
    ordered=sorted(range(30),key=lambda w:hashlib.sha256(('mei9-channel-cross-v1|'+str(w)).encode()).hexdigest());roles={w:('fit' if j<15 else 'cal' if j<21 else 'query') for j,w in enumerate(ordered)};split_cross=np.array([roles[int(w)] for w in words]);branches=[k for k in d if k not in ['user','word','row','sample_id']];results={};private={}
    for protocol,split in [('same_word_assumed',split_same),('cross_word_assumed',split_cross)]:
        fit=split=='fit';cal=split=='cal';query=split=='query';results[protocol]={};private[protocol]={}
        if protocol=='cross_word_assumed':assert not set(words[fit])&set(words[cal]) and not set(words[fit])&set(words[query]) and not set(words[cal])&set(words[query])
        for branch in branches:
            x=StandardScaler().fit(d[branch][fit]).transform(d[branch]);assert np.isfinite(x).all();rows=[]
            for u in users:
                model=SVC(C=1.,gamma='scale',class_weight='balanced').fit(x[fit],y[fit]==u);cs=model.decision_function(x[cal]);cy=y[cal]==u;ts=np.r_[np.nextafter(cs.min(),-np.inf),np.unique(cs),np.nextafter(cs.max(),np.inf)]
                tau=min(ts,key=lambda t:(abs(float((cs[~cy]>=t).mean())-float((cs[cy]<t).mean())),float((cs[~cy]>=t).mean())))
                s=model.decision_function(x[query]);truth=y[query]==u;tar=float((s[truth]>=tau).mean());far=float((s[~truth]>=tau).mean());rows.append(dict(user=int(u),threshold=float(tau),tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(truth,s)),genuine_n=int(truth.sum()),impostor_n=int((~truth).sum())))
            private[protocol][branch]=rows;results[protocol][branch]={k:float(np.mean([v[k] for v in rows])) for k in ['tar','far','hter','auc']};print('RESULT',protocol,branch,results[protocol][branch],flush=True)
            (out/'results_private.json').write_text(json.dumps(private,indent=2));(out/'summary_public.json').write_text(json.dumps(dict(state='partial',results=results),indent=2))
        np.save(out/(protocol+'_split_private.npy'),split)
    summary=dict(state='complete',users=9,original_recordings=len(y),branches=branches,results=results,split_counts={p:dict(Counter(s)) for p,s in [('same_word_assumed',split_same),('cross_word_assumed',split_cross)]},backend='each branch independent balanced RBF SVC C1 gamma scale;StandardScaler fit-only;peruser CalEER threshold;all8other identities retained;closed-set',frontend='DC removal;logpower40geometriccenters;mean/std/4timeblockmeans240dim peraxis;XYZconcat720dim;allusers same;no perrecord amplitude normalization;HMD10..300Hz;Audio50..7600Hz plus10..300Hz control;source hashes checked',identity_names='private mapping inherits raw9user_order;not oldU1..U9 mapping',limits=['word labels conditional on inherited order hypothesis;not verified strict content','missing last records and nonexact pairing exclusions disclosed','same recording conditions may contribute','XYZ has greater feature dimension than singleaxis','HMD is not ABCS BC;do not conflate sensor modality','handcrafted independent-channel baseline,not generated or teacher feature comparison'],elapsed_seconds=time.time()-start)
    (out/'summary_public.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--raw-root',default='E:/desk/数据操作');main(p.parse_args())
