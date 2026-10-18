"""Timestamp-paired Data0723 mei learning curve with one shared frontend.

Read only previous TRAIN/validation source files. Original test rows stay closed.
"""
import os, sys, json, zipfile, hashlib, argparse
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy import signal, ndimage

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);args=p.parse_args()
    root=args.root;out=root/'data';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root.parent))
    from data0723_frontend_helpers import create_mel_filterbank
    fb=create_mel_filterbank(16000,512,80,20,7500)
    membership=json.loads((root/'personal_membership.json').read_text())
    source=json.loads(args.manifest.read_text())
    original=[r for r in source['records'] if r['user']=='wangjingmei']
    assert len(original)==400
    # Same timestamp/key occurs in different task folders. Protect validation AND old test
    # source keys using metadata alone; never open original test arrays.
    reserved_keys={r['key'] for r in original if r['split']=='test'}
    held_keys={r['key'] for r in original if r['split']!='train'}
    canonical={}
    for row in original:
        if row['split']=='train' and row['key'] in held_keys:continue
        if row['split']=='validation' and row['key'] in reserved_keys:continue
        if row['split'] not in ['train','validation']:continue
        canonical.setdefault((row['split'],row['key']),row)
    rows=list(canonical.values())
    assert sum(r['split']=='train' for r in rows)==180 and sum(r['split']=='validation' for r in rows)==60
    rng=np.random.default_rng(20261018);orders={}
    for task in ['0-9','word']:
        names=sorted(r['task']+'__'+r['key'] for r in rows if r['split']=='train' and r['task']==task)
        orders[task]=np.array(names)[rng.permutation(len(names))].tolist()
    subsets={str(n):orders['0-9'][:a]+orders['word'][:b] for n,a,b in [(20,8,12),(70,27,43),(180,70,110)]}
    assert set(subsets['20'])<set(subsets['70'])<set(subsets['180'])
    bands=[(2,10),(10,40),(40,100),(100,200),(200,450),(450,490)]
    exported={};source_audit=[];raw_hashes={};crop_rows=[]
    def features(x,a,g,split,domain,task):
        assert len(x)>=800 and len(a)>=12800 and np.isfinite(x).all() and np.isfinite(a).all()
        x=x-x.mean(0);energy=[]
        for xx in [x]+[signal.sosfiltfilt(signal.butter(4,b,btype='bandpass',fs=1000,output='sos'),x,axis=0) for b in bands]:
            for ms in (50,250):
                power=np.maximum(ndimage.uniform_filter1d(xx*xx,size=ms,axis=0,mode='reflect'),0)
                energy.extend([np.log10(np.sqrt(power[:,j])+1e-12) for j in range(3)])
                energy.append(np.log10(np.sqrt(power.sum(1))+1e-12))
        energy=np.array(energy);last=min(len(x)-800,int(np.floor((len(a)-12800)/16)))
        starts=list(range(0,last+1,400))
        if starts[-1]!=last:starts.append(last)
        ss=[];ee=[];yy=[]
        for b in starts:
            aa=a[b*16:b*16+12800];w=x[b:b+800].T
            assert aa.shape==(12800,) and w.shape==(3,800)
            _,_,z=signal.stft(w,fs=1000,nperseg=128,noverlap=118,nfft=128,axis=-1)
            ss.append((20*np.log10(abs(z)+1e-12)).reshape(195,81))
            ix=np.minimum(b+np.arange(81)*10,energy.shape[1]-1);ee.append(energy[:,ix])
            _,_,az=signal.stft(aa,fs=16000,nperseg=512,noverlap=352)
            yy.append(10*np.log10(fb@abs(az)**2+1e-10))
            crop_rows.append(dict(domain=domain,recording=g,split=split,start_s=b/1000))
        source_audit.append(dict(domain=domain,recording=g,task=task,split=split,duration=min(len(x)/1000,len(a)/16000),crops=len(starts)))
        exported[g]=dict(spec=np.array(ss),energy=np.array(ee),audio=np.array(yy),split=split,domain=domain)
    with np.load(root/'personal_train_val_raw.npz') as personal:
        for g,split in sorted(membership.items()):
            x=personal[g+'__acc'].copy();a=personal[g+'__audio'].copy();fs=int(personal[g+'__fs'])
            if fs!=16000:a=signal.resample_poly(a,16000//np.gcd(fs,16000),fs//np.gcd(fs,16000))
            features(x,a,'personal__'+g,'train' if split=='train' else 'validation','personal','personal')
    with zipfile.ZipFile(args.archive) as zf:
        for k,row in enumerate(rows):
            prefix=row['task']+'/'+row['user']+'/'
            ap=prefix+row['key']+'.npy';yp=prefix+row['key'].replace('accel_data','audio_data',1)+'.npy'
            ab=zf.read(ap);yb=zf.read(yp)
            import io
            acc=np.load(io.BytesIO(ab)).astype('float64');mic=np.load(io.BytesIO(yb)).astype('float64')
            assert acc.ndim==2 and acc.shape[1]==4 and mic.ndim==2 and mic.shape[1]==2
            assert np.isfinite(acc).all() and np.isfinite(mic).all()
            t=acc[:,0];order=np.argsort(t,kind='stable');t=t[order];xyz=acc[order,1:4]
            keep=np.r_[True,np.diff(t)>0];t=t[keep];xyz=xyz[keep];t=t-t[0]
            duration=min(t[-1],len(mic)/48000);grid=np.arange(0,duration,.001)
            x=np.stack([np.interp(grid,t,xyz[:,j]) for j in range(3)],1)
            a=signal.resample_poly(mic.mean(1),1,3)
            g=row['task']+'__'+row['key'];raw_hashes[g]=(row['split'],hashlib.sha256(ab+yb).hexdigest())
            features(x,a,g,row['split'],'mei',row['task'])
            if (k+1)%40==0:print('MEI_EXPORTED',k+1,flush=True)
    # Prevent exact duplicate source pairs across gradient fit / external validation.
    th={h for s,h in raw_hashes.values() if s=='train'};vh={h for s,h in raw_hashes.values() if s=='validation'}
    assert not th&vh,'Duplicate raw source pairs across split'
    assert len(th)==180 and len(vh)==60,'Further exact duplicates need auditing'
    personal_train=[g for g,d in exported.items() if d['domain']=='personal' and d['split']=='train']
    # Smallest nested mei training set + original20 personal TRAIN; no large-set-only calibration benefit.
    calibration=personal_train+subsets['20']
    specs=np.concatenate([exported[g]['spec'] for g in calibration]);eng=np.concatenate([exported[g]['energy'] for g in calibration]);audio=np.concatenate([exported[g]['audio'] for g in calibration])
    lo,hi=np.percentile(specs,[1,99]);mu=eng.mean((0,2))[None,:,None];sd=np.maximum(eng.std((0,2)),.05)[None,:,None];alo,ahi=np.percentile(audio,[1,99])
    def save(name,names):
        xx=[];yy=[];ids=[]
        for g in names:
            d=exported[g];spec=2*(d['spec']-lo)/(hi-lo)-1;energy=(d['energy']-mu)/sd
            xx.append(np.concatenate([spec,energy],axis=1));yy.append(np.clip(2*(d['audio']-alo)/(ahi-alo)-1,-1,1));ids.extend([g]*len(spec))
        x=np.concatenate(xx).astype('float32');y=np.concatenate(yy).astype('float32')
        assert x.shape[1:]==(251,81) and y.shape[1:]==(80,81) and np.isfinite(x).all() and np.isfinite(y).all()
        np.savez_compressed(out/(name+'.npz'),x=x,y=y,ids=np.array(ids),activity_fraction=np.ones(len(ids),dtype='float32'))
        return dict(recordings=len(names),windows=len(ids))
    counts={}
    for n,names in subsets.items():counts['mei'+n]=save('mei'+n+'_train',names)
    for domain in ['mei','personal']:
        names=[g for g,d in exported.items() if d['domain']==domain and d['split']=='validation']
        counts[domain+'_validation']=save(domain+'_val',names)
    counts['personal20']=save('personal20_train',personal_train)
    protocol=dict(source='Data0723/wangjingmei raw timestamp-matched npy pairs, NOT old nine-user spectral arrays; personal=previous20TRAIN/4REUSEDval rawvAccel',counts=counts,nested_subsets=subsets,split='Original task-wise280TRAIN contains duplicate source keys and crosses VAL/test metadata. Protect all originalVAL/test keys, remove their TRAIN copies, keep one canonical TRAIN row per source key:180uniqueTRAIN/60VAL, original60test metadata protected and arrays NOTopened. Cross-split raw content hashes checked. Old test was historically evaluated, not a pristine confirmation set. No auth model trained.',preprocessing='SAME all domains/subsets: nominal first-sample-zero pairing; source timestamps sorted/deduplicated and interpolate signedXYZ at1000Hz; subtract fullrecord DC; main spectrum0-500Hz XYZ STFT128/hop10 plus56multiscale energies, no phase. Audio mono16k, original HTKMel80 nfft512/hop160. Fullrecord .8s crops/.4s stride, include final validcrop. No padding, stretching, lag search, per-user/window gain or normalization.',source_conversion='Data0723 actual source48k stereo -> mean mono and polyphase16k; existingpersonal16k unchanged. Timestamp interpolation is source ingestion, common downstream frontend. Physical source type/unit may differ; cross-domain comparison is not a data-size causal test.',normalization=dict(scope='ONE pooled calibration using original20personalTRAIN plus nested20meiTRAIN only, fixed IDENTICAL for every new route; does not use additional70/180-only records or anyVAL/test. Each low-data mei model has no feature stats from larger mei subsets.',spec_lo_hi=[float(lo),float(hi)],energy_mean=mu.ravel().tolist(),energy_std=sd.ravel().tolist(),audio_lo_hi=[float(alo),float(ahi)]),calibration_recordings=calibration,sampler='record-uniform; uniform crop per record, activity_fraction ones is compatibility field, no audio activity filtering',exact_raw_train_val_duplicate_pairs=len(th&vh),unique_train_raw_pairs=len(th),unique_val_raw_pairs=len(vh),test_source_keys_blocked_from_train=len(reserved_keys),source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),source_audit=source_audit,crops=crop_rows,scope='Within-mei nested20/70/180 isolates gradient-training record count under fixed frontend/model/updates/validation; source/timing/session/content diversity grows with record count. Single subject per model, not identity accuracy.')
    (root/'data_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('COUNTS',json.dumps(counts),flush=True)
if __name__=='__main__':main()
