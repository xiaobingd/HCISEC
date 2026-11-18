"""Record-disjoint mei/ke authentication on the current frozen shared frontend."""
import os,sys,json,argparse,zipfile,hashlib,io
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy import signal,ndimage

def partition(rows,sizes,seed):
    assert sum(sizes)==len(rows)
    rng=np.random.default_rng(seed);pools={}
    for task in sorted({r['task'] for r in rows}):
        rr=sorted([r for r in rows if r['task']==task],key=lambda r:r['key']);pools[task]=[rr[i] for i in rng.permutation(len(rr))]
    groups=[]
    for count in sizes:
        remaining=sum(map(len,pools.values()));tasks=sorted(pools);allocation={t:int(np.floor(count*len(pools[t])/remaining)) for t in tasks}
        while sum(allocation.values())<count:
            t=max(tasks,key=lambda t:(count*len(pools[t])/remaining-allocation[t],len(pools[t])-allocation[t]));allocation[t]+=1
        g=[]
        for task,n in allocation.items():g.extend(pools[task][:n]);pools[task]=pools[task][n:]
        groups.append(g)
    return groups

def frontend(row,zf,norm,audio=False):
    p=row['task']+'/'+row['user']+'/';a=np.load(io.BytesIO(zf.read(p+row['key']+'.npy'))).astype('float64');assert a.ndim==2 and a.shape[1]==4 and np.isfinite(a).all()
    ix=np.argsort(a[:,0],kind='stable');t=a[ix,0];x=a[ix,1:4];keep=np.r_[True,np.diff(t)>0];t=t[keep];x=x[keep];t-=t[0]
    grid=np.arange(0,min(t[-1],2.),.001);x=np.stack([np.interp(grid,t,x[:,j]) for j in range(3)],1);x-=x.mean(0);assert 1997<=len(x)<=2000
    bands=[(2,10),(10,40),(40,100),(100,200),(200,450),(450,490)];energy=[]
    for xx in [x]+[signal.sosfiltfilt(signal.butter(4,b,btype='bandpass',fs=1000,output='sos'),x,axis=0) for b in bands]:
        for ms in (50,250):
            power=np.maximum(ndimage.uniform_filter1d(xx*xx,size=ms,axis=0,mode='reflect'),0)
            energy.extend([np.log10(np.sqrt(power[:,j])+1e-12) for j in range(3)]);energy.append(np.log10(np.sqrt(power.sum(1))+1e-12))
    _,_,zz=signal.stft(x.T,fs=1000,nperseg=128,noverlap=118,nfft=128,axis=-1);assert zz.shape==(3,65,201)
    lo,hi=norm['spec_lo_hi'];mu=np.array(norm['energy_mean'])[:,None];sd=np.array(norm['energy_std'])[:,None];frame=np.minimum(np.arange(201)*10,len(x)-1)
    result=np.concatenate([(2*(20*np.log10(abs(zz)+1e-12)-lo)/(hi-lo)-1).reshape(195,201),(np.array(energy)[:,frame]-mu)/sd]).astype('float32');assert result.shape==(251,201) and np.isfinite(result).all()
    if not audio:return result,None
    from data0723_frontend_helpers import create_mel_filterbank
    mic=np.load(io.BytesIO(zf.read(p+row['key'].replace('accel_data','audio_data',1)+'.npy'))).astype('float64');assert mic.shape==(96000,2)
    aa=signal.resample_poly(mic.mean(1),1,3);_,_,az=signal.stft(aa,fs=16000,nperseg=512,noverlap=352);fb=create_mel_filterbank(16000,512,80,20,7500);alo,ahi=norm['audio_lo_hi'];yy=np.clip(2*(10*np.log10(fb@abs(az)**2+1e-10)-alo)/(ahi-alo)-1,-1,1).astype('float32');assert yy.shape==(80,201)
    return result,yy

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['development','final'],default='development');args=p.parse_args();root=args.output;root.mkdir(exist_ok=True);sys.path.insert(0,str(args.source.parent))
    dp=json.loads((args.source/'data_protocol.json').read_text());norm=dp['normalization'];meta=json.loads(args.source_manifest.read_text());canonical={}
    for row in sorted(meta['records'],key=lambda r:(r['task'],r['key'])):canonical.setdefault((row['user'],row['key']),row)
    assert sum(u=='wangjingmei' for u,k in canonical)==300 and sum(u=='chengkeyi' for u,k in canonical)==298
    ganrows=[canonical[('wangjingmei',r['recording'].split('__',1)[1])] for r in dp['source_audit'] if r['domain']=='mei' and r['split']=='train'];assert len(ganrows)==180
    valrows=[canonical[('wangjingmei',r['recording'].split('__',1)[1])] for r in dp['source_audit'] if r['domain']=='mei' and r['split']=='validation'];check=partition(valrows,[30,30],20261019)[0]
    forbidden={r['key'] for r in ganrows+check};available=[r for (u,k),r in canonical.items() if u=='wangjingmei' and k not in forbidden];assert len(available)==90
    target=partition(available,[30,10,10,10,30],42);ke=partition([r for (u,k),r in canonical.items() if u=='chengkeyi'],[120,30,30,118],43)
    groups=dict(gan_train=ganrows,gan_check=check,head_train=target[0]+ke[0],head_validation=target[1]+ke[1],enrollment=target[2],calibration=target[3]+ke[2],final_test=target[4]+ke[3])
    memberships={k:[r['user']+'|'+r['key'] for r in rows] for k,rows in groups.items()}
    for a,aa in memberships.items():
        for b,bb in memberships.items():
            if a!=b:assert not set(aa)&set(bb),(a,b)
    # The frozen calibration's mei members must remain in generation gradient TRAIN.
    assert set(dp['nested_subsets']['20'])<=set(r['recording'] for r in dp['source_audit'] if r['domain']=='mei' and r['split']=='train')
    with zipfile.ZipFile(args.archive) as zf:
        if args.stage=='development':
            hashes={};acc_hashes={}
            for key,row in canonical.items():
                prefix=row['task']+'/'+row['user']+'/';ab=zf.read(prefix+row['key']+'.npy');yb=zf.read(prefix+row['key'].replace('accel_data','audio_data',1)+'.npy');hashes[key]=hashlib.sha256(ab+yb).hexdigest();acc_hashes[key]=hashlib.sha256(ab).hexdigest()
            for name in ['pair','raw_acc']:
                hs=hashes if name=='pair' else acc_hashes;hgroups={g:{hs[(r['user'],r['key'])] for r in rows} for g,rows in groups.items()}
                for a in groups:
                    for b in groups:
                        if a!=b:assert not hgroups[a]&hgroups[b],('duplicate_content',name,a,b)
            counts={g:{u:sum(r['user']==u for r in rows) for u in ['wangjingmei','chengkeyi']} for g,rows in groups.items()}
            protocol=dict(scope='Exploratory two-user same-collection-session authentication: mei enrolled; ke identity seen in head training but final recordings disjoint. Not unseen-impostor-identity open-set, not cross-day/person-general evidence.',counts=counts,memberships=memberships,groups=groups,shared_normalization=norm,split_seed=42,split_method='Unique(user,key) canonical0-9/word; CGtrain exactly previous180; seededtaskstratified30 ofprevious60VAL forGANcheck. Remaining90mei=30head/10headval/10enroll/10thresholdcal/30final. All298ke=120head/30headval/30thresholdcal/118final, taskstratified. No sources deleted by authentication outcome.',assertions=dict(source_ids_disjoint=True,raw_pair_hashes_disjoint=True,raw_ACC_file_hashes_disjoint=True,normalization_frozen=True,mei_calibration_members_CGtrain=True),source_hashes={u+'|'+k:h for (u,k),h in hashes.items()},limits='Both users same shared processing but actual original signal provenance and sessions can confound identity. Historical60VAL/test mei previously explored; final authentication split locked before head training, not pristine independently collected confirmation. No per-user normalization. Cached CGTRAIN arrays unchanged; final signals only exported after heads locked.')
            (root/'split_manifest.json').write_text(json.dumps(protocol,indent=2));selected=['gan_check','head_train','head_validation','enrollment','calibration']
        else:
            assert (root/'heads_locked.json').exists(),'Lock head states and calibration thresholds before final export'
            locked=json.loads((root/'split_manifest.json').read_text());assert locked['memberships']==memberships;selected=['final_test']
        for group in selected:
            xx=[];yy=[];ids=[];tasks=[];labs=[]
            for row in groups[group]:
                x,y=frontend(row,zf,norm,audio=(group=='gan_check'));xx.append(x);ids.append(row['user']+'|'+row['key']);tasks.append(row['task']);labs.append(0 if row['user']=='wangjingmei' else 1)
                if y is not None:yy.append(y)
            data=dict(x=np.array(xx),ids=np.array(ids),tasks=np.array(tasks),labels=np.array(labs))
            if yy:data['y']=np.array(yy)
            np.savez_compressed(root/(group+'.npz'),**data);print('EXPORTED',group,len(xx),flush=True)
    print('DATA_STAGE_DONE',args.stage,flush=True)
if __name__=='__main__':main()
