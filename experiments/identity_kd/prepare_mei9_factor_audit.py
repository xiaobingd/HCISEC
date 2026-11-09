"""One raw-HMD frontend for factor ablation and encoder comparisons."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,re,hashlib,time
from pathlib import Path
from datetime import datetime
from collections import Counter,defaultdict
import numpy as np
from scipy import signal
from sklearn.metrics import normalized_mutual_info_score

def main(a):
    start=time.time();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'));wanted=set(np.load(a.previous)['sample_id']);rows=[r for r in manifest['records'] if r['sample_id'] in wanted];assert len(rows)==2695
    bank=[];amplitude=[];quality=[];dates=[];clock=[];rates=[];change_rates=[];meta=[]
    for r in rows:
        p=Path(r['acc_file'])
        if not p.exists():p=Path(a.raw_root)/r['user']/'acc'/p.name
        original=np.load(p,allow_pickle=False);assert hashlib.sha256(memoryview(np.ascontiguousarray(original)).cast('B')).hexdigest()==r['acc_array_sha256']
        t=original[:,0].astype(float);v=original[:,1:].astype(float);order=np.argsort(t,kind='stable');t=t[order];v=v[order];positive=np.diff(t)>0;duplicate_t=float(1-positive.mean());keep=np.r_[True,positive];t=t[keep];v=v[keep];dt=np.diff(t);changed=np.r_[True,np.any(v[1:]!=v[:-1],axis=1)];span=t[-1]-t[0];rates.append(1/np.median(dt));change_rates.append((changed.sum()-1)/span)
        quality.append([len(t),np.median(dt),np.std(dt),np.quantile(dt,.95),t[0],t[-1],duplicate_t,1-changed.mean(),(changed.sum()-1)/span])
        q=np.clip(np.arange(2000)/1000.,t[0],t[-1]);xyz=np.stack([np.interp(q,t,v[:,k]) for k in range(3)]);xyz-=xyz.mean(1,keepdims=True);spec=[];amp=[]
        for x in xyz:
            f,tt,z=signal.stft(x,fs=1000,nperseg=256,noverlap=192,nfft=1024,boundary=None,padded=False);lp=np.log(np.abs(z)**2+1e-20);b=np.stack([np.interp(np.geomspace(10,300,40),f,lp[:,j]) for j in range(lp.shape[1])],axis=1);assert b.shape==(40,28);spec.append(b.astype('float32'));amp.extend(np.log(np.r_[np.sqrt(np.mean(x*x)+1e-20),np.max(np.abs(x)),np.quantile(np.abs(x),[.5,.9,.99])]+1e-20))
        bank.append(spec);amplitude.append(amp);stamp=re.search(r'(\d{14})\.npy$',p.name);assert stamp;date=datetime.strptime(stamp.group(1),'%Y%m%d%H%M%S');dates.append(date.strftime('%Y-%m-%d'));clock.append(date.timestamp());meta.append(r)
        if len(meta)%400==0:print('PREPARED',len(meta),flush=True)
    user=np.array([r['user_id'] for r in meta]);word=np.array([r['word_id'] for r in meta]);role=np.array([r['role'] for r in meta]);days_by={u:len(set(np.array(dates)[user==u])) for u in np.unique(user)}
    np.savez_compressed(out/'common_frontend_private.npz',spectrum=np.array(bank),amplitude=np.array(amplitude,dtype='float32'),quality=np.array(quality,dtype='float32'),user=user,word=word,role=role,sample_id=np.array([r['sample_id'] for r in meta]),record_date=np.array(dates),clock=np.array(clock))
    p=dict(state='complete',records=len(meta),identities=9,spectrum_shape=list(np.array(bank).shape),source='OpenVR vVelocity XYZ,not rawIMU;source hash verified',common_frontend='timestamp interpolate1000Hz2sec;DCremoval;256Hann64hop1024FFT;40logfreq10..300Hz x28frames;no resizing',amplitude='15logstats:eachaxis RMS/peak/absolute p50,p90,p99;afterDC',quality='9collection proxies:samplecount,dtmedian/std/p95,start/end,duplicateTimestampFraction,unchangedVectorFraction,changeEventsPerSec;not physical device/wear labels',unique_capture_dates=len(set(dates)),users_with_multiple_capture_dates=sum(n>1 for n in days_by.values()),capture_date_count_per_user_histogram=dict(Counter(days_by.values())),user_date_nmi=float(normalized_mutual_info_score(user,dates)),nominal_poll_rate_quantiles=np.quantile(rates,[0,.1,.5,.9,1]).tolist(),vector_change_rate_quantiles=np.quantile(change_rates,[0,.1,.5,.9,1]).tolist(),device_identifiers_found=False,verified_redon_labels_found=False,limits=['timestamp day is not a wearing/device label','polling/vector-change event rates do not establish sensor analog bandwidth','gain/axis perturbation controls are simulations,not measured redon','wordlabels inherit unverified order mapping'],elapsed_seconds=time.time()-start)
    (out/'source_audit_public.json').write_text(json.dumps(p,indent=2));print('COMPLETE',json.dumps(p),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--previous',required=True);p.add_argument('--out',required=True);p.add_argument('--raw-root',default='E:/desk/数据操作');main(p.parse_args())
