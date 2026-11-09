"""ABCS closed-set cross-sentence RBF, matched query texts, physical channels separate."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,time,wave,zlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

def main(a):
    start=time.time();audit=Path(a.audit);root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    coverage=json.loads((audit/'available_text_coverage_private.json').read_text(encoding='utf-8'))['1'];users=coverage['cohort']
    texts=sorted(coverage['texts'],key=lambda t:hashlib.sha256(('abcs-split-v1|'+t).encode()).hexdigest());assert len(texts)==125 and len(users)==8
    roles={t:('fit' if i<75 else 'cal' if i<100 else 'query') for i,t in enumerate(texts)}
    rows=json.loads((audit/'transcript_mapping_private.json').read_text(encoding='utf-8'))
    archive={x['name']:x for x in json.loads((audit/'archive_central_private.json').read_text())}
    mapping={}
    for r in rows:
        if r['present'] and r['speaker_id'] in users and r['normalized_text'] in roles:
            key=(r['speaker_id'],r['normalized_text']);assert key not in mapping,'Unexpected multiple utterances: lock independent-repeat policy first';mapping[key]=r
    assert len(mapping)==1000
    features={0:[],1:[]};labels=[];split=[];records=[];seen={};clip={0:[],1:[]};durations=[]
    centers=np.geomspace(50,1200,40)
    def pool(v):return np.concatenate([v.mean(1),v.std(1)]+[v[:,q].mean(1) for q in np.array_split(np.arange(v.shape[1]),4)])
    for u,name in enumerate(users):
        for text in texts:
            r=mapping[name,text];path=root/r['relative_wav'];raw=path.read_bytes();entry=archive['ABCS_database/'+r['relative_wav']]
            assert len(raw)==entry['size'] and zlib.crc32(raw)==entry['crc32']
            with wave.open(str(path),'rb') as f:
                assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,2,2)
                pcm=f.readframes(f.getnframes())
            h=hashlib.sha256(pcm).hexdigest();assert h not in seen,'Duplicate source recording';seen[h]=r['pair_id']
            pair=np.frombuffer(pcm,dtype='<i2').reshape(-1,2);assert len(pair)>=2048
            durations.append(len(pair)/16000)
            for ch in [0,1]:
                x=pair[:,ch].astype(float)/32768.;clip[ch].append(float((np.abs(x)>=.999).mean()));x-=x.mean()
                f,t,z=signal.stft(x,fs=16000,nperseg=1024,noverlap=768,nfft=2048,boundary=None,padded=False)
                p=np.abs(z)**2;b=np.stack([np.interp(centers,f,np.log(p[:,it]+1e-12)) for it in range(p.shape[1])],axis=1)
                assert np.isfinite(b).all() and b.shape[1]>=4
                n=(b-b.mean())/max(b.std(),1e-4);energy=np.r_[np.log(np.sqrt(np.mean(x*x)+1e-12)),np.log(np.quantile(np.abs(x),[.1,.5,.9,.99])+1e-12)]
                features[ch].append((pool(b),pool(n),np.r_[pool(n),energy]))
            labels.append(u);split.append(roles[text]);records.append(dict(pair_id=r['pair_id'],speaker=name,text=text,role=roles[text]))
        print('PREPARED_USER',u+1,flush=True)
    y=np.array(labels);split=np.array(split);fit=split=='fit';cal=split=='cal';query=split=='query';result={};private={}
    assert Counter(split)=={'fit':600,'cal':200,'query':200}
    for ch in [0,1]:
        result[str(ch)]={};private[str(ch)]={}
        for k,method in enumerate(['global_scale','instance_norm','instance_norm_plus_energy']):
            x=np.stack([v[k] for v in features[ch]]);sc=StandardScaler().fit(x[fit]);x=sc.transform(x);metrics=[];details=[]
            for u in range(8):
                m=SVC(C=1.,gamma='scale',class_weight='balanced').fit(x[fit],y[fit]==u)
                cs=m.decision_function(x[cal]);cy=y[cal]==u
                thresholds=np.r_[np.nextafter(cs.min(),-np.inf),np.unique(cs),np.nextafter(cs.max(),np.inf)]
                tau=min(thresholds,key=lambda t:(abs(float((cs[~cy]>=t).mean())-float((cs[cy]<t).mean())),float((cs[~cy]>=t).mean())))
                s=m.decision_function(x[query]);truth=y[query]==u;tar=float((s[truth]>=tau).mean());far=float((s[~truth]>=tau).mean())
                row=dict(tar=tar,far=far,hter=(1-tar+far)/2,auc=float(roc_auc_score(truth,s)));metrics.append(row);details.append(dict(user=users[u],threshold=float(tau),metrics=row))
            result[str(ch)][method]={key:float(np.mean([r[key] for r in metrics])) for key in metrics[0]};private[str(ch)][method]=details;print('RESULT',ch,method,result[str(ch)][method],flush=True)
    public=dict(status='complete',protocol='closed-set cross-sentence;all8identities in fit;actual normalized texts globally disjoint75fit25cal25query;every query text matched across all8people;NOT same-text fit/test;NOT unseen-user open-set',users=8,common_texts=125,fit_per_user=75,cal_per_user=25,query_per_user=25,impostors_per_target=175,total_original_wavs=1000,exact_pcm_duplicates=0,crc_failures=0,frontend='16kHz stereo separated physically;DC removal;64msHann16mshop;logpower interpolated50..1200Hz40centers;whole utterance variable duration,no waveform time warping;240dim pooled spectrum or245with energy',normalization='global scaler fit only on600fit recordings;instance normalization one mean/std per original recording;identical rules for allusers',backend='8 independent balanced RBF SVC heads;C1 gamma scale;each per-user threshold determined only on Cal via FAR/FRR balance;no test-based choice or attacker removal',channel_semantics='UNCONFIRMED;report both physical channels,do not infer BC from better authentication',results=result,max_clipping_fraction={str(ch):max(clip[ch]) for ch in [0,1]},duration_range_seconds=[min(durations),max(durations)],elapsed_seconds=time.time()-start,limits=['official transcription structurally verified,not independent listening/ASR checked','8users selected by shared-text metadata coverage,not authentication results','partial release subset;missing z01','same speaker/session conditions may contribute','no same-sentence independent repeats','handcrafted RBF diagnostic,not original772dim baseline or Audio-guided teacher','no Audio-teacher benefit established'])
    for name,obj in [('summary_public.json',public),('results_private.json',private),('split_private.json',records)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(public),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);main(p.parse_args())
