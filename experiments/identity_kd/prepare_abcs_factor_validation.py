"""Common ABCS text cohort; fixed no-padding windows and real sentence groups."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,zlib,wave,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import signal

def main(a):
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True);audit=Path(a.audit);root=Path(a.root);start=time.time();cov=json.loads((audit/'available_text_coverage_private.json').read_text(encoding='utf-8'))['1'];users=cov['cohort'];texts=sorted(cov['texts'],key=lambda t:hashlib.sha256(('abcs-split-v1|'+t).encode()).hexdigest());assert len(users)==8 and len(texts)==125
    lookup={(r['speaker_id'],r['normalized_text']):r for r in json.loads((audit/'transcript_mapping_private.json').read_text(encoding='utf-8')) if r['present'] and r['speaker_id'] in users and r['normalized_text'] in texts};archive={r['name']:r for r in json.loads((audit/'archive_central_private.json').read_text())}
    def role(j):return 'fit' if j<60 else 'develop' if j<75 else 'registration' if j<87 else 'calibration' if j<100 else 'query' if j<124 else 'unused'
    bank={k:[] for k in ['spectrum','low_spectrum','mid_spectrum','other_spectrum','center_spectrum','amplitude','user','role','text','text_index','sample_id']};seen=set();records=[];durations=[]
    def power(x):
        f,t,z=signal.stft(x,fs=16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);return f,np.log(np.abs(z)**2+1e-20)
    def spec(p,centers):
        f,lp=p;b=np.stack([np.interp(centers,f,lp[:,j]) for j in range(lp.shape[1])],axis=1);return np.stack([np.interp(np.linspace(0,b.shape[1]-1,16),np.arange(b.shape[1]),row) for row in b]).astype('float32')
    for u,name in enumerate(users,1):
        for j,text in enumerate(texts):
            rrole=role(j)
            if rrole=='unused':continue
            r=lookup[name,text];path=root/r['relative_wav'];raw=path.read_bytes();entry=archive['ABCS_database/'+r['relative_wav']];assert len(raw)==entry['size'] and zlib.crc32(raw)==entry['crc32']
            with wave.open(str(path),'rb') as f:
                assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,2,2);pcm=f.readframes(f.getnframes())
            digest=hashlib.sha256(pcm).hexdigest();assert digest not in seen;seen.add(digest);pair=np.frombuffer(pcm,dtype='<i2').reshape(-1,2);assert len(pair)>=20000;durations.append(len(pair)/16000);offset=(len(pair)-20000)//2;pair=pair.astype(float)/32768.;center=pair[offset:offset+20000,0].copy();center-=center.mean();pair-=pair.mean(0,keepdims=True);x=pair[:,0]
            powers=[power(pair[:,ch]) for ch in [0,1]]
            for key,centers,ch in [('spectrum',np.geomspace(50,7600,40),0),('low_spectrum',np.geomspace(10,300,40),0),('mid_spectrum',np.geomspace(50,1200,40),0),('other_spectrum',np.geomspace(50,7600,40),1)]:
                b=spec(powers[ch],centers);assert b.shape==(40,16);bank[key].append(b[None])
            bank['center_spectrum'].append(spec(power(center),np.geomspace(50,7600,40))[None])
            bank['amplitude'].append(np.log(np.r_[np.sqrt(np.mean(x*x)+1e-20),np.max(np.abs(x)),np.quantile(np.abs(x),[.5,.9,.99])]+1e-20))
            for key,value in [('user',u),('role',rrole),('text',text),('text_index',j),('sample_id',r['pair_id'])]:bank[key].append(value)
            records.append(dict(user=u,speaker=name,text=text,role=rrole,pair_id=r['pair_id'],relative_wav=r['relative_wav'],center_control_samples=[offset,offset+20000],main='whole source utterance,time representation normalized16frames'))
        print('PREPARED_USER',u,flush=True)
    d={k:np.asarray(v) for k,v in bank.items()};assert d['spectrum'].shape==(992,1,40,16)
    for rr in ['fit','develop','registration','calibration','query']:
        groups=[set(d['text'][(d['user']==u)&(d['role']==rr)]) for u in range(1,9)];assert all(s==groups[0] for s in groups)
    groups=[set(d['text'][d['role']==rr]) for rr in ['fit','develop','registration','calibration','query']];assert all(not groups[i]&groups[j] for i in range(5) for j in range(i));np.savez_compressed(out/'inputs_private.npz',**d);(out/'manifest_private.json').write_text(json.dumps(records,ensure_ascii=False))
    summary=dict(state='complete',users=8,common_texts=125,used_common_texts=124,records=992,role_counts=dict(Counter(d['role'])),per_user_counts=dict(fit=60,develop=15,registration=12,calibration=13,query=24),unused_texts=1,text_role_overlap=0,all_query_texts_matched_across_users=True,crc_failures=0,exact_pcm_duplicates=0,source_duration_range=[min(durations),max(durations)],window='whole original utterance for main;time-frequency representation interpolated16frames,no waveform stretching/zero padding;center1.25sec only raw-stat control',frontend='16kHz DC removal,4096Hann/1024hop/16384FFT,1x40x16;primary50..7600Hz;10..300/50..1200Hz raw controls',channels='channel0 provisional BC per user;channel1 independent raw-stat control,not fused;official channel index not independently confirmed',split='reuse previous abcs-split-v1 text order;60fit15dev12register13Cal24query1unused;Cal-query texts separate from final-query texts',limits=['partial archive subset,z01 unavailable','no independent same-text repeats for same-content training/testing','no verified redon/session/device labels','official transcript structure checked,not independent listening/ASR','whole-utterance normalized time differs from VibraVox fixed2sec frontend;within-dataset methods matched','shared prompts do not imply framewise phoneme alignment'],elapsed_seconds=time.time()-start)
    (out/'preprocessing_public.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);main(p.parse_args())
