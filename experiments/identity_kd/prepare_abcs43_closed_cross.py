"""All43 known identities, globally disjoint exact sentence roles, private exports."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,unicodedata,wave,zlib
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import signal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def h(t):return hashlib.sha256(t.encode()).hexdigest()
def canon(t):return ' '.join(unicodedata.normalize('NFC',t).casefold().split())
def role(t):
    x=int(h('closed3-ABCS-text-v1|'+t)[:8],16)%100
    return 'fit' if x<45 else 'develop' if x<55 else 'registration' if x<70 else 'calibration' if x<80 else 'query'

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);old=json.loads(Path(a.old_manifest).read_text(encoding='utf8'));users=old['users'];assert len(users)==43;pool=defaultdict(dict)
    for r in sorted(json.loads(Path(a.mapping).read_text(encoding='utf8')),key=lambda r:r['relative_wav']):
        if r['present'] and r['speaker_id'] in users:pool[r['speaker_id']].setdefault(canon(r['normalized_text']),r)
    records=[]
    for u,name in enumerate(users):
        for rr,n in [('fit',60),('develop',15),('registration',12),('calibration',8),('query',24)]:
            texts=sorted([t for t in pool[name] if role(t)==rr],key=lambda t:h('closed3-ABCS-record-v1|'+t));assert len(texts)>=n,(u,rr,len(texts))
            for t in texts[:n]:records.append(dict(pool[name][t],user=u,role=rr,text=t))
    assert len(records)==5117;vocab={t:i for i,t in enumerate(sorted({r['text'] for r in records}))};d=dict(user=np.array([r['user'] for r in records]),role=np.array([r['role'] for r in records]),text_id=np.array([vocab[r['text']] for r in records]),sample_id=np.array([f'ABCS-C3-{i:05}' for i in range(len(records))]));fit=np.flatnonzero(d['role']=='fit');reg=np.flatnonzero(d['role']=='registration');dev=np.concatenate([np.flatnonzero((d['user']==u)&(d['role']=='develop'))[:4] for u in range(43)]);cal=np.concatenate([np.flatnonzero((d['user']==u)&(d['role']=='calibration')) for u in range(43)]);ix=dict(fit=fit,dev=dev,register=reg,calA=np.concatenate([np.flatnonzero((d['user']==u)&(d['role']=='calibration'))[:4] for u in range(43)]),calB=np.concatenate([np.flatnonzero((d['user']==u)&(d['role']=='calibration'))[4:8] for u in range(43)]),query=np.flatnonzero(d['role']=='query'));teacher=np.unique(np.r_[np.concatenate([np.flatnonzero((d['user']==u)&(d['role']=='fit'))[:4] for u in range(43)]),reg]);allowed=set(np.r_[fit,dev,reg].tolist());archive={r['name']:r for r in json.loads(Path(a.archive).read_text(encoding='utf8'))}
    def process(item):
        i,r=item;p=Path(a.raw)/r['relative_wav'];blob=p.read_bytes();entry=archive['ABCS_database/'+r['relative_wav']];assert len(blob)==entry['size'] and zlib.crc32(blob)==entry['crc32']
        with wave.open(str(p),'rb') as f:assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,2,2);pcm=f.readframes(f.getnframes())
        pair=np.frombuffer(pcm,dtype='<i2').reshape(-1,2).astype('float64')/32768.;acc=pair[:,0].copy();acc-=acc.mean();freq,_,z=signal.stft(acc,16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);b=np.stack([np.interp(np.geomspace(50,7600,40),freq,col) for col in np.log(abs(z)**2+1e-20).T],1);grid=lambda bb,n:np.stack([np.interp(np.linspace(0,bb.shape[1]-1,n),np.arange(bb.shape[1]),row) for row in bb]).astype('float32');b16=grid(b,16);aa=grid(b,32);m=b16.mean(1);sd=b16.std(1);seg=np.stack([b16[:,q].mean(1) for q in np.array_split(np.arange(16),4)]);amp=np.log(np.r_[np.sqrt(np.mean(acc*acc)+1e-20),abs(acc).max(),np.quantile(abs(acc),[.5,.9,.99])]+1e-20);x=np.r_[m-m.mean(),sd,(seg-m).ravel(),m.mean(),amp].astype('float32');st=np.r_[m,sd,seg.ravel()].astype('float32');bb=np.zeros_like(aa);w=None;l=None
        if i in allowed:
            mic=pair[:,1].copy();mic-=mic.mean();f,_,zz=signal.stft(mic,16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);bb=grid(np.stack([np.interp(np.geomspace(50,7600,40),f,col) for col in np.log(abs(zz)**2+1e-20).T],1),32)
        if i in set(teacher):
            v=pair[:,1];n=min(len(v),32000);start=max((len(v)-32000)//2,0);v=v[start:start+n].copy();v-=v.mean();w=np.pad(v,(0,32000-n)).astype('float32');l=n/32000
        return i,x,st,aa,bb,w,l,hashlib.sha256(pcm).hexdigest()
    xx=[];ss=[];aa=[];bb=[];ww=[];ll=[];seen=set()
    with ThreadPoolExecutor(max_workers=8) as workers:
        for i,x,s,av,bv,w,l,digest in workers.map(process,enumerate(records)):
            assert digest not in seen;seen.add(digest);xx.append(x);ss.append(s);aa.append(av);bb.append(bv)
            if w is not None:ww.append(w);ll.append(l)
            if (i+1)%500==0:print('CLOSED_ABCS_PREPARE',i+1,len(records),flush=True)
    d.update(x=np.stack(xx),statistics=np.stack(ss));np.savez_compressed(out/'inputs_private.npz',**d);aa=np.stack(aa)[:,None];bb=np.stack(bb)[:,None];am=float(aa[fit].mean());asd=float(aa[fit].std());bm=float(bb[fit].mean());bsd=float(bb[fit].std());aa=np.clip((aa-am)/(3*asd),-1,1);valid=np.array(sorted(allowed));bb[valid]=np.clip((bb[valid]-bm)/(3*bsd),-1,1);np.savez_compressed(out/'spectra_private.npz',a=aa,b=bb,audio_indices=valid,input_mean=np.array(am),input_std=np.array(asd),audio_mean=np.array(bm),audio_std=np.array(bsd));np.savez_compressed(out/'teacher_audio_private.npz',indices=teacher,wave=np.stack(ww),lengths=np.array(ll,dtype='float32'));(out/'split_private.json').write_text(json.dumps({k:v.tolist() for k,v in ix.items()}));(out/'manifest_private.json').write_text(json.dumps(dict(users=users,records=records),ensure_ascii=False),encoding='utf8')
    # Text-only identity diagnostic; never enters signal encoders or authenticators.
    vec=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=20000);text=[r['text'] for r in records];tx=vec.fit_transform([text[i] for i in fit]);model=LogisticRegression(C=1,max_iter=500,class_weight='balanced',random_state=42).fit(tx,d['user'][fit]);query=ix['query'];acc=float((model.predict(vec.transform([text[i] for i in query]))==d['user'][query]).mean());pub=dict(state='prepared',dataset='ABCS',users=43,records=len(records),role_counts=dict(Counter(d['role'])),used_counts={k:len(v) for k,v in ix.items()},teacher_audio_records=len(teacher),audio_spectrum_records=len(valid),crc_checks=len(records),exact_pcm_duplicates=0,query_audio_model_forwards=0,query_audio_spectrum_calculations=0,text_roles_global_overlap=0,text_only_query_identity_accuracy=acc,text_only_chance=1/43,identity_selection='same metadata-fixed43,allincluded;no authentication-score selection',text_selection='sha256 closed3-ABCS-text-v1 45fit10dev15reg10cal20query;fixed record hash caps',channel_mapping='ch0 provisional vibration,ch1 provisional Audio,not physically verified',stereo_note='whole WAV bytes read for CRC/ACC;query ch1 not transformed or model-forwarded',frontend='full utterance;16k/4096Hann1024hopFFT16384,40geom50..7600;GAN32time,stat16time;Basefit globalmodality normalization',files={p.name:p.stat().st_size for p in out.glob('*.npz')});(out/'preparation_public.json').write_text(json.dumps(pub,indent=2));print(json.dumps(pub),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--old-manifest',default='outputs/abcs43_shape_v4_20261110/manifest_private.json');p.add_argument('--mapping',default='outputs/abcs_content_audit_20261109/transcript_mapping_private.json');p.add_argument('--archive',default='outputs/abcs_content_audit_20261109/archive_central_private.json');p.add_argument('--raw',default='D:/BaiduNetdiskDownload/ABCS_database');main(p.parse_args())
