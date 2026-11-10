"""Reuse fixed59 cohort, all identities known, new closed-set legal Audio cache."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import signal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main(a):
    source=Path(a.source);paired=Path(a.paired);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);old=dict(np.load(source/'inputs_private.npz'));records=json.loads((source/'manifest_private.json').read_text())['records'];users=sorted(np.unique(old['user']).tolist());assert users==list(range(59));fit=np.flatnonzero(old['role']=='fit');reg=np.flatnonzero(old['role']=='registration');devrows={u:np.flatnonzero((old['user']==u)&(old['role']=='develop')) for u in users};assert min(map(len,devrows.values()))>=10;ix=dict(fit=fit,register=reg,calA=np.concatenate([v[:4] for v in devrows.values()]),calB=np.concatenate([v[4:8] for v in devrows.values()]),dev=np.concatenate([v[8:10] for v in devrows.values()]),query=np.flatnonzero(old['role']=='query'));teacher=np.unique(np.r_[np.concatenate([np.flatnonzero((old['user']==u)&(old['role']=='fit'))[:4] for u in users]),reg]);allowed=np.unique(np.r_[fit,reg,ix['dev']]);as_set=set(allowed.tolist());t_set=set(teacher.tolist());text=old['text'];tid={v:j for j,v in enumerate(sorted(set(text)))};x=old['x'];m=x[:,:40]+x[:,240:241];stats=np.c_[m,x[:,40:80],x[:,80:240]+np.tile(m,(1,4))].astype('float32');d=dict(x=x,statistics=stats,user=old['user'],role=old['role'],text_id=np.array([tid[v] for v in text]),sample_id=old['sample_id']);np.savez_compressed(out/'inputs_private.npz',**d);(out/'split_private.json').write_text(json.dumps({k:v.tolist() for k,v in ix.items()}))
    for k in ['fit','register','dev','calA','calB']:
        assert not set(d['text_id'][ix[k]])&set(d['text_id'][ix['query']])
    if not (out/'spectra_private.npz').exists():
        def spec(w):
            w=w.astype('float64');i=(len(w)-32000)//2;w=w[i:i+32000].copy();assert len(w)==32000;w-=w.mean();freq,_,z=signal.stft(w,16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(z)**2+1e-20);b=np.stack([np.interp(np.geomspace(50,7600,40),freq,col) for col in lp.T],1);return np.stack([np.interp(np.linspace(0,b.shape[1]-1,32),np.arange(b.shape[1]),v) for v in b]).astype('float32')
        def read(item):
            i,r=item
            with np.load(paired/'paired'/r['file']) as z:
                w=z['acc'].astype('float32');assert hashlib.sha256(w.tobytes()).hexdigest()==r['acc_sha256'];aa=spec(w);bb=np.zeros_like(aa);tw=None
                if i in as_set:
                    v=z['audio'].astype('float32');assert hashlib.sha256(v.tobytes()).hexdigest()==r['audio_sha256'];bb=spec(v)
                    if i in t_set:j=(len(v)-32000)//2;tw=v[j:j+32000].copy();tw-=tw.mean()
            return i,aa,bb,tw,r['acc_sha256']
        aa=np.empty((len(x),1,40,32),dtype='float32');bb=np.zeros_like(aa);waves=[];waveindices=[];seen=set()
        with ThreadPoolExecutor(max_workers=8) as workers:
            for i,av,bv,w,digest in workers.map(read,enumerate(records)):
                assert digest not in seen;seen.add(digest);aa[i,0]=av;bb[i,0]=bv
                if w is not None:waveindices.append(i);waves.append(w)
                if (i+1)%500==0:print('VIB_CLOSED_PREPARE',i+1,len(x),flush=True)
        assert np.array_equal(waveindices,teacher);am=float(aa[fit].mean());asd=float(aa[fit].std());bm=float(bb[fit].mean());bsd=float(bb[fit].std());aa=np.clip((aa-am)/(3*asd),-1,1);bb[allowed]=np.clip((bb[allowed]-bm)/(3*bsd),-1,1);np.savez_compressed(out/'spectra_private.npz',a=aa,b=bb,audio_indices=allowed,input_mean=np.array(am),input_std=np.array(asd),audio_mean=np.array(bm),audio_std=np.array(bsd));np.savez_compressed(out/'teacher_audio_private.npz',indices=teacher,wave=np.stack(waves),lengths=np.ones(len(teacher),dtype='float32'))
    vec=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=20000);tx=vec.fit_transform(text[fit]);model=LogisticRegression(C=1,max_iter=500,class_weight='balanced',random_state=42).fit(tx,d['user'][fit]);accuracy=float((model.predict(vec.transform(text[ix['query']]))==d['user'][ix['query']]).mean());pub=dict(state='prepared',dataset='VibraVox',users=59,records=len(x),used_counts={k:len(v) for k,v in ix.items()},teacher_audio_records=len(teacher),audio_spectrum_records=len(allowed),query_audio_model_forwards=0,query_audio_spectrum_calculations=0,fit_register_query_exact_text_overlap=0,dev_cal_text_note='old develop is partitioned by recording;the same prompt may occur across different identities in dev/Cal,never in final query',text_only_query_identity_accuracy=accuracy,text_only_chance=1/59,selection='existing metadata-fixed59,allincluded,no auth-score selection',frontend='center2s/DC16k4096Hann1024hopFFT16384,40geom50..7600,32time;all-known-fit global normalization',sensor='single-axis forehead accelerometer and headset microphone,not HMDvelocity',files={p.name:p.stat().st_size for p in out.glob('*.npz')});(out/'preparation_public.json').write_text(json.dumps(pub,indent=2));print(json.dumps(pub),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--paired',required=True);p.add_argument('--out',required=True);main(p.parse_args())
