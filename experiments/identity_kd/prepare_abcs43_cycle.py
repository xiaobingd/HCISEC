"""Build anonymous GAN spectra from the unchanged ABCS43 private manifest.

Stereo WAV bytes are read for CRC/ACC. Query channel1 is never transformed.
Only Base fit/develop and Cal/Test registration Audio spectra are exported.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,wave,hashlib,zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import signal

def main(a):
    prior=Path(a.prior);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d=dict(np.load(prior/'inputs_private.npz'));m=json.loads((prior/'manifest_private.json').read_text(encoding='utf8'));records=m['records'];assert len(records)==2445 and len(d['x'])==2445;fit=np.flatnonzero((d['user']<23)&(d['role']=='fit'));dev=np.flatnonzero((d['user']<23)&(d['role']=='develop'));audio_indices=np.unique(np.r_[fit,dev,np.flatnonzero(d['role']=='registration')]);allowed=set(audio_indices.tolist());archive={r['name']:r for r in json.loads(Path(a.archive).read_text(encoding='utf8'))}
    def spec(w):
        w=w.astype('float64');w-=w.mean();f,_,z=signal.stft(w,16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(z)**2+1e-20);b=np.stack([np.interp(np.geomspace(50,7600,40),f,lp[:,j]) for j in range(lp.shape[1])],1);return np.stack([np.interp(np.linspace(0,b.shape[1]-1,32),np.arange(b.shape[1]),row) for row in b]).astype('float32')
    def read(item):
        i,r=item;assert int(r['user'])==int(d['user'][i]) and r['role']==d['role'][i];p=Path(a.raw)/r['relative_wav'];blob=p.read_bytes();entry=archive['ABCS_database/'+r['relative_wav']];assert len(blob)==entry['size'] and zlib.crc32(blob)==entry['crc32']
        with wave.open(str(p),'rb') as f:assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,2,2);pcm=f.readframes(f.getnframes())
        pair=np.frombuffer(pcm,dtype='<i2').reshape(-1,2).astype('float64')/32768.;aa=spec(pair[:,0]);bb=spec(pair[:,1]) if i in allowed else np.zeros_like(aa);return i,aa,bb,hashlib.sha256(pcm).hexdigest()
    aa=np.empty((2445,40,32),dtype='float32');bb=np.zeros_like(aa);seen=set()
    with ThreadPoolExecutor(max_workers=8) as workers:
        for i,x,y,digest in workers.map(read,enumerate(records)):
            assert digest not in seen;seen.add(digest);aa[i]=x;bb[i]=y
            if (i+1)%250==0:print('ABCS_GAN_PREPARE',i+1,2445,flush=True)
    am=float(aa[fit].mean());ast=float(aa[fit].std());bm=float(bb[fit].mean());bst=float(bb[fit].std());aa=np.clip((aa-am)/(3*ast),-1,1)[:,None];bb[audio_indices]=np.clip((bb[audio_indices]-bm)/(3*bst),-1,1);bb=bb[:,None];assert not bb[~np.isin(np.arange(2445),audio_indices)].any();path=out/'spectra_private.npz';np.savez_compressed(path,a=aa,b=bb,fit=fit,dev=dev,audio_indices=audio_indices,normalization=np.array([am,ast,bm,bst]),row_ids=np.arange(2445));summary=dict(state='prepared',users=43,base=23,cal=8,test=12,records=2445,fit_records=len(fit),develop_records=len(dev),registration_audio_records=240,allowed_audio_spectrum_records=len(audio_indices),query_audio_feature_calculations=0,query_audio_model_forwards=0,crc_checks=2445,exact_pcm_duplicates=0,frontend='whole paired utterance,DC remove,16k/STFT4096hop1024FFT16384/40geom50..7600logpower,timeinterpolation32;globalBasefitmeanSDclip3',channels='ch0 provisional vibration,ch1 provisional Audio,physical mapping not independently confirmed',stereo_note='stereo WAV bytes read for ACC/CRC; query ch1 not transformed or model-forwarded',private_npz_bytes=path.stat().st_size);(out/'preparation_public.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prior',required=True);p.add_argument('--out',required=True);p.add_argument('--raw',default='D:/BaiduNetdiskDownload/ABCS_database');p.add_argument('--archive',default='outputs/abcs_content_audit_20261109/archive_central_private.json');main(p.parse_args())
