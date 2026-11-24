"""Fresh raw local power features, only 1280 fit-word paired recordings."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile

BANDS=[(20,50),(50,80),(80,120),(120,180),(180,250),(250,350),(350,450)]
def local_power(wave,rate):
    wave=wave-wave.mean(axis=-1,keepdims=True)
    f,t,z=signal.stft(wave,rate,nperseg=rate//10,noverlap=rate//10-rate//50,nfft=rate//10,boundary=None,padded=False)
    power=abs(z)**2
    if power.ndim==3:power=power.sum(0)
    bands=np.stack([np.log(np.maximum(power[(f>=lo)&(f<hi)].mean(0),1e-20)) for lo,hi in BANDS],1)
    return bands,power,f,t
def main(root):
    out=root/'conditional_correspondence/raw_common_dynamics';out.mkdir(exist_ok=True)
    raw=np.load(root/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');idx=np.flatnonzero(raw['fit'])
    assert len(idx)==1280
    manifest=json.loads((root/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
    assert [r['sample_id'] for r in manifest]==list(raw['sample_id'])
    acc=raw['acc'][idx];xa=[];motion=[]
    for wave in acc:
        b,p,f,t=local_power(wave,1000);xa.append(b);motion.append(np.log(np.maximum(p[f<20].mean(0),1e-20)))
    groups={}
    for pos,i in enumerate(idx):groups.setdefault(manifest[i]['record_group'],[]).append((pos,int(i)))
    aa=np.zeros_like(xa,dtype='float32');energy=np.zeros((1280,len(t)),dtype='float32')
    assert len(groups)==160
    for count,(stem,records) in enumerate(groups.items()):
        path=Path(stem);meta=json.loads(path.with_name(path.name+'_meta.json').read_text(encoding='utf-8-sig'))
        rate,pcm=wavfile.read(path.parent/meta['full_audio']);assert rate==48000 and pcm.dtype==np.int16
        pcm=pcm.astype('float32')/32768
        if pcm.ndim==2:pcm=pcm[:,0]
        for pos,i in records:
            rec=manifest[i];assert rec['audio_timebase_hz']==44100
            center=(rec['audio_start_s']+rec['audio_end_s'])/2
            frames=np.rint(center*44100).astype(int)+np.arange(132300)-66150
            valid=(frames>=0)&(frames<len(pcm));wave=np.zeros(132300,dtype='float32');wave[valid]=pcm[frames[valid]]
            wave=signal.resample_poly(wave,160,441)
            b,p,f,at=local_power(wave,16000);assert len(at)==len(t) and np.max(abs(at-t))<1e-8
            aa[pos]=b;energy[pos]=np.log(np.maximum(p[(f>=300)&(f<3500)].mean(0),1e-20))
        if (count+1)%40==0:print('RAW_COMMON_GROUPS',count+1,flush=True)
    np.savez_compressed(out/'local_power_private.npz',audio=aa,acc=np.array(xa,dtype='float32'),motion=np.array(motion,dtype='float32'),audio_activity=energy,
        sample_id=raw['sample_id'][idx],user=raw['user'][idx],word=raw['word'][idx],repeat=raw['repeat'][idx],indices=idx,times=t)
    info=dict(records=1280,users=10,words=16,recording_groups=160,bands_hz=BANDS,window_ms=100,hop_ms=20,frames=len(t),seconds=3,
        acc='raw3axis,1000Hz,sumaxis-power,removeperwindowconstant',audio='fullPCM header48000 interpreted44100,resample16k,not utterancepad',
        activity='300-3500Hz Audio log-power proxy,notgroundtruthVAD',motion='ACC<20Hz log-power afterperwindowDCremoval,notgroundtruthmotion',
        fit_only=True,final_query_audio_read=False,physical_sync_verified=False,windows_may_overlap=True)
    (out/'extraction_public.json').write_text(json.dumps(info,indent=2));print('RAW_COMMON_EXTRACTION_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
