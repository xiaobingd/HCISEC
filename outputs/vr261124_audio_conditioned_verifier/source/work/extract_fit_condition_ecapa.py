"""Frozen official ECAPA on 1280 fit-only Audio records; local offline inference."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'work/audio_multiword_runtime'),str(ROOT/'work/verifier_deps')]
sys.path.append(str(ROOT/'work/esmb_audio_deps'))
import hashlib
import json
import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
from torch import nn
from torch.nn import functional as F

def main():
    torch.set_num_threads(2)
    base=ROOT/'outputs/vr261119_10x40_raw_acc';out=base/'conditional_correspondence/audio_conditioned_verifier';out.mkdir(exist_ok=True)
    d=np.load(base/'phrase_auth/frontend_private.npz');reg=d['fit'];indices=np.flatnonzero(reg)
    assert len(indices)==1280
    path=out/'fit_ecapa_private.npz'
    if path.exists():
        c=np.load(path)
        if 'length_aware' in c and bool(c['length_aware']):
            assert np.array_equal(c['sample_id'],d['sample_id'][indices]);print('REGISTERED_ECAPA_CACHE_VERIFIED');return
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization
    modelpath=ROOT/'outputs/ecapa_voxceleb_20261103/model/embedding_model.ckpt'
    sha=hashlib.sha256(modelpath.read_bytes()).hexdigest()
    assert sha=='0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2'
    net=ECAPA_TDNN(input_size=80,channels=[1024,1024,1024,1024,3072],kernel_sizes=[5,3,3,3,1],dilations=[1,2,3,4,1],attention_channels=128,lin_neurons=192)
    net.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True));net.eval()
    fbank=Fbank(n_mels=80);normalizer=InputNormalization(norm_type='sentence',std_norm=False)
    manifest=json.loads((base/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf8'))['records']
    assert [r['sample_id'] for r in manifest]==list(d['sample_id'])
    waves=np.zeros((1280,48000),dtype='float32');lengths=np.zeros(1280,dtype='float32')
    for j,i in enumerate(indices):
        fs,pcm=wavfile.read(manifest[i]['audio_file']);assert fs==48000 and pcm.dtype==np.int16
        pcm=pcm.astype('float32')/32768
        if pcm.ndim==2:pcm=pcm[:,0]
        wav=signal.resample_poly(pcm,160,441);assert len(wav)<=48000
        waves[j,:len(wav)]=wav;lengths[j]=len(wav)/48000
    # Throughput only: choose CPU threads without using validation/authentication outcomes.
    import time
    benchmark=[];reference=None
    for threads in (2,4,8):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            b=torch.tensor(waves[:8]);valid=torch.tensor(lengths[:8])
            warm=F.normalize(net(normalizer(fbank(b),valid),valid).squeeze(1),dim=1)
            start=time.perf_counter();check=F.normalize(net(normalizer(fbank(b),valid),valid).squeeze(1),dim=1)
        elapsed=time.perf_counter()-start
        if reference is None:reference=check
        error=float((reference-check).abs().max());assert error<1e-4
        benchmark.append(dict(threads=threads,seconds=elapsed,max_error=error))
        print('FIT_ECAPA_SPEED',threads,elapsed,flush=True)
    best=min(benchmark,key=lambda a:a['seconds'])['threads'];torch.set_num_threads(best)
    (out/'extraction_speed_public.json').write_text(json.dumps(benchmark,indent=2))
    features=[]
    with torch.inference_mode():
        for i in range(0,1280,8):
            batch=torch.tensor(waves[i:i+8]);valid=torch.tensor(lengths[i:i+8]);feat=normalizer(fbank(batch),valid)
            z=F.normalize(net(feat,valid).squeeze(1),dim=1);assert z.shape==(len(batch),192) and torch.isfinite(z).all()
            features.append(z.numpy())
            if (i+8)%64==0:print('REGISTERED_ECAPA',i+8,flush=True)
    np.savez_compressed(path,embedding=np.concatenate(features),indices=indices,sample_id=d['sample_id'][indices],length_aware=np.array(True))
    (out/'fit_audio_extraction_public.json').write_text(json.dumps(dict(records=1280,fit_words=sorted(set(d['word'][reg].tolist())),calibration_audio=0,query_audio=0,
        model='official VoxCeleb ECAPA frozen,not locally finetuned',sha256=sha,interpreted_audio_rate=44100,resampled_rate=16000,padding_seconds=3,length_aware_right_padding=True),indent=2))
    print('REGISTERED_ECAPA_COMPLETE',flush=True)

if __name__=='__main__':main()
