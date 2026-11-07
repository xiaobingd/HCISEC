"""Fetch pinned official ERes2NetV2 assets; optionally extract real Audio targets."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');a=ap.parse_args()
root=Path('/autodl-fs/data/eres2netv2_control_20261108');root.mkdir(exist_ok=True)
manifest=root/'asset_manifest.json'
if manifest.exists():meta=json.loads(manifest.read_text());revision=meta['source_revision']
else:
    req=urllib.request.Request('https://api.github.com/repos/modelscope/3D-Speaker/commits/main',headers={'User-Agent':'HCISEC-research'})
    revision=json.load(urllib.request.urlopen(req,timeout=45))['sha'];meta=dict(source_revision=revision,model_id='iic/speech_eres2netv2_sv_zh-cn_16k-common',model_revision='v1.0.1',files={})
for name in ['ERes2NetV2.py','pooling_layers.py','fusion.py']:
    target=root/'speakerlab/models/eres2net'/name;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        content=urllib.request.urlopen('https://raw.githubusercontent.com/modelscope/3D-Speaker/'+revision+'/speakerlab/models/eres2net/'+name,timeout=45).read();target.write_bytes(content)
    meta['files'][str(target.relative_to(root))]=hashlib.sha256(target.read_bytes()).hexdigest()
for folder in [root/'speakerlab',root/'speakerlab/models',root/'speakerlab/models/eres2net']:(folder/'__init__.py').touch()
weights=root/'pretrained_eres2netv2.ckpt'
if not weights.exists():
    url='https://modelscope.cn/api/v1/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/repo?Revision=v1.0.1&FilePath=pretrained_eres2netv2.ckpt'
    urllib.request.urlretrieve(url,weights.with_suffix('.part'));weights.with_suffix('.part').replace(weights)
digest=hashlib.sha256(weights.read_bytes()).hexdigest()
assert digest=='0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c',digest
meta['weights_sha256']=digest;meta['asset_state']='downloaded_and_hash_verified';manifest.write_text(json.dumps(meta,indent=2));print('ASSETS_READY',digest,flush=True)
if not a.extract:sys.exit(0)
import numpy as np
import torch
from torch.nn import functional as F
import torchaudio.compliance.kaldi as kaldi
sys.path.insert(0,str(root))
from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
net=ERes2NetV2(feat_dim=80,embedding_size=192).to(device)
net.load_state_dict(torch.load(weights,map_location='cpu'));net.eval()
src=Path('/autodl-fs/data/identity_kd_all9_raw_full_multisource_20261103/raw9_full_features_private.npz');d=np.load(src)
ix=np.flatnonzero((d['user']!=9)&(d['role']=='fit'));path=root/'audio_embedding_private.npy';progress=root/'extraction_progress.json'
done=json.loads(progress.read_text())['done'] if progress.exists() else 0
z=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(d['user']),192))
for j in range(done,len(ix)):
    i=ix[j];w=torch.tensor(d['wave'][i:i+1].copy());feat=kaldi.fbank(w,num_mel_bins=80,sample_frequency=16000,dither=0);feat=feat-feat.mean(0,keepdim=True)
    with torch.no_grad():v=F.normalize(net(feat[None].to(device)),dim=1).cpu().numpy()[0]
    assert np.isfinite(v).all();z[i]=v;z.flush();progress.write_text(json.dumps(dict(done=j+1,total=len(ix),device=device)))
    if j%20==0:print('EXTRACT',j+1,len(ix),flush=True)
meta['extraction_state']='complete';meta['preprocessing']='real Audio,16kHz2s;Kaldi80fbank,dither0,per-frequency time mean removal; official FBank defaults, no waveform amplitude multiplication';manifest.write_text(json.dumps(meta,indent=2));print('EXTRACTION_COMPLETE',flush=True)
