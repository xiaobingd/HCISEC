"""Local official ECAPA inference; verify parity against archived Base features."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,time,json,hashlib
from pathlib import Path
sys.path.insert(0,'work/esmb_audio_deps')
import numpy as np
import torch
from torch.nn import functional as F
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization

torch.set_num_threads(4)
root=Path('outputs/esmb_registration_20261109');modelpath=Path('outputs/ecapa_voxceleb_20261103/model/embedding_model.ckpt')
sha=hashlib.sha256(modelpath.read_bytes()).hexdigest();assert sha=='0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2'
model=ECAPA_TDNN(input_size=80,channels=[1024,1024,1024,1024,3072],kernel_sizes=[5,3,3,3,1],dilations=[1,2,3,4,1],attention_channels=128,lin_neurons=192)
model.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True));model.eval()
for p in model.parameters():p.requires_grad_(False)
features=Fbank(n_mels=80);features.eval();normalizer=InputNormalization(norm_type='sentence',std_norm=False);normalizer.eval()
@torch.no_grad()
def encode(wave):
    x=torch.tensor(np.asarray(wave));length=torch.ones(len(x));return F.normalize(model(normalizer(features(x),length),length).squeeze(1),dim=1).numpy()
raw=np.load(root/'base_audio_embeddings_private.npy').copy();g=np.load(root/'golden_wave_private.npz');start=time.time();actual=encode(g['wave']);golden=raw[g['index']];err=float(np.max(np.abs(actual-golden)));cos=float(np.min((actual*golden).sum(1)))
print('GOLDEN',err,cos,'seconds',time.time()-start,flush=True)
assert err<1e-4 and cos>.99999,'local ECAPA inference differs from archived official pipeline'
while not (root/'registration_wave_private.npz').exists() or (root/'registration_wave_private.npz').stat().st_size<47602371:time.sleep(2)
data=np.load(root/'registration_wave_private.npz');indices=data['index'];assert len(indices)==720 and np.all(indices%200<120)
start=time.time()
for j in range(0,len(indices),8):
    raw[indices[j:j+8]]=encode(data['wave'][j:j+8])
    if j%80==0 or j+8>=len(indices):
        np.save(root/'registration_audio_private.npy',raw)
        (root/'audio_status.json').write_text(json.dumps(dict(state='running',completed=min(j+8,len(indices)),total=len(indices),seconds=time.time()-start,golden_error=err)))
        print('REGISTRATION_AUDIO',min(j+8,len(indices)),len(indices),'elapsed',time.time()-start,flush=True)
assert np.all(np.linalg.norm(raw[indices],axis=1)>.9)
assert np.all(np.linalg.norm(raw[np.concatenate([np.arange(u*200+160,u*200+200) for u in range(10)])],axis=1)==0)
np.save(root/'registration_audio_private.npy',raw)
(root/'audio_status.json').write_text(json.dumps(dict(state='complete',registration_records=720,golden_error=err,golden_min_cos=cos,seconds=time.time()-start,model_sha256=sha,final_audio_access=False)))