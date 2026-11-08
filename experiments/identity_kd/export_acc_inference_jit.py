"""Export frozen, float32 ECAPA inference plus an equivalence fixture."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/root')
from run_ecapa_pilot import teacher_load
torch.set_num_threads(1)
root=Path('/autodl-fs/data/vr_acc_pretrain_control_20261108')
model=teacher_load(root)
ck=torch.load('/autodl-fs/data/vibravox_dual_teacher_20261105/acc_teacher/best_backbone_private.pt',map_location='cpu',weights_only=False)
model.mods.embedding_model.load_state_dict(ck['embedding_model']);del ck
model.eval()
assert model.mods.mean_var_norm.norm_type=='sentence' and model.mods.mean_var_norm.mean_norm and not model.mods.mean_var_norm.std_norm
class Inference(nn.Module):
    def __init__(self):
        super().__init__();self.features=model.mods.compute_features;self.encoder=model.mods.embedding_model
    def forward(self,w):
        lengths=torch.ones(w.shape[0],device=w.device)
        f=self.features(w)
        # SpeechBrain sentence mean removal uses .data, which tracing would
        # incorrectly freeze. Keep the identical operation input-dependent.
        f=f-f.mean(dim=1,keepdim=True)
        return F.normalize(self.encoder(f,lengths).squeeze(1),dim=1)
a=np.load(root/'vr_acceleration_transfer_private.npz');raw=a['acceleration'];ix=np.flatnonzero(a['user']!=9)[-3:]
waves=np.stack([resample_poly(raw[i],16,1,axis=1).astype('float32') for i in ix])
wrapped=Inference().eval()
with torch.inference_mode():
    traced=torch.jit.freeze(torch.jit.trace(wrapped,torch.tensor(waves[0]),check_trace=False).eval())
    expected=np.stack([F.normalize(model.encode_batch(torch.tensor(w)).squeeze(1),dim=1).numpy() for w in waves])
    actual=np.stack([traced(torch.tensor(w)).numpy() for w in waves])
assert np.allclose(expected,actual,rtol=1e-4,atol=2e-5),float(np.max(np.abs(expected-actual)))
traced.save(str(root/'vibravox_acc_inference_private.pt'))
np.savez(root/'jit_fixture_private.npz',waves=waves,expected=expected)
print('EXPORTED',float(np.max(np.abs(expected-actual))),flush=True)
