"""Same raw VR frontend, VoxCeleb initialization vs VibraVox ACC backbone.

Three fixed axes are embedded independently then averaged and normalized.
No target-specific axis choice or local supervised training is performed here.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import resample_poly
import torch
from torch.nn import functional as F
sys.path.insert(0,'/root')
from run_ecapa_pilot import teacher_load

root=Path('/autodl-fs/data/vr_acc_pretrain_control_20261108');root.mkdir(exist_ok=True)
source=Path('/autodl-fs/data/identity_kd_all9_raw_full_multisource_20261103')
for name in ['model','deps']:
    if not(root/name).exists():(root/name).symlink_to(source/name,target_is_directory=True)
torch.set_num_threads(1);device='cuda' if torch.cuda.is_available() else 'cpu'
a=np.load(root/'vr_acceleration_transfer_private.npz');meta=np.load(source/'raw9_full_features_private.npz');assert np.array_equal(a['sample_id'],meta['sample_id'])
ix=np.flatnonzero(a['user']!=9)
for mode in ['voxceleb_init','vibravox_acc']:
    if (root/(mode+'_complete.json')).exists():continue
    model=teacher_load(root)
    if mode=='vibravox_acc':
        ck=torch.load('/autodl-fs/data/vibravox_dual_teacher_20261105/acc_teacher/best_backbone_private.pt',map_location='cpu',weights_only=False);model.mods.embedding_model.load_state_dict(ck['embedding_model']);del ck
    model.eval();path=root/(mode+'_embedding_private.npy');progress=root/(mode+'_progress.json');done=json.loads(progress.read_text())['done'] if progress.exists() else 0
    z=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(a['user']),192))
    for j in range(done,len(ix)):
        i=ix[j];waves=resample_poly(a['acceleration'][i],16,1,axis=1).astype('float32');vectors=[]
        for axis in range(3):
            with torch.no_grad():v=F.normalize(model.encode_batch(torch.tensor(waves[axis:axis+1],device=device)).squeeze(1),dim=1).cpu().numpy()[0]
            vectors.append(v)
        v=np.mean(vectors,axis=0);z[i]=v/max(float(np.linalg.norm(v)),1e-12);z.flush();progress.write_text(json.dumps(dict(done=j+1,total=len(ix),device=device)))
        if j%20==0:print('EXTRACT',mode,j+1,len(ix),flush=True)
    (root/(mode+'_complete.json')).write_text(json.dumps(dict(state='complete',input='timestamp-derived acceleration XYZ,2s1000Hz->16kHz,DCremoved;no per-user axis selection',aggregation='mean of three unit axis embeddings,thenL2normalize',mode=mode)))
    del model
print('BOTH_ACC_CACHES_COMPLETE',flush=True)
