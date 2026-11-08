"""Same raw VR frontend, VoxCeleb initialization vs VibraVox ACC backbone.

Three fixed axes are embedded independently then averaged and normalized.
No target-specific axis choice or local supervised training is performed here.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
import time
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
raw_acceleration=a['acceleration']
ix=np.flatnonzero(a['user']!=9)
for mode in ['voxceleb_init','vibravox_acc']:
    if (root/(mode+'_complete.json')).exists():continue
    model=teacher_load(root)
    if mode=='vibravox_acc':
        ck=torch.load('/autodl-fs/data/vibravox_dual_teacher_20261105/acc_teacher/best_backbone_private.pt',map_location='cpu',weights_only=False);model.mods.embedding_model.load_state_dict(ck['embedding_model']);del ck
    model.eval();path=root/(mode+'_embedding_private.npy');progress=root/(mode+'_progress.json');done=json.loads(progress.read_text())['done'] if progress.exists() else 0
    z=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype='float32',shape=(len(a['user']),192))
    def infer(waves,chunk):
        vectors=[]
        with torch.inference_mode():
            for start in range(0,len(waves),chunk):
                vectors.append(F.normalize(model.encode_batch(torch.tensor(waves[start:start+chunk],device=device)).squeeze(1),dim=1).cpu().numpy())
        vectors=np.concatenate(vectors).reshape(-1,3,192).mean(axis=1)
        return vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    # Every candidate keeps all three axes, same samples, and float32 precision.
    probe=np.stack([resample_poly(raw_acceleration[i],16,1,axis=1).astype('float32') for i in ix[done:min(done+2,len(ix))]]).reshape(-1,32000)
    reference=infer(probe,1);bench={};errors={}
    for chunk in [1,3,6]:
        times=[]
        for repeat in range(2):
            start=time.perf_counter();value=infer(probe,chunk)
            if device=='cuda':torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            errors[str(chunk)]=float(np.max(np.abs(value-reference)))
            assert np.allclose(value,reference,rtol=1e-4,atol=2e-5),(chunk,errors[str(chunk)])
        bench[str(chunk)]=float(np.median(times)/(len(probe)/3))
    chunk=int(min(bench,key=bench.get));batch=2 if chunk==6 else 1
    report=dict(seconds_per_recording=bench,max_absolute_error=errors,selected_axis_batch=chunk,recordings_per_batch=batch,baseline_axis_batch=1,speedup=bench['1']/bench[str(chunk)],precision='float32',device=device)
    (root/(mode+'_speed_benchmark.json')).write_text(json.dumps(report,indent=2));print('BENCHMARK',mode,json.dumps(report),flush=True)
    started=time.perf_counter();last_saved=done
    def save_progress(count):
        z.flush();tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(dict(done=count,total=len(ix),device=device,axis_batch=chunk,elapsed_seconds=time.perf_counter()-started,run_start_record=done)));tmp.replace(progress)
    for j in range(done,len(ix),batch):
        selected=ix[j:j+batch];waves=np.stack([resample_poly(raw_acceleration[i],16,1,axis=1).astype('float32') for i in selected]).reshape(-1,32000)
        z[selected]=infer(waves,chunk)
        count=j+len(selected)
        if count-last_saved>=8 or count==len(ix):save_progress(count);last_saved=count
        if j%20==0:print('EXTRACT',mode,count,len(ix),flush=True)
    (root/(mode+'_complete.json')).write_text(json.dumps(dict(state='complete',input='timestamp-derived acceleration XYZ,2s1000Hz->16kHz,DCremoved;no per-user axis selection',aggregation='mean of three unit axis embeddings,thenL2normalize',mode=mode)))
    del model
print('BOTH_ACC_CACHES_COMPLETE',flush=True)