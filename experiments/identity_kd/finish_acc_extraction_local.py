"""Resume frozen extraction locally after validating exported inference."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.signal import resample_poly
import torch

root=Path('outputs/acc_extraction_acceleration_20261108');root.mkdir(exist_ok=True,parents=True)
ap=argparse.ArgumentParser();ap.add_argument('--start-record',type=int);ap.add_argument('--reuse-benchmark',action='store_true');args=ap.parse_args()
net=torch.jit.load(str(root/'vibravox_acc_inference_private.pt'),map_location='cpu').eval()
fixture=np.load(root/'jit_fixture_private.npz');bench={}
for threads in ([] if args.reuse_benchmark else [1,2,4]):
    torch.set_num_threads(threads);times=[]
    with torch.inference_mode():
        for rep in range(2):
            start=time.perf_counter();actual=np.stack([net(torch.tensor(w)).numpy() for w in fixture['waves']]);times.append((time.perf_counter()-start)/len(actual))
            assert np.allclose(actual,fixture['expected'],rtol=1e-4,atol=2e-5),float(np.max(np.abs(actual-fixture['expected'])))
    bench[str(threads)]=float(np.median(times))
if args.reuse_benchmark:
    report=json.loads((root/'benchmark.json').read_text());bench=report['seconds_per_record'];threads=report['selected_threads']
else:
    threads=int(min(bench,key=bench.get))
    report=dict(seconds_per_record=bench,selected_threads=threads,fixture_max_error=float(np.max(np.abs(actual-fixture['expected']))),precision='float32',device='local_cpu')
torch.set_num_threads(threads)
(root/'benchmark.json').write_text(json.dumps(report,indent=2));print('LOCAL_BENCHMARK',json.dumps(report),flush=True)
a=np.load('outputs/mei8_strict_open_residual_20261108/vr_acceleration_transfer_private.npz');raw=a['acceleration'];ix=np.flatnonzero(a['user']!=9)
progress=root/'vibravox_acc_progress.json';done=args.start_record if args.start_record is not None else json.loads(progress.read_text())['done'];start_done=done;started=time.perf_counter()
z=np.load(root/'vibravox_acc_embedding_private.npy',mmap_mode='r+')
if args.start_record is None:assert np.all(np.linalg.norm(z[ix[:done]],axis=1)>.9)
with torch.inference_mode():
    for j in range(done,len(ix)):
        i=ix[j];w=resample_poly(raw[i],16,1,axis=1).astype('float32');axes=net(torch.tensor(w)).numpy();v=axes.mean(0);z[i]=v/max(float(np.linalg.norm(v)),1e-12)
        if (j+1-done)%8==0 or j+1==len(ix):
            z.flush();tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(dict(done=j+1,total=len(ix),device='local_cpu',threads=threads,elapsed_seconds=time.perf_counter()-started,run_start_record=start_done)));tmp.replace(progress)
        if (j+1)%40==0:print('EXTRACT',j+1,len(ix),round(time.perf_counter()-started,2),flush=True)
validate=ix[start_done:] if args.start_record is not None else ix
assert np.isfinite(z[validate]).all() and np.allclose(np.linalg.norm(z[validate],axis=1),1,atol=1e-5)
print('COMPLETE',len(ix),time.perf_counter()-started,flush=True)
