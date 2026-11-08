"""Merge a validated local suffix into the stopped server extractor cache."""
import argparse
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--cut',type=int,default=1915);args=ap.parse_args()
root=Path('/autodl-fs/data/vr_acc_pretrain_control_20261108');source=np.load(root/'vr_acceleration_transfer_private.npz');ix=np.flatnonzero(source['user']!=9)
server=np.load(root/'vibravox_acc_embedding_private.npy',mmap_mode='r+');local=np.load(root/'vibravox_acc_local_suffix_private.npy')
progress=root/'vibravox_acc_progress.json';done=json.loads(progress.read_text())['done']
assert done>=args.cut,(done,args.cut)
assert server.shape==local.shape
assert np.isfinite(server[ix[:args.cut]]).all() and np.allclose(np.linalg.norm(server[ix[:args.cut]],axis=1),1,atol=1e-5)
assert np.isfinite(local[ix[args.cut:]]).all() and np.allclose(np.linalg.norm(local[ix[args.cut:]],axis=1),1,atol=1e-5)
overlap=ix[args.cut:min(done,len(ix))]
error=float(np.max(np.abs(server[overlap]-local[overlap]))) if len(overlap) else None
if len(overlap):assert np.allclose(server[overlap],local[overlap],rtol=1e-4,atol=2e-5),error
server[ix[args.cut:]]=local[ix[args.cut:]];server.flush()
assert np.allclose(np.linalg.norm(server[ix],axis=1),1,atol=1e-5)
report=dict(state='complete',rows=len(ix),server_prefix_recordings=args.cut,local_suffix_recordings=len(ix)-args.cut,overlap_recordings=len(overlap),overlap_max_error=error,precision='float32',input='timestamp-derived acceleration XYZ,2s1000Hz->16kHz,DCremoved;no per-user axis selection',aggregation='mean of three unit axis embeddings,thenL2normalize',mode='vibravox_acc',acceleration='validated frozen JIT inference on localCPU plus serverCPU;all axes and all records retained')
tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(dict(done=len(ix),total=len(ix),device='server_cpu+local_cpu')));tmp.replace(progress)
(root/'vibravox_acc_complete.json').write_text(json.dumps(report,indent=2));(root/'parallel_merge_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
