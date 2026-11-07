"""Rebuild timestamp-derived acceleration for fixed, user-independent transfer.

Private output: do not publish the waveform archive or original manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_full_raw9 import read_acc

ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
manifest=json.loads(args.manifest.read_text(encoding='utf-8'));records=manifest['records'];waves=np.zeros((len(records),3,2000),dtype='float32');found=0
for i,row in enumerate(records):
    if row['user_id']==9:continue
    rawpath=row['acc_file'].replace('\\\\','\\');path=Path(rawpath)
    if not path.exists():
        # Reverse the known UTF-8-as-GBK directory-name corruption only.
        parts=[]
        for part in path.parts:
            try:parts.append(part.encode('gbk').decode('utf-8'))
            except (UnicodeError,LookupError):parts.append(part)
        path=Path(*parts)
    assert path.exists(),str(path)
    raw=np.load(path,allow_pickle=False)
    digest=hashlib.sha256(memoryview(np.ascontiguousarray(raw)).cast('B')).hexdigest()
    assert digest==row['acc_array_sha256'],row['sample_id']
    _,derivative,_=read_acc(path);waves[i]=derivative;found+=1
    if found%100==0:print('REBUILT',found,flush=True)
args.out.parent.mkdir(exist_ok=True,parents=True)
np.savez(args.out,acceleration=waves,sample_id=np.array([r['sample_id'] for r in records]),user=np.array([r['user_id'] for r in records]))
args.out.with_suffix('.json').write_text(json.dumps(dict(rows=found,shape=list(waves.shape),excluded_user=9,input='timestamp derivative of HMD velocity; interpolate1000Hz2s,per-axis DC removal; no user-specific processing; no learned scaling',warning='computed derivative,not physical accelerometer; waveform transfer changes frontend versus ResNet spectrum; compare bothECAPA checkpoints onexactlysameinput',all_source_payload_hashes_verified=True),indent=2))
print('COMPLETE',found,flush=True)
