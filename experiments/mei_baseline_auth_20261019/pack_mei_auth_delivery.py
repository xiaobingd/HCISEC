"""Private local delivery of model weights plus review, no public signal upload."""
import argparse,zipfile,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
review=['summary.json','private_scores.json','generation_plots.npz','cyclegan_protocol.json','head_protocol.json','head_histories.json','full_history.json','fresh_local_history.json']
models=['baseline_weights.pt','heads_locked.json','split_manifest.json','head_protocol.json','cyclegan_protocol.json']+[r+'_'+str(seed)+'_head.pt' for r in ['fusion','raw_ACC','cycle_ACC'] for seed in [42,43]]
with zipfile.ZipFile(root/'delivery.zip','w',zipfile.ZIP_DEFLATED) as z:
    for group,names in [('review',review),('models',models)]:
        for name in names:z.write(root/name,group+'/'+name)
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in models};z.writestr('models/file_sha256.json',json.dumps(hashes,indent=2))
print('DELIVERY', (root/'delivery.zip').stat().st_size,flush=True)
