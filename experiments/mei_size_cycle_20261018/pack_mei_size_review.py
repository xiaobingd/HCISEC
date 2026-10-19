"""Small private review bundle, excluding checkpoints and full source arrays."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root;res=root/'results';out=root/'review';out.mkdir(exist_ok=True)
runs=json.loads((res/'results.json').read_text());assert len(runs)==8
long_runs=json.loads((root/'results_long/results.json').read_text());assert len(long_runs)==2
runs+=long_runs
small_long=json.loads((root/'results_long20/results.json').read_text());assert len(small_long)==2
runs+=small_long
references={};indices={}
for domain in ['mei','personal']:
    d=dict(np.load(root/'data'/(domain+'_val.npz')));ug=np.unique(d['ids'])
    names=([g for g in ug if str(g).startswith('0-9__')][:2]+[g for g in ug if str(g).startswith('word__')][:2]) if domain=='mei' else ug[:4]
    ii=[np.flatnonzero(d['ids']==g)[0] for g in names];indices[domain]=ii
    for k in ['x','y','ids']:references[domain+'_'+k]=d[k][ii]
np.savez_compressed(out/'plot_references.npz',**references)
for row in runs:
    name=row['route']+'_'+str(row['seed']);domain='personal' if row['route']=='personal20' else 'mei';folder=root/'results_long20' if row['route']=='mei20_long' else root/'results_long' if row['route']=='mei180_long' else res;d=dict(np.load(folder/(name+'_predictions.npz')))
    np.savez_compressed(out/(name+'_plot_predictions.npz'),**{k:v[indices[domain]] for k,v in d.items()})
(out/'results.json').write_text(json.dumps(runs,indent=2))
(out/'protocol.json').write_bytes((res/'protocol.json').read_bytes())
(out/'long_protocol.json').write_bytes((root/'results_long/protocol.json').read_bytes())
(out/'small_long_protocol.json').write_bytes((root/'results_long20/protocol.json').read_bytes())
for f in res.glob('*history.json'):(out/f.name).write_bytes(f.read_bytes())
for f in (root/'results_long').glob('*history.json'):(out/f.name).write_bytes(f.read_bytes())
for f in (root/'results_long20').glob('*history.json'):(out/f.name).write_bytes(f.read_bytes())
with zipfile.ZipFile(root/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in out.iterdir():z.write(f,'review/'+f.name)
print('BUNDLE', (root/'review.zip').stat().st_size)
