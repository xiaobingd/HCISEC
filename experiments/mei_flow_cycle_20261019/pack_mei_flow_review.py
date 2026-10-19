"""Collect completed full-cycle literature pilot phases and private review plots."""
import json,zipfile,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);args=p.parse_args();root=args.study;out=root/'review';out.mkdir(exist_ok=True)
parts=[root/'results',root/'results_decoupled'];runs=[];controls=[];phases=[]
for part in parts:
    runs.extend(json.loads((part/'results.json').read_text()));controls.extend(json.loads((part/'controls.json').read_text())['runs']);phases.append(dict(folder=part.name,protocol=json.loads((part/'protocol.json').read_text())))
    for pattern in ['*_plots.npz','*_history.json']:
        for path in part.glob(pattern):(out/path.name).write_bytes(path.read_bytes())
assert len(runs)==len(controls)==8 and len({(r['seed'],r['route']) for r in runs})==8
objects=dict(results=runs,controls=dict(runs=controls,scope='Paired random noise, no best-of-N, fixed repeatedVAL, no identity claim.'),protocol=dict(phases=phases,scope='Original3routes6fits plus2decoupled exploratory fits. Same180TRAIN60reusedVAL, fixed endpoints; research-inspired Mel adaptations, not original algorithms reproduced.'))
for name,value in objects.items():(out/(name+'.json')).write_text(json.dumps(value,indent=2))
with zipfile.ZipFile(root/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in out.iterdir():z.write(path,path.name)
print('PACKED',len(runs),(root/'review.zip').stat().st_size)
