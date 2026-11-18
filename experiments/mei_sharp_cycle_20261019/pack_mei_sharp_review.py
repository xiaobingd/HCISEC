"""Collect completed phases and private review plots, never model weights."""
import json,zipfile,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);args=p.parse_args();root=args.study
parts=[root/'results',root/'results_structure',root/'results_highpass'];out=root/'review';out.mkdir(exist_ok=True)
runs=[];controls={};phases=[]
for part in parts:
    rr=json.loads((part/'results.json').read_text());runs.extend(rr)
    phases.append(dict(folder=part.name,protocol=json.loads((part/'protocol.json').read_text())))
    cc=json.loads((part/'controls.json').read_text())
    for r in cc['runs']:controls.setdefault((r['seed'],r['route']),r)
    for pattern in ['*_plots.npz','*_history.json']:
        for path in part.glob(pattern):(out/path.name).write_bytes(path.read_bytes())
assert len(runs)==12 and len({(r['seed'],r['route']) for r in runs})==12
for name,value in [('results',runs),('protocol',dict(phases=phases,scope='All fixed endpoints; structure/highpass added during repeated VAL exploration, not independent confirmation. Unchanged private signal preprocessing; no test/auth read.')),('controls',dict(runs=list(controls.values()),scope=cc['scope']))]:
    (out/(name+'.json')).write_text(json.dumps(value,indent=2))
with zipfile.ZipFile(root/'review.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in out.iterdir():z.write(path,path.name)
print('PACKED',len(runs),(root/'review.zip').stat().st_size)
