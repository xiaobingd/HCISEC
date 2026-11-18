"""Training-only sensitivity check; deliberately does not evaluate validation."""
from pathlib import Path
import json
source=Path('work/verify_delay_tail.py').read_text(encoding='utf-8')
prefix=source.split('results={};grids={}')[0]
scope={};exec(compile(prefix,'delay_analysis_prefix','exec'),scope)
np=scope['np'];scope['lags']=np.arange(-.5,.501,.02);scope['taus']=np.array([0.,.03,.06,.10,.16,.25,.40,.60,.80])
out={}
for j in [1,3]:
 grid=scope['fitgrid'](j);best=max(grid,key=lambda c:c['train_loo_relative_mse_gain']);near=[c for c in grid if c['train_loo_relative_mse_gain']>=best['train_loo_relative_mse_gain']-.005]
 out[scope['names'][j]]=dict(best=best,within_half_percent_gain=dict(candidates=len(near),delay_range_s=[min(c['delay_s'] for c in near),max(c['delay_s'] for c in near)],tau_range_s=[min(c['release_tau_s'] for c in near),max(c['release_tau_s'] for c in near)]),grid=grid)
 print(scope['names'][j],best,out[scope['names'][j]]['within_half_percent_gain'],flush=True)
scope['OUT'].joinpath('train_boundary_sensitivity.json').write_text(json.dumps(dict(scope='TRAIN ONLY sensitivity, no validation evaluation or replacement of original selected settings',results=out),indent=2),encoding='utf-8')
