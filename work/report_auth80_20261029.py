"""Independent post-fit metric and label/count verification; no model selection."""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/autodl-fs/data/mei_auth80_20261029')
M=json.loads(Path('/autodl-fs/data/mei_strict_baseline_20261029/manifest_private.json').read_text())
rows={}
for u in range(1,10):
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text())
 scores=np.load(str(folder/'scores_private.npz'));locked=json.loads((folder/'locked_before_queries.json').read_text())
 expected=np.concatenate([np.full(len(M['sources'][str(v)]['roles']['test']),v-1) for v in range(1,10) if v!=u])
 assert np.array_equal(scores['attack_labels'],expected)
 for route,r in a['routes'].items():
  g=scores[route+'__final_genuine'];n=scores[route+'__final_impostor'];t=locked['thresholds'][route]
  assert len(g)==len(M['sources'][str(u)]['roles']['test']) and len(n)==len(expected)
  assert r['genuine_n']==len(g) and r['impostor_n']==len(n)
  auc=((g[:,None]>n)+.5*(g[:,None]==n)).mean()
  assert abs(auc-r['auc'])<1e-12 and abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
  assert abs((r['frr']+r['far'])/2-r['hter'])<1e-12
  assert len(r['attack_by_user'])==8
  for v,b in r['attack_by_user'].items():
   s=n[expected==int(v)-1];assert len(s)==b['n'] and int((s>=t).sum())==b['accepted']
  r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum())
 rows[str(u)]=a
macro={}
for route in ['baseline_rbf','auto_without_diffusion']:
 macro[route]={budget:{k:float(np.mean([a[budget][route][k] for a in rows.values()])) for k in ['tar','far','auc','hter']} for budget in ['routes','fit50_reference']}
out=dict(status='PASS',targets=9,routes=18,results=rows,macro=macro,verification='Recomputed accepted counts, AUC, HTER and eight-source counts from saved scores and strict manifest labels')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(status='PASS',macro=macro)))

