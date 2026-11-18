"""Verify fixed fit80 references and saved diffusion-supplement test scores."""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/autodl-fs/data/mei_auth80_diffusion_20261029')
PRIOR=Path('/autodl-fs/data/mei_auth80_20261029')
M=json.loads(Path('/autodl-fs/data/mei_strict_baseline_20261029/manifest_private.json').read_text())
rows={}
for u in range(1,10):
 folder=ROOT/('U%d'%u);a=json.loads((folder/'results_public.json').read_text())
 ref=json.loads((PRIOR/('U%d'%u)/'results_public.json').read_text())
 assert a['reference']==ref['routes']
 lock=json.loads((folder/'locked_before_queries.json').read_text());before=json.loads((PRIOR/('U%d'%u)/'locked_before_queries.json').read_text())
 for k,w in before['weights']['auto_without_diffusion'].items():assert lock['weights'][k]==w
 assert [k for k in lock['weights'] if k.startswith('diff_')]==a['selected_diffusion'] and len(a['selected_diffusion'])<=1
 data=np.load(str(folder/'scores_private.npz'));g=data['final_genuine'];n=data['final_impostor'];t=lock['threshold'];r=a['route']
 labels=np.concatenate([np.full(len(M['sources'][str(v)]['roles']['test']),v-1) for v in range(1,10) if v!=u])
 assert np.array_equal(labels,data['attack_labels'])
 assert len(g)==len(M['sources'][str(u)]['roles']['test'])==r['genuine_n'] and len(n)==len(labels)==r['impostor_n']
 assert abs(((g[:,None]>n)+.5*(g[:,None]==n)).mean()-r['auc'])<1e-12
 assert abs((g>=t).mean()-r['tar'])<1e-12 and abs((n>=t).mean()-r['far'])<1e-12
 assert abs((1-r['tar']+r['far'])/2-r['hter'])<1e-12
 assert len(r['attack_by_user'])==8
 for v,b in r['attack_by_user'].items():
  s=n[labels==int(v)-1];assert len(s)==b['n'] and int((s>=t).sum())==b['accepted']
 if not a['selected_diffusion']:
  prior=np.load(str(PRIOR/('U%d'%u)/'scores_private.npz'))
  for role in ('final_genuine','final_impostor'):assert np.array_equal(data[role],prior['auto_without_diffusion__'+role])
 r['genuine_accepted']=int((g>=t).sum());r['attack_accepted']=int((n>=t).sum());rows[str(u)]=a
macro={k:float(np.mean([a['route'][k] for a in rows.values()])) for k in ('tar','far','auc','hter')}
reference={k:float(np.mean([a['reference']['auto_without_diffusion'][k] for a in rows.values()])) for k in macro}
out=dict(status='PASS',targets=9,results=rows,macro=macro,reference_macro=reference,verification='Independent saved-score counts, pairwise AUC, fixed fit80 references, eight-source labels, unchanged non-diffusion weights and no-selection score identity')
(ROOT/'verified_summary_public.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='PASS',macro=macro,reference=reference)))

