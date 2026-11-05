"""Independent score/count check and aggregate table for completed targets only."""
import json,sys,time
from pathlib import Path
import numpy as np
R=Path('/autodl-fs/data/mei_strict_autosupplement_20261029')
def main():
 out={};checked=0
 for u in range(1,10):
  p=R/('U%d'%u)
  if not (p/'COMPLETE.json').exists():continue
  row=json.loads((p/'results_public.json').read_text());lock=json.loads((p/'locked_before_final.json').read_text());z=np.load(str(p/'scores_private.npz'))
  for route,m in row['routes'].items():
   g=z[route+'__final_genuine'];n=z[route+'__final_impostor'];t=lock['thresholds'][route];auc=float(((g[:,None]>n[None,:])+.5*(g[:,None]==n[None,:])).mean())
   assert abs(auc-m['auc'])<1e-12 and abs(np.mean(g>=t)-m['tar'])<1e-12 and abs(np.mean(n>=t)-m['far'])<1e-12
   assert len(g)==m['genuine_n'] and len(n)==m['impostor_n']
   assert len(m['attack_by_user'])==8
   for v,a in m['attack_by_user'].items():
    s=n[z['attack_labels']==int(v)-1];assert len(s)==a['n'] and int((s>=t).sum())==a['accepted']
   checked+=1
  clean={k:v for k,v in row.items() if k!='diffusion_training'}
  clean['diffusion_training']={k:v for k,v in row['diffusion_training'].items() if k not in ('fit_ids','selection_ids')}
  out[str(u)]=clean
 public=dict(status='PASS',completed_targets=list(out),routes_verified=checked,results=out)
 (R/'verified_results_public.json').write_text(json.dumps(public,indent=2))
 lines=['# Strict optional supplementary features','','Only completed targets; all eight known attack identities retained. Not an unknown-attacker test.','', '|User|Route|TAR|FAR|AUC|Selected weights|','|---|---|---:|---:|---:|---|']
 for u,row in out.items():
  for route,m in row['routes'].items():lines.append('|U%s|%s|%.2f%%|%.2f%%|%.4f|%s|'%(u,route,100*m['tar'],100*m['far'],m['auc'],json.dumps(m['weights'])))
 (R/'results_report_public.md').write_text('\n'.join(lines)+'\n');print('AUTOSUPPLEMENT_VERIFIED',checked,flush=True)
if __name__=='__main__':
 if '--watch' not in sys.argv:main()
 else:
  previous=-1
  while True:
   try:
    status=json.loads((R/'status_public.json').read_text());n=len(list(R.glob('U*/COMPLETE.json')))
    if n!=previous:main();previous=n
    if status['state']=='complete':main();break
    if status['state']=='failed':raise RuntimeError(status.get('error','training queue failed'))
   except (FileNotFoundError,json.JSONDecodeError):pass
   time.sleep(15)
