"""Independent result/score and split verification for both public-data pilots."""
import sys,json,hashlib,zipfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/autodl-fs/data/public_baseline_compare_20261028');r=json.loads((ROOT/'results_public.json').read_text());scores=np.load(str(ROOT/'private_scores.npz'));verified=[]
for dataset,owners in r.items():
 p=json.loads((ROOT/dataset/'protocol_private.json').read_text());d=np.load(str(ROOT/dataset/'inputs.npz'))
 assert not set(p['owners'])&set(p['unknown'])
 for owner,record in owners.items():
  groups=p['roles'][owner];assert len(set(sum(groups.values(),[])))==sum(map(len,groups.values()))
  train=set(groups['gan_train']+groups['gan_val']);auth=set(groups['auth_train']+groups['calibration']+groups['final']);assert not train&auth
  known=next(u for u in p['owners'] if u!=owner)
  assert record['checkpoint_sha256']==hashlib.sha256((ROOT/dataset/owner/'models/checkpoint_epoch_200.pth').read_bytes()).hexdigest()
  for route,row in record['routes'].items():
   prefix=dataset+'_'+owner+'_'+route;g=scores[prefix+'_g'];assert len(g)==8
   for kind,users in [('known',[known]),('unknown',p['unknown'])]:
    n=np.concatenate([scores[prefix+'_'+u] for u in users]);m=row[kind];t=row['threshold']
    assert len(n)==m['attacker_n'] and int(np.sum(n>=t))==m['attacker_accepted']
    assert abs(m['tar']-np.mean(g>=t))<1e-12 and abs(m['far']-np.mean(n>=t))<1e-12
    auc=roc_auc_score(np.r_[np.ones(len(g)),np.zeros(len(n))],np.r_[g,n]);assert abs(m['auc']-auc)<1e-12
    assert abs(m['hter']-(m['far']+m['frr'])/2)<1e-12
   lock=json.loads((ROOT/dataset/owner/('auth_'+route+'_locked_private.json')).read_text());assert lock['threshold']==row['threshold']
   assert not set(lock['fit_units'])&set(sum((p['roles'][u]['final'] for u in p['unknown']),[]))
   verified.append([dataset,owner,route])
assert len(verified)==16
out={'verified_routes':verified,'n_routes':len(verified),'checks':['checkpoint SHA','unit role disjointness','unknown excluded','locked threshold','genuine and attacker counts','FAR/TAR/HTER','known and unknown AUC'],'status':'PASS'}
(ROOT/'verification_public.json').write_text(json.dumps(out,indent=2))
# Scalar/identity aggregates only; private source indices/raw tensors excluded.
meta={}
for dataset in r:
 p=json.loads((ROOT/dataset/'protocol_private.json').read_text());meta[dataset]={k:p[k] for k in ['dataset','owners','unknown','scales_bc_mel','shared_normalization','clip_fractions']}
(ROOT/'data_summary_public.json').write_text(json.dumps(meta,indent=2))
with zipfile.ZipFile(str(ROOT/'review_private.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for name in ['results_public.json','verification_public.json','data_summary_public.json','private_scores.npz','confound_controls_public.json']:
  if (ROOT/name).exists():z.write(str(ROOT/name),name)
 for dataset,owners in r.items():
  for owner in owners:
   p=ROOT/dataset/owner/'auth_final_generation.png';z.write(str(p),dataset+'_'+owner+'_generation.png')
print('ALL_PUBLIC_BASELINE_RESULTS_VERIFIED',len(verified),flush=True)
