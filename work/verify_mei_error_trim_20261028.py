"""Recompute final counts/AUC and audit removal identities/roles independently."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
R=Path('/autodl-fs/data/mei_error_trim_20261028')
def main():
 results=json.loads((R/'results_public.json').read_text());removed=json.loads((R/'removed_training_private.json').read_text());scores=np.load(str(R/'scores_private.npz'))
 source=json.loads(Path('/autodl-fs/data/complete_baseline_teacher_20261028/source_manifest.json').read_text());names=['bo']+['user%d'%v for v in range(2,10)]
 checked=0
 for u,features in results.items():
  for mode,row in features.items():
   for drop in removed[u][mode]:
    assert drop['user'] not in (3,8) and drop['row']//10<15 and drop['votes']>=2
    assert drop['row'] in source[names[drop['user']-1]]['auth_indices']
   for v,c in row['removed_by_identity'].items():
    assert c['removed']<=int(np.floor(.1*c['total']))
    assert c['removed']==sum(d['user']==int(v) for d in removed[u][mode])
   reference=None
   for arm,detail in row['routes'].items():
    key='U%s_%s_%s'%(u,mode,arm);s=scores[key+'_scores'];uid=scores[key+'_users'];idx=scores[key+'_rows'];t=detail['fit']['threshold'];y=uid==int(u)
    assert np.isfinite(s).all() and np.all(idx//10>=20)
    if reference is None:reference=(uid,idx)
    else:assert np.array_equal(uid,reference[0]) and np.array_equal(idx,reference[1])
    for scope in ('all','unknown','known'):
     mask=np.ones(len(y),bool) if scope=='all' else y|np.isin(uid,[3,8]) if scope=='unknown' else ~np.isin(uid,[3,8])
     yy=y[mask];ss=s[mask];m=detail[scope]
     assert int(np.sum(ss[yy]>=t))==m['genuine_accepted']
     assert int(np.sum(ss[~yy]>=t))==m['attack_accepted']
     assert int(yy.sum())==m['genuine_count'] and int((~yy).sum())==m['attack_count']
     assert abs(roc_auc_score(yy,ss)-m['AUC'])<1e-12
    assert set(detail['by_attacker'])==set(str(v) for v in range(1,10) if v!=int(u))
    checked+=1
 out=dict(status='PASS',routes_checked=checked,all_final_rows_retained=True,all_eight_attack_sources_retained=True,unknown_users_absent_from_removals=True,cap_and_word_role_checks=True)
 assert checked==20;(R/'verification_public.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
