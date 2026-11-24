import argparse,json
from pathlib import Path
import numpy as np

def main(root):
    out=root/'conditional_correspondence/linear_shared_identity';read=lambda n:json.loads((out/n).read_text())
    p=read('protocol_public.json');s=read('summary_public.json');selection=read('selection_public.json');audit=read('exclusion_audit_public.json');rows=read('metrics_private.json')
    assert len(selection)==540 and len(audit)==135 and len(rows)==19440
    assert set(p['fit_words']).isdisjoint(p['internal_development_words'])
    for a in audit:
      assert len(a['mapping_training_users'])==8 and not set(a['mapping_training_users'])&set(a['held_pair'])
      assert set(a['mapping_training_users'])|set(a['held_pair'])==set(range(1,11))
      assert a['held_records_in_fit_or_selection']==0
    for a in selection:
      best=min(a['candidates'],key=lambda r:(-round(r['inner_auc'],12),r['dimension'],-r['ridge']))
      assert a['dimension']==best['dimension'] and a['ridge']==best['ridge'] and a['inner_auc']==best['inner_auc']
      assert len(a['candidates'])==9 and a['dimension'] in (1,2,4)
    for a in rows:
      assert a['user'] in a['held_pair'] and 0<=a['top1']<=1 and 0<=a['auc']<=1 and 0<=a['top1_10way']<=1
    for a in s['per_user']:
      rr=[r for r in rows if (r['user'],r['method'],r['scope'],r['repeats'])==(a['user'],a['method'],a['scope'],a['repeats'])]
      assert len(rr)==54
      for key in ['top1','auc','margin','top1_10way']:assert abs(a[key]-np.mean([r[key] for r in rr]))<1e-12
    with np.load(out/'linear_models_private.npz') as models:
      assert len(models.files)==4320
      assert all(np.isfinite(models[k]).all() for k in models.files)
    result=dict(status='PASS',outer_identity_folds=135,correct_and_mismatch_models=1080,nested_candidate_selections=540,workpoints=19440,held_identity_exclusion=True,hyperparameters_independently_recomputed=True,per_user_aggregation_independently_recomputed=True,final_authentication_performed=False)
    (out/'independent_verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
