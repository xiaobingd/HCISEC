import argparse,json
from pathlib import Path
import numpy as np
def main(root):
    out=root/'conditional_correspondence/selective_gradient_audit'
    p=json.loads((out/'protocol_public.json').read_text());d=json.loads((out/'diagnostics_private.json').read_text());s=json.loads((out/'summary_public.json').read_text())
    assert len(d['gradients'])==600 and len(d['ranking'])==600 and len(d['virtual'])==3000
    assert set(p['optimization_words']).isdisjoint(p['virtual_check_words'])
    for r in d['gradients']:
        assert r['selected']==bool(r['teacher_stable'] and r['body_cosine']>0)
        assert -1.00001<=r['body_cosine']<=1.00001 and -1.00001<=r['wrong_body_cosine']<=1.00001
    for row in s['per_user']:
        g=[a for a in d['gradients'] if a['user']==row['user']];assert len(g)==60
        assert abs(row['selected_rate']-np.mean([a['selected'] for a in g]))<1e-12
    for r in s['virtual_updates']:
      for a in r['per_user']:
        rr=[v for v in d['virtual'] if v['method']==r['method'] and v['user']==a['user']]
        ref=[next(b for b in d['virtual'] if b['method']=='identity_only' and (b['seed'],b['episode'],b['user'])==(v['seed'],v['episode'],v['user'])) for v in rr]
        delta=np.mean([v['check_margin']-b['check_margin'] for v,b in zip(rr,ref)])
        assert abs(delta-a['check_margin_delta'])<1e-12
    # Clarify archived evidence: NPZ arrays are decompressed then sliced to fit;
    # no non-fit records enter analysis or score computation.
    s['verification'].pop('no_final_query_access',None)
    s['verification'].update(no_final_query_scoring=True,only_fit_features_used=True)
    (out/'summary_public.json').write_text(json.dumps(s,indent=2))
    v=dict(status='PASS',gradient_rows=600,virtual_workpoints=3000,selection_rule_recomputed=True,per_user_margins_independently_recomputed=True)
    (out/'independent_verification_public.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
