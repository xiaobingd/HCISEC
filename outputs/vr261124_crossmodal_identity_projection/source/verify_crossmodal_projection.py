"""Independently check folds, word roles, alpha locks and baseline equivalence."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main(root):
    out=root/'conditional_correspondence/crossmodal_identity_projection';read=lambda n:json.loads((out/n).read_text())
    data=np.load(root/'phrase_auth/frontend_private.npz');u,w=data['user'],data['word']
    protocol=read('protocol_public.json');audits=read('episode_audit_public.json');selection=read('alpha_selection_public.json')
    rows=read('per_seed_user_public.json');locks=read('thresholds_private.json')
    assert len(audits)==30 and len(selection)==60 and len(rows)==720 and len(locks)==240
    for r in audits:
      e=np.load(out/f"episodes_seed{r['seed']}_held{r['held_target']}_private.npz")['indices']
      assert e.shape==(400,2,16) and hashlib.sha256(e.tobytes()).hexdigest()==r['episode_sha256']
      assert not np.isin(u[e],[r['held_target'],r['inner_development_user']]).any()
      assert np.array_equal(u[e[:,0]],u[e[:,1]])
      assert np.isin(w[e],protocol['train_words']).all()
      for step in e:
        assert len(set(w[step[0]]))==len(set(w[step[1]]))==1
        assert not np.intersect1d(w[step[0]],w[step[1]]).size
    for r in selection:
      best=min(r['candidates'],key=lambda a:(a['hter'],a['eer'],a['alpha']))
      assert r['alpha']==best['alpha'] and r['alpha'] in protocol['alpha_grid']
      assert r['inner_development_user']!=r['held_target']
      assert all(a['genuine_trials']==8 and a['attack_trials']==64 for a in r['candidates'])
    assert len({(r['seed'],r['user'],r['method'],r['family']) for r in rows})==720
    for r in rows:
      assert r['genuine_trials']==40 and r['attack_trials']==360
      assert abs(r['far']-r['false_accepts']/360)<1e-12 and abs(r['frr']-r['false_rejects']/40)<1e-12
      assert abs(r['hter']-(r['far']+r['frr'])/2)<1e-12
    for r in locks:
      if r['method'].endswith('_Bthreshold'):
        b=next(a for a in locks if a['seed']==r['seed'] and a['user']==r['user'] and a['method']=='Audio_correct')
        assert b['threshold']==r['threshold']
    old=json.loads((root/'conditional_correspondence/enrollment_identity_quality_fixed_background/per_seed_user_public.json').read_text())
    error=0.
    for r in rows:
      if r['method']!='raw_ACC':continue
      b=next(a for a in old if a['seed']==r['seed'] and a['user']==r['user'] and a['family']==r['family'] and a['method']=='acc_crossword12' and a['backend']=='cosine' and a['words']==1)
      error=max(error,max(abs(r[k]-b[k]) for k in ('eer','hter','far','frr','auc')))
    assert error<1e-10 and len(list(out.glob('*_private.pt')))==60
    result=dict(status='PASS',episode_tasks=12000,training_identity_exclusion=True,word_role_separation=True,alpha_rule_recomputed=True,baseline_max_error=error,models=60,workpoints=720)
    (out/'independent_verification_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
