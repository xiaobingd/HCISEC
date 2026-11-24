"""Independent episode isolation, threshold inheritance and baseline regression checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main(root):
    out=root/'conditional_correspondence/audio_conditioned_verifier'
    read=lambda name:json.loads((out/name).read_text())
    rows=read('per_seed_user_public.json');locks=read('thresholds_private.json')
    data=np.load(root/'phrase_auth/frontend_private.npz');u,w=data['user'],data['word']
    protocol=read('protocol_public.json');audits=read('episode_audit_public.json')
    assert len(rows)==720 and len({(r['seed'],r['user'],r['method'],r['family']) for r in rows})==720
    for r in rows:
        assert r['genuine_trials']==40 and r['attack_trials']==360
        assert abs(r['far']-r['false_accepts']/360)<1e-12
        assert abs(r['frr']-r['false_rejects']/40)<1e-12
        assert abs(r['hter']-(r['far']+r['frr'])/2)<1e-12
    assert len(locks)==240 and len(audits)==30
    for a in audits:
        e=np.load(out/f"episodes_seed{a['seed']}_held{a['held_target']}_private.npz")
        s,q=e['support_indices'],e['query_indices']
        assert s.shape==(400,9,12) and q.shape==(400,9,8)
        assert hashlib.sha256(s.tobytes()+q.tobytes()).hexdigest()==a['episode_sha256']
        assert not (u[s]==a['held_target']).any() and not (u[q]==a['held_target']).any()
        assert np.isin(w[s],protocol['gate_optimization_words']).all() and np.isin(w[q],protocol['gate_optimization_words']).all()
        owner=u[s[:,:,0]]
        assert (u[s]==owner[:,:,None]).all()
        assert (u[q[:,:,:4]]==owner[:,:,None]).all() and (u[q[:,:,4:]]!=owner[:,:,None]).all()
        for step in range(400):
            assert not np.intersect1d(w[s[step]],w[q[step]]).size
        assert a['destroyed_audio_moment_max_error']<1e-5
    for a in locks:
        if a['method'].startswith('B_'):
            ref=next(b['threshold'] for b in locks if b['seed']==a['seed'] and b['user']==a['user'] and b['method']=='B_correct_audio')
            assert a['threshold']==ref
    old=json.loads((root/'conditional_correspondence/enrollment_identity_quality_fixed_background/per_seed_user_public.json').read_text())
    maxerr=0.
    for r in rows:
        if r['method']!='ungated_ACC':continue
        b=next(b for b in old if b['seed']==r['seed'] and b['user']==r['user'] and b['family']==r['family'] and b['method']=='acc_crossword12' and b['backend']=='cosine' and b['words']==1)
        for key in ('eer','far','frr','hter','auc'):maxerr=max(maxerr,abs(r[key]-b[key]))
    assert maxerr<1e-10
    gates=read('gate_diagnostics_public.json')['weight_ranges']
    assert len(gates)==240 and all(.75<=g['gate_min']<=g['gate_max']<=1.25 for g in gates)
    assert len(list(out.glob('*_private.pt')))==120
    checks=dict(status='PASS',independently_checked_episodes=12000,held_target_excluded=True,support_query_word_overlap=0,baseline_max_absolute_error=maxerr,perturbations_reuse_correct_audio_threshold=True,private_checkpoints=120)
    (out/'independent_verification_public.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
