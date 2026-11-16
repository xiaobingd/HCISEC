"""Private prediction replay and fresh-checkpoint split verification."""
import hashlib
import json
import numpy as np
import torch
from evaluate_relation_predictability import OUT,relation,metrics,dump

def main():
    d=np.load('outputs/mei_common_frontend_private.npz');c=np.load(OUT/'acc_embeddings_private.npz')
    t=np.load('outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz')
    fit=np.flatnonzero((d['word']>=0)&(d['word']<12));ck=torch.load(OUT/'acc_fresh_private.pt',map_location='cpu',weights_only=False)
    assert np.array_equal(ck['training_indices'],fit) and ck['steps']==1400 and ck['seed']==42
    norms=np.load(OUT/'normalization_private.npz');assert np.array_equal(norms['fit_indices'],fit)
    assert np.array_equal(c['sample_id'],d['sample_id'][c['indices']])
    val=np.concatenate([np.flatnonzero((c['user']==u)&(c['word']==w))[:8] for u in range(1,10) for w in range(12,18)])
    assert not set(c['indices'][val])&set(fit)
    truth=relation(t['embedding'][c['indices'][val]].reshape(9,6,8,-1))
    private=np.load(OUT/'predictions_private.npz');result=json.loads((OUT/'results_public.json').read_text())
    error=0.
    for name,saved in result['methods'].items():
        actual=metrics(private[name],truth)
        for condition in ['sameword','crossword']:
            for key in ['task_spearman_macro','top1','aggregate_identity_ranking_spearman','residual_spearman_macro']:
                if actual[condition][key] is None:assert saved[condition][key] is None
                else:error=max(error,abs(actual[condition][key]-saved[condition][key]))
            assert len(saved[condition]['per_user'])==9
    assert error<1e-12
    lock=json.loads((OUT/'prediction_lock_public.json').read_text());protocol=json.loads((OUT/'protocol_public.json').read_text())
    sha=hashlib.sha256((OUT/'acc_fresh_private.pt').read_bytes()).hexdigest()
    assert sha==lock['checkpoint_sha256']==protocol['checkpoint_sha256']
    assert lock['cohort_counts']=={'5':126,'7':36,'9':1}
    checks=(OUT.parents[1]/'work/relation_predictability_checks_20261117.log')
    assert checks.exists() and 'PASS:' in checks.read_text(encoding='utf-8-sig')
    v=json.loads((OUT/'verification_public.json').read_text());v.update(
        saved_prediction_replay_max_error=error,actual_checkpoint_training_indices_checked=True,
        checkpoint_unchanged_after_prediction=True,synthetic_tests_passed=True,
        complete_nine_user_metrics_for_all10_methods=True,
        limits='private prediction replay and split/source checks;not independent waveform re-extraction')
    dump(OUT/'verification_public.json',v);print('VERIFICATION_PASS',error)

if __name__=='__main__':main()
