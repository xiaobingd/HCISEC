"""Checkpoint/cache parity, data isolation, threshold and margin score replay."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
import hashlib
import json
import numpy as np
import torch
from crossword_representation_losses import IdentityEncoder
from evaluate_local_relation_auth_controls import OUT,dump
from evaluate_crossword_encoder_distill import metric,threshold

def main():
    torch.set_num_threads(4)
    metadata=np.load(OUT/'split_normalization_private.npz');f=np.load('outputs/mei_common_frontend_private.npz')
    fit=metadata['fit_indices'];cal=metadata['cal_indices'];val=metadata['validation_indices']
    assert len(fit)==648 and len(cal)==216 and len(val)==432
    assert not set(fit)&set(cal) and not set(fit)&set(val) and not set(cal)&set(val)
    assert np.all(f['word'][fit]<12) and set(f['word'][val])==set(range(12,18))
    assert np.array_equal(f['sample_id'][metadata['used_indices']],metadata['sample_id'])
    spec=f['spectrum'];assert np.allclose(metadata['mean'],spec[fit].mean((0,3),keepdims=True))
    assert np.allclose(metadata['std'],spec[fit].std((0,3),keepdims=True).clip(1e-4))
    training=json.loads((OUT/'training_summary_public.json').read_text());assert len(training)==12
    for seed in [42,43,44]:assert len(set(r['batch_sha256'] for r in training if r['seed']==seed))==1
    rng=np.random.RandomState(91)
    selected=np.r_[rng.choice(metadata['fit_local'],8,replace=False),rng.choice(metadata['cal_local'],8,replace=False),rng.choice(metadata['val_local'],8,replace=False)]
    x=torch.tensor((spec[metadata['used_indices'][selected]]-metadata['mean'])/metadata['std'])
    parity=0.
    for r in training:
        prefix=f"{r['method']}_seed{r['seed']}";path=OUT/f'{prefix}_private.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert np.array_equal(ck['fit_indices'],fit) and ck['seed']==r['seed'] and ck['method']==r['method']
        assert ck['warmup']==400 and ck['continuation']==400
        model=IdentityEncoder();model.load_state_dict(ck['encoder']);model.eval()
        with torch.no_grad():actual=model(x).numpy()
        saved=np.load(OUT/f'{prefix}_features_private.npz')['embedding'][selected]
        parity=max(parity,float(np.max(abs(saved-actual))))
    assert parity<1e-5,parity
    scores=np.load(OUT/'scores_private.npz');locks=json.loads((OUT/'calibration_locks_private.json').read_text())
    rows=json.loads((OUT/'all_seed_user_private.json').read_text());error=0.;margin_error=0.;threshold_error=0.
    for r in rows:
        owner=int(r['user'][1:]);key=f"{r['method']}_seed{r['seed']}_{r['backend']}_{r['user']}"
        ycal=np.tile(np.repeat(np.arange(1,10),2)==owner,264)
        tau=threshold(ycal,scores[key+'_calibration']);threshold_error=max(threshold_error,abs(tau-locks[key]['threshold']))
        value=scores[key+'_'+r['condition']];y=np.tile(np.repeat(np.arange(1,10),4)==owner,60)
        actual=metric(y,value,tau);error=max(error,max(abs(actual[k]-r[k]) for k in actual))
        cases=value.reshape(60,9,4);negative=np.arange(9)!=owner-1
        margin=cases[:,owner-1].mean(1)-cases[:,negative].mean(2).max(1)
        # Evaluation converts each case margin to a Python float before pooling.
        gap=float(margin.astype('float64').mean()/locks[key]['impostor_std'])
        worst=float((cases[:,owner-1].min(1)-cases[:,negative].reshape(60,-1).max(1)).astype('float64').mean()/locks[key]['impostor_std'])
        margin_error=max(margin_error,abs(gap-r['hard_identity_margin']),abs(worst-r['worst_sample_margin']))
    assert error==0 and threshold_error==0 and margin_error<1e-10,(error,threshold_error,margin_error)
    v=json.loads((OUT/'verification_public.json').read_text());v.update(
        actual_checkpoint_fit_indices_verified=True,all12_checkpoint_cache_parity_max_error=parity,
        independent_threshold_replay_error=threshold_error,margin_replay_max_error=margin_error,
        complete432_seed_user_condition_rows=True,normalization_independently_recomputed=True,
        limits='checkpoint/cache and score replay;not independent raw extraction;exploratory words12..17')
    dump(OUT/'verification_public.json',v);print('VERIFY_PASS',parity,margin_error,flush=True)

if __name__=='__main__':main()
