# user4 CycleGAN identity retention: controlled training probes

## Finding

The current user4 model has room for improvement, but neither adding more epochs nor the two single-variable loss changes tested here improved its identity verification. The main observable bottleneck is weak held-out ACC-to-audio pairing: the original generator's MAE is 0.2960 on 15 CycleGAN test pairs, compared with 0.2382 for a mean audio spectrogram estimated **only from the 119 CycleGAN training pairs**. Its output diversity is 0.708 of real audio. This does not prove the generated audio has no identity information, but it means visual plausibility and cycle reconstruction alone cannot establish a useful forward mapping.

The 150 CycleGAN samples remain isolated from the authentication half. Within the CycleGAN half the fixed split is 119 train, 16 validation, 15 test. The authentication evaluation uses the unchanged user4 protocol, with 15 final genuine and 240 final impostor examples. All checkpoints were evaluated with the corrected user4-target relative-response formula. N0 was used for training comparisons; the N1/N2 input transformations are only frozen-generator stress tests.

## Controlled comparisons

| user4 checkpoint | change from epoch-50 baseline | held-out ACC→audio MAE ↓ | output diversity/real ↑ | paired signature top-1 | C4 relative cycle AUC ↑ | C7 all-feature AUC ↑ | C7 HTER ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| epoch 50 baseline | none | 0.2960 | 0.708 | 0/15 | 0.7817 | 0.8506 | 0.2208 |
| epoch 100 baseline | 50 more epochs | 0.2844 | 0.627 | 0/15 in prior evaluator | 0.7328 | 0.8650 | 0.2000 |
| epoch 50 without sample contrast | remove `0.50 * idc_audio + 0.15 * idc_acc` | 0.2933 | 0.696 | 1/15 | 0.7603 | 0.8336 | 0.2854 |
| epoch 50 stronger paired structure | change `4.00 * forward_structure` to `8.00` | 0.2979 | 0.714 | 2/15 | 0.7778 | 0.8228 | 0.3104 |

The epoch-100 C7 result was recomputed with the corrected user4-target formula. The epoch-100 top-1 value came from an earlier evaluator and should not be directly ranked against the new evaluator's top-1 values. The 50/100 quality figures came from the earlier fixed CycleGAN test evaluation; the two new ablations were also evaluated on the same 15 samples. A 1- or 2-sample top-1 difference is too small to establish an improvement.

The original `SampleIdentityPreserveLoss` treats every other window in a batch as a negative even though all windows belong to user4. Thus it targets sample correspondence, not stable user identity. Removing it worsened the frozen authentication features, so it should not be dropped without a replacement. Doubling the paired structure weight slightly increased signature top-1 but worsened MAE, cycle error, and C7 authentication. Continuing to epoch 100 improved MAE but reduced diversity and C4; epoch choice must include identity diagnostics rather than visual quality alone.

## Recommended next training experiment

1. Rebuild ACC and microphone spectrograms using scaling parameters fitted on the **119 CycleGAN training pairs only**; apply them unchanged to CycleGAN validation/test and the separate authentication half. Preserve raw amplitude, band energy, and frequency profile as auxiliary channels or targets rather than normalizing each window to the same shape. This removes the known per-user full-data normalization leakage in the existing `acc.npy` files.
2. Train a paired ACC→audio baseline with the same generator capacity on the 119 pairs. If it still fails to beat the training-mean audio predictor on the 16 validation and 15 test pairs, investigate alignment, preprocessing, and information shared by the two sensors before tuning adversarial weights. The time stamps themselves have already been confirmed by the user; this diagnostic concerns spectral framing and model input representation.
3. For CycleGAN, replace the sample-negative identity term with a two-part objective: positive paired spectral-profile agreement plus consistency of stable low-frequency/harmonic statistics across different **training windows of user4**. Retain a variance or diversity constraint so the generator cannot collapse to a mean spectrum. Do not treat other user4 windows as identity negatives. Tune the weights on the 16 CycleGAN validation samples using forward quality, diversity, and paired retrieval; hold the 15 CycleGAN test samples and all authentication final samples for one-time evaluation.
4. If a discriminative identity teacher is needed, train it on a disjoint development cohort and freeze it before fitting user4's generator. Do not use user4's authentication half or the final attackers to train that teacher. This is a later experiment because a single-user CycleGAN training set by itself does not contain user-class negatives.

Selection should require improvement in forward conversion **and** identity separation. The current 15-sample CycleGAN test set and 15 final genuine authentication samples are too small for fine-grained ranking of close variants; repeat the selected configuration across seeds before drawing a general conclusion.

Code: `work/user4_train_cyclegan_reference.py`, `work/user4_train_cyclegan_no_sample_negatives.py`, `work/user4_train_cyclegan_structure8.py`, `work/eval_user4_identity_ablation.py`, `work/innovation1_user1_response_gap_pilot.py`.

Machine-readable results: `work/user4_identity_corrected_20260929/{baseline,n1,n2,epoch100}/result.json`, `work/user4_identity_loss_ablation_20260929/auth_probe_corrected/result.json`, `work/user4_structure8_20260929/auth_probe/result.json`, and `work/user4_identity_optimization_quality_test.json`.

