# Innovation 1: 30 final genuine samples for user1 and user4

The target authentication half was repartitioned to use 30 final genuine examples. The original target train and model-validation assignments, all attacker assignments, and each user's 150 CycleGAN samples remain unchanged. Fifteen target examples were moved from threshold development into final genuine testing. Target development therefore contains 15 samples for user1 and 14 for user4. Each final attack set still contains 240 examples. All groups are pairwise disjoint, and the CycleGAN/auth separation was checked from the saved manifests.

The generator checkpoints and feature evaluator are unchanged, with the corrected user-target response-gap formula. N0 input scale was used. Thresholds were selected from the smaller target development set and unchanged attacker calibration set; the 30 final genuine and 240 final attackers were excluded from training, calibration, and threshold selection.

| target | feature | final AUC | final FAR | final FRR | final HTER |
|---|---|---:|---:|---:|---:|
| user1 | C4 relative cycle gap | 0.9732 | 0.0458 | 0.1667 | 0.1062 |
| user1 | C6 relative cycle + response gap | 0.9899 | 0.0083 | 0.2000 | 0.1042 |
| user4 | C4 relative cycle gap | 0.8372 | 0.0958 | 0.5000 | 0.2979 |
| user4 | C7 all features | 0.8935 | 0.1417 | 0.2000 | 0.1708 |

For comparison, the prior 15-genuine protocol gave C4 AUC 0.9789/0.7817 for user1/user4. The user1/user4 C4 separation persists with 30 final genuine samples. User4 C4 AUC rises, but its selected operating threshold rejects 15 of 30 genuine samples, so this is not yet a usable standalone authenticator. The C7 combination is better at that operating point, accepting 24 of 30 genuine samples while accepting 34 of 240 attackers.

The 30-sample result is not simply a larger version of an unchanged test set: 15 former target-development examples entered final testing, reducing the data available for threshold calibration. Thus changes in FAR/FRR reflect both the new test composition and the recalibrated threshold. The existing per-user full-data normalization in `acc.npy` also remains a limitation.

Reproducibility: `experiments/innovation1_final30/target_auth_protocol_150_final30.py` and `experiments/innovation1_final30/innovation1_response_gap_pilot.py`. Aggregate and per-source metrics are in `experiments/innovation1_final30/results/user1_result.json` and `user4_result.json`. Full sample-index manifests remain in the local/remote experiment outputs: `work/innovation1_final30_20260929/user1/` and `work/innovation1_final30_20260929/user4/`.

