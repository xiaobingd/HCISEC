# ACC-to-audio identity-head experiment

This experiment uses five isolated targets: `bo` (`user1`), `user2`, `user3`, `user4`, and `user5`. Each target has 150 CycleGAN samples separated from its authentication samples. The CycleGAN split is 119 train, 16 validation, and 15 test; authentication uses the remaining half.

The ACC-to-audio generator uses U-Net skips, bilinear upsampling, a shift-tolerant paired structure loss (±8 frames), frequency-envelope matching, cycle consistency, and identity contrast. The selected checkpoints are user1/bo epoch 50, user2 epoch 50, user3 epoch 100, user4 epoch 50, and user5 epoch 50.

## Generation test

| target | MAE | cosine | checkerboard ratio | paired retrieval |
|---|---:|---:|---:|---:|
| bo | 0.3172 | 0.6377 | 0.761 | 26.7% |
| user2 | 0.3241 | 0.6291 | 0.703 | 13.3% |
| user3 | 0.2634 | 0.7772 | 0.602 | 13.3% |
| user4 | 0.2960 | 0.7092 | 0.669 | 6.7% |
| user5 | 0.3158 | 0.6852 | 0.848 | 6.7% |

## Frozen-feature identity probe

The probe trains only a standardized linear classifier on the original ACC shallow feature, CycleGAN bottleneck, cycle reconstruction difference, or all three. Thresholds are selected on development data and evaluated on 15 genuine and 240 impostor final samples.

| target | combined AUC | combined EER | combined HTER | bottleneck AUC | shallow AUC |
|---|---:|---:|---:|---:|---:|
| bo | 0.912 | 0.119 | 0.215 | 0.917 | 0.874 |
| user2 | 0.686 | 0.344 | 0.392 | 0.615 | 0.678 |
| user3 | 0.565 | 0.450 | 0.465 | 0.529 | 0.604 |
| user4 | 0.836 | 0.204 | 0.219 | 0.815 | 0.736 |
| user5 | 0.751 | 0.273 | 0.292 | 0.748 | 0.693 |

Macro averages are AUC 0.750, EER 0.278, and HTER 0.316 for the combined representation, versus AUC 0.717 and HTER 0.370 for shallow ACC alone. The result supports user-dependent feasibility: the CycleGAN bottleneck adds identity information for bo, user4, and user5, but user2 and user3 remain weak. Generation similarity alone is not a reliable identity criterion.

