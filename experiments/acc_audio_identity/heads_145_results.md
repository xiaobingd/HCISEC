# SupCon authentication-head results for user1, user4, user5

The original three-source contrastive head was trained with the isolated 150-sample authentication protocol and the aligned CycleGAN checkpoints selected previously:

- `user1` / bo: CycleGAN epoch 50
- `user4`: CycleGAN epoch 50
- `user5`: CycleGAN epoch 50

The CycleGAN generators were frozen. Only the feature fusion layer and projection head were trained with supervised contrastive loss for 100 epochs. Each reported value averages five registration seeds (42, 123, 456, 789, 2026). Thresholds were calibrated using development genuine samples and impostor calibration samples, then evaluated on 15 final genuine samples and 240 final impostors.

| target | FAR | FRR | HTER | EER | AUC | balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| bo (`user1`) | 9.0% | 15.0% | **12.0%** | 10.6% | 0.960 | 0.880 |
| user4 | 23.0% | 25.0% | **24.4%** | 23.9% | 0.810 | 0.756 |
| user5 | 33.0% | 35.0% | **33.9%** | 34.7% | 0.758 | 0.661 |
| mean | 22.0% | 24.9% | **23.4%** | 23.0% | **0.843** | 0.766 |

Compared with the frozen linear probe, SupCon improves the three-user combined representation from AUC/HTER of 0.833/0.242 (bo, user4, user5 average from the same target protocol) to 0.843/0.234. The improvement is modest overall but substantial for bo. user5 remains the limiting target, so the method should be described as user-adaptive rather than universally reliable.

Full machine-readable output: `outputs/heads_145_evaluation.json`.

