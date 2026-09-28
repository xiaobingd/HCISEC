# Why user1/bo performs best

The evidence points to a data-distribution advantage, especially in the original ACC branch. It is not because ACC and Audio are better aligned for user1.

| target | ACC mean | ACC std | Audio mean | Audio std | ACC shallow AUC | CycleGAN bottleneck AUC |
|---|---:|---:|---:|---:|---:|---:|
| user1/bo | -0.569 | 0.363 | -0.131 | 0.503 | 0.874 | 0.917 |
| user2 | -0.388 | 0.413 | -0.163 | 0.484 | 0.678 | 0.615 |
| user3 | -0.400 | 0.412 | -0.228 | 0.487 | 0.604 | 0.529 |
| user4 | -0.464 | 0.392 | -0.242 | 0.479 | 0.736 | 0.815 |
| user5 | -0.410 | 0.411 | -0.150 | 0.518 | 0.693 | 0.748 |

User1's ACC baseline is much darker and has a narrower global spread than the other selected users. A six-statistic centroid analysis gave user1 a nearest-user distance of about 3.22 standardised units, while user2 and user3 had nearest-user distances below 1.4. This makes user1 an outlier that a classifier can reject easily.

The ACC→Audio pairing does not explain the advantage. User1 had mean paired time-profile correlation 0.305 and frequency-profile correlation 0.240, both lower than most other users. Its flat paired cosine was only 0.152. The authentication advantage therefore begins in the original ACC distribution and is amplified by the head; it is not evidence that CycleGAN reconstructs user1's audio more faithfully.

The preprocessing code uses percentile mapping to [-1, 1] and intentionally retains individual differences. It does not normalize each window to zero mean and unit variance. Consequently, sensor amplitude, placement, contact pressure, or recording gain can remain as identity shortcuts. This is valid for a device-specific biometric, but it weakens the claim that the identity comes specifically from the cross-modal conversion.

The clean follow-up is an amplitude-normalized ablation: equalize each window's mean and variance before the frozen feature extractor, then retrain the same head. If user1 remains strong while other users improve, the feature is genuinely structural; if user1 collapses, its current result is mainly a global ACC-level shortcut.

