# Personal cross-modal teacher and identity-constrained generator, word-major split

Date: 2026-10-27

## Question

Does a per-user cross-modal identity teacher improve one-class authentication from ACC-to-Audio generated features when prompt words are isolated between fitting, registration, calibration, and final test?

## Data and isolation

- Dataset: the nine-user Mei ACC/Audio arrays. U3 and U8 were excluded from loading, teacher negatives, generator training, registration, and attack evaluation.
- The prior same-content audit describes the arrays as word-major, with ten consecutive repetitions per word. The user clarified that missing samples are at the tail; all users therefore use the common first 297 rows. This ordering/tail assumption is not independently verified against a transcript manifest.
- Identical existing preprocessing for every user: use the first 297 rows and crop the final spectrogram column to obtain 80x80 inputs.
- Prompt groups (word IDs are zero-based): teacher fit 0–11 (120 samples/user); checkpoint selection 12–14 (30); registration 15–20 (60); genuine-only radius calibration 21–23 (30); final test 24–29 (57, due to the partial final word block).
- Every repetition of a word remains in the same partition. Final test words are absent from teacher/generator fitting, model selection, and threshold calibration.
- For each target, the personal teacher is a separate target-vs-six-other-users ACC/Audio embedding model, trained on the fit prompts and frozen for that target's generator. The six other identities are known teacher negatives, so this is held-out-word/recording evaluation, not unseen-attacker-identity evaluation.

## Models and decision rules

- Teacher loss: binary target-vs-rest loss, supervised contrastive loss (weight 0.15), and paired cross-modal cosine consistency (weight 0.5).
- Each target has a personal ACC-to-Audio generator trained with paired Audio L1, frequency-gradient L1, time-gradient L1, and frozen-teacher identity loss. Identity-loss weights: 0, 0.25, and 0.5; all other settings are held fixed.
- Primary authentication: positive-only cosine prototype from the 60 registration samples; radius calibrated from 30 genuine-only development samples to target 10% FRR. Final test has 57 genuine queries and 342 impostor queries. Reported TAR/FAR are measured on the untouched final test.
- Secondary reference: logistic binary head trained on registration samples (60 positive and 360 known-user negatives); decision threshold calibrated only from the target's genuine development samples. This is not the positive-only/open-set protocol.

## Positive-only prototype results

Each cell is `AUC / FAR / TAR`; FAR and TAR are fractions. Macro rows are unweighted means over seven target users.

| Target | Raw ACC teacher embedding | G, identity weight 0 | G, identity weight 0.25 | G, identity weight 0.5 |
|---|---:|---:|---:|---:|
| U1 | 0.996 / 0.006 / 0.807 | 0.562 / 0.991 / 0.912 | 0.964 / 0.129 / 0.965 | 0.964 / 0.123 / 0.982 |
| U2 | 0.564 / 0.743 / 0.807 | 0.547 / 0.880 / 0.895 | 0.565 / 0.848 / 0.930 | 0.537 / 0.737 / 0.807 |
| U4 | 0.658 / 0.775 / 0.982 | 0.549 / 0.833 / 0.912 | 0.503 / 0.839 / 0.895 | 0.535 / 0.898 / 0.965 |
| U5 | 0.683 / 0.775 / 0.947 | 0.646 / 0.772 / 0.947 | 0.705 / 0.743 / 0.965 | 0.714 / 0.646 / 0.930 |
| U6 | 0.551 / 0.974 / 0.982 | 0.552 / 0.845 / 0.912 | 0.585 / 0.863 / 0.947 | 0.606 / 0.915 / 0.982 |
| U7 | 0.460 / 0.994 / 0.912 | 0.554 / 0.810 / 0.825 | 0.504 / 0.827 / 0.825 | 0.543 / 0.711 / 0.789 |
| U9 | 0.749 / 0.567 / 0.965 | 0.551 / 0.889 / 0.930 | 0.709 / 0.547 / 0.825 | 0.740 / 0.491 / 0.895 |
| **Macro mean** | **0.666 / 0.690 / 0.915** | **0.566 / 0.860 / 0.905** | **0.648 / 0.685 / 0.907** | **0.663 / 0.646 / 0.907** |

## Secondary binary-head results

Unweighted macro means over seven targets (`AUC / FAR / TAR`):

| Route | Macro AUC | Macro FAR | Macro TAR |
|---|---:|---:|---:|
| Raw ACC teacher embedding | 0.762 | 0.546 | 0.902 |
| G, identity weight 0 | 0.716 | 0.691 | 0.957 |
| G, identity weight 0.25 | 0.738 | 0.608 | 0.947 |
| G, identity weight 0.5 | 0.733 | 0.595 | 0.930 |

## Interpretation

1. The correct word-major identity-teacher/generator pipeline completed for U1, U2, U4, U5, U6, U7, and U9. The first attempted round-major split was stopped and is not included in these results.
2. The identity term sharply improved U1 over its paired-only generated route (prototype AUC 0.562 to 0.964; FAR 0.991 to 0.123 at weight 0.5), but raw ACC remained substantially better for U1 (FAR 0.006).
3. Across all seven users, weight 0.5 reduced prototype macro FAR by 4.4 percentage points versus raw ACC (0.690 to 0.646), while macro TAR fell by 0.8 points (0.915 to 0.907) and macro AUC was essentially unchanged/slightly lower (0.666 to 0.663). Weight 0.25 had almost no macro FAR gain over raw ACC.
4. The per-user response is heterogeneous. U5 and U9 show some FAR reduction at weight 0.5, but U2/U4/U6/U7 remain weak. The binary-head reference also favors raw ACC in macro AUC and FAR.
5. Paired cross-modal similarity is not itself proof of preserved identity. Stronger evidence requires consistent separation on independent content and recordings. These results support a user-specific effect, not a general claim that the teacher constraint solves identity preservation.

## Limitations

- The target FRR calibration uses only 30 genuine development examples, so test rates are coarse and uncertain. No confidence intervals or multi-seed replication were run.
- Attacker identities are the six other users already seen as teacher negatives and generator-training users. No unseen-identity attack claim is made.
- Same-session recordings only; no cross-session result.
- Correct content grouping depends on the word-major order and tail-missing clarification; a transcript/recording manifest would be needed for independent verification.

## Artifacts

- Raw summary JSON: `outputs/mei_personal_teacher_word_major_exclude_u3_u8_20261027_results.json`.
- Server output directory: `/autodl-fs/data/mei_personal_teacher_word_major_exclude_u3_u8_20261027/`.
- Resumable training script: `work/train_mei_personal_teacher_word_grouped_exclude_u3_u8_20261027.py`.
