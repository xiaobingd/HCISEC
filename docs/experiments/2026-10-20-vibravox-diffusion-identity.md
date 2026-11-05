# VibraVox cross-modal identity pilot and owner-specific follow-up, 2026-10-20

This is the publishable aggregate report. Raw recordings, paired manifests, utterance-level scores, model weights, and local experiment scripts remain in the private workspace and server; they are not attached to this report.

Official source: https://huggingface.co/datasets/Cnam-LMSSC/vibravox , `speech_clean` train Parquet. This pilot uses the simultaneously recorded **forehead single-axis accelerometer** and **headset microphone**, not another bone-conduction channel. Four female speakers were selected by preview availability before scoring. The first 24 utterances for each selected speaker across the first 30 Parquet shards were retained. `indexed_manifest_96.json` records sensor paths, sentence IDs and SHA-256 hashes. The 96 paired audio recordings are kept locally and should not be published to the project GitHub repository.

## Paired signal check

Both sensors were resampled to 8 kHz, filtered 70–1600 Hz, Hilbert-enveloped at 100 Hz and trimmed by 300 ms on both ends. Maximizing envelope correlation over ACC lags 0–200 ms gave **median 0.872** for matched utterances. A 200-repeat within-speaker wrong-utterance permutation gave mean null median **0.251** (empirical p=0.00498). This confirms shared temporal activity in this subset; it does not establish identity information. See `pair_activity.json` and `work/vibravox_pair_activity_20261020.py`.

## Raw identity sanity check

This is **not a Diffusion result**. Both signals used the same deterministic 16 kHz STFT frontend, 24 fixed frequency groups, per-utterance log spectral-shape summaries and temporal activity statistics; absolute energy was excluded. For each of 5 seeds and each of 4 target rotations, 24 utterances per speaker were split into 12 head TRAIN, 4 VAL, 4 calibration, 4 FINAL. One other speaker was a TRAIN negative; the remaining two were unseen attackers. Logistic regression used a fixed C=0.1, standardized on head TRAIN only. A threshold calibrated from the genuine calibration utterances to accept at least 3/4 was locked before FINAL. No segments from the same utterance crossed roles.

Across 20 correlated fits, ACC: final AUC **0.806**, TAR **0.638**, unseen FAR **0.231**. Microphone: AUC **0.853**, TAR **0.725**, unseen FAR **0.225**. The owner-level ACC AUC values ranged **0.556–1.000**. Only 4 owner FINAL utterances were available in each fit; the 20 fits reuse the same 96 recordings. These are descriptive, not reliable performance estimates. No sentence ID is shared across the four selected speakers, so content effects remain unresolved. See `raw_identity_pilot.json`, `vibravox_pilot_identity.png`, and `work/vibravox_raw_pilot_20261020.py`.

## Frozen Diffusion transfer

After SSH access recovered, the **frozen ESMB 0001 Diffusion V3 EMA600** ACC→Audio network was applied to VibraVox forehead ACC. The model had never seen VibraVox. Each utterance supplied two nonoverlapping 0.8 s ACC windows; both used the original ESMB 16 kHz/STFT512/hop160/loglinear80×80 frontend and **fixed original ESMB ACC scale**. Across 192 windows, only 0.037% / 0.322% of pixels were clipped at the lower/upper limits. No VibraVox microphone entered the auth input. At t500, three deterministic noise draws were averaged and the window features were averaged per utterance. `z_cond = GAP(H(x500, ACC, 500) − H(x500, 0, 500))`; the same noisy image appears in both terms. A randomly initialized U-Net of the same architecture is a control.

The head was the **same fixed logistic-regression protocol** as the raw ACC pilot, with identical 5 seeds, target rotations and utterance-level role splits. For an input-matched raw comparison, we summarized the **same two ACC windows** into 160 dimensions (mean/std across time for each of 80 frequency rows). Mean results over 20 correlated fits:

| Feature | Final AUC | TAR | Unseen FAR |
|---|---:|---:|---:|
| Raw ACC, same windows | 0.827 | 73.8% | 24.4% |
| Frozen trained `z_cond` | **0.863** | **78.8%** | 26.3% |
| Full trained t500 bottleneck | 0.820 | 77.5% | 28.1% |
| Trained image-only (ACC condition zero) | 0.745 | 80.0% | 36.3% |
| Random U-Net condition increment | 0.723 | 76.3% | 38.1% |

`z_cond` improved AUC for **two of four** target speakers and worsened it for the other two. Their average per-speaker AUCs (raw→`z_cond`) were S1 **0.994→0.881**, S2 **0.756→0.956**, S3 **0.962→0.862**, S4 **0.594→0.750**. Thresholded FAR also worsened for S1/S3 and improved for S2/S4. This is evidence of a **transferable but uneven input-dependent response**, not consistent superiority over raw ACC, proof of physiological identity, or proof that iterative denoising adds identity. The response is measured from the trained conditional encoder at a single t500 call. Speaker content is not matched across the four selected speakers, there is no second recording session, and each fit has only four final target utterances. Repeated split means cannot be treated as 20 independent trials.

See `frozen_transfer_auth.json`, `frozen_diffusion_transfer.png`, `extract_protocol.json`, and `work/vibravox_diffusion_transfer_extract_20261020.py` / `work/vibravox_diffusion_transfer_head_20261020.py`. The remote extraction directory is `/autodl-fs/data/vibravox_diffusion_transfer_20261020`.

The frozen-model transfer is a stress test, not a final authentication result. The following owner-specific experiment uses separate generator and authentication utterances; same-text/cross-session checks remain necessary to distinguish identity from content and recording conditions.

## Owner-specific Vibravox model (follow-up)

The metadata index was extended to the first 90/201 official Parquet shards, yielding 77–87 utterances for the four selected speakers. Because `2BuQxDAGnW` had the poorest raw-ACC result in the first pilot, it was selected **after that result** as the exploratory owner. The final expanded set contains 80 owner utterances, 60 seen-negative utterances (`95rDQfTOk7`), and 24 utterances each for two unseen attackers. This target choice introduces selection bias; do not read the result as independent confirmation.

The model used the original paired Diffusion V3 training code, 200 epochs, two directions and **two centrally placed nonoverlapping 0.8 s windows per utterance**. It saw only 45 owner utterances for generator TRAIN and 5 distinct owner utterances for generator VAL/checkpoint selection. One ACC/audio preprocessing method was applied to everyone. The two fixed normalization scales were fitted from owner GAN TRAIN only and applied unchanged to owner and attackers. The best checkpoint was epoch 200. GAN VAL cycle reconstruction MSE fell from **0.424 at epoch 50** to **0.086 at epoch 200**; SSIM rose from **0.110 to 0.274**. This cycle metric alone does not measure whether the generated Audio matches the real paired Audio. See `owner80_protocol.json`, `owner80_train.log` and `owner80_val_epoch200.png`.

For a direct paired check on six completely unseen owner FINAL utterances, generated-vs-real Audio **dynamic** spectrogram correlation averaged **0.446** after removing each frequency row's time-constant color. Pairing with another owner utterance gave **−0.011**. This shows speech activity/content timing is partly preserved. But raw ACC vs real paired Audio, processed the same way, scored **0.551**. Generation has not improved the measured cross-modal dynamics. `owner80_final_pairs.png` shows that generated Audio follows some timing and low-frequency patterns while missing much of the real Audio texture and harmonics. This structural correlation is not intelligibility or identity verification. See `generation_eval.json`.

The generator was then frozen. Five *correlated* authentication-head splits were drawn solely from the 30 owner utterances outside GAN TRAIN/VAL and the 60 negative utterances. Per split: 16 owner/24 negative head TRAIN, 4/6 head VAL, 4/6 calibration, 6/24 FINAL; both unseen attackers' 24 utterances each remained FINAL only. The fixed logistic head uses identical settings and role splits for all features, with scalers fit on head TRAIN and threshold from the four owner calibration utterances. Each final owner count is six, so thresholded rates are noisy.

| ACC-only auth feature | FINAL unseen AUC | TAR | Unseen FAR |
|---|---:|---:|---:|
| Raw ACC, same windows | **0.995** | 76.7% | **2.1%** |
| Owner Diffusion `z_cond` | 0.927 | 86.7% | 13.3% |
| Owner Diffusion full t500 bottleneck | 0.951 | 80.0% | 7.5% |
| Random U-Net `z_cond` | 0.720 | 80.0% | 40.4% |
| Raw ACC + `z_cond` | 0.996 | **86.7%** | 1.7% |
| Raw ACC + full bottleneck | 0.999 | 80.0% | 0.4% |

The owner-trained conditional features are meaningfully better than the random network, yet **the raw ACC is the stronger stand-alone identity feature**. Fusion gives small apparent gains on reused test utterances; both fusion variants were inspected after earlier FINAL results, so those gains need a fresh test. The one original split was more variable (`z_cond` TAR 2/6 at FAR 0/48), showing why a single six-utterance verdict is unstable. Results are in `owner80_auth.json`, `owner80_auth_multisplit.json` and `owner80_auth_comparison.png`.

The original 24-utterance pilot had suggested this owner's raw ACC was weak (mean AUC 0.594), but in the larger, differently partitioned 80-utterance trial raw ACC reached 0.995. The pilot ranking was therefore **not a stable person-level property** and must not be used as evidence that this person's ACC lacks identity information.

The current evidence supports paired ACC/Audio timing and a learned input-dependent feature, but it does **not** show that Diffusion adds speaker identity beyond original ACC or that its generated Audio has sufficient fidelity. Every speaker has one recording context and sentence IDs are unmatched across speakers. This project still needs held-out recording sessions and content-balanced genuine/impostor tests to claim a transferable biometric.

## Content-overlap audit (2026-10-25)

I checked the transcript fields across all **20,981** `speech_clean/train` rows. The full train split contains **20,964 distinct normalized texts**; only **8 text strings** are repeated across speakers, and each is shared by exactly two speakers. None of the four speakers in the original pilot share a text with one another, even when their full train recordings are considered (694 utterances; 694 distinct texts). There are no common utterances across three or more users, so VibraVox cannot support a sufficiently sized, exact same-text multi-user train/calibration/final protocol for these speakers.

As a limited negative control, a text-only character n-gram TF-IDF classifier on the same owner80 head split gave unseen AUC **0.403–0.431** and unseen FAR **75.8–77.1%** at a threshold calibrated to accept about three of four owner-calibration utterances. Surface text alone did not classify this owner, but this does **not** prove the ACC/Diffusion result is content-independent: the text samples are not repeated across users, each final owner fold has only six utterances, and phonetic content can interact with ACC features even when a simple text classifier performs poorly.

The experimental conclusion must therefore be restricted to **identity discrimination under unmatched sentence content**. To verify content independence, collect a controlled set where all participants read the same prompts. Keep prompt sets disjoint by role: owner-only prompts for generator training; matched owner/seen-negative prompts for head training and calibration; and new prompts recorded from the owner plus unseen attackers for FINAL. Within FINAL, each genuine and attacker comparison must use the same exact prompt, while no FINAL prompt appears in earlier roles. Repeat across recording sessions. The VibraVox corpus's current train split does not provide enough repeated cross-speaker prompts for that protocol.

Audit artifacts: `global_text_overlap.json`, `shared_text_scan.json`, and `text_only_auth.json`; analysis scripts `work/vibravox_global_text_overlap_20261025.py`, `work/vibravox_scan_shared_text_fast_20261025.py`, and `work/vibravox_text_only_auth_20261025.py`.

