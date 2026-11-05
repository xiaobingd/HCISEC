> 2026-10-29内容标签更正：真实Audio审计与用户确认支持前100条数字按10项轮流重复、后段指令按20项轮流重复。原统一row%30或row//10标签不能证明同词或跨词隔离。原数值保留；相关内容隔离、词探针和内容双头的解释需重新审计。见docs/experiments/2026-10-29-content-generalization-innovation-assessment.md。

# Same-content identity check after CycleGAN and Diffusion

Server run: 2026-10-25. Source arrays: `train_core/data/acc.1.npy` through `acc.9.npy`.

## Strict same-content subset

Only U1, U2, U5, and U6 were used for the primary result because their complete 300-row arrays passed the earlier paired-Audio order audit. Under round-major ordering, prompt ID is `row % 30` and repetition round is `row // 30`.

| Representation | Same-prompt rank-1 | Same-prompt verification AUC | Descriptive EER | Prompt count |
|---|---:|---:|---:|---:|
| Raw ACC (CycleGAN rows) | 50.8% | 0.740 | 33.5% | 30 |
| CycleGAN → Audio → ACC | 31.3% | 0.548 | 46.9% | 30 |
| Raw ACC (Diffusion holdout) | 37.5% | 0.648 | 39.3% | 14 |
| Diffusion ACC → Audio → ACC | 37.5% | 0.651 | 41.4% | 14 |

Chance is 25% rank-1 and 0.50 AUC for four users. For each method, PCA and scaling were fit on enrollment data only; queries were compared to each user's centroid for the same prompt. EER is an uncalibrated descriptive estimate.

## Splits and models

- **CycleGAN:** U1 original single-user model trained on chronological rows 0–159. Enrollment rows 180–239 and query rows 240–299 were outside that conversion training pool. U2/U5/U6 were unseen by this U1-specific converter. Enrollment and query each contain two repeated recordings per prompt.
- **Diffusion:** Used the existing nine-user EMA checkpoint and recreated the seed-42 70/30 split. Authentication uses the 30% held-out portion (90 records per user) that was not used to fit the conversion model. We kept the 14 prompt IDs for which every evaluated user had at least two held-out recordings; remaining held-out repeats enroll the same-prompt centroid and the latest repeat is queried.
- **Input processing:** the stored arrays were used uniformly, cropped to 80×80 only because both converters require it. No user-specific filters or normalization were added.

## Interpretation

CycleGAN reduced rank-1 from 50.8% to 31.3% and same-prompt AUC from 0.740 to 0.548; its transformed ACC is close to chance. It did not preserve the raw ACC identity separation in this check. The U1-specific converter's U1 rank-1 fell from 88.3% to 51.7%, while the three unseen users were near chance after conversion. That pattern is consistent with information compression or target-user bias, but does not by itself identify the mechanism.

Diffusion left rank-1 unchanged at 37.5%; AUC increased only 0.002 (0.648 to 0.651), while descriptive EER became slightly worse. This is no meaningful overall gain. Per-user changes were mixed: U2 improved, U5 declined, U1 rose slightly, and U6 was unchanged.

This is **not a nine-user matched-content result**. Five arrays are short and their prompt alignment was not verified; no exact row-ID/transcript labels were present. Also, CycleGAN and Diffusion use different held-out recordings and 30 vs 14 common prompts, so the two pipelines are not a paired per-record comparison. A full nine-user claim needs corrected per-record prompt IDs for those five users.

Raw metrics are reported separately for each pipeline's evaluation set, so use the within-pipeline raw-vs-converted comparison rather than comparing the two raw rows directly.

## Reproduction

Remote script: `/root/autodl-tmp/train_core/exp_same_content_auth.py`  
Remote log: `/root/autodl-tmp/train_core/exp_same_content_auth.log`  
Remote JSON: `/root/autodl-tmp/train_core/exp_same_content_auth_20261025/results.json`
