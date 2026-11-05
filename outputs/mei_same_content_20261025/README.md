> 2026-10-29内容标签更正：真实Audio审计与用户确认支持前100条数字按10项轮流重复、后段指令按20项轮流重复。原统一row%30或row//10标签不能证明同词或跨词隔离。原数值保留；相关内容隔离、词探针和内容双头的解释需重新审计。见docs/experiments/2026-10-29-content-generalization-innovation-assessment.md。

# Same-content ACC identity check (2026-10-25)

## Result

Using four users with complete 300-row sequences (U1, U2, U5, U6), the paired-Audio order audit supports **round-major** ordering: 30 prompts are cycled through over repeated rounds. We used nine rounds (270 examples per user) and held out three repetition rounds at a time. Each prompt therefore appears in both enrollment and test, while the held-out recording itself is unseen.

| Input | 4-way identity top-1 | Same-prompt verification AUC | Descriptive EER |
|---|---:|---:|---:|
| ACC | 65.0% (chance 25%) | 0.773 | 30.9% |
| Paired Audio control | 96.2% (chance 25%) | 0.845 | 23.3% |

Across the 30 prompts, ACC AUC averaged 0.771 (SD 0.056). Mean same-user/same-prompt similarity was 0.214; other-user/same-prompt similarity was -0.059. This indicates measurable user information remains when prompt content is held fixed for this subset. Its discrimination is moderate and below the paired Audio control.

![Same-prompt ACC and Audio results](/outputs/mei_same_content_20261025/same_prompt_results.png)

## Order and cohort checks

The arrays are preprocessed 80×81 spectrograms and contain no per-row transcript IDs. We compared two possible row layouts. Round-major order yielded above-chance cross-user prompt retrieval and audio prompt recognition; word-major order was near chance. This supports, but does not prove from transcript labels, the round-major interpretation.

Five arrays have only 297 or 299 rows. Audio prompt transfer was at/below chance for at least one of these users, so their row positions cannot safely be treated as exact same-prompt labels. The nine-user result is retained in `summary.json` as exploratory only. The table above is restricted to the four complete sequences.

## Protocol

- Source: `E:\Edge下载\train_core\data\acc.1.npy`–`acc.9.npy` and paired `audio.1.npy`–`audio.9.npy`.
- Same stored preprocessing for each participant; no per-user filters or scaling. Time axis downsampled by stride 4 for compact representation.
- For each of 3 folds, six repetition rounds per prompt/user form enrollment templates; three rounds form test queries. Fold test slots are shared across users.
- Fold-fitted shared `StandardScaler` and randomized PCA (64 dimensions), followed by logistic regression for top-1 identity. Same-prompt verification compares each query to each user's centroid for that exact prompt.
- EER is descriptive from pooled out-of-fold scores; no calibrated deployment threshold is claimed.

This is within-cohort, same-session, closed-set evidence. It does not establish open-set security, cross-day reliability, physiological causality, or an added benefit from CycleGAN.

Reproduction: `work/mei_same_content_auth_20261025.py` and `work/mei_same_content_strict_subset_20261025.py`.
