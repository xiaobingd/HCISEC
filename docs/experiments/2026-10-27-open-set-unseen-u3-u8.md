> 2026-10-29词序审计更正：用户确认前100条是10个英文数字、后段是20个VR指令；Audio诊断支持前段10项/后段20项轮流重复。本报告原row//10分词不能证明真实跨词隔离，旧数值保留但跨词解释撤回。详见2026-10-29-mei-audio-order-audit.md。

# Open-set attacks from unseen users U3 and U8

Date: 2026-10-27

## Protocol

- Reused the completed per-target teachers and ACC-to-Audio generators for U1, U2, U4, U5, U6, U7, and U9. No model was retrained or selected on open-set results.
- U3 and U8 were absent from each target's teacher and generator training. Their ACC arrays were used only for final attack scoring.
- Kept the prior word-major assumption: ten consecutive repetitions per word; missing rows are at the tail; first 297 rows; crop to 80x80. Registration prompts 15–20 (60 target samples), calibration prompts 21–23 (30 target samples), final prompts 24–29 (57 samples/identity).
- Each target was attacked with 57 final-test samples from U3 and 57 from U8 (114 unknown attacks/target). The threshold was calibrated only with the target's 30 genuine calibration samples at 10% target FRR.
- Primary metric is the positive-only cosine prototype. A binary logistic reference trained on known users is included as a secondary diagnostic.

## Positive-only cosine prototype results

Cells are `TAR / open-set FAR / AUC`. TAR/FAR are measured on the 57 genuine and 114 unseen-identity queries. FAR is also listed by target below as the primary operating-point result.

| Target | ACC teacher feature | G, λ=0 | G, λ=0.25 | G, λ=0.5 |
|---|---:|---:|---:|---:|
| U1 | 80.7% / 0.0% / 1.000 | 91.2% / 100.0% / 0.604 | 96.5% / 1.8% / 0.997 | 98.2% / 0.0% / 0.999 |
| U2 | 80.7% / 91.2% / 0.415 | 89.5% / 89.5% / 0.512 | 93.0% / 93.9% / 0.477 | 80.7% / 89.5% / 0.429 |
| U4 | 98.2% / 64.9% / 0.765 | 91.2% / 86.0% / 0.514 | 89.5% / 83.3% / 0.504 | 96.5% / 91.2% / 0.507 |
| U5 | 94.7% / 100.0% / 0.497 | 94.7% / 92.1% / 0.467 | 96.5% / 99.1% / 0.546 | 93.0% / 96.5% / 0.507 |
| U6 | 98.2% / 98.2% / 0.487 | 91.2% / 98.2% / 0.484 | 94.7% / 100.0% / 0.508 | 98.2% / 96.5% / 0.523 |
| U7 | 91.2% / 100.0% / 0.440 | 82.5% / 91.2% / 0.475 | 82.5% / 85.1% / 0.470 | 78.9% / 79.8% / 0.463 |
| U9 | 96.5% / 90.4% / 0.447 | 93.0% / 95.6% / 0.480 | 82.5% / 82.5% / 0.559 | 89.5% / 76.3% / 0.582 |
| **Macro mean** | **91.5% / 77.8% / 0.579** | **90.5% / 93.2% / 0.505** | **90.7% / 77.9% / 0.580** | **90.7% / 75.7% / 0.573** |

## Open-set vs known-impostor FAR

Comparison of macro FAR for the same positive-only prototype route. The known-impostor column is from the prior final test containing the six known non-target users; open-set FAR uses only U3/U8.

| Route | Known-user FAR | U3/U8 open-set FAR |
|---|---:|---:|
| ACC teacher feature | 69.0% | 77.8% |
| G, λ=0 | 86.0% | 93.2% |
| G, λ=0.25 | 68.5% | 77.9% |
| G, λ=0.5 | 64.6% | 75.7% |

The binary logistic reference had macro open-set FAR 63.8% for ACC teacher features and 72.3% for G at λ=0.5; its open-set AUCs were 0.682 and 0.646 respectively. It did not outperform its known-user FAR (54.6% and 59.5%).

## Interpretation and limits

- Unknown identities are generally accepted more often than known impostors. The gap is 8.8 percentage points for ACC teacher features and 11.1 points for G at λ=0.5.
- U1 remains an outlier: U3/U8 open-set FAR is 0% for the ACC prototype and G λ=0.5. In contrast, U5's G λ=0.5 open-set FAR is 96.5%, compared with 64.6% on known impostors. U9 improves to 76.3% from 90.4% raw ACC, but remains far too high for reliable authentication.
- The personal identity constraint does not make the open-set route robust across users. The macro FAR change from λ=0.25 to λ=0.5 is modest; generated AUC is slightly below ACC teacher features.
- Only two unknown identities were tested, both from the same collection/session and under the stated word-order/tail-missing assumption. These measurements are an initial open-set check, not a population-level generalization estimate.

## Artifacts

- Local results: `outputs/mei_open_set_unseen_u3_u8_20261027_results.json`.
- Server output: `/autodl-fs/data/mei_open_set_unseen_u3_u8_20261027/open_set_results.json`.
- Evaluator: `work/eval_mei_open_set_unseen_u3_u8_20261027.py`.
