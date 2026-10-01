# Round06 decision: close the objective pair

Root accepts independent aggregate review `9a6242fccc372734d18e8605a5f437b4e49600165099659dba907024ea312a91` on 2026-10-01 Asia/Shanghai. Both arms completed60 epochs/112860 updates normally, with all66860 validation labels retained and no TEST evaluation.

| Objective | Selected epoch | Validation pooled raw-f R² | Fixed60 R² |
|---|---:|---:|---:|
| Original LE+Ls | 49 | 0.404797567 | 0.389052216 |
| LE+normalized raw-f | 18 | 0.422028167 | 0.393659486 |

The candidate gains .0172306 against the contemporaneous control but trails the retained v2 reference .447169401 by .0251412. It fails the frozen requirement of +.003 above both. Close this objective pair without extension, coefficient/LR tuning or confirmation seeds. Keep original Round05 MTO epoch45 as the validation-selected v2 reference, checkpoint `round05_scratch_preparation/runs/control/geometry_best.pt`, SHA256 `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e` on the server campaign.

The selected raw-f candidate worsens pooled MAE, energy error and true q90/q99 RMSE despite reduced false-bright SSE. Paired component-bootstrap intervals cross zero and condition on reused validation and selected checkpoints. They are not seed confirmation or fresh-generalization intervals. The repeated seed11 control differs materially from Round05 despite matching declared settings, initial tensors and order. Document execution-graph and concurrency differences; no cause is established. No scores are averaged and no labels are removed.

Authorize complete lightweight code/log/metric/analysis/review/decision/command records to be downloaded and hash-verified under D:\MTO\archives\, independently inspected as exact staged bytes, then non-force published to main/research with remote verification. Update README with both selected/fixed60 results, the failed dual gate, retained reference and uncertainty/TEST limitations. Preserve ancestry from c26a860b972a1c01267f8b766393772173515b92. Private models, optimizer tensors, raw data, predictions, identities and caches stay on the server.

After closeout publication, authorize proposal development only for the deferred shared PSD-congruence hypothesis, using original LE+Ls and a matched original/scalar/tensor comparison. Require a concrete phase/symmetry-safe mathematical contract, fixed normalization/gate inputs, exact controls and budget, and independent proposal review before any implementation or updates. The existing algebraic feasibility evidence does not establish a physical response operator or predictive benefit. Retain the v2 reference as an additional allocation threshold so a weak contemporaneous control cannot alone promote a candidate. No architecture, objective, coefficient or seed search is authorized by this proposal scope.

The single-model R².60 goal remains unmet. TEST remains sealed; no extension, extra seed, fit, inference or TEST scoring is authorized by this decision.
