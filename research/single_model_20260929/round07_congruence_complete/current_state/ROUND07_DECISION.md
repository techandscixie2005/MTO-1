# Round07 decision: close the shared-transform study

Root accepts independent scientific review `e5eeefb45d78f903d13b7f90ae04b4b00f5992ac26fb18ff3337f94939256bf9` on 2026-10-02 Asia/Shanghai. All three arms completed60 epochs/112860 updates normally. Every66860 validation label remained included; TEST stayed sealed.

| Model | Selected epoch | Selected raw-f R² | Fixed60 R² (rounded) |
|---|---:|---:|---:|
| Original | 52 | .4091040813 | .3845332 |
| Scalar | 49 | .4293390126 | .3937752 |
| Tensor | 34 | .4187551766 | .3277767 |

The tensor gains .0096511 over contemporaneous original but trails scalar by .0105838 and the retained .44716940136585204 reference by .0284142. It fails its frozen triple+.003 gate. Close this study without extension, coefficient/bound tuning or confirmation seeds. The scalar is the strongest model in this round but does not replace the retained reference. Never average the controls or predictions.

Retain Round05 original MTO epoch45, private campaign checkpoint `round05_scratch_preparation/runs/control/geometry_best.pt`, SHA256 `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`. Its .447169401 validation result lacks independent-seed or TEST confirmation. The .60 goal remains unmet.

The tensor lowers false-bright SSE but worsens MAE, energy and q99 errors against both controls. Gates were active and anisotropic context appeared, without identifying physical transition directions or explaining a causal mechanism. Selected paired component-bootstrap intervals span zero; fixed60 tensor intervals lie below both controls. These are conditional reused-validation summaries, not seed uncertainty or fresh confirmation. Preserve all state/tail metrics and negative evidence, including repeated-control variability.

Authorize complete lightweight results/code/logs/commands/reviews/decisions to be downloaded and hash-verified under D:\MTO\archives\ FIRST, then exact independently reviewed staged publication and non-force main/research push with remote byte verification. Preserve ancestry from271d61f6dc9e21009c31d10467fe704e44b974ad. Delegated history review suffices for accurate README updates. Final report/manifest bindings must complete before packaging. Keep all checkpoints, learned coefficients, optimizers, datasets, private identities/splits/predictions and caches server-only.

After publication, authorize a read-only bottleneck audit and next-proposal development. Examine original MTO representation/readout information flow, train/validation trajectories, known repeated-control variability, and prior capacity/depth/learning-rate schedule trials from existing source and aggregate records. Compare scientifically justified alternatives against this history. Prefer one falsifiable question that changes a substantive limitation over another small readout correction. Do not infer bad labels or a QC mechanism from large errors; retain every valid label. Consult primary literature when needed. No new target decoding, model inference, implementation, synthetic optimizer updates or fits are included in this audit authority.

Any next pilot needs its own exact independently reviewed protocol and root decision. No test scoring, new split, architecture sweep, extra seeds or production continuation is authorized here.
