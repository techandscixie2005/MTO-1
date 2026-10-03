# Round05 preparation decision: matched training from initialization

Root selects the four-arm fresh-initialization MTO factorial as the next concrete pilot preparation. The user has authorized continuing research; no additional user approval is needed for this in-scope decision. This document authorizes implementation and bounded technical preflight, not the production training runs.

Accepted proposal: `round05_scratch_preparation/PROTOCOL_PROPOSAL.md`, SHA256 `2089e93b5fc886ce3eaa7f97cacce647d4e72492a5da74148d347e1826a5a1df`. Preserve that proposal snapshot when writing the final protocol. The QC specialist may develop a separate response/density proposal; unresolved speculative architectures do not block the required fresh baseline.

## Why this experiment

The prior F/decorrelation factorial continued a validation-selected epoch33 backbone at low learning rate; the frozen-F experiments could not alter its molecular representation. Neither tested joint learning from random initialization at the original scratch learning rate. This is the new rationale, not a claim that the previous negative results disappear. Round04's fixed scalar maps have now also failed their matched controls and are closed.

The four arms are original MTO, right-operand F, raw-state decorrelation, and both. Treat these as empirical architectural/regularization choices. Raw M is not an identified wavefunction, the penalty is not established electronic orthogonality, and F is not an identified physical dipole operator. Ordinary geometry-only polar-vector heads and naive transition-density heads are not approved substitutes: phase-insensitive loss alone does not resolve their point-group/electronic-character obstruction.

## Fixed scientific scope

Use only the separately versioned QM9S v2 partition after independent source, synthetic, identity and final disjointness review. Preserve the fixed seed20260930/90-5-5 whole-component assignment protocol and every valid row. It is a new partition of historically used data, not an independent external holdout.

All four arms start with identical freshly initialized base tensors, seed11 and independently fixed data-order seed11; no old checkpoint, learned calibration, residuals or full-corpus target statistics. Recompute normalization, energy means and bright thresholds from the new TRAIN only. Keep the proposed original LE+Ls objective, eta0 architecture, AMSGrad LR0.001, batch64, weight decay0, clipping5, FP32, AMP/TF32 off, two CPU threads, and fixed common learning rate. The right adapter uses the existing identity initialization; the raw-state penalty uses the existing definition and lambda0.001. Do not add objectives, schedulers, augmentation, dropout or calibration to this factorial.

Prepare a fixed60-epoch pilot in every arm, selecting checkpoints by full-validation pooled raw-f SSE, including initialization and retaining the earliest exact tie. Report both selected-best and fixed60 comparisons, all states and TRAIN-defined bright-tail/false-bright errors. A best-selected interaction is descriptive pipeline evidence, not mechanistic synergy. The contemporaneous fresh control is the comparator; the old-split calibrated score is historical context only.

Keep the proposal's allocation rule: if no noncontrol arm gains at least0.003 validation R² over the matched control, close this bounded pilot without selective extension. If the gate passes, only a separately reviewed decision after round analysis/archive/publication may continue all four unchanged to100. Paired independent-seed confirmation remains a subsequent stage; no prediction averaging. The0.60 objective does not justify test-driven tuning, changing the benchmark or relaxing these rules.

## Authorized preparation and access limits

Science owns implementation. History independently reviews it and owns the split audit. Synthetic checks and a bounded TRAIN-only next-update/resume fixture are allowed after the split is frozen and resources admitted. Record exact fixture, update count, resource identity and temporary artifacts. Do not turn a preflight into a pilot or inspect validation performance to adjust settings.

New TEST target values and predictions must remain inaccessible to training, statistics and selection. A full-corpus numeric target load followed by slicing cannot be described as sealed-test access. Inspect actual container formats and use partition-isolated or streamed selected-row decoding. Raw-byte hashing or transport is different from numeric decoding; document the boundary and prove the fields/indices accessed. If the boundary cannot be maintained, report it before proceeding. Validation is only for the declared evaluator; technical parity/resume fixtures use TRAIN, not validation/test.

Verify shared initialization and data order, unchanged mathematics, O(3)/permutation behavior, masks, finite/live gradients, exact resumable next-update state and one-checkpoint geometry-only export. Keep complete optimizer/RNG/order/split/source provenance in private resumable checkpoints. Logs and published records contain only code/settings, hashes and aggregate diagnostics.

## Before production execution

Require the final frozen split and protocol, independent source/preflight PASS, resource budget/admission, exact source closure, D-first preparation archive and inspected non-force publication before root issues the bound execution decision. No production fit is currently launched or authorized by this document. Preferred GPUs1/2/4/6 require fresh occupancy/health/UUID checks and per-device shared locks; GPUs3/7 remain excluded and unrelated jobs untouched. Prebind physical UUIDs before interpreter imports. Do not infer failure from an interrupted agent or missing observation.

The proposed60-epoch budget is approximately10–12 GPU-hours total, subject to measured technical overhead; this estimate is not a timeout or a reason to kill a healthy run. Keep the existing four-hour monitor, registered identities and resumable state. Every completed/failed round follows download-and-hash-verification on D before inspected commit/push. Publication ancestry now starts from verified `8497e0ba10efc3226197c894f46507acdad8ec54` or its verified descendant on main/research.
