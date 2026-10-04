# Independent historical schedule and semantics audit

Scope: existing source, settings and published aggregate records only. No model construction, target/prediction/checkpoint-array decoding, numerical experiment, inference or fit. Round09 is closed at6e23b9d9c7eeb479b478ebb69ec061024de62455. This note supports review of a future proposal; it does not authorize implementation.

## Bounded historical finding

The audited record set contains initial learning-rate and schedule experiments. No true matched comparison of original PSD MTO, LE+Ls, zero decay, fresh current-v2 initialization, equal60epochs and a single midpoint LR drop versus fixed.001 was found in those records. This is not a repository-wide proof of absence.

| Historical evidence | Exact schedule evidence | Why it does not answer the proposed contrast |
|---|---|---|
| Full-EA campaign | Initial LR.001/.0003/.003; ReduceLROnPlateau factor.5, patience50, relative threshold1e-4, minimum1e-6 | LE+full-tensor LA; joint-loss selection, adaptive stopping and unequal248/226/227 completed epochs. Best joint losses .523346/.537891/.539912 are not raw-f R². |
| Original eta0 and seed23/37 | Legacy plateau; eta0 completed236, selected33; replication budget100, selected53/44 | Old split and different realized budgets; no matched fixed-LR schedule arm. Seed23/37 native R² .368709/.394455 are historical results, not schedule effects. |
| Direct-f scratch | LR.001 epochs1–60, .0003 epochs61–85, .0001 epochs86–100 | Direct-f readout, LE+Lf, different initialization and100epoch budget. Its .373493 result does not isolate a schedule or current original-PSD capacity. |
| Current v2 original controls Round05–09 | Fixed.001,60epochs, zero decay | Selected R² .447169/.404798/.409104/.417636/.415284 demonstrates realized control variation, not independent-seed confirmation or its cause. |

The source-backed historical ledger contains23 configurations and92 exact input hashes. It already distinguishes completed full-EA trials from a failed seed23 attempt. Its model/objective distinctions must remain intact. Later weak decorrelation, raw-f, congruence, transport and fixed all-parameter coupled-L2 results do not test the proposed time-dependent LR recipe. Late validation decline alone does not establish optimization failure, overfitting or the remedy.

## Smallest identifiable proposal

Science's proposed candidate uses.001 for completed-epoch labels1–30 and.0003 for31–60. The contemporary control remains.001 for all60. Both use originalPSD/LE+Ls/WD0, common freshseed11 and order11, identical data/statistics/masks/135-tensor optimizer membership, batch64/clip5/AMSGrad and112,860 updates. The one boundary and.0003 level are unvalidated choices fixed before the future run; they were considered after historical/reused-validation evidence and must not be called an untouched prospective holdout design.

This contrasts the complete reduced-late-LR recipe against fixedLR. Equal updates do not equal accumulated LR: the epoch-LR sums are.039 versus.060. A gain would not identify an optimal drop boundary, prove that schedule shape matters independently of lower average LR, or establish an exploration/refinement mechanism. No third arm or sweep is requested here.

## Exact indexing and recovery requirements

- Epoch0 is the common eligible initialization evaluation, with initial optimizerLR.001 and zero updates. It is not one of the60 training epochs.
- Each training epoch has1,881 updates. Candidate updates1–56,430 use.001;56,431–112,860 use.0003.
- Assign the absolute groupLR from arm and the next one-based training epoch immediately before that epoch's first update. Do not repeatedly multiply the current value or depend on a scheduler counter/validation metric.
- Checkpoint after epoch30 stores the LR actually used,.001. Restore model, full Adam/AMSGrad state, RNG/order and history; then assign.0003 before epoch31. No moment, bias-correction step, order-generator or RNG reset/rescaling.
- An interruption during31 replays from committed30 using the same transition. Resume after31 retains the completed snapshot and uses.0003 for32. A completed60 run must refuse restart.
- Future metadata should freeze all60 expected per-arm LRs and their provenance. Validate saved LR against the completed epoch before setting the next LR. Update inherited constant-LR assertions, first-epoch/terminal checks and history consistently; never bypass roster/optimizer-option checks.

## Evaluation boundary

Keep earliest strict minimum native pooled validation SSE over0–60, all66,860 labels including zeros, and report fixed60 separately. Candidate seed-allocation screen requires at least+.003 over both its contemporaneous selected control and retainedRound05 .44716940136585204. State/tail/energy regressions remain relevant; no automatic seed job, extension, TEST release or prediction averaging. Any later confirmation must use independent fresh seeds with within-seed matched controls and individual one-checkpoint results. Conditional component bootstrap does not include checkpoint-selection or training-seed uncertainty.

Existing reference records and exact bytes are in REFERENCE_HASHES.json. The concrete protocol/settings must still receive their own final independent binding before root decides preparation.
