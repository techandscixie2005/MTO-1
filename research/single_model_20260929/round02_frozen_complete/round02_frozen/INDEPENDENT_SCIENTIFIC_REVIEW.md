# Independent round02 result review

**PASS for integrity and interpretation. Neither objective meets the predeclared promotion criterion.** No new recipe replaces the calibrated eta0 baseline, and no independent-seed fit follows from this round.

| One-model recipe | Selected epoch | Native validation R² | Gain over own epoch0 | Fixed historical calibration R² | Fixed20 native R² |
|---|---:|---:|---:|---:|---:|
| Frozen base + F, LE+Ls | 2 | .4054078302 | +.0001136998 | .4175387222 | .4031255423 |
| Frozen base + F, LE+Lf | 1 | .4053875246 | +.0000934045 | .4178651545 | .4021579798 |

The calibrated eta0 anchor remains approximately.41811924 validation R². The learned frozen h-only historical residual had a larger, still small native validation gain of about.001496; that older recipe has not been exported as one composite checkpoint. All comparisons remain exposed historical or reused-validation evidence, with no fresh holdout.

## Integrity checks

Fresh CPU review verified all64 frozen code/data hashes, both normal20-epoch completions,37,620updates each, exact0..20 history rows, native-SSE earliest-minimum selection, actual selected/last checkpoint hashes, optimizer/RNG state and20 paired orders. The orders equal the original full-model factorial. Original model tensors are bitwise unchanged in selected and final checkpoints; only4,016 adapter parameters changed. Frozen/nonpersistent buffer checks pass in the terminal geometry loader. Both original source checkpoints and cached arrays stay server-only.

The reviewer independently decoded only the selected validation label rows and recomputed native/fixed-calibrated pooled, ten-state, TRAINq90/q99-tail and energy metrics for all66,860 labels. The selected true/predicted-brightness bins were independently recomputed. Final-epoch geometry replay agrees with history within1e-7; all native bin counts/SSEs partition the whole score and every state. Plot, metadata inventory, analysis/replay sources, inputs and outputs are verified. No second GPU inference or test evaluation was performed by this review.

The terminal replay's first import-order error occurred before model inference; its corrected run and initial log are preserved. Earlier resume-device and synthetic-fixture checks were preflight issues, not fit failures. The two production fits completed normally.

## What the result supports

- The interventions were active: fixed-TRAIN mean relative F movement reaches6.05% for trace and4.97% for raw-f at epoch20. No gradient clipping occurred; maximum preclip norms were.516 and2.280, below5. Neither frozen representations nor an inert adapter explains away the result.
- Online TRAIN raw-f losses decline approximately1.46%/1.58% from epoch1 to20. These are batch averages at changing parameters, not end-epoch fixed-model TRAIN evaluation or independent generalization evidence.
- Selected energy MAE and true-bright tails improve slightly, but native MAE worsens. By epoch20 both native pooled R² values regress; q99 true-bright SSE still improves from about30.9031 to30.0882/30.0406. Eight states improve SSE, while S7 and S9 worsen. This differs from round01's S7/S8 concentration.
- The final true-dim/predicted-bright bin adds approximately1.1562/1.3433 SSE relative to the common zero-update anchor, while total SSE rises only about.3435/.4968. These fixed-threshold bins retain every label; their membership can change with predictions, so this is descriptive error localization, not proof of a causal set of molecules.
- Selected raw-f minus trace native R² is−.0000203056. Both gain intervals over the zero-update anchor include zero: trace[−.0004265,+.0006124], raw-f[−.0002474,+.0003788]. The matched native objective-comparison interval also includes zero. These molecule-bootstrap intervals are descriptive after reused-validation checkpoint selection.
- Fixed calibration ranks the two selected recipes differently and favors raw-f over trace by.0003264, but both remain below the unchanged calibrated baseline. This cannot support a new strongest-recipe claim or override native checkpoint selection.

## Decision scope and limitations

A later, separately authorized TRAIN-only identity-baseline pass adds descriptive moment evidence: all1,203,550 valid labels give native TRAIN R².64610356 and analytic affine slope/intercept1.00359075/.00009052, compared with the historical validation pair.85118302/.00372519. The reviewer verified source/provenance and pooled/per-state moment algebra without repeating inference. No new coefficients were applied or scored. This difference motivates investigating in-sample feature distribution, but is not causal proof that it explains the adapter failures or that an out-of-sample readout will transfer to the full source. See `TRAIN_CALIBRATION_REVIEW.json`.

Neither recipe reaches+.003 native R² over epoch0, and raw-f does not establish the additional+.003 advantage over its active trace control. No seed confirmation is warranted under the frozen rule. The result does not prove every frozen pre-CG adapter or raw-f objective will fail; it rules out promotion of these two fixed recipes. Do not expand the schedule or tune coefficients using these outcomes without a new explicit protocol.

Fixed coefficient1 changes objective weighting and initial total gradient scale; it was not fitted to match trace. Freezing removes parameter drift but does not remove the base model's in-sample TRAIN distribution or inherited errors. Available results alone do not establish residual-distribution mismatch or Adam-reset causality. Epoch33 Adam moments were not saved; the retained optimizer is from completed236 and cannot be substituted as a matched state.

The empirical adapter is O(3)-equivariant, but its scalar gates do not establish electronic sign covariance. No transition-density/NTO reconstruction is supported. No new test has been scored and no fresh holdout exists. The frozen root decision in ../current_state/ROUND02_DECISION.md closes this round without promotion and specifies a label-clean internal-TRAIN affine-transfer study as the next preparation. This result review itself authorizes no new fit; the separate exact protocol, independent review, archive-first publication and resource gates apply.

## Evidence and retained checkpoints

- `INDEPENDENT_TERMINAL_REVIEW.json` and `independent_terminal_review.py`: independent audit.
- `ROUND02_RESULTS.json`, `ANALYSIS_RECEIPT.json`, `TERMINAL_REPLAY.json`, `CHECKPOINT_INVENTORY.json`: complete metrics and provenance.
- Trace selected checkpoint: `/home/inspur/MTO-1/research/single_model_20260929/round02_frozen/runs/trace/best.pt`, SHA256 `af4d4bded1f0c21ad464bd4b3c6b4d04dabfbbdac8643b7471ad6d46bd525037`.
- Raw-f selected checkpoint: `/home/inspur/MTO-1/research/single_model_20260929/round02_frozen/runs/raw_f/best.pt`, SHA256 `12f72255109c6a67d4d3ef740b75bbc23df76ef862f94bf6a2f33d58c0467e32`.

Each selected file embeds the whole base and adapter, source config and stats. `predictor.load_predictor` loads that one checkpoint for geometry-only E/A inference; native f follows the original energy/trace relation. It does not reopen the original base checkpoint, cache or QC labels. The unchanged calibrated eta0 composite remains the strongest recorded eligible recipe; its deployment and calibration-consistency limitations are documented in the archived round01 report.
