# Round04 preparation result

The four-coefficient f-only spline protocol and complete future fit/export/evaluation runner are prepared. No real-data head fit or outer-validation comparison was executed, so this preparation establishes no accuracy gain. Existing calibrated eta0 remains the incumbent.

## Evidence

| Check | Result |
|---|---|
| Fixed scalar basis | Two historical TRAIN knots; four coefficients; shared across every state; no E/state/context input |
| Synthetic least-squares recovery | Maximum coefficient error 1.51e-16 |
| Exact affine nesting | Maximum numerical error 2.22e-16 |
| In-sample calibration design | Rank 4; condition 10.3199979469; 1600 inputs at/above upper knot |
| Held-out calibration design | Rank 4; condition 9.6517799555; 1252 inputs at/above upper knot |
| Permitted calibration input audit | 240,710 identical valid rows; five input/mask/identity fields; no target/energy field decoded |
| Frozen base / one checkpoint | Original eta0 parameters/buffers unchanged; E/A bitwise equal on synthetic geometries |
| Scalar NumPy/Torch parity | Nonlinear fixture 1.3877787807814457e-17; affine fixture zero |
| One-file access | No external model/data/cache file opened during guarded load/forward |
| Independent adversarial checks | 18 deny-before-import/no-side-effect cases plus immutable recovery and repeated-evaluation refusal passed |

The design condition numbers are well below the fixed 1e8 limit. This establishes numerical support for the chosen basis, not reliable rare-tail estimates or predictive improvement. The 4546 true zero labels are inherited from the pinned Round03 receipt; targets were not reopened to count them.

## Provenance and staged checks

Source manifest SHA256: `79e5e6b6aa87f04808ddbf5e0f1b766a52226d71ef8edcb7b5bf1a0fc0e7467f` (51 source/dependency entries and four binary-input hash references).

The documents-only snapshot and first stub/preflight snapshot are retained separately. The design audit and synthetic molecular inference receipts record the exact original source versions they exercised. `PREFLIGHT_CONTINUITY.json` proves their scripts/wrapper/backbone and mathematical helper definitions remain unchanged; only durable output writing and the separately gated runner were added. Current synthetic recovery tests cover the extracted shared solver.

Earlier in-house synthetic/guard receipts bind an earlier execution-gate version. The final independent adversarial receipt binds the current gate `70078ab5506d7a1930687093a31d8ac1eb80dc16bd2d521b454a117a2a5e04bd` and production stages `04496de870fc1d5619b6168bdfffe8af28f369bd5b7dfbd9d5624bcd4669d2aa`. This distinction is retained rather than presenting old receipts as tests of later bytes.

The complete runner checks separate authorization and exact manifest/review/publication bindings before data/model-stage imports or writes. It retains durable per-arm STARTED markers and immutable private coefficient tensors, supports receipt/export recovery without another solve, and refuses completed or interrupted validation automatically. Tests use temporary hand-set tensors and replace real data/solver/export access with failures or spies. No production artifacts or execution authorization exist.

## Remaining limitations and authority

- The model receives only scalar native f. Identical f inputs get identical corrections regardless of the underlying molecule or transition.
- Linear tail extrapolation can still be steep or nonmonotone. Output clamping prevents negative values but does not bound large positive errors.
- The hypothesis follows previous validation diagnostics; historical outer-validation exposure and differing source training size/quality remain. No physical mechanism, independent-seed result or fresh holdout is established.
- Corrected f can be inconsistent with unchanged E/A. Both eventual exports still contain one original full eta0 plus one scalar map and need no auxiliary source or QC label at inference.
- Rank/parity/gate tests do not replace an empirical comparison. The future production path has been implemented and source-reviewed but deliberately not run on targets or validation values.

`RUNNER_HANDOFF.md` contains the exact blocked future commands and authorization schema. Preparation PASS, publication and root closeout do not authorize fitting. The saved heartbeat remains monitoring-only unless a later instruction authorizes the new execution scope. Any future candidate must exceed both its source-matched affine and the incumbent by at least 0.003 validation R² before a separate seed-confirmation allocation is considered; all state/tail tradeoffs remain visible.

Monitor must first download and verify lightweight records to the new D-drive archive, then inspect staged files and commit/push. All coefficient/model tensors, exact new learned slopes, raw labels, indices, predictions and caches remain server-only. This preparation contains no fitted head weights.
