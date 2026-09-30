# Round04 preparation decision

Heartbeat: 2026-09-29T23:50:29Z. This decision permits preparation only. It does not authorize a real-data fit, validation scoring, test access or a new production experiment.

## Accepted scientific question

Round03 passed its declared preparation threshold and was published as `4f9ae50647f957ac7d68ab1a20200a79fa527cf9`. The calibrated eta0 incumbent remains strongest. The next minimal question is whether a slightly larger shared scalar function improves on the corresponding affine maps, and whether that increment differs between the two frozen prediction sources.

Accept the proposed four-coefficient continuous piecewise-linear map with fixed knots at TRAIN q90 `0.0549` and q99 `0.2412`. Use the fixed original-TRAIN standard deviation `0.050109692115345626` as common input scaling. The basis is `[1, f/s, (f-k90)+/s, (f-k99)+/s]`, followed by the same final nonnegative clamp as the affine controls. No E, state index, context or hidden representations are added.

The proposed fitting method is one unconstrained FP64 SVD least-squares solve per source, on the unclamped output, with `rcond=1e-12`. Require full rank four and condition number at most `1e8`; stop without fallback on a failed contract, nonfinite value, deficient rank or excessive condition number. No optimizer, epoch selection, learned knots, ridge, monotonicity constraint, output cap or hyperparameter search is introduced. This tests added scalar flexibility/capacity, not an isolated physical mechanism. The hypothesis follows previous validation diagnostics and is not independent confirmation.

Each eventual candidate must gain at least `0.003` pooled validation raw-f R² over both its corresponding frozen affine control and the incumbent before allocation to a separately designed seed-confirmation study. A held-out-source advantage additionally requires at least `0.003` over the in-sample piecewise-linear map. Report every endpoint and all state/tail tradeoffs; no automatic promotion follows from a point estimate. Exact ties retain the incumbent. No predictions are averaged.

## Preparation scope in this heartbeat

After independent protocol review passes, science may implement the solver, scalar map, one-checkpoint wrapper and fail-closed execution gates in the new Round04 preparation directory. Preserve the preceding documents-only snapshot and every completed round.

Allowed checks:

- Synthetic known-coefficient recovery, affine nesting, continuity, linear extrapolation, clamp behavior, numerical parity and explicit failure cases.
- Read-only audit of the already frozen internal calibration cache: exact producer/schema/hash, row/mask/zero counts, finite values, and design rank/condition/support under the fixed basis. No coefficients may be solved from these real labels. Do not alter the protocol after seeing those diagnostics.
- CPU-only export/access and numerical parity on synthetic molecular geometries with hand-set coefficients and the unchanged full eta0 base. Do not reuse the Round03 validation replay as a calibration fixture or reopen it for these checks.
- Independent source and preflight review. Keep synthetic tests distinct from experimental accuracy evidence.

Set CPU-only visibility before Python imports. No GPU allocation, new source prediction generation, outer-validation value inspection, test access, real-data head fitting or production evaluation is allowed. Hash references from completed receipts may be read without decoding their validation arrays.

Fitted-head export parity on real calibration geometries is a post-fit gate in a future authorized execution. It must not be described as a prerequisite that silently performs the real fit during preparation. A future run must require an explicit execution decision bound to the final source manifest, independent review and verified preparation publication. An absent authorization must fail before real-data fitting or validation access. No executable preparation step may manufacture that approval.

## Preservation and publication

Both eventual deployed predictors use the same frozen full eta0 and one scalar map in one checkpoint. The temporary source and caches are absent from inference. E/A remain unchanged auxiliary outputs and need not reconstruct the corrected f.

Keep checkpoints, model/head tensors, fitted coefficient arrays, optimizer states, raw data, indices, prediction arrays and caches on the server. Lightweight records may contain hashes, shapes, fixed settings, code, logs, aggregate diagnostics, reviews and decisions. Synthetic hand-set test constants are identified as such.

After preparation and independent review finish, FIRST download and hash-verify its lightweight records to a new `D:/MTO/archives/single_model_20260929/round04_fonly_preparation` archive. THEN inspect exact staged files, commit, non-force push and verify ancestry from `4f9ae506` or its verified descendant. This publication does not authorize a fit. Update current operational handoffs after publication without changing sealed snapshots. Preserve the existing four-hour monitor and unrelated jobs.
