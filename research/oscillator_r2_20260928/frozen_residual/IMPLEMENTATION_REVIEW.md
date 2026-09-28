# Independent frozen-residual implementation review

Date: 2026-09-29. PASS for the single authorized seed11,20epoch,37620update head-only experiment. No training or model inference was run by this reviewer. Runtime launch must still pass source, cache, completion and healthy-GPU/shared-lock gates. Root standing authorization supplies execution authority.

Reviewed handoff SHA256b48240e4e20a8841610fea906363ea48875ed0c4dbde16f9507ca701bfe4f89d. IMPLEMENTATION_REVIEW.json contains passed=true and the exact12-file source_hashes map expected by launch/train. Train SHA256b77bb5dff6c58d281de3b4683a673002d2beb8724480be8b2877cf16913d640d; refreshed preflight075632210e2a6d9d7225212178b23fe3a2788049560f98aad84e0f42e4c97d04.

## Findings and resolved correction

The original selection code subtracted an epsilon, while the protocol, final assertion and summarizer required the exact SSE minimum. Executor changed this to strict sse<best_sse, preserving earliest exact ties, and reran preflight. Validation diagnostics now include base/signed/delta quantiles and extrema. The FP32 note now states numerical agreement with qualified tolerances rather than attributing all differences uniquely to batch shape. Refreshed hashes were independently checked after these edits.

## Model, cache and loss

The exact sealed retained-residual initial head is copied; its final affine weight/bias are zero. Only4161 head parameters enter Adam. No source backbone/router/CG/E/tensor-base parameter enters the production optimizer: training operates on immutable cached h/base/E, with no source-model forward in its loop. Model feature generation uses eval mode. Inspected source has no BatchNorm/running batch statistics; dropout is configured0, while decoder LayerNorm operates within feature vectors. Thus train/eval switching does not introduce a learned batch-statistic difference in the frozen source or linear/SiLU head.

The train32 and native64 base paths correctly remain distinct. Training uses raw-f FP32 SSE/.002510981243894732. Native evaluation promotes E and matrix entries before trace/product and adds the FP32 head delta in FP64 before abs. Constant LE is reported alongside Lf; its omission from the differentiable head loss is algebraically harmless. Adam AMSGrad1e-4, default(.9,.999)/eps1e-8, decay0, clip5, molecule-batch64, FP32 and disabled AMP/TF32 match the protocol. No oracle E/f enters cached features.

Independently checked both server cache byte hashes, train/validation IDs and indices against source, their disjointness, raw validation truth against the frozen export, all-valid masks, scalar finiteness and exact pooled training variance. Only train/validation rows are retained/consumed. Standard NPZ member decoding can transiently materialize all raw-label rows before train indexing; this is storage behavior, not evidence of test labels affecting fitting/selection, and the implementation should not be described as avoiding every physical read of test-containing bytes. No test split is selected, evaluated or cached.

## Numerical qualification

The first batch2-vs-cache64 check failed abs tolerance at1.9744e-6; that observation remains preserved. The matched64 check passed combined abs1e-6/rel1e-5, despite nonzero h entries having max absolute differences about2.1e-6. That is mathematically compatible with the declared elementwise tolerance. Cache generation and replay are not claimed bitwise equal, even at the same batch shape. Gradient/loss/one-step agreement and epoch0 R2 difference6.08e-10 make the observed numerical differences immaterial to this screen's .01 R2 decision threshold.

Head evaluation batches1024 cached molecules for efficiency. Linear/SiLU operations have no cross-molecule dependence; changing this batch shape can still cause small FP32 numerical differences and is not exact arithmetic equivalence. No validation score selected precision or batch shape.

## Selection, diagnostics and recovery

All20 archived molecule-order hashes and1881updates/epoch (last batch35) passed preflight. Initial, best and final validation arrays are saved server-side, with native64 predictions and base/delta/signed/E/truth/ID columns. Epoch0 is eligible. Full checkpoint-fixed training-f metrics are computed each epoch separately from online loss, preventing the earlier train-f ambiguity. Baseline energy/base cache fields are unchanged throughout; final summary explicitly checks exact initial/final equality.

The checkpoint stores optimizer, RNGs, permutation, cursor, selection and history. Production resume logic was inspected; a separate copied two-step preflight reproduced parameters exactly after restoring optimizer/order/cursor/RNG. This is not a claim that every filesystem interruption was experimentally simulated. Mid-epoch interruption can undercount reported training wall time because that counter updates at epoch completion; optimizer/scientific state remains recoverable, and receipts/log timestamps retain timing evidence.

Launch and worker both require independent review and preflight hashes; worker takes shared GPU and own-run locks. Healthy/idle admission, ongoing ECC/foreign-process guards, finite loss/gradient checks and final cache hashes are present. GPU5 is explicitly refused by the current launcher/worker pending a separate guarded microcheck; this run can use clean1/4/6. GPUs0/2/3/7 are not admitted. This narrower implementation is safe and does not silently use guarded hardware.

Summary uses frozen eta0 and original connectivity grouping, preserves fixed selection, reports molecule/group uncertainty, and limits causal claims. Raw arrays/checkpoints are generated only remotely; later archive tools must enforce their exclusion. One-seed validation results cannot authorize test access, an extension, a hyperparameter grid or model promotion.