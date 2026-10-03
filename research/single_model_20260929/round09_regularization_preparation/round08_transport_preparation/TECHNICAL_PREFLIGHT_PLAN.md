# Round08 bounded technical preflight: fixed before execution

Root authority ROUND08_PREPARATION_DECISION.md SHAa227764a9540cfa4f89dd9ed8ea48aa923d65bb4152f2e90ff734073601379a1. Independent exact source/tolerance review precedes CPU or GPU numerical execution. No validation/TEST fixture or production fit. All old sealed evidence remains unchanged.

## CPU synthetic scope and criteria

Use synthetic_checks.py with empty CUDA visibility before imports, OMP/MKL2. Pure branch fixtures cover actual irrep-major128×(3+5+7) layout through an independent literal-slice/edge-loop reference, operational i/index0→j/index1, degree mean, isolated receiver/empty edges, blocks2/3 only and exactly768 active coefficients. Source-only T perturbation at fixed edge scalar and receiverT must change the neighbor receiver residual but leave the local receiver residual unchanged. This demonstrates implemented feature dependence, not a collision of real molecular representations.

At theta0, added residual is exactly zero and theta gradient equals contraction of H with a fixed probe. New-path T/edge-scalar gradients are exactly zero there, then finite/nonzero on a hand-set nonzero-theta fixture. Particular symmetric/zero channels can still be inactive. Positive and negative hand-set coefficients are checked. O(3), reflections and atom/edge permutations use invariant scalar gates shared over magnetic components; no learned frame or electronic phase interpretation.

A synthetic wrapped Interaction_Block fixture counts exactly one Attention call, independently computes original message/update and the added residual from pre-block T, and distinguishes the incorrect T+ut insertion. Original and both identity-initialized wrappers preserve the original function. Full native-dtype models use only fixed synthetic CH4/H2O geometries for identity, reflection/permutation/translation and their own nonzero-mode one-checkpoint export checks. No historical/trained model or dataset is opened; only explicitly allowed temporary synthetic checkpoint files. Fresh base/inherited/schema hashes must match across arms; CPU constructor RNG isolation exact. Do not convert whole backbone buffers through FP64→FP32.

Mask checks retain inherited LE+Ls: valid labels selected before squared errors; invalid synthetic target NaNs do not affect loss/gradients. No target or mask is a transport input. Initialization/base/order/statistics are unchanged; no trainable congruence/readout or right-F.

## Fixed numerical bounds

Standalone module FP64: atol1e-10/rtol1e-8. Module/full-model FP32 and GPU E/A/model+Adam replay: atol2e-6/rtol1e-5. Loss replay: atol1e-6/rtol1e-5. Same-incoming zero-theta algebra, initialization tensors, CPU constructor RNG, restored RNG/order/integer state: exact. Full CUDA forward/export is numerical parity, not guaranteed bitwise identity. Record actual maximum errors and bitwise flags. Native-f errors are reported from the same outputs; no separate forward probe. No outcome-driven tolerance relaxation.

Gradient liveness means finite norm>0 on the fixed nontrivial synthetic fixture and active TRAIN branch; no lower-norm tuning target. Independent analytical gradient comparisons use the dtype-specific bounds above. Degree/edge/mode/parameter-count contracts exact. Diagnostic norms divide by max(norm,1e-12) and are named **regularized relative norms**; record zero and below-floor counts. This fixed floor is diagnostics only, never activation/gradient normalization. Atom sums/counts yield exact atom-weighted epoch summaries; raw-M/gradient summaries retain molecule weighting. These are trajectory aggregates, not fixed-checkpoint TRAIN scores.

## GPU scope, only after source PASS

First128 ascending v2 TRAIN rows, two consecutive64 minibatches. For each original/local/neighbor: update1, update2, restore update1 and replay update2. Exactly9 executed discarded optimizer updates total, using production objective_and_diagnostics and original LE+Ls. No extra target subset, fitting probe or adaptive weight/LR. Actual-step retained graphs may supply term gradient diagnostics before the same backward/update. Report branch/base gradients, theta movement, regularized residual magnitudes/counts, loss/clip/time/peak memory and exact RNG/order replay. Technical states are server-only/disposable and never production initialization.

Initial wrapper/native parity uses the first two already decoded TRAIN geometries; trained wrappers compare only with their own one-file export on those geometries. Partition reader must record exactly128 selected TRAIN numeric rows/fields and zero VAL/TEST rows. UUID-prebound child, fresh physicalGPU1 admission, shared lock and owned registration are mandatory. Wrapper and child refuse completed/prior fixture directories before data access. Preserve any failure and await explicit reviewed recovery rather than retrying or adding steps.

Final metadata freeze binds actual source/CPU/GPU/reviewer/negative-gate receipts, both historical audit namespaces, unchanged185-source Round07 dependency lineage where referenced, split/TRAIN stats and all60 production order hashes. Full production remains blocked pending root acceptance, D-first reviewed publication and distinct exact execution authority. best.pt is a selected snapshot; last.pt is the committed-epoch resume point. No numerical stage repeats just to refresh a hash.
