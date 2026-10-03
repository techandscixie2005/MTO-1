# Round09 implemented protocol — original MTO with coupled Adam decay

Preparation is authorized by ROUND09_PREPARATION_DECISION.md SHA6b10b8b5a7d75e6a5c46537f8c8b9d126dcc2694d7cfec687e31f120c038d139. The unchanged sealed PROTOCOL_PROPOSAL.md and PROPOSAL_SETTINGS.json define the scientific question. This document records its implementation. Production is blocked until separate bound execution authority follows independent preparation review, root acceptance and verified D-first publication. TEST remains sealed.

## Predictor, initialization and parameter membership

Both ordered arms, zero_decay and coupled_l2, use the same original-mode MTO: DetaNet128/3 blocks/l≤3 and original coherent PSD decoder E=softplus(E-head+offset), C=beta I/sqrt(3)+Q, A=C Cᵀ. The inherited transport wrapper bypasses transport in both arms; dormant right-F and transport parameters are frozen. No new readout, dropout, raw-f objective or active tensor path is introduced.

Both rebuild fresh seed11 tensors; no historical checkpoint or disposable technical state initializes production. Base SHA231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2 and full-schema SHAf9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3 are common. CPU constructor isolation and exact fixture RNG replay do not imply bitwise deterministic CUDA trajectories. Production setup occurs after construction; resume restores saved RNGs.

PARAMETER_ROSTER.json SHA290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943 was frozen before any optimizer call. It records one registration-ordered group of135 tensors/1,552,092 elements, exactly matching the entire unwrapped original trainable name/shape/dtype/numel list. Include every original bias, LayerNorm, embedding, radial alpha/beta and energy offset. Exclude exactly11 frozen right_adapter/transport tensors and all buffers. No new no-decay groups or dimensional exemptions. Both arms verify the same roster before optimizer construction and on resume.

## One optimizer difference, fixed semantics

Use torch2.5.1 Adam with AMSGrad, LR.001, betas(.9,.999), eps1e-8, foreach=None, fused=None, capturable/differentiable/maximize false. zero_decay passes weight_decay0; coupled_l2 passes1e-4. This is coupled Adam, not AdamW.

Compute unchanged task LE+Ls and backward, record task gradients, clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True), calculate detached diagnostics, then call Adam.step. For each parameter with grad!=None, Adam adds lambda*p to the already clipped task gradient, then updates first/second moments and the AMSGrad maximum before bias-corrected parameter update. No second clip and no pre-clip norm loss are added. The total effective gradient need not obey the task clip5 bound. grad=None skips parameter, decay and state; an explicit zero gradient still permits decay. Default backend dispatch is retained.

The fixed1e-4 is an unvalidated probe, not an optimum or guaranteed weak update. It may affect radial denominators, offsets and adaptive moments. Record finite radial beta minimum absolute value and offset range; stop on nonfinite results without clamping, exemptions or tuning. The bounded historical audit found WD0 in23 inspected recipes; that is not a repository-wide absence claim or proof of the cause of prior validation deterioration.

## Data and loss

Use sealed v2 split SHA c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155 and verifier395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2. Counts TRAIN120355/VAL6686/TEST6686. Disjointness is under audited conservative identity rules. This is a new partition of historically exposed data, not external fresh data: TEST includes5989 formerTRAIN/361formerVAL/336formerTEST rows. Current TEST targets are not decoded or scored.

The unchanged selected-row reader validates indices before numeric target decoding, recording fields and authorized rows. Raw-byte hashing/decompression is distinguished from numeric target decoding. Only production authority permits the separate VAL reader. No TEST reader is exposed.

TRAIN statistics SHA d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae supplies sE²=.5376833706691944,sA²=.13245475393297818,n_ref18 and fixed E-state means. LE is valid-E mean squared error/sE²; Ls is valid-E-and-A trace mean squared error/(3*sA²). Original base_loss retains linear trace extraction before masking and valid indexing before squared errors. All benchmark masks must be true; fail rather than exclude changed rows. No tensor/Q/raw-f loss, E² weighting or decorrelation penalty. Logged task total excludes optimizer decay.

Native evaluation remains FP64 f=2*E_eV*trace(A)/(3*27.211386245988), compared with all printed raw-f labels including zeros. Predicted E is positive by softplus, A PSD. No teacher forcing, calibration, clipping, averaging, output permutation or target reconstruction. Reject nonfinite/invalid outputs.

## Fixed production budget and selection

Each arm: seed11/order11,60 epochs plus eligible epoch0, batch64,1881 batches/epoch,112860 updates; LR.001 fixed, no schedule/early stop/warmup, FP32/no AMP/no TF32, two CPU threads. Fresh tensors and60 epoch-order hashes are pinned; graphs/diagnostics are common across the arms except the intended optimizer decay option.

Select earliest strict minimum validation native-f SSE across epochs0–60, all66860 labels. Also retain fixed60 output. Candidate allocation requires selected coupled_l2 R² at least.003 above BOTH selected contemporaneous zero_decay and retained Round05 control45 R².44716940136585204. A pass permits only a later root decision; no automatic seed/extension. A fail closes this coefficient/budget. Better-control evidence may be retained descriptively by root without a decay claim or automatic allocation. No TEST release.

Saved-output analysis reports all states, MAE/RMSE, energy, fixed TRAIN q90=.0549/q99=.2406 tails, complete true/pred-q99 bins, error concentration, counts and selected/fixed60 paired component bootstrap2000 draws/seed20260930/recomputed SST. Model-dependent bins are descriptive. Bootstrap resamples whole validation identity groups with all their molecules/states and variable molecule totals; it is conditional on already-selected checkpoints and does not capture selection or seed uncertainty. The retained reference is aggregate-only, without a paired CI.

## Diagnostics and recovery

TRAIN loss/raw-f/gradient/norm statistics are pre-update minibatch trajectory aggregates, not fixed-final-checkpoint TRAIN evaluation. Both arms compute the same detached diagnostics. Coupled-term/effective-gradient norms use exactly grad!=None parameters. Ratio norm(lambda*p)/max(norm(clipped task gradient),1e-12) is explicitly regularized; log zero/below-floor counts. Never use it to adapt lambda. Frozen original-mode gate diagnostics remain zero. Technical checks verify frozen parameters and buffers unchanged.

last.pt is the atomic completed-epoch transaction: model, Adam/AMSGrad, RNG/order, history, best references, config/statistics, roster/decay/source/split. Resume only last.pt, replaying an interrupted uncommitted epoch. best.pt retains selected model/optimizer/RNG snapshot but is not the current committed transaction. Preserve failed attempts, unique registry IDs and source identity. A completed arm refuses rerun; no blind restart or changed settings.

Geometry-only export contains one model, full config/statistics, original mode/transform contract, strict tensor/buffer metadata and training provenance. Forward inputs are (z,pos,batch,n,edge_index). Load and forward must not reopen training data, prior weights, normalization or cache files. Only source code is external; weights/arrays stay server-only.

## Actual bounded preparation and remaining authority

CPU_SOURCE_REVIEW ca35f1c8 binds295 pins. CPU_PREFLIGHT dfca7977 passed once: three fresh constructors, exact roster freeze, five toy optimizer calls with manual FP64 moment/AMSGrad verification, zero full-model updates, synthetic masking/symmetry and strict one-file export. PARAMETER_ROSTER is immutable. Actual toy arithmetic max error2.22e-16; this is engineering evidence, not accuracy.

The separately reviewed GPU fixture is fixed at first128 TRAIN rows, two64 batches, update1/update2/replay2 per arm, six discarded updates total. Predeclare model/Adam atol2e-6/rtol1e-5, loss atol1e-6/rtol1e-5, exact RNG/order; enforce E/A parity and report native-f errors from the same outputs. No VAL/TEST technical fixture or extra update. All fixture states are prohibited as production initialization. Actual completion is established only by GPU_PREFLIGHT.json and its independent result review; this document does not infer a pass from source readiness.

Final metadata freeze rehashes executed sources, receipts, roster, installed optimizer source, inherited dependency closure and60 prescribed orders, without model or target decoding. All preparation records remain separate from production authority. RUNNER_HANDOFF.md gives the exact blocked command/schema and fresh resource/registration/recovery rules. Memory/time from the128-row fixture are indicative, not guarantees for full-TRAIN batches.

## Resource-only retry amendment

The first GPU1 admission rejected occupied memory before a scientific child, registration, data decode or update. The original empty attempt and nine-file snapshot ab9c820e remain preserved, independently verified by b1b9e401. Root resource decision673bcc503705fe6063e14f02d3573738d82a12907f18e730d41d760c7ae277e8 permits one distinct reviewed retry using ordered devices4,6,2,1. Fresh independent inventory selectedGPU4; final strict admission occurs under its shared lock. run_gpu_preflight_retry01.py/resource_retry_entry.py bind this resource authority and then run the unchanged original registered barrier/scientific fixture. Original technical review5c4b82c0 remains the mathematical binding; retry review and launch receipts separately bind resource/path changes. CPU and roster checks do not repeat. No other numerical setting or update budget changes, no automatic fallback and no production authority.


The authorized GPU retry completed once: GPU_PREFLIGHT SHAa32a7fe8759e1ca171c6dee2efca91374c5f1e749e5a95235cacb6682add77ac, exactly6 discarded TRAIN updates/0VAL/TEST, exit0. Model/Adam replay max1.19e-7/1.49e-8, loss0, RNG/order exact. PREPARATION_REPORT.md reports actual resource identity, metrics and limits; final result/source review remains a separate gate. Detached effective-gradient norms are FP64 reconstructions for reporting, not promises of bitwise optimizer-intermediate equality.

