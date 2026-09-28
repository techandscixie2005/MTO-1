# Corrected native DetaNet versus MTO direct-f: matched scratch protocol

2026-09-29. PREPARATION ONLY. Root requested this next architecture study while the approved seed23/37 jobs proceed. Exactly two arms,100epochs each; execution requires protocol approval, implementation/preflight review and explicit root launch authority. Do not modify active seed jobs, sealed studies, data split or Gram outputs. No test access, hyperparameter/seed search, case filtering or automatic extension.

## Question and architectural contrast

Does native atomwise scalar readout or MTO molecular routing/coupling generalize better when both learn direct positive oscillator strengths from scratch under the same properly scaled raw-f objective? The historical native E/f result (~.0463 validation f R2) used unscaled joint losses and unconstrained f, so it is inconclusive. This pair removes that scaling/positivity and pretrained-adaptation confound.

Both already use the SAME complete DetaNet backbone:128features,3 interaction blocks, l<=3,32 trainable Bessel radial functions,8 attention heads,radius5Angstrom,dropout0. This is a readout-family control on a shared backbone, not a backbone replacement. The common core has1,371,840 parameters. Preserve native readout width; do not widen it to equalize total parameter count.

- Native-direct: official scalar20 DetaNet atom MLP128->128->20 with its original activation, followed by molecular SUM. Split20 outputs into10 E and10 f logits; apply positivity only AFTER summation. No mean-pooling, n_ref factor, extra global network, state queries or tensor output head.
- MTO-direct: original MTO molecular composition/context and state-conditioned routing, reference/excited-state CG coupling, decoder scalar trunk128, legacy scalar energy head and direct-f head128->32->1 with SiLU. Remove unused beta/tensor_gate property heads. All retained parameters train from scratch; no teacher warmup, tensor/A loss, frozen eta0, residual correction or distillation. This arm uses h from the established direct-f screen but starts fresh.

Native scalar atom features already mix geometry/equivariant information through message passing. Native atomwise summation lacks MTO's global routing/context and nonlinear cross-atom coupling before property output. MTO-direct uses CG scalar contractions but omits full cross-channel tensor Gram information outside h; neither arm is the original tensor-product f readout. Adding softplus after native summation supplies molecular nonlinearity/positivity but does not reproduce MTO contractions. Different routing, state structure, parameter counts and gradients are intentional parts of the readout-family comparison; do not attribute a result to only one component. Report actual total/trainable/core counts after the buffer changes below (historical approximate totals1.391M native and1.552M MTO are references, not equality requirements).

## Matched fresh initialization and positive output

Fixed pair seed11 and order_seed11. Build one fresh random common DetaNet core and copy its complete parameters and relevant buffers exactly into both arms. Verify one-to-one semantic state mapping and identical latent scalar/tensor outputs on fixed training examples. If official versus vendored naming differs, document the mapping; no silent shape-only mapping. Confirm the official/vendored computational source equivalence already found in the feasibility audit. Independently initialize each arm's own readout with seed11 in a documented RNG context; do not pretend readout weights of different shape can be identical. Record all initialization hashes and RNG states. No pretrained checkpoint is loaded into either trainable model.

Compute train-only per-state means mu_E[a],mu_f[a] from raw labels, using fixed masks; all ten means must be strictly positive and finite. Reuse sE2=.5378066634062587 and sf2=.002510981243894732, sf=.050109692115345626 after verifying against immutable train statistics. Per-state means affect only fixed output offsets; losses use pooled fixed denominators below.

Both emit:

    E_a = softplus(u_E,a + invsoftplus(mu_E[a]))
    f_a = sf * softplus(u_f,a + invsoftplus(mu_f[a]/sf))

Offsets and sf are FIXED buffers, including replacing MTO's original trainable energy_offset with a fixed buffer. Use stable invsoftplus(x)=x+log(-expm1(-x)), evaluated in FP64 then stored in model dtype. Zero the FINAL affine weights and biases producing all E/f logits in both arms. For native this is the20-output atomic MLP layer before summation; for MTO it is energy_head's final affine and strength_head's last affine. Keep earlier readout layers randomly initialized. Consequently both arms initially emit identical statewise training means independent of molecule size. Verify in FP32 within declared rounding tolerance. First-step zero backbone/earlier-readout gradients are expected from zero final weights; verify they become nonzero on subsequent fixed training steps. No output clipping, sorting, true-energy input, f/E product, log-f target or predicted-A reconstruction enters learning.

## Exact objective and budget

Same connectivity split120355train/6686val/6686untouchedtest, same immutable z/pos/radius-edge cache, IDs and ten ordered state slots. Only training and validation rows are consumed. Use raw printed f; training tensors are FP32, raw labels retained separately in FP64 for metrics. E uses the same source raw label with original FP32 training conversion. Assert masks/all expected counts; do not silently discard a target.

    L = mean_mask_E[(Epred-Etrue)^2]/sE2
        + mean_mask_f[(fpred-ftrue)^2]/sf2

Equal coefficient1 on the normalized terms, no A/trace/spectrum loss or learned loss weights. Gradient flows through each direct-f head and common core; E is predicted independently and is not an input to f. Per-state selection/weighting is forbidden. Record separate energy/f terms and gradient/clipping behavior so scale imbalance is visible.

Adam AMSGrad,betas(.9,.999),eps1e-8,weight_decay0,global clip5,FP32,AMP/TF32off,CPU threads2. Fixed learning rates:1e-3 epochs1..60,3e-4 epochs61..85,1e-4 epochs86..100, set before the first update of each epoch. Preserve optimizer moments at schedule boundaries. No validation scheduler, early stop or arm-specific schedule. Exactly100epochs,1881updates/epoch (batch64, last35),188100updates/arm. Use the same stored NumPy default_rng(11) permutations of original training indices for both arms; verify every order hash.

This is one controlled screen, not an established optimum for either family. Equal updates/examples are primary fairness; record wall/GPU time and memory because native may be cheaper. Prior MTO149.6s/epoch suggests~4.2GPU-hours/100epochs; native runtime is unmeasured. No promise of convergence or automatic second seed/extension.

## Selection, outputs and decision

Evaluate initial epoch0 and each epoch on the fixed validation set. Select strictly minimum pooled raw-f FP64 validation SSE over epochs0..100, earliest exact tie, independently per arm. Also record normalized joint-objective best for context, but raw-f is the primary selection and must not be silently substituted. Save initial, raw-f-best, joint-best and final100 checkpoints and validation arrays server-only with IDs/global indices/raw truth, E/f predictions, exact selection epoch/metric and source/config hashes. Unlike tensor models, direct f is the forward scalar output promoted to FP64; do not recompute it from E or any tensor.

Report pooled raw-f SSE/SST/R2/MAE/RMSE, per-state metrics, fixed q90=.0546/q99=.2377 bright/complement SSE, energy metrics, signed/negative counts (positive heads should yield none), max predictions, bias and molecule-error concentration. Include the previously fixed ID14562 diagnostic without giving it a fitting/selection role. Full checkpoint-fixed TRAIN metrics at initial, selected and final distinguish training fit from generalization; online training means are not a substitute. Save optimizer/gradient norm and clipping summaries each epoch.

Primary contrast: selected native minus selected MTO direct-f at equal100epoch budget, with2000 paired whole-molecule and connectivity-group bootstrap draws,seed20260928, all states/groups retained and SST recomputed. Existing eta0 and fixed equal-three are contextual references, not matched scratch controls. A>=.01 R2 difference with>=80% positive molecule draws and group robustness is exploratory evidence favoring that family; converse evidence favors MTO at larger capacity. Bootstrap after checkpoint selection and prior validation reuse is descriptive. Neither arm improving the known baseline is no promotion. No untouched-test inference is authorized, even if a threshold is met.

Both poor with high training error leaves optimization/budget and representation unresolved. Lower training error but poor validation favors a generalization explanation. One paired seed cannot establish robustness; an eventual useful result needs confirmation. This pair does not prove a tensor-readout or full invariant-readout ceiling, and it does not test a deeper/new backbone.

## Implementation and preflight gates

Work in a new isolated scratch_readout_comparison directory. Reuse sealed code read-only; avoid the existing ArchitectureMTO.build default because it loads pretrained checkpoints. Pin actual executed source dependencies, dataset/split/normalization/raw-label sources, configs/protocol, common initialization, order plan and fixed comparison arrays. Inspect imported modules so native and MTO resolve the intended equivalent DetaNet implementation. Do not mutate active seed or earlier source files.

Before launch require: exact common-core weight/buffer and latent-forward identity; fresh reproducible pair initialization and absence of pretrained loading; identical mean predictions and positive E/f; parameter counts and fixed offsets; rotation/atom-permutation invariance with remapped edges; raw-label/ID/mask/native-metric alignment; loss scaling arithmetic and finite gradients through both heads/core after initial zero-gradient step; same full order hashes/update counts; fixed LR boundary tests; two-step serialized resume restoring model/optimizer/LR/RNG/order/cursor within justified FP32 tolerance; transactional versioned best-checkpoint/prediction commit recovery. Evaluation must not consume order RNG or alter subsequent training mode. Initial/selected/final outputs and completeness receipts must be preflight-covered. Do not raise tolerances after looking at validation outcomes.

Use per-run locks plus shared physical-device locks, PID/start-time/UUID/cwd receipts, source-hash checks, finite-loss/gradient gates and ongoing ECC/remap/occupancy checks. GPU2 may be used only when healthy idle and the Gram worker is confirmed exited/released. A second device may be naturally freed healthyGPU4, or guardedGPU5 ONLY after an explicit NEW admission/microcheck and independent review; prior guarded eligibility alone is insufficient. Do not stop G3, preempt unrelatedGPU0, use faulty3/7 or alter seed jobs on1/6. If only GPU2 is available, sequential paired execution is acceptable and preserves controls; do not weaken admission to obtain concurrency. Capture hardware/time differences as limitations.

Implementation handoff must include protocol approval, exact source/config/initial/order hashes and bounded preflight evidence. Root and independent implementation review must approve before launching either100epoch arm. Failures pause/stop for review; no silent seed replacement, fallback loss, refit or extended budget.