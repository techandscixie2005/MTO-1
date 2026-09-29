# Frozen-base raw-M cache feasibility

**Verdict: technically feasible, conditional on a separately approved study.** Caching raw M can remove repeated backbone/MTO computation while preserving the mathematical geometry-to-output function. A small CPU test passed exact output/gradient parity. No real-data cache, new fit, GPU work or checkpoint export was performed.

## Evidence and scope

`cpu_cache_feasibility.py` loaded the pinned eta0 checkpoint and training normalization, then used two synthetic geometries. With only the4,016 right-adapter parameters trainable:

- Full geometry and cached-M paths gave bitwise identical E/A at identity, a nonzero adapter, and after one discarded synthetic Adam step.
- Gradients of the original LE+Ls objective with respect to every adapter parameter were bitwise identical.
- Every frozen parameter and every named buffer, including nonpersistent buffers, remained bitwise unchanged. Frozen parameter gradients were absent and the optimizer held only adapter parameters.
- Both outputs remain trainable through F: synthetic gradient norms were1.6390 for summed E and .18646 for summed trace(A). Frozen decoder weights do **not** freeze decoder outputs when its inputs change. This is not a strictly dipole-only update.

Results and source hashes are in `CPU_CACHE_FEASIBILITY.json`. This establishes a synthetic CPU implementation check, not real TRAIN/GPU parity or an accuracy result. The test duplicates the short decoder expression only for verification; production full/cached paths should call one shared post-M helper to prevent drift.

## Why eval-mode caching is possible here

The current DetaNet configuration uses dropout0 and norm=False. The decoder and new adapter use LayerNorm, which has no running batch statistics. Source inspection found no geometry augmentation in `Data.batch` and no batch-dependent running normalization in this model. The runtime module audit found no BatchNorm and only zero-probability Dropout modules.

Keep the whole model in eval mode, freeze all parameters, then enable gradients only for `right_adapter`; eval mode does not disable autograd. Fixed buffers include MTO normalization, tensor-product/cartesian constants and nonpersistent element/mass constants. Check **all named buffers**, not only `state_dict`, because the latter omits nonpersistent buffers. Snapshot/hashes before and after training must establish unchanged base weights/buffers.

## Minimal cache and training boundary

1. Pin source checkpoint, model/config/normalization code hashes, geometry/edge/split hashes, FP32 and device/arithmetic settings. Cache only the frozen TRAIN geometries with original atom, edge and state order, in eval/no-grad mode.
2. Store all raw states including M0, for each of0e/1o/2e. Per molecule: `(11,16,1)`, `(11,16,3)`, `(11,16,5)` =1,584 FP32 values =6,336bytes. All120,355 TRAIN molecules need762,569,280bytes, about727MiB before file headers and index metadata. Contiguous server-only arrays can be memory mapped or placed on the eventual healthy GPU; no throughput claim has been measured.
3. Detach raw M. Keep cached M0 unchanged. Apply F only to excited-state right operands before CG(M0,F(Mk)). Recompute CG and the frozen decoder **with autograd enabled** so gradients reach F.
4. Preserve both original skips exactly: scalar invariants of the untransformed M0/Mk and untransformed excited2e tensors. Do not cache decoder h or already-combined CG output, and do not replace raw skips with transformed features.
5. Use the same target masks and original LE+Ls for an unchanged-objective study. Keep raw printed f only for the current primary evaluation unless a new loss comparison is explicitly registered. Raw-M decorrelation is constant under this freeze and must be excluded as an active factor.
6. Optimize only F. Keep source weights/buffers bitwise unchanged, check parameter-ID membership in the optimizer, and preserve adapter Adam/AMSGrad state plus RNG/order state for resume. Cache manifest and ID-to-row mapping must be part of the resume contract; never confuse global dataset indices with cache row positions.
7. Evaluate validation checkpoints through the **full geometry model** with the existing all-label raw-f evaluator. Epoch0 remains eligible, fixed calibration remains secondary without refit, and no test inference occurs. There is no need to cache validation labels/features for this proposal.
8. Export one complete model checkpoint containing the unchanged base, trained F, buffers, adapter-enabled setting, config and normalization. Deployment recomputes M from geometry; neither a training cache nor an external base checkpoint nor QC label is required.

The cache contains derived training data and stays server-only. Do not archive/commit it or any cached targets, weights or optimizer states. Archive its lightweight provenance/hash/coverage records only.

## Required parity checks before any cache-backed fit

- On a fixed TRAIN subset, compare cached and full paths at identity and a nonzero F, including E, A, raw f and F gradients. Then compare the exact next optimizer update from matched optimizer/RNG state. The synthetic CPU test supplies a prototype, not a substitute for the actual device/data check.
- Same-batch, same-device FP32 can match bitwise; different GPU reductions, cache-creation batching or device changes can introduce ordinary floating-point differences. Predeclare a justified numerical tolerance from TRAIN-only checks. Do not tune it against validation improvement.
- Validate cache readback hashes, record count, shapes, finite values and ID alignment. Preserve original masks and all primary labels, including zero f.
- Noise, dropout, coordinate augmentation, changed edges/cutoffs or any trainable upstream parameter invalidate a once-only cache. A later rotation-augmentation design would have to rotate all cached irreps, including M0, correctly; it is not part of the current proposal.

## Scientific controls and historical distinction

The historical frozen h-only residual operated after the old readout and selected epoch2 at validation R² .4067899. It already optimized raw printed-f MSE divided by TRAIN variance .002510981243894732, with `f=abs(base_f+.050109692115345626*MLP(h))`; LE was constant because E was frozen. Its20epochs used Adam AMSGrad1e-4, batch64/order11. It did not optimize LE+Ls, and no separate frozen-head E² objective is documented. Evidence: `research/oscillator_r2_20260928/frozen_residual/FROZEN_RESIDUAL_PROTOCOL.md` and `train.py` lines194–195.

This proposal acts before CG and changes both scalar/tensor coupling branches and potentially E/A while keeping raw skips. It is a new controlled question, not evidence that either location is better. A pre-CG raw-f study would not be the first frozen-base/raw-f experiment; its new rationale is the input location and changed E/A coupling. A pre-CG LE+Ls study differs from the historical residual in objective as well as location.

Freezing the backbone, changing adapter location/capacity, changing the objective or changing its optimizer scale are separate factors. A frozen identity model is a fixed epoch0 reference, not an equally trained capacity control. A simpler right-channel linear adapter could probe nonlinearity, but has fewer active parameters and should not be called capacity matched without an explicit design. The historical residual has4,161 extra parameters versus4,016 here, but its inputs, outputs, loss and selected training path differ.

If a native-f loss is later studied, compare it with original LE+Ls using the **same** frozen-base F, initialization, learning rate, order and budget. Any normalizer or loss weight must come from TRAIN-only evidence and be declared before outcomes. Preserve original printed raw-f labels and distinguish effects on energy/trace from direct f optimization. Neither a new loss nor learning rate is selected by this feasibility review.

The existing full-model factorial remains unchanged and must finish first. A null result there does not by itself identify learning rate, representation drift, loss mismatch or adapter optimization as the cause. Use its prespecified mechanism diagnostics before deciding whether this cached frozen-base study deserves resources.

Reproduce this bounded CPU feasibility test:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
  /home/inspur/MTO-1/research/single_model_20260929/reports/frozen_cache_feasibility/cpu_cache_feasibility.py
```
