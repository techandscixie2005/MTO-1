# Round02: frozen shared pre-CG adapter and objective comparison

This protocol is a new pilot following the negative full-model factorial. All four previous arms selected epoch0. Fixed20 deterioration concentrated in rare large predictions for dim targets; the original F changed its right input by about0.75% on a fixed TRAIN subset and raw-M decorrelation measurably changed features. These observations motivate restricting the trainable pathway; they do not establish why the old continuation regressed or predict this pilot will succeed.

The original backbone was trained on these same TRAIN molecules, so its cached representations are in-sample. Validation representations can have a different error distribution. Freezing the backbone does not remove inherited overfit or guarantee correction of large false-bright predictions.

## Hypotheses and matched controls

Keep the exact shared4016-parameter identity-initialized equivariant adapter before CG(M0,F(Mk)), k=1..10. Freeze every original parameter and every buffer, including nonpersistent constants; keep the entire model in eval mode. Train only F. Both original raw-state invariant and2e skip paths remain unchanged. F can affect both E and A through the same frozen decoder. No E-output freeze or extra energy branch is introduced.

- `trace`: LE+Ls, exactly the original normalized energy and trace losses.
- `raw_f`: LE+Lf, with Lf = mean over every valid printed-f label of `(C_F*E*trace(A)-f)^2` divided by the historical TRAIN population variance `.002510981243894732`. Coefficient1; no fitted gradient scaling. LE uses the same normalization/mask as trace. Zero oscillator strengths remain valid targets.
- The same zero-update model is the epoch0 anchor for both arms. It is a fixed reference, not an actively trained control. The trace arm is the active matched control for the objective comparison.

No raw-M penalty is applied because raw M is constant with the original parameters frozen. No target clipping, Huber loss, outlier removal, special sampling, validation-fitted caps or target reordering is permitted. The squared raw-f loss retains sensitivity to rare large prediction errors. Gradient audits describe scale but never change the objective.

AMSGrad learning rate1e-5 fixed for20epochs; batch64; seed/order11; adapterseed11; weightdecay0; global F-gradient clip5; FP32; AMP/TF32off;2CPUthreads. Both arms have the same order, initialization, trainable capacity and fixed schedule. Reset runtime RNG after construction. Keep the same schedule as round01 so the trace arm changes primarily which parameters can adapt. No adaptive schedule or interim stopping decisions.

Checkpoint selection minimizes pooled native raw-f SSE on all valid fixed validation labels, with earliest exact tie and epoch0 eligible. The fixed historical calibration is secondary and never refit or used for selection. Report energy, every state, TRAINq90/q99 true-bright tails and the prespecified q99 true/predicted brightness bins; retain all labels. Compare selected recipes and fixed20 separately. No test inference or scoring occurs; the shared Data loader holds the original arrays, but train/validation are the only indexed inference splits.

A gain of at least.003 R2 over the zero-update anchor is a priority for paired independent starting-seed confirmation. A claim for the raw-f objective additionally requires at least.003 R2 improvement over the active trace arm and inspection of energy/state/tail tradeoffs. A same-checkpoint continuation seed is not independent full-training-seed confirmation. No predictions are averaged.

The historical frozen h-only residual already optimized normalized printed-f error (4161parameters, AMSGrad1e-4), obtaining only about+.00150 validation R2. This study differs in pre-CG placement and E/A sensitivity. It is not the first freezing or raw-f objective experiment. Nonlinearity alone cannot be causally credited without a simpler active adapter control. O(3) equivariance does not imply electronic phase-odd covariance; raw M is a learned feature, and physical interpretations remain hypotheses.

## Cache and reproducibility

Cache all raw TRAIN states including M0 in FP32:0e/1o/2e shapes[N,11,16,1/3/5]. The current base has dropout0, no BatchNorm or geometry augmentation. Cache fixed geometry/edge order in eval mode and pin model, data, source, dtype, GPU and index-to-ID mapping. Cache arrays and cached labels stay server-only. No validation cache is needed. Both full geometry and cached paths call one `from_raw` helper, preserving the original coupling and decoder arithmetic.

Before fitting, require actual TRAIN/GPU parity of E/A/f and adapter gradients at identity and nonzero F; compare two Adam updates, including one with nonempty optimizer state. Compare cached generation batch64 with independently batched TRAIN32. Tolerances are fixed in config before checks, and actual deviations are recorded. Check all original parameter/buffer hashes before and after discarded updates. Verify optimizer membership is exactly F. A separate serialization test must reproduce the next cached update and order bitwise, including optimizer/RNG. Replay full geometry validation epoch0 against the known anchor. Test updates are discarded.

Training keeps atomic `last.pt` at every complete epoch with full model, F optimizer, RNG/order state, config, cache contract, all history and selector state. An interrupted epoch replays from the previous complete epoch; SIGINT/TERM finishes the current epoch. Selected checkpoint transactions retain `selected_epoch_###.pt` and recover the best alias on resume. No weights, optimizer states, raw datasets, prediction arrays or cache arrays are archived or committed.

Every epoch reports TRAIN energy/trace/f losses, clip fraction, max gradient norm, adapter movement on the frozen256-TRAIN subset, sample-order hash and unchanged original parameter/buffer verification. These diagnostics cannot alter the schedule. Selected-checkpoint validation uses full geometry inference and replays at completion. The final checkpoint embeds the entire original model plus F, config and statistics; inference recomputes M from geometry and never requires a cache or QC labels.

The shared validation metric field `base_objective` always means LE+Ls. For the raw_f arm, its own validation objective is `energy_loss + raw_f.sse / raw_f.count / train_f_variance`; neither objective selects checkpoints. The fixed coefficient1 produces a larger F-gradient norm for the raw_f objective in the preflight (about2.04–3.43 times trace across the fixed TRAIN batches); disclose this scale difference rather than fitting a coefficient.

The initial serialization test exposed an Adam scalar step-counter device mismatch in the test comparison after loading directly to CUDA. Production resume and the test now load the checkpoint on CPU, then let the optimizer move moment tensors to parameter devices while keeping non-capturable step counters on CPU. The rerun matches next weights, optimizer state and order bitwise. The initial preflight log is preserved; no production fit failed or used those discarded updates.

The initial CPU composite-checkpoint fixture left F trainable while the loader disabled all gradients. The strict bitwise assertion failed under those differing execution configurations. The fixture now uses the same all-frozen inference configuration as the loader and passes bitwise E/A replay with no base-checkpoint, dataset or cache access. Both audit logs are retained; this does not alter fitting, metrics, cache or weights.

## Gates and commands

Preparation is permitted under monitor allocation; fitting requires this protocol, source freeze, independent PASS and archive-first verified remote publication. GPU admission checks exactUUID, idle compute/memory, uncorrectedECC and remap status under shared `/tmp/mto_pouter_gpu_INDEX.lock`. GPU0 is unrelated;3/7 are excluded. Registered production process identities are monitored every four hours.

From this round directory with the pinned environment Python:

```sh
python prepare_cache.py --gpu 2
python preflight.py
python resume_preflight.py
python inference_preflight.py
python freeze_round.py
python launch_round.py --review INDEPENDENT_PRELAUNCH_REVIEW.json --publication-receipt ../ops/ROUND02_PUBLICATION_RECEIPT.json
```

The bare `python` above means `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`. `VARIANCE_PROVENANCE.json`, cache receipt, preflight reports and parent source closure are part of the sealed manifest. The launcher refuses missing or mismatched review/publication receipts and registers owned PID/start-time/UUID without disturbing unrelated jobs.
