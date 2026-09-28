# Independent implementation review: matched scratch readouts

PASS for the exact source and code maps in IMPLEMENTATION_REVIEW.json, under root's standing authorization for exactly one native-direct and one MTO-direct seed11 run, 100 epochs each. This review does not launch either worker. Fresh device admission and shared locks remain mandatory. GPU5 is not admitted by this review: its new microcheck and independently bound admission review are still required. Healthy idle GPU2, naturally freed healthy GPU4, or sequential GPU2 are permitted by the protocol.

## Scientific implementation

The approved protocol 9be3d271f86afac91a2b20957b4edee86bad6a14b0bd9058c564254ad8b151b2 is unchanged. Both arms copy the same fresh 1,371,840-parameter core by semantic keys and buffers. Actual official native versus vendored MTO imports are recorded and their executed DetaNet source hashes match. The native last-block hook versus MTO latent outputs differs by at most 2.3842e-7 scalar and 5.9605e-8 tensor. This is floating-point agreement, not bitwise forward identity.

Native has 1,391,188 trainable parameters; MTO-direct has 1,551,986. Native atomwise MLP plus molecular sum differs from MTO routing, state coupling and scalar trunk; this remains a readout-family comparison on the shared backbone. MTO h omits the full cross-channel Gram information. Results cannot isolate one architectural component or prove a backbone/invariant-readout ceiling.

Train-only raw-label means and fixed variance denominators match the protocol. Stable inverse-softplus and f scaling are computed in FP64 before buffer conversion. Both initial heads reproduce the same per-state means (maximum E error 3.134e-7 and f error 3.232e-9), with fixed offsets and zero final affine weights. Both show zero initial core gradients and finite nonzero core gradients on the next step. The nonconstant rotation/permutation probe is confined to disposable preflight models; saved fresh initial states are independently recreated and verified afterward.

## Training, selection and records

The implementation consumes train/validation rows only and keeps raw FP64 labels separate from FP32 training targets. Preflight verifies exact source batch construction, validation ID/index/raw-label/mask alignment, all 100 order hashes, 1,881 updates per epoch and the prescribed learning-rate boundaries. Objective is E MSE/sE2 plus direct f MSE/sf2, AMSGrad with clip5 and no validation scheduler, oracle energy input or warm start.

Raw-f selection uses strict FP64 pooled validation SSE over epochs0..100 with earliest ties; joint-objective selection is separate. Initial/best/final arrays and checkpoints are saved server-only. Versioned selected artifacts are committed through last.pt before aliases, and interrupted-alias recovery is tested. Full fixed-checkpoint train metrics distinguish fit from generalization. Saved-array primary SSE is checked exactly; repeat-forward comparisons use declared FP32 tolerances. Per-state, tail, concentration and fixed-case diagnostics and paired molecule/connectivity bootstrap remain descriptive after validation selection and repeated validation use. Historical eta0 and fixed equal-three are contextual references.

Serialized two-step resume parameter differences are 1.1921e-7 native and 8.6986e-7 MTO, below the prespecified 3e-6 bound, with loss agreement. This supports bounded numerical resume equivalence, not bitwise training reproducibility or a guarantee for every future interruption. Launch is deliberately fail-closed against existing receipts/failure markers; any later restart requires an explicit reviewed operational handoff, with no scientific setting changes.

## Provenance and resource checks

Independently recomputed all 58 source hashes and 10 code/config hashes against both handoff and preflight. Preflight used clean GPU2, with unchanged ECC/remap counters and no foreign GPU process. Launcher and worker independently acquire/check the physical lock, health, occupancy, UUID, receipt PID/start time, source maps and review/preflight hashes. GPU5 additionally requires a newly reviewed admission receipt bound by exact hash; prior guarded eligibility alone is insufficient. GPUs0/1/3/6/7 are not launcher choices. No active seed job or sealed prior study was modified.

No scratch launch receipt existed at sealing. No new training, model inference, test-label access, or scientific-source mutation was performed by this review. The worker will still need fresh healthy-idle resource gates at launch. No extension, tuning, ensemble search, promotion or test access is authorized.
