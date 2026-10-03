# Repeated seed11 control: comparison boundary

The declared original-PSD LE+Ls recipe selected validation R²0.44716940136585204 at epoch45 in Round05 and0.4047975672725648 at epoch49 in Round06 (difference−0.04237183409328724). This is realized variability between two runs with the same declared scientific recipe and seed, not independent-seed confirmation. It is not evidence that the raw-f objective caused the difference between those two controls.

## What is matched

The frozen records match the original-base and full-schema initialization hashes, all60 TRAIN order hashes, new TRAIN statistics, v2 split/validity and the declared optimizer/schedule/batch/clip/budget. The copied reader, fresh-model factory, runtime, metrics, model configuration and environment records are byte-identical. Both original controls used physical GPU1 UUID GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800, empty-parent/UUID-bound-child setup, OMP/MKL threads2 and unbuffered output. Each had one registered original attempt and completed60epochs/112860steps. No warmstart, resume, extension, label exclusion or TEST evaluation is recorded.

Round05 control launch receipt SHA32065a00ec955ad827482b97f611cc92df3bb6e20041062565bfdcd87dd6cc80; Round06 trace-control launch receipt SHA89aa0d8fe62b92501de95dfb25481d595527a56a649d08beba0cda1221267c16. Initial validation R² values differed by about6e-11, consistent with the documented numerical rather than bitwise full-GPU parity policy.

## Execution differences and limits

The scientific control loss remains the exact original base loss, verified synthetically before launch. The training program is not byte-identical: Round05 constructs the raw-M decorrelation autograd graph and adds zero times that value for its control; Round06 computes decorrelation under no_grad and additionally computes the unused raw-f loss diagnostic. Round06 also records one extra loss statistic. These alter executed graph/workload without changing the declared control loss. Source hashes: Round05 training c0147d25cf2a2b6cd0956a57bfcab069657618bc22712efa29feb1b6a9533cdd; Round06 training821fae3660f82026f53bbb267e2f12b431da4963935c5ec63e6fbd607867fc2b.

Round05 scheduled four concurrent GPU fits; Round06 scheduled two. First recorded epoch times were162.675s and155.430s; sums over60 recorded epochs were9566.683s and9348.306s, respectively. These observations neither isolate system-load effects nor establish a nondeterministic-kernel cause. No trajectory rerun, kernel isolation, gradient comparison between completed runs or additional experiment was performed for this note.

Within Round06, the fresh common-source two-arm comparison is the intended objective contrast. Its candidate must still improve by+.003 over both the contemporary control and retained Round05 reference. A gain over the lower contemporary control alone cannot promote the candidate. Any component-bootstrap interval is conditional on these selected checkpoints and reused validation; it does not quantify training or seed variability. Preserve the fixed dual gate and do not reinterpret a control fluctuation as independent confirmation or a physical mechanism.
