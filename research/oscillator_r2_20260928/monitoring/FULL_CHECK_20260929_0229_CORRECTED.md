# Scheduled monitoring report — 2026-09-29 02:29 +08

Completed at 2026-09-29 02:31:01 +08. This is a historical health check, not a statement of current status. The report covers the MTO experiment processes and excludes the unrelated SpecGPT job on GPU0.

## Completed loss pilot

The replacement queue (PID 785736) reported `ALL_COMPLETE`. Control, weighted, and direct-f-matched each completed 20 training epochs and selected epoch 0. CPU recomputation from their saved validation predictions gave pooled raw native-f R² values 0.4052941077, 0.4052941089, and 0.4052941280, respectively. The best difference was 2.04e-8; this is a numerical tie. Epoch-20 R² values were 0.3093488848, 0.3065721866, and 0.3173645027. No arm was promoted; no test data were read.

## Legacy chan64 campaign

G1 and G4 had natural `FIT_COMPLETE` receipts. G2 had an intentional administrative stop at epoch 215/cursor 114496; the original campaign remained partial and this was not treated as a trainer failure. G3 was live at PID 566521, start_ticks 1242573486, on GPU4, epoch 281/cursor 108800, 528380 steps, LR 0.00025. Its latest completed validation objective components were [0.5477895469, 0.0273874588, 0.1360195182, 0.3843825699]. The legacy status did not expose native-f R². No G3 terminal or failure marker was present at check time.

## Frozen residual head-only run

The reviewed fixed 20-epoch screen completed 37,620 steps. Selected epoch 2 validation R² was 0.406789903, +0.001495785 versus eta0 0.405294118; epoch 20 was 0.404299111. The gain was below the +0.01 threshold. No promotion, test evaluation, or extension was authorized.

## Frozen-Gram attempt 1

PID 846525 was absent, a `FAILED.json` marker existed, and no `PROBE_COMPLETE` marker existed. The failure occurred before fitting at frozen-cache identity validation (`h` max difference 2.384185791015625e-06). It was an implementation/preflight gate failure, not a model result. No retry was launched at this check.

## Resource snapshot and action

At 02:31, GPU2 and GPU6 were idle (10 MiB each; 32°C and 33°C; 0% utilization; corrected and uncorrected ECC counters zero). GPU4 was occupied by G3 (5257 MiB, 51°C, 43% utilization; ECC counters zero; row-remap pending/failure clear). GPUs3 and 7 remained excluded. GPU0 remained reserved for unrelated SpecGPT. No intervention was needed; preserve G3 and all unrelated work.

The next scheduled check was 2026-09-29 06:29:29 +08. This report is reconstructed from the completed check record and receipts; it does not include any later launch or review.
