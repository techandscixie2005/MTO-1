# G4 completion and GPU6 reassignment

G4 reached natural `FIT_COMPLETE` after epoch 223 with `early_stop`. Its best validation objective was 0.136369137227 at epoch 71, followed by 152 epochs without improvement. Two learning-rate reductions occurred; the last was epoch 173, so the 50-epoch cooldown ended at 223. No administrative stop was needed.

G4's configuration sets `supervise_E=false`. Its previously audited validation oscillator-strength R² 0.348335 uses **oracle true energy** and is diagnostic only. G4 has no comparable deployable native-`f` score.

G4's original PID 566532 had exited. The architecture queue subsequently assigned GPU6 (UUID GPU-2431a641-8045-a10b-4aa0-2769010e0708) to its matched original-control worker PID 790200. Exact config, completion marker, launch receipts, best/last checkpoints and validation-audit hashes are in [G4_COMPLETION_REASSIGNMENT.json](G4_COMPLETION_REASSIGNMENT.json). No process was signaled, no GPU was reset, and no test prediction was opened.
