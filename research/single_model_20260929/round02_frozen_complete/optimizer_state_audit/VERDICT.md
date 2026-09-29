# Selected eta0 optimizer-state availability

**An exact preserved-Adam comparison from selected epoch33 is unavailable from the retained checkpoints.** No reconstruction or fit is proposed.

- `experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/best.pt` is epoch33 and contains only model, epoch, validation values, config and fingerprint. It has no optimizer, scheduler or RNG state.
- `last.pt` retains135 Adam parameter states, scheduler and RNG, but corresponds to completed epoch236: next epoch237, cursor0,443,916steps and learning rate.000125. Pairing these later moments with epoch33 weights would not restore the original optimizer state or constitute a matched preserved-moment control.
- `trainer.py:112` saves the model-only best checkpoint. Lines78–80 overwrite the full `last.pt`; lines64–70 restore complete state on ordinary resume. The source run has only these two `.pt` files. The available repository inventory with eta0-named paths revealed no other epoch33 optimizer snapshot. Unknown external backups were not searched or assumed to exist.

Earlier20-epoch loss pilots explicitly reset Adam in `research/oscillator_r2_20260928/SCIENTIFIC_AUDIT_AND_PROTOCOL.md:21`; the architecture pilot states the same in `architecture/ARCHITECTURE_PROTOCOL.md:43`. No matched gentle preserved-moment control is documented in these audited continuations.

The later clean-initialization inspection also checked the on-server completed backup `backups/mto_completed_20260927.tar.gz`. Its eta0 best/last checkpoint members are byte-identical to the current files by SHA256; it does not restore epoch33 moments. The archive was not downloaded or extracted to disk.

The original training trajectory naturally continued after epoch33 with its existing moments under lr.001 and later plateau reductions. Its original validation LE+Ls was.1722269 at33,.1909843 at34 and.2123658 at236, with best epoch33 retained. These are original objective values, not pooled raw-f R², and this trajectory differs in learning rate, schedule and order from the new continuations. It neither isolates nor proves an effect of Adam resetting on the observed raw-f regression.

`audit_metadata.py` and `OPTIMIZER_METADATA.json` preserve the bounded metadata evidence and exact checkpoint/source hashes. Only lightweight records are exportable; weights and optimizer arrays stay on the server. Round02 fitting and frozen source closures were untouched.
