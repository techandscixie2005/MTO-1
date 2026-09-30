# First-epoch metadata inspection

The first read-only invocation of inspect_first_epoch.py reached the `completed_epoch >= 1` assertion while the control worker still had only its epoch-0 commit. It exited 1 before writing a receipt. The production workers continued normally; no FAILED.json existed and no model, optimizer or training file was changed. This was an early readiness check, not a training failure or a replay.

A subsequent status snapshot showed all four epoch-1 commits. The same unchanged inspector then passed and wrote ops/FIRST_EPOCH_METADATA.json. It loaded the committed checkpoint tensors on CPU only to inspect optimizer/RNG/order metadata; it constructed no model, performed no inference and opened no raw target arrays. Production has advanced naturally throughout.

Observed first-epoch durations including validation were 162.67 s control, 166.80 s adapter, 162.90 s decorrelation and 168.07 s both. Every checkpoint contains 1881 Adam updates, matched frozen initial hashes, the prescribed order hash, all five RNG categories and immutable selected-version hashes. Decoder ledgers cover only new TRAIN and validation, with zero TEST numeric rows.

At the observed pace, the fixed 60 epochs take approximately 2 h 43 min to 2 h 48 min per concurrent arm, excluding changing system load. This is an estimate, not a completion promise. Early validation scores do not change the frozen protocol or select an experimental direction.
