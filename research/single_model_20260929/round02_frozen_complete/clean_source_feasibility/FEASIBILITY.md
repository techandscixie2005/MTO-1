# Clean internal-TRAIN source: preparation feasibility

**Feasible from a fresh constructor; no source fit or split-array creation occurred in this audit.** This is a distribution-transfer experiment hypothesis, not a fresh external holdout or evidence of improved accuracy.

The pinned `model_factory.py:build` seeds PyTorch and constructs MTOEA without loading learned weights. The existing `initial_model.pt` and initialization hash must not be reused unchanged: `models_ea.py` initializes the energy offset from full-TRAIN `E_state_mean`, and existing `normalization.json` contains full-TRAIN target statistics. Recompute E_state_mean, sE2, sA2 and any later f statistics on the source-fit subset only; n_ref may be recomputed from its geometry. Rebuild random initialization and pin its new hash in a separate runner. Use no full-TRAIN eta0 weights, teacher distillation targets or selected source checkpoint as initialization.

The proposed source checkpoint is the fixed epoch33 state, not a checkpoint selected on its held-out20%, outer validation or test. Held-out E/A/f labels must be absent from gradients, normalization, model selection and schedule decisions. Source-training metrics may be monitored, but source epoch/schedule cannot adapt to held-out outcomes. The existing trainer's outer-validation selection, target-derived initial hash and early stopping therefore require an isolated, reviewed runner rather than direct reuse unchanged.

## Group policy and exact metadata

The pinned `identity_audit_v2.json` SHA256 is `9d384425a90dd88fbc68f8b609a303872910bb21c69e0d0a19bcccaee49cb3c4`. Identity keys match dataset molecule IDs and use the existing conservative connectivity/bond-order/stereo-agnostic grouping. Original TRAIN has120,127 groups for120,355 molecules, including218 duplicate groups (446records), and266 unresolved groups (278records).

Preserve unresolved groups in the source-fit subset, matching their original train-only policy. Shuffle sorted resolved original-TRAIN group keys with NumPy RNG seed20260930; take complete groups from the prefix until at least24,071 held-out rows; preserve original TRAIN index order within both subsets. This deterministic metadata-only audit gives exactly96,284 source-fit and24,071 held-out molecules (24,032 held-out groups). No group is divided or molecule excluded; original outer splits remain unchanged.

- Source-fit index bytes SHA256: `b7b00dfe514ae57f3dd609a60dc259625007a5f43288ff75e2c83a3a7e71da37`.
- Held-out index bytes SHA256: `9547ca89feff3301f5e4c6fcd736be7a9186878a9fcb1fec8695c5c12d60707b`.

## Cost and interpretation

Original eta0 averaged149.48seconds per full-TRAIN epoch (first33:150.91seconds). Linear training-size scaling suggests about65.8 GPUminutes for33epochs on80%; allow roughly60–75minutes plus setup/export, subject to measured hardware performance. This is an estimate, not a launch or schedule approval. A smaller source trained for the same number of epochs receives fewer updates and may have different accuracy and error structure; the comparison cannot isolate only membership effects.

A global affine readout uses observable f in a consistent scale across temporary and full sources. Arbitrary h/M features can change their learned basis across sources and would need extra alignment justification. Even observable residual relations from the smaller held-out source may transfer poorly to the full source. The same held-out-row full-source control, untouched outer validation and separate one-checkpoint packaging are required by the pending protocol. No prediction averaging is implied.

The source train/held-out partition is internal to already used TRAIN data. It must not be described as fresh final holdout confirmation. The current best eligible calibrated eta0 and all completed-round records remain intact.
