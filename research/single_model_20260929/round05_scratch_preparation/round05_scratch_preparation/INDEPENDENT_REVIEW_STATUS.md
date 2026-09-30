# Independent review handoff — Round05 preparation complete

## Current state

- Final preparation PASS; no production fit is authorized or started.
- Frozen source: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/FROZEN_MANIFEST.json`, SHA `66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1`, 82 files.
- Final independent JSON: `INDEPENDENT_PREPARATION_REVIEW.json`, SHA `ef7d90250fd0491fba1432201416de983afa86ee6ae7ebf6e086735f5e10ff19`. Markdown SHA `5b831eb968861d3fbbbf0f79f563eeebcfc44a9b563d2ac87281bc53a47d068b`.
- Final bindings: all82 current hashes,161 receipt source entries and60 TRAIN order hashes checked. Only private TRAIN indices opened, no model/target/prediction values.

## Completed technical work; do not repeat

- New v2 split is materialized and independently reconstructed:120355/6686/6686, all133727 rows retained, zero overlap under audited rules. Split SHA c8ce66dd..., independent verification395d415f.... New TEST contains5989 oldTRAIN/361 oldVAL/336 oldTEST; historically exposed partition, not fresh external evidence.
- New TRAIN-only statistics and CPU synthetic checks PASS. Original CPU dtype-fixture failure preserved.
- First GPU full-forward bitwise check failed before updates; zero-update diagnosis and root post-diagnostic amendment234b0ea5... preserved.
- Successful retry GPU_PREFLIGHT c0a62e24... used exactly128 TRAIN molecules and12 discarded updates, no VALID/TEST values. Independent GPU review4071eca0.... Model replay1.19e-7/Adam2.98e-8/loss0, exact RNG/order/shared-M identity.
- Two parent-directory fsync additions after the successful preflight were independently checked by exact diff/AST. Executed training17d2bb26... snapshot preserved; finalc0147d25...; IO continuity review876e2ea4.... No extra model updates or I/O behavior test.

## Next authorized work

Root must accept preparation, then publisher archives lightweight records to D first and inspects/stages/pushes only exact reviewed files. Independent reviewer remains available for the publisher wrapper/staged bytes. No checkpoint, optimizer, raw/prediction/index array, cache or private identity table may be published. Entire private_preflight* and private_partition are excluded. Preserve failed attempts and diagnostic snapshots.

A distinct exact production authorization after publication is still required. Four future arms are fresh control/F/decor/both,60epochsLR.001AMSGradbatch64clip5, original LE+Ls with raw-M lambda.001 where enabled; no new TEST scoring. No production entry is to be called from this handoff.

Current Git publication base is main/research8497e0ba10efc3226197c894f46507acdad8ec54 or its subsequently verified descendant. Historical notes/snapshots remain intact.
