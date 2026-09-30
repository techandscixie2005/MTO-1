# QM9S v2: frozen identity-disjoint partition

**The private split is materialized and independently verified.** All 133,727 previously valid QM9S records are retained: TRAIN 120,355, VALIDATION 6,686 and TEST 6,686. No model was fitted or scored in this preparation. No target or prediction array was decoded.

**This is a new partition of historically used data, not an independent external dataset.** Its TEST contains 5,989 former TRAIN, 361 former validation and 336 former test records. Existing models and calibrations are therefore ineligible for this benchmark. Train from fresh random initialization, calculate target statistics only on v2 TRAIN, select one checkpoint using v2 validation, and leave v2 TEST unscored until a separately frozen final evaluation decision. R² 0.60 remains an unachieved research objective; this preparation supplies no new accuracy result.

## Frozen construction and verification

The protocol fixed PCG64 seed20260930 before any scoring, with no target stratification. Original explicit-H keys, conservative heavy-atom connectivity plus full elemental formula, all five prescribed connectivity candidates for ambiguous rows, and complete element-pair distance matches at L∞≤1e-4Å were united into whole components. Components were ordered by minimum molecule ID, shuffled once, and assigned TEST then VALID until each reached6,686; the remainder and ambiguous linked components went to TRAIN. No boundary adjustment or seed search occurred; neither held partition overshot.

The result has114,572 components, largest14 records. Original full-key unions merged258 rows; additional heavy/formula unions merged18,897. Eighty-three near-geometry pairs were already inside these groups. All278 ambiguous records remain in263 TRAIN-only components; there were no candidate-generation failures. Every row appears exactly once.

The separately written verifier imports neither builder nor grouping core. It reconstructed RDKit keys, geometry distances with SciPy pdist, union-find components, PCG64 assignment, ambiguity masks and old-exposure intersections from the allowed identity inputs. All saved arrays, component labels and signatures matched exactly. Cross-partition row, molecule ID, component, original-key, heavy/formula-key and near-geometry overlap counts are zero under these rules.

This is an audited conservative identity rule, not an absolute proof of chemical identity. Bond perception can be ambiguous; formula/heavy-graph grouping intentionally merges some stereo, bond-order and tautomeric variants. Distance signatures can merge homometric structures. Conformers sharing the declared graph/formula family stay together. These conservative merges reduce leakage risk without using labels.

## Historical exposure, retained for disclosure

| New partition | Old TRAIN | Old validation | Old test | Old inner source fit | Old inner calibration |
|---|---:|---:|---:|---:|---:|
| TRAIN |108,319|6,004|6,032|86,598|21,721|
| VALIDATION |6,047|321|318|4,835|1,212|
| TEST |5,989|361|336|4,851|1,138|

The two inner columns partition the old TRAIN intersections; they are not additional disjoint datasets. These counts were computed after the fixed assignment and never used to select memberships. The original split remains unchanged.

## Target access and primary metric

The builder and verifier decoded only `ids.npy`, `z.npy` and `pos.npy` from the curated dataset. They used pinned identity/split JSON and old index metadata for grouping/provenance. Hashing opaque dataset bytes is distinct from decoding target values. No E, A, f, target mask, prediction or checkpoint was decoded. All identity/signature/index arrays remain server-only.

Keep the original ten-state printed raw-f target, including valid zero values, and pooled `1−SSE/SST` over all valid molecule-state entries. Do not substitute spectrum scores, selected states, method mixtures or exclusions. Future reports must include per-state errors, MAE, TRAIN-derived bright tails and false-bright bins. New TRAIN statistics, label-access isolation, initializer and evaluator still require the separate model-preparation review. No statistics or training approval follows from this split verification alone.

## Exact artifacts

Server root: `/home/inspur/MTO-1/research/single_model_20260929/dataset_audit_20260930`.

| Record | SHA256 |
|---|---|
| SOURCE_MANIFEST.json |`2bcd4d256179ad99f6b88b34f50bba1a99facc9389a6438535662410c6783a38`|
| CORPUS_AUDIT.json |`644b6e6d5e2c7af1b2a6f234715a8d5acdaed9d596f746eba9f897c539f45a1c`|
| SPLIT_MANIFEST.json |`c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155`|
| INDEPENDENT_SPLIT_VERIFICATION.json |`395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2`|
| private_partition/train_indices.npy |`8ff3ca690cf63051701d66c3dd438228457bc3b4d7b92fa5511bd943de1176cc`|
| private_partition/val_indices.npy |`be13afee03e4e72418b3979040c765cd94ee02c8a1b220cc87d83e08d7d239fb`|
| private_partition/test_indices.npy |`c5c65be9219a4c8c3d797dd19bde006b25459e2d5e50d425a32737ef6cad3692`|

`SPLIT_MANIFEST.json` additionally pins component IDs, ambiguity masks, molecule IDs and private identity signatures. Its public metadata contains shapes/hashes/paths, not membership values. All code and environment versions are pinned; NumPy2.2.6, SciPy1.15.3, RDKit2025.09.4, Python3.10.19.

## Failure, repair and resumability

The first materialization stopped before writing arrays because Python dictionary equality treated reloaded JSON histogram string keys differently from the rebuilt integer keys. `SERIALIZATION_REPAIR.md`, the original source/review snapshot and the failed process/log remain preserved. Root approved a narrow normalization/comparison fix. The22 synthetic checks include multi-digit key round-trip and changed-count rejection. Independent review confirmed unchanged grouping code, settings and protocol.

The renewed audit reproduced every scientific aggregate and every planned array hash from the original audit; only source provenance changed. The reviewed repair01 materialization then completed normally, followed by independent reconstruction PASS. Each operation has its own registered wrapper/child identity, log and terminal receipt under `ops/`; no blind retry or unrelated-job modification occurred. Immutable writes retain identical existing bytes and reject changed outputs; owned wrappers refuse already completed stages. Do not rerun completed stages to recreate evidence.

Before Git publication, download and verify the exact lightweight allowlist on D. Include both failed and repaired records. Exclude the entire `private_partition` directory, including its JSON identity table. No compatible overlap-cleared external holdout has yet been verified; external QC-method-shift candidates remain separately documented in `PUBLIC_DATA_AND_QC_REVIEW.md`.
