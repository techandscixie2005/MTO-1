# Target-blind v2 split — current handoff

Updated 2026-09-30. Round04/main publication is complete at 8497e0ba10efc3226197c894f46507acdad8ec54. Dataset work resumed under root's accepted protocol; no fitting or test scoring is authorized here.

Server namespace: `/home/inspur/MTO-1/research/single_model_20260929/dataset_audit_20260930`. Local mirror: `research_state/dataset_audit_20260930`.

Builder/settings/core/synthetic checks are implemented. Scientific source manifest SHA256 `1315f7fcb6112c893ad0855b2144af7cf4711baadd7e6438ae5f96075d8dd042`. Twenty-one synthetic checks passed; receipt SHA256 `1ca088f8db35f63c4d46eb3cd98e6fc86bfb6bc46cae548f7cd6cf11856de9a9`. Science's `PRE_CORPUS_REVIEW.json` is PASS, SHA256 `f64de41f384536d0bc1510f3f75fe59f87881d2b76b0bf05b77db7902fc03a45`. Operational wrapper source also reviewed PASS. Await exact aggregate approval `MATERIALIZATION_REVIEW.json` before `materialize`.

Corpus audit is COMPLETE, exit0 in127.81seconds. Registered wrapper PID1286793/start1260315910 and child PID1286795/start1260315920 are historical identities (boot793423dc-86b6-4fb9-9a9d-e74cfa8135a3). Authoritative command/log/terminal receipt are in `ops/audit_attempt`. CORPUS_AUDIT.json SHA256 `0419a5a856e084ad4e75cc67113d3dd290adf7826d235cedc9d2cf41df78dc80`. Do not relaunch this completed audit.

Planned counts TRAIN120355 / VALID6686 / TEST6686, no boundary overshoot; all133727 rows retained in114572 conservative components, largest14. All278 ambiguous rows are in263 TRAIN-only components. All row/ID/component/original-key/heavy-formula-key/near-geometry cross-partition counts are0. Eighty-three geometry matches were already within graph groups. No candidate failure. Only ids/z/pos decoded. Proposed TEST exposure is5989 oldTRAIN+361oldVAL+336oldTEST; proposed VALID6047+321+318. These are disclosure counts, not assignment inputs. No private v2 partition arrays exist yet; awaiting science's exact aggregate approval for materialization.

Operational `run_owned_builder.py` is outside the scientific closure. Launch with empty CUDA visibility, OMP/MKL two threads before Python. It registers wrapper identity with the existing CPU monitor, saves child identity/command/log and terminal receipt under `ops/{audit,materialize}_attempt`, and refuses duplicate stage outputs/attempts. Inspect its process identity and receipts after interruption before any action; never blindly relaunch.

Expected private outputs: `private_partition/{train_indices,val_indices,test_indices,component_index,train_only_mask,ids}.npy` and `identity_components.json`. They must never be archived. `CORPUS_AUDIT.json` and `SPLIT_MANIFEST.json` expose only paths/hashes/counts and old-exposure intersections. Independent science verification after materialization must reconstruct/check original/full and heavy/formula keys plus geometry links, without targets.

External data and QC findings remain in `PUBLIC_DATA_AND_QC_REVIEW.md`; no compatible independent external holdout is verified. This v2 split is a new partition of historically used data. Fresh initialization and v2 TRAIN-only target statistics remain required before the separately reviewed Round05 pilot.
