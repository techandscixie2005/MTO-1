# Existing auxiliary-label inventory

2026-09-30, bounded read-only metadata/source review. No array shard, molecule record, raw log, train/validation/test label, or new dataset was loaded. No new extraction or fitting was performed.

## Verdict

The published TD extraction already answers the availability question. It retains optional **velocity-gauge transition dipoles/strengths and magnetic transition dipoles**. Ground-state total energy, permanent dipole, orbital energies, HOMO/LUMO gaps, MO/AO integrals, full X/Y, transition densities and NTOs are **not fields in this extraction**. Their existence in the unavailable original logs is not established by this audit.

| Field | Existing location | Training-supervision status |
|---|---|---|
| Velocity transition dipole and printed velocity oscillator strength | NPZ schema `velocity_dipole_au`, `oscillator_strength_velocity`; JSONL also retains `velocity_dipole_strength_au2` | Potential auxiliary targets after a train-only coverage/unit/phase/convention audit. Current primary masks do not certify these optional fields. |
| Magnetic transition dipole | NPZ/JSONL `magnetic_dipole_au` | Potential auxiliary target after coverage, units, convention, electronic phase and axial-parity checks. No training-readiness claim. |
| Printed excited-state spin squared and symmetry text | JSONL only `spin_squared`, `symmetry_printed` | Metadata, not currently materialized NPZ targets. Manifest reports all successful roots as `Singlet-?Sym`; no useful verified spatial irrep labels. Spin-squared coverage/range not summarized. |
| Charge and ground-state multiplicity | NPZ/JSONL | All 133727 successful records have `(0,1)`, so the published inventory has no target variation. |
| Wavelength, electric-table strength, f reconstructed from E and mu | NPZ/JSONL | Retained consistency/rounding information derived from or redundant with current excitation labels, not independent electronic structure. Keep printed raw-f primary. |
| Ground-state energies/permanent dipoles/orbital energies/gaps | Not parsed by the existing extractor | Unavailable as audited training targets. Do not assume a standard QM9 label table is present or aligned just from dataset naming. |

## Provenance and limits

Authoritative extraction directory: `/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925/`. Sources inspected: `README.md`, `dataset_manifest.json`, `validation_report.json`, `source_spotcheck_report.json`, and `extract_qm9s.py`.

- Extractor lines64–65 parse velocity/magnetic tables; lines125–127 store optional fields. Lines129–130 define scalar and vector masks using E/wavelength/length-f and length-dipole only. Lines169–195 define NPZ fields. There is no separate optional velocity/magnetic validity mask.
- The parser counts velocity blocks but does not use that count for its single-calculation mask; it discards the magnetic block count. Electric root-set consistency is checked, but equivalent optional-table root-set/duplication checks are not shown. A later training-only audit must resolve these issues before trusting optional labels.
- Full published verification says JSONL/NPZ equality and `A=mu mu^T` equality for 1,337,270 states. The source spotcheck reports108 logs/810 labeled states. These receipts do not establish a train-only optional-field coverage rate, length/velocity physical agreement, or full-TDDFT amplitude reconstruction.
- The manifest reports route `#p td=(nstates=10) b3lyp/TZVP nosymm geom=check guess=read`, Gaussian16B.01 and DoRPA=T for successful calculations. Complete X/Y vectors and their normalization are not supplied by the extraction. Existing response audit explains why X-only NTOs cannot substitute for full X/Y reconstruction.
- A bounded directory inventory found only `QM9S/qm9s_td_extracted_20260925` under `/home/inspur/datasets`, and no `/home/inspur/MTO-1/data` directory. This is not a claim about every filesystem path or external source.

Any later auxiliary study must join IDs to the **existing frozen TRAIN split**, audit optional-field validity and geometry conventions there, retain all current primary target labels, and use geometry/atomic numbers alone at inference. Do not use validation/test auxiliary labels for pretraining. Gauge-invariant targets can avoid arbitrary whole-state sign, but their usefulness and transformation rules still need validation. No auxiliary objective is selected by this inventory.

Existing detailed phase/full-TDDFT limitations: `research_state/response_operator_candidate.md`; response-head counterexamples: `research_state/response_feasibility/VERDICT.md`. Existing inventory suffices; this audit stops here.
