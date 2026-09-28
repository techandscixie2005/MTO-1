# ID14562: bounded source/provenance check

Date: 2026-09-29. Addendum to FINAL20_REPLAY_SCIENTIFIC_INTERPRETATION.md; that file is preserved unchanged. No model inference, training, test-label use, case removal or benchmark change.

## Result

The available per-molecule extraction record explicitly contains ten observed state labels with oscillator_strength=0.0 and ten matching oscillator_strength_table=0.0 fields. The inspected extraction code requires an explicit numeric f= token in each Gaussian Excited State line; it does not invent a zero when that token is absent. Missing array scalar fields initialize to NaN and masks false, not zero. The zero labels are therefore supported by the preserved extraction record and its parsing path, rather than a missing-data zero-fill convention.

This is not a fresh byte-level inspection of the original Gaussian log. The source manifest points to historical E:\DATA\QM9S\tdlog1-20000\014562-td.log. The exact path is unavailable on the current Windows host (read-only Test-Path=false), and the inspected remote dataset directory contains the extraction artifacts rather than raw logs. No wider filesystem/chemistry search was undertaken. The original log hash is preserved in both the molecule record and source manifest, but could not be freshly recomputed here.

## Available calculation/label evidence

- Source filename014562-td.log,45081 bytes; recorded source SHA2566a6691ad4f42f01dfb9d18a0f812126c316882760e1d190077edc92de7def8ba.
- One normal termination, zero error terminations; normal_termination=true and response_convergence_reported=true.
- unambiguous_single_calculation=true; geometry_present=true; one electric-dipole table and one preceding geometry; warnings=[]; all states have unique root indices and scalar/vector fields present.
- All ten state labels are Singlet-?Sym, each with printed spin_squared0.0. Ground calculation charge0 and multiplicity1. These fields do not supply a specific spatial-symmetry assignment.
- Route: #p td=(nstates=10) b3lyp/TZVP nosymm geom=check guess=read; DoRPA=T; Gaussian ES64L-G16RevB.01 20-Dec-2017; formulaC8H2; ten atoms.
- Electric-table oscillator strengths and velocity-table oscillator strengths are zero in every state. Higher-precision electronic-transition dipoles are small but nonzero, and were checked by the extraction against the rounded electric table. E/dipole-derived f ranges from4.72e-13 to1.31122e-7; S7 is1.31122e-7 and S8 is2.72283e-9. These support printed zeros at1e-4 precision, not mathematically exact zero transitions. They are many orders below the model's S7/S8 predictions.

The positive convergence/validity flags and consistent tiny dipoles provide no indication of a parser default, absent label or recorded failed calculation for this case. They do not independently certify the electronic-structure method or the unpublished raw log. Retain the original raw labels and masks; no exception or exclusion is justified.

## Narrow provenance

Root: /home/inspur/datasets/QM9S/qm9s_td_extracted_20260925

Target extraction: records/part-00002.jsonl.gz, line4562 (one-based), molecule_id14562. Only this target line was decoded as a scientific record; surrounding compressed bytes were streamed without interpreting other molecule labels. The shard manifest spans IDs10001-15000.

SHA256:

- records/part-00002.jsonl.gz:20a6a2cdf28bf74bd00b16ea3485c130d63338bf6e6ee2b6ca492ecaa8926dd8
- Target decompressed record line, UTF-8 including newline:b87e2a9f9e53b6e5a6cedc9e73916515634ad3644c8ba145c9c363b9c5279d87
- extract_qm9s.py:71ddb1625f6aa5e73e9c38b738b23e80d0100303049804d60219835eda86fa9b
- source_manifest.jsonl:47514c25411c0cccb0ac79b17903c263ba14ecdecbc04fd1b7fcfda954a53510
- dataset_manifest.json:fc848263753bb8311a07bdf39299193f8b6fb3ff0f35fef329c81a8255580959

The target source-manifest entry was read directly by its ID-ordered line and matched the extracted source filename/size/hash. Parser checks covered required STATE f token, scalar presence/root masks, dipole-table agreement, normal/convergence flags and NaN initialization. No source re-extraction or QC calculation was run. The accepted frozen-head protocol remains unchanged.