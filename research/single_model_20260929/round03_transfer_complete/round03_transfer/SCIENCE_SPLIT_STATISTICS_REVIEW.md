# Independent implementation review of the split and statistics

Reviewer: science_implementation. Author of prepare_split_statistics.py and its metadata audit: history_baseline. This review is separate from history's review of the fitting code.

Result: PASS for the frozen Round03 source protocol. I read prepare_split_statistics.py, grouped_split in clean_source_feasibility/audit_clean_source.py, the selected_rows streaming helper, split_manifest.json, fit_normalization.json and STATISTICS_AUDIT.json. The real-data source preflight also verified every decoded row against the source-fit index array and checked the original loss and gradients bitwise.

The split groups original TRAIN indices by audited identity key before shuffling sorted resolved keys with NumPy default_rng20260930. Any unresolved member marks its entire group source-fit-only. A whole-group prefix reserves24,071 rows; both output arrays preserve original TRAIN order. The generator checks disjointness, complete TRAIN assignment and group integrity. Its identity/dataset/split/label hashes are pinned. The resulting96,284/24,071 counts and recorded index-byte hashes match the coordinator's metadata-only dry run. Labels do not enter selection.

The statistics reader passes sorted source-fit indices to selected_rows. It interprets only those rows of raw E/A and their masks. Passing compressed bytes for other rows while seeking does not decode or retain them as labels. It checks IDs against metadata, all962,840 E/A masks and finiteness. No f or held-out target is decoded. Energy normalization is the population mean squared deviation from each source-fit state mean; A normalization is the mean squared Frobenius norm. The independent second-moment/block identities in the author audit agree within1e-11/1e-12. n_ref is the median source-fit atom count18.

Values sE2=0.5384852554041385 and sA2=0.13127702708928576 differ appropriately from full-TRAIN values. Only the source-fit E_state_mean initializes energy offsets. The source uses original FP32 curated E/A for optimization and FP64 raw-label normalization, preserving the original convention. The numerical source preflight confirms the clean model's softplus energy offsets equal these supplied state means and that construction opens no learned checkpoint or full-training normalization file.

This is a source-code/data-access and supplied numerical-audit review, not a second independent recomputation of all raw statistics. The original outer validation has prior exposure; the internal source split does not create a fresh outer holdout.

Reviewed execution manifest:93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37. This review does not change its sources or numerical settings.
