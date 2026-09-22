# Archive provenance

The complete successful seed-11 run was copied byte-for-byte from the cluster on 2026-09-22. `remote_snapshot_manifest.json` records all 145 source files, their SHA-256 hashes, sizes, and the verified completion of all six fits. Git internals, Python bytecode and transient lock files were excluded.

`README.md` is the publication entry point. The original cluster README is preserved verbatim as `publication/RUNNING_ON_CLUSTER.md`; all frozen Python, configuration, Slurm scripts, split files, training checkpoints and completion markers retain their original bytes. Source commit on the cluster: `20b4ab6` (full ID in the snapshot manifest).

The large prepared `data/spectra.npy` cache is represented by ordered, lossless gzip chunks listed in `storage_map.json`. These reconstruct the exact original byte stream and match the original frozen data hash. Every other prepared data array is included directly. All 12 best/last checkpoints, including optimizer/scheduler/RNG state, are included directly as ordinary Git blobs. No Git LFS download or external checkpoint link is required.

The external upstream 601-point NPZ / original raw QM9S files are not copied from their separate historical workspace. Their source paths and hashes are preserved in the frozen data metadata, and the complete prepared 240-point data actually consumed by this run is included. The original author reference and provenance also remain in the repository's earlier experiment archive.

`audit/analysis_1k/` and `audit/utf8_recovery_20260921/` preserve dated intermediate analysis and the failed report's partial outputs. Their statements about a pending report describe that earlier snapshot, not the final run status. Current final metrics are under `reports/`; slight last-digit differences between preserved first-pass test metrics and the successful GPU report rerun are retained rather than silently rewritten. No training was repeated for that report repair.

The additional `analysis/spectrum_comparison/` files contain the four fixed common test-molecule plots requested after all stages completed. They were generated from saved predictions; no smoothing, new training, or test-driven checkpoint selection was introduced.

The frozen Slurm entry points preserve the original cluster paths and Python environment. Moving the archive to another machine is not an automatic cluster-job migration: retain the frozen evidence, create a separate runnable configuration if retraining elsewhere, and do not fabricate or rewrite fingerprint/completion records.
