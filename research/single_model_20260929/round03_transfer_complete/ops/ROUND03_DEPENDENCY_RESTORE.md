# Round03 source restoration

ROUND03_ARCHIVE_PREPARATION.json maps every archived executable/config/statistics dependency to its original absolute path and SHA256. Restore that layout and verify against round03_transfer/FROZEN_MANIFEST.json. data_metadata contains statistics/hashes only. The exact runtime/package versions are recorded in ENVIRONMENT.json.

Provide separately preserved raw datasets, identity metadata and original outer splits whose paths and hashes are recorded. The internal TRAIN split generator must reproduce the sealed fit/calibration identities; do not regenerate the outer split. No identity membership, index arrays, model or optimizer data is in this archive.

prepare_split_statistics.py regenerates the allowed internal partition and fit-only normalization under the recorded policy. The fixed33 clean source uses random initialization; the existing eta0 checkpoint is needed only by the separate frozen affine comparison. All commands and exact publication/review/admission gates are in round03_transfer/PROTOCOL.md.

Persistent monitor/register code accepts arbitrary completed_epoch fields and train-only status/log progress; no validation fields or20epoch assumption are required.
