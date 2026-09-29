# Round02 source restoration

ROUND02_ARCHIVE_PREPARATION.json maps each exact archived source/config/statistics file to its original absolute server path and SHA256. Restore this layout and verify hashes against round02_frozen/FROZEN_MANIFEST.json before reproducing. data_metadata maps back to data and contains only statistics/hashes. No raw data, split membership or weights are included.

Provide separately preserved server data, frozen split and starting checkpoint identified by manifest hashes. Do not regenerate splits. CACHE_COMPLETE.json records reproducible TRAIN cache construction, checksums and parity; recreate cache with prepare_cache.py when needed. Cache arrays remain server-only. ENVIRONMENT.json records the pinned interpreter/packages.

The root PROTOCOL.md and source files are frozen parent dependencies. Round02 objectives, commands and launch gates are in round02_frozen/PROTOCOL.md and config.json. The exact independent review and all initial/preflight logs are preserved.
