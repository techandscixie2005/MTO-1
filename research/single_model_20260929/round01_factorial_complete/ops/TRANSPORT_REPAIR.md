# Publication transport repair

The first preparation attempt downloaded and verified the initial round bundle before Git staging. Direct SSH22, SSH443 and HTTPS443 to GitHub failed. Read-only HTTPS ref access succeeded through the already enabled Windows system proxy127.0.0.1:7890. No global or repository configuration was changed.

The partial publication clone attempted to fetch missing historical blobs during `write-tree`. Scoped HTTPS URL rewriting fixes child-fetch transport, but fetching historical datasets/model artifacts is unnecessary. The revised helper uses `write-tree --missing-ok` only to preserve unchanged historical base objects. Every new source/record blob is created locally from a hash-verified downloaded record. Diff paths must be limited to the explicit allowlist under this campaign. Existing parent69bf39f is verified on GitHub. A separate index preserves the unborn worktree HEAD and its untracked files. Cancelled index lock files were renamed and retained after confirming no publication Git process remained.

The helper revision is separately downloaded to the D: archive before use. The initial round archive remains immutable. Publication uses non-force push and verifies the remote branch commit. No credentials are displayed, committed, or copied; existing credential-manager or SSH credentials are used only through normal authentication.
