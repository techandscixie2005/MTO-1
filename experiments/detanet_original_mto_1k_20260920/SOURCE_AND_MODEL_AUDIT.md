# Original DetaNet / planned MTO / global-gate 1k audit

Workspace: `/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920`, inside the verified requested `/data/run01/sczc698/xxy/MTO`. The historical workspace has no `.git`; the original Git SHA, branch and Git diff are unavailable, not invented. `audit/prechange_git.json` records the actual failed Git commands. `audit/prechange_source.tar.gz` and `audit/prechange_sha256.json` preserve the existing source/configuration. Existing jobs and historical results were not cancelled or replaced. No applicable AGENTS.md was found in the inspected ancestry or target workspace.

## Source and licensing

The author source is fixed to [WeiHuQLU/DetaNet commit 4f92e643ab64651b91c4a1392cf389ddfd0d89f0](https://github.com/WeiHuQLU/DetaNet/tree/4f92e643ab64651b91c4a1392cf389ddfd0d89f0). Selected source files, the two example notebooks, paper PDFs, MIT license and the UV checkpoint were downloaded individually. Repository history and unrelated checkpoints were not downloaded. SHA256 values are in `audit/reference_sha256.json`.

[Paper archive DOI](https://doi.org/10.24433/CO.5808137.v3) could not be obtained: browser access failed and an independent HTTPS request returned HTTP 403. This GitHub version has **not** been verified byte-for-byte against the paper archive. The paper and supplement were read from the author's checked-in PDFs; the training-hyperparameter page and supplementary UV formula page were rendered and visually inspected.

`reference/detanet_model` is untouched reference source, imported through its own Python package namespace. `vendor/detanet_model` is a distinct byte-identical source copy. The adapter subclasses the vendor class and makes only `Embedding.elec` and `mass` nonpersistent buffers for device/dtype migration. Their values and checkpoint keys are unchanged. Reference constants are manually migrated in the independent test harness. All weight loading is strict. No algorithm patch is currently applied.

## Constructor and operator audit

`models.UV` contains the exact `uv_model` parameters: 128 features, lmax 3, 3 blocks, trainable Bessel/32 radial features, 8 attention heads, rc 5 Å, learnable swish, dropout 0, no additional cutoff, maximum atomic number 9, no atom reference, scale 1, 240 scalar outputs, no tensor output, atomic sum, norm false and no derivative output.

| Operator | Verified implementation |
|---|---|
| Embedding | Nuclear embedding plus the author's normalized static electronic table, Linear and learnable Swish; original initialization |
| Radial | Original trainable Bessel alpha initially 0..31, beta initially 2, prefactor sqrt(2/5); original 5 Å radius |
| Geometry | Original `i,j=edge_index`, `rij=pos[j]-pos[i]`; SH normalization `component`, normalized directions |
| Attention | Original per-edge reshape `[edge,8,features/8]`, head-by-head matrix, softmax on final dimension; no replacement attention |
| Tensor products | Original scalar-to-1o/2e/3o message paths and original featurewise update products |
| Update | Original scatter to `index[1]` and both residual additions; no RMS added |
| A readout | Original per-atom `128→128→240` MLP; original biases, Xavier initialization, intermediate learnable Swish, no last activation; then atomic sum |
| B/C interface | Built-in latent mode, scalar_outsize 0, summation false, scale None; exact S/T comparison against A's last block; no detach or extra layer |
| MTO interface | Select only 0e/1o/2e using actual irrep metadata; keep 3o inside every backbone block |

The original cutoff helper is suspicious, but is disabled in this constructor and remains untouched. The original scalar/latent path fails for an isolated single atom because an empty attention tensor cannot be reshaped with an inferred dimension. This was reproduced and recorded as upstream behavior, not silently fixed. The 1k audit checks that every atom has a neighbor and that the default 32-neighbor cap does not bind. The code's default graph cap, direction, self-loop exclusion and radius-boundary behavior are tested separately from fixed-edge operator equivalence.

GPU attempt 1298912 failed an initially incorrect test assumption that a requested cap of 32 always yields at most 32 neighbors. Inspection of the installed Python wrapper and the [torch_cluster 1.6.3 CUDA source](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cuda/radius_cuda.cu) explains the behavior: the wrapper requests 33 candidates before removing self-loops, while CUDA scans indices in order. In the dense 40-atom test, atoms 0..32 have 32 neighbors and atoms 33..39 retain 33. The corrected test asserts this exact degree sequence and exact edge set, rather than relaxing a numerical tolerance or changing the model. It additionally compares CPU/GPU edge sets for all 1000 actual molecular geometries. The initial failing log and test source are preserved in `logs/attempt_1298912/`.

Attempt 1298941 passed GPU float64 but exposed a separate FP32 test-harness issue: `torch.cdist` uses a GEMM path and may produce a nonzero diagonal. Treating `d>0` as self-loop exclusion incorrectly added five self edges in the independent reference mask. The direct coordinate-difference reference with explicit index-based diagonal exclusion fixed this without touching model construction. The next GPU run recorded a maximum spurious diagonal distance of 0.0013810679 Å, with differences only at indices (0,0), (8,8), (26,26), (32,32), (34,34). Every actual 1k CPU/GPU edge set matched; maximum degree was 26. All 7 GPU float64 and all 7 GPU float32 model tests then passed. Historical failures remain preserved.

Attempt 1298945 subsequently stopped during the engineering resume check: comparing optimizer moments after another GPU backward pass failed at `exp_avg`. CUDA scatter uses atomic additions and is not bitwise deterministic. The revised audit distinguishes exact disk-restored model/optimizer/scheduler state from subsequent numerical evolution: it requires the loaded tensors to match exactly, retains the original next-step model/loss tolerance, records gradient differences, and checks each Adam moment update against its actual gradient with an eight-machine-epsilon arithmetic-rounding bound. Multi-step mid-epoch training/order/RNG replay is additionally tested exactly on CPU within the allocated job. This does not promise bitwise-identical long GPU trajectories. The failed log is retained; revised engineering status is reported separately, never inferred from model-test success.

## Numerical evidence

Initial CPU float64 suite: 7 tests, 19.062 seconds, exit code 0. Independent random-weight and official-UV-weight runs both strictly loaded. Embedding, radial, all three block S/T outputs, per-atom MLP, final spectra and common parameter gradients had maximum absolute difference **0**. Official checkpoint use is restricted to this audit; every new training run constructs fresh parameters.

CPU real-molecule O(3) maximum spectrum absolute difference was 5.3291e-15; H 8.8818e-16, M 1.3878e-16, CG 3.9031e-18. Full per-check absolute/relative errors, dtype thresholds and gradient norms are recorded in `logs/acceptance_cpu_float64.json`. GPU results must be read from their own logs; CPU success is not GPU success. Final status is recorded separately in `RUN_STATUS.md`.

| Model | Total parameters | Common backbone | Readout | Parameters with gradients in measured CPU batch |
|---|---:|---:|---:|---:|
| A: detanet_original_uv | 1,419,568 | 1,371,840 | 47,728 | 1,370,032 |
| B: detanet_mto_planned | 1,552,092 | 1,371,840 | 180,252 | 1,552,092 |
| C: detanet_mto_global_gate | 1,552,092 | 1,371,840 | 180,252 | 1,552,092 |

The active count is the observed count with non-None gradients, not a claim that each parameter must have a nonzero gradient for every sample. A retains upstream tensor computations whose final outputs do not reach its scalar readout. No parameters were removed or added to force a parameter match. Every common initialization tensor is checked using torch.equal. B and C have identical full initial state dictionaries and parameter keys.

## Grid, broadening and training deviations

The author's calculation notebook cell 16 uses `torch.linspace(1.5,13.5,240)`. The spacing is 12/239 eV, despite the supplement describing 0.05 eV intervals. The actual FP32 notebook coordinates are preserved as the common target grid. The available 1k source spectra have 601 numeric CSV-header coordinates from 1.5 to 13.5 eV. They are preserved and converted with fixed piecewise-linear `numpy.interp`, without extrapolation. The mean and maximum 601→240→601 interpolation MSE are 2.28707e-9 and 7.48528e-8. No interpolation method was selected on the test set.

**Original model code/architecture equivalence, with target data grid adaptation. This is not a complete reproduction of the paper experiment.** The retained historical 1k identities and test set are not described as previously untouched. The decompressed source payload SHA, identity uniqueness, split intersections, shapes, finite coordinates/labels, distances and neighbor counts are checked. Geometry units inherit the QM9S/author Å convention; no coordinate conversion is applied. Earlier row alignment is preserved and explicitly distinguished from the new checks.

The subsequent independent native-data audit decoded only `z`, `pos` and `number` for these 1000 selected source rows and compared their original SMILES. Every value matched the preserved 1k sample exactly. All selected native record schemas were checked: no native UV/broadened-spectrum field was present. See `audit/native_1k_alignment.json`. No full-dataset tensors or statistics were recomputed.

The source directories' 11 notebooks were scanned as code, without executing them; matching cells and file hashes are in `audit/source_notebook_cells.json`. Their available UV references read the existing CSV; no provenance-confirmed UV CSV broadening generator was found. The original coordinate/identity alignment is therefore verified independently, while the source broadening remains explicitly unknown.

The supplementary PDF prints a unit-integral Gaussian with `c=F/(2 sqrt(ln 2))`, calling F=0.5 eV a half-width. The GitHub implementation instead computes `ci=sigma/2*sqrt(2 ln 2)` and divides by `ci*2*pi`; that kernel has area approximately 1/sqrt(2*pi), not one. These are different formulas; neither is silently rewritten as a standard deviation of 0.5 eV. The installed source dataset has transition energies and dipoles but no independent oscillator-strength or broadened-spectrum field in the inspected native record. A provenance-confirmed CSV generation program has not been obtained. Therefore `source_broadening_status=unknown`; no candidate fitted width is promoted to provenance.

B/C use the planned unit-integral Gaussian with sigma=0.2 eV as an explicitly disclosed initial assumption. No unknown unit/area conversion is applied. N_ref=18 and RMS=0.024564612974391966 are computed only on the adapted 800 training records. Label scaling occurs in the common loss adapter, never inside A. The tensor readout LayerNorm acts only on invariants and is retained from the prior full-architecture implementation; the backbone's former atom/type RMS is absent. Small nonzero router/output initializations and per-state energy offsets remain engineering choices; the explicit initial energy range is 2–13 eV in `protocol.json`, not a claimed paper hyperparameter.

The paper's training page describes Adam+AMSGrad, batch 64, initial lr=1e-3, loss checks every 50 epochs and halving on a plateau; its maximum is 1,000,000 epochs. The notebook trainer defaults to Adam_amsgrad lr=5e-4, while its executed example selects AdamW lr=1e-5; supplementary example text mentions 5e-4/AdamW. These descriptions are not identical. This experiment uses the user's new 1k protocol: five seeds, Adam/AMSGrad, no weight decay/AMP/TF32, validation ReduceLROnPlateau and the specified 200/1000 epoch and early-stop gates. It is not the paper's full-data training protocol.

## Interpretation

A↔B is a whole-model comparison of direct spectra versus planned MTO plus physical decoding. B↔C isolates atom-dependent selection under the same queries, projections, CG and decoder. Neither comparison alone measures every contribution of the whole MTO+CG system. Isotropic spectral supervision cannot uniquely recover real states, orbitals or tensor directions. Unfavorable results are retained.
