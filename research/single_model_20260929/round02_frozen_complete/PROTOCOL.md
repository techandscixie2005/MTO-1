# Single-model right-adapter and raw-state decorrelation pilot

Frozen before fitting. This round continues one eta0 checkpoint in four independent arms. Predictions are never averaged. The data split and every valid raw-f label stay fixed. The exposed historical test is not evaluated.

## Rationale and baseline

The strongest verified native MTO baseline is eta0 seed11 epoch33, validation pooled raw-f R2 0.4052941183410983. Its original selection used LE+Ls; the checkpoint hash is in round_config.json. Earlier 20-epoch whole-model continuation at learning rate 1e-4, post-decoder residuals, and changed intensity losses failed to improve meaningfully. This pilot changes the pre-CG right representation and uses a tenfold smaller matched learning rate. It does not repeat those post-decoder or loss variants. A calibration wrapper may be a stronger eligible single-model deployment benchmark; report that separately without changing the factorial source.

The backbone, raw M construction, tensor product weights, original invariant skips, raw 2e skip, and PSD decoder are unchanged at initialization. CG outputs 0e and 2e features and the decoder predicts C C-transpose. It is not an explicit physical transition-dipole vector model.

## Factorial arms and exact definitions

| Arm | Shared right adapter | Raw-state penalty coefficient |
|---|---|---:|
| control | disabled | 0 |
| adapter | enabled | 0 |
| decorrelation | disabled | 0.001 |
| both | enabled | 0.001 |

Only the tensor-product right operand Mk, k=1,...,10, is transformed. The left M0 and all original skip inputs remain raw. The same parameters apply to every k. For each irrep t, F_t(x)=x+W_t[tanh(g_t(u(x))) times x]. The invariant input concatenates the 0e channels and mean-square 1o/2e channel norms, compressed by signed log1p. LayerNorm is applied only to these invariant scalars; the gate MLP is 48-32-48 with SiLU. W_t is a bias-free 16x16 channel map shared across magnetic components and initialized to zero. Thus initialization is exactly the old function while the W_t gradient is nonzero. There are 4,016 extra trainable parameters in adapter arms; controls store frozen dormant adapter parameters for a shared checkpoint schema. This is not an active capacity-matched comparison. A positive adapter result would require a later simpler channel-map control before attributing benefit specifically to nonlinearity.

For each excited state concatenate [M(0e),M(1o)/sqrt(3),M(2e)/sqrt(5)], flattening channels and magnetic components. Divide by max(norm,1e-8), then average squared pairwise cosine over upper-triangle valid excited-state pairs. Validity is mask_E AND mask_A AND mask_f; invalid states are zeroed before arithmetic. M0 is excluded because the regularizer targets excited-query redundancy and imposing reference-state independence would add a separate assumption. Zero or one valid state gives graph-connected zero; zero norms stay included. This is a weak learned-feature diversity hypothesis, not wavefunction orthogonality and not a penalty on mu or A.

The coefficient 0.001 was fixed using four predetermined train-only batches of eight molecules (sample-seed20260929). Initial penalty0.397-0.478, weighted/base gradient norm ratio0.000369-0.001683; raw state norms1.82-9.46 are far above epsilon. No validation or test sweep selected this coefficient. Test numerics, input-index hash, and gradient values are in architecture/PREFLIGHT.json.

## Training, selection and comparison

All arms start from identical eta0 weights. Full-model Adam AMSGrad is reset, learning rate1e-5, weight decay0, batch64, gradient clipping5, FP32, AMP/TF32 off, 2CPU threads, fixed20epochs, seed/order11, no scheduler. The base objective remains LE+Ls with original normalization sE2 and 3*sA2. Construction RNG is isolated; runtime RNG resets explicitly and a separate NumPy generator controls matched batches. Each completed epoch records its sample-order hash.

Each arm selects one checkpoint by smallest pooled validation raw-f SSE, with epoch0 eligible and the earlier checkpoint retained on exact ties. R2 uses one global mean and float64 accumulation. Evaluation also reports per-state errors, energy errors, and bright-tail SSE/RMSE/MAE at training-only q90/q99 cutoffs frozen in FROZEN_MANIFEST.json. All valid f labels contribute to the primary metric; no subset drives selection. Epoch0 must reproduce the existing R2 within1e-5 (CUDA reduction tolerance), while CPU output identity was exact.

Before any pilot fit, a secondary comparison was prescribed: apply the same historical fixed map max(0,0.8511830211044088*f+0.003725185373211049) to each arm and report pooled/per-state/tail raw-f errors. No recalibration occurs and these metrics never select checkpoints. The fixed calibrated eta0 wrapper is an eligible single-checkpoint baseline (replayed validation R2 0.4181192453); its historical OOF estimate was0.41665766. Calibration coefficients were fit on the reused validation split and the map is not consistent with the original E/A identity. Its historical-test evidence is exposed. This secondary report keeps the stronger deployment benchmark visible while the factorial primary remains native f.

Prioritize a candidate only if its validation R2 improves by at least0.003 over both the matched control and epoch0. Smaller differences are inconclusive. Tail regressions and energy tradeoffs must be reported. Promotion requires paired control/candidate confirmation from genuinely independent starting training seeds23 and37 or from scratch; different continuation shuffles of the same source are not full training-seed replication. Keep every predictor single-checkpoint. No new untouched valid holdout is available in the current source; exposed historical-test evidence cannot become fresh confirmation.

## Resume, resources and archiving

last.pt atomically stores the full model, optimizer, all RNG states, independent order generator, completed epoch, selected epoch, and history at each epoch boundary. A process killed mid-epoch reruns that epoch from its last completed checkpoint. SIGTERM/SIGINT request an orderly stop after the current epoch. Each newly selected epoch is retained server-side so transaction recovery can restore the best corresponding to the durable state. A final selected-checkpoint replay verifies metrics and regenerates prediction arrays.

Launch uses GPUs1/2/4/6 only after checking their UUIDs, idle compute state, memory, uncorrected ECC and pending/failed remaps under shared /tmp/mto_pouter_gpu_INDEX.lock locks. GPU0 is unrelated;3/7 have uncorrected ECC;5 is reserve. PID/start-time/boot-ID/argv/UUID are registered with the dedicated persistent four-hour monitor. No unrelated process is signaled.

Code/settings/tests/review are frozen and downloaded to D:\MTO\archives before commit/push and launch. On completion, download lightweight logs/config/results/analysis/code first, then inspect staged files and push. Checkpoints, optimizer state, prediction arrays, raw data and caches remain server-only.

## Quantum-chemical label gate and future directions

The audited extraction contains E, raw f, transition dipoles, derived A=mu mu-transpose, velocity/magnetic dipoles, masks and metadata. It does not contain complete MO coefficients, AO dipole/overlap integrals, TDDFT X/Y, transition densities, or NTOs. Original logs were not copied to this server and the historical E: source directory is not currently mounted. The recorded Gaussian route is full TDDFT (DoRPA=T), B3LYP/TZVP, nosymm, ten singlets. All printed state labels are Singlet-?Sym; these are not useful symmetry classes. The E/mu reconstruction of f has mean absolute residual2.48655e-5 and maximum7.49881e-5, consistent with limited printed precision. This audit does not establish transition-amplitude reconstruction accuracy.

Physical extensions therefore need a separate audited label-generation step before supervision. For real full-TDDFT, reconstruct electric dipoles using both X and Y with the program's spin and normalization conventions; velocity moments use the corresponding antisymmetric combination. Test recovered mu and raw f including weak/bright states before using amplitudes. Phase/sign alignment and near-degenerate state-subspace comparisons must be handled explicitly. NTO factors have sign and degenerate-subspace gauge freedoms; supervise invariant/projector quantities or aligned subspaces, and keep signed coherent contributions until after summation. Inference inputs must remain atom identities and coordinates.

Primary reference check: [PySCF TDDFT documentation](https://pyscf.org/user/tddft.html) distinguishes full response from TDA. [Official restricted TDDFT source](https://pyscf.org/_modules/pyscf/tdscf/rhf.html) contracts length moments with X+Y, anti-Hermitian moments with X-Y, uses spin-dependent normalization, and explicitly documents that its basic get_nto routine ignores Y. These conventions must not be copied blindly into Gaussian labels.

A learned electron-hole factorization or shared response map can still be tested as a geometry-only hypothesis, but current supervision cannot identify it as a physical transition density. A polar1o coherent-sum/outer-product decoder was already tried in pouter campaigns; any revisit must preserve missing skip information or supply audited amplitude supervision, rather than merely repeat the same rank-one head. Shared operator and explicit-integral hypotheses should be staged after this factorial evidence, with reconstruction first and capacity-matched controls.
