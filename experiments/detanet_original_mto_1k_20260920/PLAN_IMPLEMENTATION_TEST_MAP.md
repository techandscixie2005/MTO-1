# Planning → formula → implementation → executable acceptance

The **complete** v2.0 2026-09-10 general plan and the mathematical summary were obtained from `D:\MTO大纲` and preserved under `planning/` with SHA256 values in the sync manifest. This is not an abstract-only audit. The relevant construction, readout, data and acceptance sections were read directly. `planning/EXPERIMENT_PLAN.md` is the prior full-architecture experimental plan; its 32-channel backbone, RMS, parameter-matched physical baseline and multi-scale chain are superseded by the latest user request.

This run covers the requested 1k comparison, not the complete future chemistry program, independent electronic-structure validation, OOD evaluation, or every ablation in the general plan. The latest request also overrides the plan's earlier label-mode branch: use the tensor factor decoder with spectrum-only supervision, with no claim of identifiable tensor orientation.

| Planning clause / latest requirement | Formula / contract | File and function | Executable test or diagnostic |
|---|---|---|---|
| §4.1; restore original backbone | 128 features, 3 blocks, 0e/1o/2e/3o; exact author operators | models.UV, CompatibleDetaNet, make_core; vendor/detanet_model | acceptance.test_01_reference_independent_layers_and_backward; test_02_latent_packing_and_initialization |
| Latest A definition | sum_i MLP_original(S_i), 240 bins, unconstrained output | SpectrumModel.forward A branch, original DetaNet.forward and MLP | reference per-atom and final-output hooks; strict official checkpoint loading |
| §3.2 channel projection | B_ikm^t = sum_b W_kb^t H_ibm^t | MolecularTensorOrbitals.project/forward | test_04: independent einsum, no nonscalar bias; test_02 packing roundtrip |
| §4.2 local invariants | u_i = [H_i^0e, mean_m((H_i^1o)^2), mean_m((H_i^2e)^2)] | invariants | test_05 and test_06 O(3) |
| §4.2 global invariant | g = [mean_i u_i, log(1+N), element counts] per molecule | MolecularTensorOrbitals.forward, pool | test_05 batch composition and permutation |
| §3.2, §4.2 signed routing | c_aik^t=tanh(phi_k^t(u_i,g,q_a)); 11 queries | MolecularTensorOrbitals.router/query | test_04 state/atom variation and saturation; test_05 c invariance |
| §3.2, §4.3 fixed pooling | F=cB; M=N_ref^(-1/2) sum_i F, N_ref=train median | MolecularTensorOrbitals.forward; data_protocol.prepare | test_04 exact F/M identities; data/frozen.json and saved n_ref buffer |
| Latest diagnostic C | replace u_i only with mean_j(u_j), broadcast | global_gate boolean in the same MolecularTensorOrbitals class | test_02 full state_dict equality; test_04 constant atom gates/nonconstant states |
| §4.4 reference/target relation | P_0a=TP(M_0,M_a), 0e and 2e, seven allowed paths | ReferenceStateCoupling | test_04 explicit path set and 7*16^3 weights; O(3) tests |
| §4.4 scalar readout | [Inv(M0),Inv(Ma),P^0e] | ReferenceStateCoupling.forward | test_06; spectrum-loss gradient test_07 |
| §4.4 tensor readout | invariant gates on [P^2e,Ma^2e] | PhysicalSpectrumDecoder.tensor_gate | test_06 covariance, test_07 tensor-gate and CG gradients |
| §4.5 positive E with eV scale | E=1 eV * softplus(h_E) | PhysicalSpectrumDecoder.forward | test_06 positivity and conversion; test_07 E gradient |
| §4.5 symmetric factor | C=beta I/sqrt(3)+Q; A=CC^T | PhysicalSpectrumDecoder; ReducedTensorProducts-derived basis | test_06 basis orthonormality, Q trace/symmetry, A PSD/covariance, trace identity |
| §4.5 physical intensity | f=(2/3)(E/27.211386245988)tr(A) | PhysicalSpectrumDecoder.forward | test_06 units/nonnegativity; test_07 E/A/f gradients |
| §4.6 unit-integral spectrum | sum_a f_a exp(-(x-E_a)^2/(2sigma^2))/(sigma sqrt(2pi)) | PhysicalSpectrumDecoder.broaden | test_06 wide-domain integral and finite-window CDF; decoder-oracle diagnostics |
| Latest pure spectral loss | MSE((prediction-target)/train_RMS) | train.update, train.evaluate | test_07 common loss; preflight.smoke; labels contain only spectra |
| §18 symmetry/engineering | rotations, reflection, translation, permutation, batching | tests/acceptance.py | actual per-check max absolute/relative errors in logs |
| Latest differentiability | finite differences through projection/router/CG/decoder | acceptance.test_07_gradients_and_gradcheck | composed-module torch.autograd.gradcheck and actual full-model gradients |
| Latest kernel capacity diagnostic | directly optimize 10 E/f, only fixed train IDs | preflight.fit_free_peaks, oracle | reports/decoder_oracle.json; distinct synthetic test, no global-bound claim |
| §18.4; latest smoke gate | fixed 32 training molecules, ≤600 steps, best <0.1 and <0.3 initial | preflight.smoke | reports/smoke_seed11.json; blocks formal training on failure |
| Latest initialization/data order | torch.equal common backbone; full B/C copy; independent shuffle RNG | build_triplet, train_one | test_02; actual trainer resume audit |
| Latest resume | model/optimizer/scheduler/RNG/order/cursor/fingerprint | checkpoint, restore, atomic_save | preflight.cpu_gpu_and_resume; fingerprint mismatch raises |
| Latest evaluation gate | finish all 15 fits before unified test; validation-selected best | report.main | all completion/fingerprint checks before test loading/prediction |

No direct-spectrum residual branch, dual head, latent detachment, state-label auxiliary loss, atom softmax, per-molecule gamma, tensor-component MLP, hand-filled Cartesian basis or test-set tuning is introduced. The spectral energy offsets and invariant LayerNorm are documented readout implementation choices rather than new physical observables.
