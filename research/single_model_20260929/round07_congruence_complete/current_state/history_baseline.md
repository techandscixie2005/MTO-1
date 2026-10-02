# Independent research review — current handoff

## Round07 closeout review sealed; D-first publication pending

Root `ROUND07_DECISION.md` SHA df59af5856490c30e035170548b31e5e0b3c631908985d915f6b9ede5f69453b accepts the completed independent scientific review and closes the fixed study. No extension, bound/coefficient tuning, confirmation seed or TEST evaluation is authorized. Retain Round05 original control epoch45, validation pooled native raw-f R²0.44716940136585204; the0.60 goal is unmet. The single geometry checkpoint remains `round05_scratch_preparation/runs/control/geometry_best.pt`, SHA e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e.

| Arm | Selected epoch | Selected R² | Fixed60 R² |
|---|---:|---:|---:|
| Original |52|0.4091040813|0.3845332343|
| Scalar |49|0.4293390126|0.3937752150|
| Tensor |34|0.4187551766|0.3277766810|

The tensor passes the+.003 comparison with the current original but fails against scalar and retained. Scalar is the strongest Round07 model, still below retained. Selected tensor reduces false-bright SSE but worsens pooled MAE, energy errors and q99 RMSE versus both controls. All selected paired component-bootstrap intervals include zero; fixed60 tensor intervals are below both controls. These are conditional reused-validation summaries, not independent-seed/fresh-holdout evidence. Gate activity and latent q anisotropy do not identify physical transition directions. TRAIN diagnostics are pre-update minibatch trajectory aggregates; extrema do not establish saturation prevalence. Full tradeoffs and all ten states remain in the reports.

### Completed evidence — do not repeat

- Source seal:185 files, `FROZEN_MANIFEST.json`68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203. Preparation review e3ae0804; protocol ee22c436; report27e1e0c7. Nine discarded technical updates and all preparation checks were completed once; their records remain immutable.
- D-first preparation publication: both refs271d61f6dc9e21009c31d10467fe704e44b974ad, canonical receipt ce3f342916bb6ecb6a76516bc8da8b64fc1c9839f406dcd9e852c6c12185d6c0. All223 new blobs remotely verified. D archive `single_model_20260929/round07_congruence_preparation`; no weights/data uploaded.
- Separate production authority dfb6a5fe356e30da37260830f8032fe7892f2749bb0afc30b6daf277652afaaa authorized exact original/scalar/tensor60 fits. Original workers1591761/1591766/1591771 on physicalGPU1/2/4 each launched once. First-epoch source review0631fad2 and metadata receipt4c43a3b3 completed once. QC owns resource monitoring and the unchanged four-hour heartbeat.
- Independent terminal JSON/source/opaque-hash receipt: `completion/INDEPENDENT_TERMINAL_METADATA.json` b827a9749e70d302139aff296de22bb2d5624525c651b3c2d2af5ad39d2fe129. All60 orders,112860 steps, history0–60, earliest minimum-SSE selection, no failures, original PIDs absent, exact checkpoint/prediction hashes and all66860 labels verified. TEST numeric rows0.
- Science's sole CPU saved-output analysis: source39bf90ec/config0c25499f, source review e49dd50440c7892f2251c3580ac1dd52251c70962e1f715e4a5ae93870822ae9. Six saved validation outputs recomputed; CPU checkpoint states, optimizer/RNG, export equality/mode/contract, dormantF and original gate verified. No model construction/inference/raw targets/TEST access.
- `ROUND07_RESULTS.json`0314d0252de7e2f93e5ecd3df1be5a3f675fab540c8fea658b4b3a5231f21209; `ANALYSIS_RECEIPT.json`62f584dacafe9f0599f95f40c6d8d706f47a89a83c7011330378c509efb0d851. Independent41-input opaque hash and aggregate-arithmetic receipt e4b663bbdab5377ce31ecaad0301498e87571413352e9d796afb63ff483f4f98. Reviewer decoded no tensors/prediction arrays and repeated no reductions/inference.
- Scientific review `completion/INDEPENDENT_SCIENTIFIC_REVIEW.md` e5eeefb45d78f903d13b7f90ae04b4b00f5992ac26fb18ff3337f94939256bf9. Root accepted its conclusion. All valid labels remain; the v2 partition is historically exposed and is not an independent external dataset.

### Next authorized steps

Final science report6c3ee0be, reproduction4fec24c3 and next-notebbbe859f are stable. Science TERMINAL_MANIFEST.json2ff6f427 binds49 lightweight files, all independently rehashed. Final INDEPENDENT_TERMINAL_REVIEW.json SHA a06502ff25aa1d9cafe97e0939876eb23e6aa9d3a724b59912df27b18f848852 and sixteen-member INDEPENDENT_CLOSEOUT_MANIFEST.json SHA58becb72f4a1d48f1668f12f5fd64acf36b71204df964610ef07b9bb77402236 are sealed. No blocking findings remain. History now reviews QC publication scripts, exact inventory and staged blobs/links. Root authorizes D-first completion publication descending271d61f6 and delegated history README approval; no extra permission gate. README content1554bb027111ae5035df9926f320d8f8d7a700492dace71cb41ffaadd7a35ce1 passed exact review, with four new links to verify in the staged tree. Keep every learned tensor, optimizer state, dataset, identity/split/prediction array and cache server-only; opaque references are never archive allowlists.

After publication only, root permits read-only bottleneck/history audit and next-proposal development from existing source and aggregates. No new target decoding, model inference, implementation, optimizer updates or fit is included. Preserve all closed snapshots.

## Completed Round06

Both authorized original attempts completed60epochs/112860steps, with all66860 validation labels and TEST0. No resume/extension/new seed/inference is pending. Frozen134-source manifest4f1908d8 and production authority2c6c7f75 remain unchanged. Science owns the one completed saved-output CPU analysis; QC owns operational monitoring/publication; history owns independent review.

Independent terminal metadata PASS91d3b8f9256373d4c077b10627fceb02f69704189f6cc2d808eeb7d068034c35 verified frozen sources, exact60orders,61rows, earliest-SSEselection, all-label/state/bin counts, dormantFzero, oneattempt/PIDabsence and opaque checkpoint/prediction hashes without decoding tensors/arrays. Analysis-source reviewc0b7c6e0 binds source75939c1b/config633b8751. One saved-array analysis completed; independent aggregate/input binding checks PASS9a6242fccc372734d18e8605a5f437b4e49600165099659dba907024ea312a91 rehashed31 inputs opaquely, no repeated model/array computation.

Selected R²: trace_control49=.4047975673; raw_f18=.4220281674. Fixed60=.3890522162/.3936594861. Candidate gains.0172306 over the paired control but trails retained Round05control45 .4471694014 by.0251412, so the frozen dual+.003 gate FAILS. Candidate improvesS7–S9 and false-bright SSE but worsens pooledMAE, energy and trueq90/q99RMSE versus pairedcontrol. Conditional5656-component bootstrap intervals includezero at selected/fixed60; not seed/selection uncertainty or fresh confirmation.

Root ROUND06_DECISION.md SHA8f1b4e6de20024c904c90f551a677b74ac56d5b213581c07672a979b0fe03a39 closes pair without extension/coefficient/LR tuning or seeds, retains originalRound05epoch45, and authorizes D-first closeout publication descending fromc26a860b. Same declared seed11 control differs materially between rounds; completion/CONTROL_REPEAT_CONTEXT.md f49ae579 records matchedinit/order/settings/UUID and changeddiagnosticgraph/concurrency without cause attribution. A PSD-congruence proposal only is permitted after archive; no implementation/fit/test authority.

Final independent review PASS is sealed at completion/INDEPENDENT_TERMINAL_REVIEW.json, SHA e11df595b17e268f5ca0b67394c601dfbc4ada32ba7b72f267586e0b4c12188d. The explicit 17-member INDEPENDENT_CLOSEOUT_MANIFEST.json has SHA 1d3b802ec0ec68a751bdd0fdf8ca2a6b212675320d420a759357bfbb2b867d01 and binds final documents, root decision and the 39-record science terminal manifest e8b9f9f0. Final narrative/commands/deferred-assessment review passed; no scientific computation was repeated.

Round06 closeout is published to both main and research at 68520927333b27c48381401c8516a6709b79c40e, parent c26a860b. D-first archive: D:/MTO/archives/single_model_20260929/round06_objective_complete; bundle f0d1a4ed689c21a91b566d68ce0610a54fee3f6801628dda74db04116fa3118e. Independent staged approval 0d596deef33f620e2815203443569ee07b3a33557615d4a4a7a91de7802f295e verified all 80 exact blob bytes, five new README links, relative figure/context links and preserved worktree. QC remotely verified all 80 blobs after the atomic non-force push. Canonical receipt ops/ROUND06_COMPLETION_PUBLICATION_RECEIPT.json has SHA 80bb3a01c2f65e08c5b7718d5bf60a3736af4572fe66f310a9dcdb46974cb889.

No Round06 publication review remains. That completed-round decision originally permitted the next proposal only; the later Round07 preparation decision above now governs current work. All private weights, arrays and identities remain server-only. TEST remains sealed and 0.60 is unmet. Preserve every completed review and archive snapshot.
