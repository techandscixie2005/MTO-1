# Independent Round06 scientific review

**PASS for result integrity; the candidate fails the frozen allocation gate.** Root closes the fixed objective comparison without extension, coefficient/LR tuning or confirmation seeds. Retain Round05 original MTO epoch45 as the v2 validation reference. No TEST evaluation or attainment of R²0.60 is claimed.

| Single model | Selected epoch | Selected pooled raw-f R² | Fixed60 R² |
|---|---:|---:|---:|
| Round06 original LE+Ls |49|0.4047975673|0.3890522162|
| Round06 LE+normalized raw-f |18|0.4220281674|0.3936594861|
| Retained Round05 original MTO |45|0.4471694014|0.4026919323|

The candidate gains0.0172306001 over its contemporaneous control but loses0.0251412340 against the retained reference. It therefore fails the requirement of at least+.003 above both. The old reference is previously published aggregate evidence on this same reused v2 validation; no reference inference or prediction averaging was performed.

## Integrity and evaluation scope

Both original registered attempts completed60 epochs/112860 updates, with61 history rows including epoch0, no failure/resume and original PIDs absent. All134 frozen source hashes, exact60 order hashes, initial tensor hashes, source/statistics/config and authorization/publication bindings agree. The actual committed optimizer/RNG/history metadata and exported-model equality passed science's reviewed CPU audit. Independent review rehashed receipts and private checkpoints/predictions opaquely; it did not load tensors, repeat inference or recompute saved prediction arrays.

Earliest minimum pooled raw-f SSE selected epochs49 and18, with all66860 valid labels and zeros retained. Every state's count is6686; all four brightness bins partition the whole evaluation. TRAIN-only q90=.0549/q99=.2406 remain fixed. No split/label changes or TEST numeric access are recorded. The geometry export matches the selected model tensor state and includes model configuration/TRAIN statistics; it is a single checkpoint with geometry-only inference. Its stored buffer fingerprint was checked for presence/format at terminal; the model was not reconstructed for a new terminal forward. Preparation already tested load/forward access guards.

Independent metadata receipt:91d3b8f9256373d4c077b10627fceb02f69704189f6cc2d808eeb7d068034c35. Analysis source review:c0b7c6e0b4725bd6de3f4d2748980220d8eac819347a101304880296a86a0b4b. Result consistency review:9a6242fccc372734d18e8605a5f437b4e49600165099659dba907024ea312a91. Science's reviewed analysis ran once on four already-saved validation outputs, plus CPU checkpoint and identity-group metadata, with no model construction/inference or raw target dataset access. Its31 bound inputs and all source/output hashes were independently rechecked.

## Tradeoffs

At the independently selected checkpoints, raw-f supervision improves SSE for states7–9 and worsens the other seven states versus its contemporaneous control. Pooled MAE worsens0.0175600→0.0185502, energy RMSE0.12518→0.15926eV, true-q90 RMSE0.09997→0.10290 and true-q99 RMSE0.22958→0.23893. False-bright SSE falls17.80755→8.22948; other error contributions offset much of that improvement. These are descriptive effects for the selected predictors, not an identified physical mechanism.

At fixed60, the candidate improves pooled R² by0.00460727 over control but both are below their own selected checkpoints and the retained reference. From raw-f epoch18 to60, TRAIN trajectory raw-f MSE falls approximately0.001226→0.000456 while validation R² falls0.42203→0.39366; false-bright SSE rises8.22948→19.57293. True bright-tail errors improve during that interval, so degradation is not uniform. TRAIN quantities are pre-update minibatch trajectory aggregates, not fixed-checkpoint TRAIN evaluations; validation base_objective remains LE+Ls in both arms.

The paired2000-draw bootstrap uses5656 audited identity components containing all6686 validation molecules. Its descriptive95% percentile intervals for candidate-minus-control R² are[-0.02842,0.07011] at selected checkpoints and[-0.02998,0.03924] at fixed60. Both cross zero. They condition on these realized predictors and reused validation; they exclude checkpoint-selection and training-seed uncertainty. No paired interval is claimed against the aggregate-only retained reference.

## Repeat-control context and limits

[CONTROL_REPEAT_CONTEXT.md](CONTROL_REPEAT_CONTEXT.md) records the material difference between repeated seed11 controls. The declared scientific settings, initial tensors, data order and GPU1 UUID match. Executed diagnostic graphs differ: Round05 control backpropagates a zero-weight decorrelation branch; Round06 logs decorrelation under no_grad and computes an unused raw-f loss diagnostic. The rounds also scheduled four versus two concurrent fits. No cause of the outcome difference has been isolated. This is realized same-declared-recipe variability, not independent-seed confirmation, evidence of a particular CUDA kernel or proof of objective causality.

This remains an internally repartitioned historically exposed corpus, with audited conservative identity rules rather than an absolute chemical-identity proof. TEST stays sealed. The scalar loss does not identify full-TDDFT X/Y amplitudes, transition densities, NTOs or phase/gauge-safe electronic operators. The completed negative allocation result does not prove every raw-f objective will fail, and it does not authorize a parameter search.

Root decision ROUND06_DECISION.md SHA8f1b4e6de20024c904c90f551a677b74ac56d5b213581c07672a979b0fe03a39 authorizes D-first closeout publication. Only after that does it permit a separate PSD-congruence proposal; no implementation, new fit, seed or TEST scoring follows from this review.
