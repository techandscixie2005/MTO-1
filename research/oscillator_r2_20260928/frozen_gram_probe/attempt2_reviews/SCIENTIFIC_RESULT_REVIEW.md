# Frozen h/base versus Gram probe: independent scientific result review

2026-09-29. Decision: no promotion, refit, ridge tuning, new inference or test access. The fixed linear Gram augmentation did not improve validation performance. The numerical computation is complete and its saved outputs are auditable; the original worker nevertheless FAILED during Markdown output under an ASCII locale. POSTPROCESS_RECOVERY.json documents additive UTF-8 report/integrity recovery without repeating fitting or inference. Preserve original FAILED.json and the absence of original PROBE_COMPLETE.json; do not describe the worker as a clean completion.

Reviewer independently inspected the saved result/recovery records and validation arrays, checked their hashes and exact shared IDs/indices/raw truth/base, recomputed both SSE/R2 values, verified shared max(0,signed) projection and base+scaled-residual decomposition, and calculated the case/state error differences below. No new model inference or fitting was performed.

| Fixed predictor | Full-train raw-f R2 | Validation raw-f R2 | Validation SSE |
| --- | ---: | ---: | ---: |
| Frozen eta0 | .64610356 | .40529412 | 94.20512 |
| A: h128 plus native base-f | .64807401 | .40442847 | 94.34224 |
| B: A plus528 Gram coordinates | .65022129 | .40281935 | 94.59714 |
| Existing equal-three ensemble | contextual | .44942146 | 87.21507 |

B minus A validation R2 is-.00160913. Molecule95% interval[-.006745,.001116], positive fraction.2805; connectivity-group interval[-.006230,.001117], positive fraction.27. A minus eta0 is-.00086564 and B minus eta0-.00247477; both corresponding intervals include zero. These estimates show no useful positive result under the fixed protocol, rather than a statistically established universal harm from Gram features. Both are far behind the existing ensemble, whose three-model inference cost must be acknowledged. No practical nomination threshold is met.

## Fitting and generalization

Both systems solve the same convex signed-residual ridge objective with lambda.001 and train-only normalization. Normal-equation residuals are1.39e-15 and2.70e-15, with finite condition estimates910 and4839. B's regularized training objective.349989 is below A's.352019, as required by the nested feature set. Projected train SSE improves from1069.5074 at eta0 to1063.5525 forA and1057.0633 forB. This is evidence that added Gram coefficients can fit additional training residual structure at this fixed representation; failure is not explained by an unfinished iterative optimizer.

The added fit is modest and does not generalize here. Effective degrees of freedom are128.21 of129 slopes forA and652.65 of657 forB: lambda.001 provides limited shrinkage relative to these standardized covariance eigenvalues. This is a limitation of the preregistered diagnostic, not permission to tune lambda on this validation result. B has more features AND coefficients; this is not a capacity-matched proof about information.

## Distribution and the fixed case

Both models trade lower bright-target SSE for worse below-q90 SSE. Relative to eta0, A gains.64775 bright SSE but loses.78487 belowq90; B gains.79711 but loses1.18913. Comparing B directly withA, extra Gram features gain.14936 bright SSE and lose.40426 belowq90, net+.25490 SSE. The common positivity projection improves validation SSE by.01913 forA and.04543 forB relative to signed predictions; the projected outcome still loses. The failure is not caused by retaining unphysical negative predictions.

S7 accounts for+.31939 of B-minus-A SSE; improvements elsewhere partly offset it. Previously fixed molecule14562 alone adds+.311984 SSE fromA toB, or122.4% of the+.254896 net difference. All other molecules together improve by.057089 SSE. This signed accounting explains why bootstrap intervals are broad and straddle zero; it is not an exclusion analysis or a reason to remove the case. All10 raw f labels for that molecule remain zero under the established source audit. Its S7 predicted f rises from frozen1.66514 toA1.69754 andB1.78717; molecule SSE rises from2.89800 to3.00388 and3.31586. The extra correction worsens an amplitude error. Do not reinterpret this as evidence that state swapping would fix it, and do not clip/drop the example.

Energy/base predictions remain frozen and identical, so this probe cannot create the much larger joint-training base spike observed in the prior residual replay. It can still add a harmful residual to an existing outlier. Frozen representation protects against one failure mechanism without guaranteeing that a learned correction will generalize.

## Scope of the conclusion and priorities

G contains all within-state pairwise contractions among the32 available l=2 tensor channels. The linear correction includes no arbitrary nonlinear h-by-G interactions, new representation, cross-state coupling or additional message passing. A negative fixed-lambda linear probe therefore does not show that scalar invariant readouts generally fail, that Gram information is useless, or that the DetaNet backbone is saturated. The model's original h-dependent tensor gate already contracts this information nonlinearly; exposing G linearly is a narrower intervention.

Together with the loss continuation screen, the evidence argues against extending the tested joint continuations or adding capacity to these frozen corrections by reflex. The frozen MLP produced only a small selected gain before overfitting; this solved linear probe also fits training residuals without a validation gain. Continue the already authorized seed robustness/diversity study. Root's next proposed paired native/MTO direct-f scratch comparison addresses a different unresolved question: properly scaled learning without the legacy warm start. It is preparation only and needs its own reviewed protocol/preflight; no further Gram search is recommended.

Provenance: RESULTS.json SHA b75610c8ed7d91b63b18adc84362453899edaed7deecefc072ed0f981b5d781a; FIT_FROZEN.json0958f141dfe290784002e91d4b0f899d37b8e00dc2d1d8dae9dc466e06d63812; POSTPROCESS_RECOVERY.json6911b141bebd2153739a9331639389b3bbf41427a4e110d3ab0d4519eb89c10e; original corrected-worker FAILED.json6f833d3bdc6a77f5d7cf445a633534e206317fb0a5d0031a899b71202903027f. Saved arrays and Gram caches stay server-only. The earlier h-tolerance failure is separately preserved under attempt1; neither technical failure is scientific evidence about R2.