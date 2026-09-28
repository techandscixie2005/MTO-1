# Legacy chan64 campaign completion and fixed validation ensembles

The legacy four-arm channel-64 campaign has ended. G1, G3, and G4 completed by early stopping. G2 was intentionally stopped during epoch 215 after a long validation plateau; its resumable checkpoint was retained, and the supervisor correctly marked the overall campaign blocked. This is a three-complete, one-administratively-incomplete campaign.

## Selected individual checkpoints

| Arm | Training objective | Outcome | Selected epoch | Selected validation loss | Native raw-f validation R² | Raw-f SSE |
|---|---|---|---:|---:|---:|---:|
| G1 | Eta=1, original tensor | Natural early stop, epoch 227 | 75 | 0.48762342 | 0.40525459 | 94.21138 |
| G2 | Eta=0, original tensor | Intentional stop within epoch 215 | 41 | 0.12957937 | unavailable | unavailable |
| G3 | Eta=1, p_outer tensor, E²-weighted trace | Natural early stop, epoch 326 | 174 | 0.52485321 | 0.36810069 | 100.09679 |
| G4 | Eta=0, p_outer tensor, E²-weighted trace | Natural early stop, epoch 223 | 71 | 0.13636914 | unavailable | unavailable |

The selected validation loss is the training objective, not oscillator-strength R². For G2 and G4, the historical audit used true E to form f because their E heads were not supervised. Their oracle-E scores were R² 0.38078050 (G2) and 0.34833500 (G4); these are diagnostic only and are excluded from deployable native-f comparisons and ensembles. At its selected checkpoint, G3 is below the earlier eta0 model (R² 0.40529412) and fixed equal-three eta0/eta01/eta1 ensemble (0.44942146). The prior validation-only CPU audit and the newly approved GPU4 replay agree within 5×10⁻⁸ in R². The replay gives G1 and G3 raw-energy MAE 0.08915 and 0.09499 eV, and raw-f MAE 0.01680 and 0.01751.

All four run manifests match their frozen source fingerprints; the dataset, raw printed labels, split and train-only scaling hashes match the pinned data manifest. G1/G3 selected checkpoint hashes match the earlier validation-only audit. G3's terminal marker records epoch 326, best epoch 174 and 613,206 updates. The G2 ADMIN_STOPPED marker records its exact within-epoch cursor and best/last checkpoint hashes; no FIT_COMPLETE was fabricated.

## Two fixed, exploratory ensembles

G1 and G3 were replayed exactly once at their joint-objective-selected best checkpoints on validation only, with healthy idle GPU4 held under the shared lock. Their native oscillator strengths were computed separately as c × predicted E × trace(predicted A), then averaged. Existing eta0/eta01/eta1 validation arrays provided the other components. Exact IDs, validation row indices, raw FP64 printed f labels, masks and state order match across all five. No weights, subsets or checkpoints were fitted after observing the results.

| Predictor | Model forwards | Native raw-f pooled R² | SSE | Change in R² vs fixed equal-three |
|---|---:|---:|---:|---:|
| Fixed eta0/eta01/eta1 equal-three | 3 | 0.44942146 | 87.21507 | reference |
| Equal G1+G3 | 2 | 0.46407773 | 84.89343 | +0.01465626 |
| Equal eta0/eta01/eta1/G1/G3 | 5 | 0.48769960 | 81.15158 | +0.03827814 |

Against fixed equal-three, the 2,000-replicate paired connectivity-group bootstrap 95% interval for the equal-two change in R² is [−0.01231, +0.05048]; it includes zero. Its SSE rises by 2.677 above the fixed validation q90 brightness threshold and by 4.015 above q99. The equal-five interval is [+0.02597, +0.05470]; its q90 SSE falls by 1.199, while q99 SSE rises by 0.624. Relative to equal-three, overall SSE falls by 2.66% for equal-two and 6.95% for equal-five. Physical state 8 SSE increases by 0.068 in the equal-five; the other nine states improve. These intervals describe paired validation variability conditional on all model and ensemble choices. They do not account for checkpoint-selection optimism.

A saved-array concentration check finds that molecule 14562 supplies 87.9% of the equal-two net SSE gain, so that gain is fragile. It supplies 16.1% of the equal-five gain; 4,091 of 6,686 molecules improve and all six previously fixed train-derived q2 strata show positive net SSE gain. The fixed q99 bright tail still worsens. No molecule or state was removed from any reported pooled score.

The five-way predictor requires five model forwards; G1/G3 each have roughly 3.45M/3.39M parameters, so forward counts are not calibrated latency ratios. The five-way validation gain is useful evidence of complementary errors, not a confirmed holdout gain. The previously accepted fixed equal-three historical-test R² is 0.48510176; that figure belongs only to the original three-model predictor. The five-model candidate has never been evaluated on test. The historical test was not accessed in this round. The coordinator retains five-way as a provisional validation candidate and closes this chan64 route without extension.

## Artifacts and limits

[CAMPAIGN_AUDIT.json] contains exact terminal status, best checkpoint and source hashes. [REPLAY_PROTOCOL.json], [REPLAY_SEAL.json], [REPLAY_REVIEW.json], [REPLAY_STARTED.json], and [REPLAY_RESULT.json] record the one-time authorized validation replay and hardware checks. [ENSEMBLE_RESULTS.json] records pooled metrics, paired molecule and connectivity-group bootstrap results, state and fixed-tail SSE. The G1/G3 per-molecule prediction NPZ files remain on the remote server and are excluded from D: and GitHub archives. Checkpoints, optimizer states, caches, and raw datasets are likewise excluded.

The original chan64 study conclusion remains that neither G1 nor G3 improved the earlier eta0 single-model native-f score. The ensemble result is a separate post-result diagnostic.
