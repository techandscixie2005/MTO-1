# Completed loss continuation: independent scientific review

2026-09-29. Decision: promote none, no extension and no test evaluation. The selected outputs are numerical ties at the unchanged eta0 baseline. Review used remote completed configuration/objective, summary, all three21-row histories, and selected validation arrays; raw-f SSE/R2 and exact cross-arm IDs/indices/raw truth were independently recomputed. Archive agent separately reports terminal/source/order integrity PASS.

| Arm | Selected epoch | Selected raw-f R2 | Final20 raw-f R2 | Added final SSE vs epoch0 |
| --- | ---: | ---: | ---: | ---: |
| Legacy control | 0 | .4052941077 | .3093488848 | 15.1983 |
| True-E-squared weighted trace | 0 | .4052941089 | .3065721866 | 15.6382 |
| Matched-scale direct native-f loss | 0 | .4052941280 | .3173645027 | 13.9286 |

All arms also select epoch0 on the old and their own validation objectives. The largest selected delta is2.04e-8: the direct-loss bootstrap positive fraction.96 is meaningless as evidence of a useful gain at that magnitude. The selected weights are the same source model evaluated through finite-precision paths. Selection eligibility of epoch0 correctly protects against harmful continuation; it is not a new winning trained model.

## What this experiment tests

All three start from the same eta0 epoch33 checkpoint, reset AdamAMSGrad, use lr1e-4, clip5, no scheduler, identical20epoch orders and37620updates. The architecture/readout is unchanged. The weighted arm multiplies squared trace residuals by detached true E squared divided by train mean E squared; true energy is a LOSS weight, not a prediction input. Direct native-f loss retains gradients through predicted E and trace, uses raw printed f, and denominator C_f^2*3*sA2*mean(E_true^2)=.011321718689897071. The later architecture screen's train-f variance denominator.002510981243894732 makes its relative f coefficient4.508882 times larger. This loss pilot therefore supplies a useful lower-weight comparison, but not an otherwise identical rerun of the architecture screen.

The continuation control deteriorates too. Thus failure cannot be assigned only to the new loss, stronger f weighting, or a particular new head. Direct f ends about.00802 R2 above the control but remains.08793 below its own starting model. That terminal ranking does not satisfy the prespecified selection/promotion rule or justify extending the arm.

## Training, energy and state behavior

Online training objectives fall from epoch1 to20: control.09772 to.06625, weighted.09259 to.06400, direct.09267 to.06412. Their own validation objectives rise from epoch0 to20 (.17223 to.19143; .15239 to.16723; .15263 to.16794). This is consistent with worse generalization during joint adaptation. These online changing-model losses are not checkpoint-fixed full-train f SSE; they cannot prove a native-f training-fit ceiling or convergence.

Final energy MAE improves from about.09139 to.08613/.08576/.08575. Final f MAE also improves slightly from.0177856 to.0175334/.0175779/.0175744, while f SSE/R2 gets substantially worse. Better average absolute error and energy accuracy do not offset the target metric's sensitivity to larger f errors.

History-derived S7 added SSE is13.8933/13.8289/12.2026; S8 adds2.1707/2.1264/2.3560. Together these account for105.7%/102.0%/104.5% of each net SSE deterioration because improvements in other states offset part of the loss. This supports concentrated high-state failure, not a claim that every state worsens. The saved loss histories do not establish final molecule/brightness/sign concentration; selected arrays are epoch0, and no final replay was performed. Do not transfer the residual arm's ID14562 decomposition to these models without evidence.

## Implications

The findings weaken the idea that simply changing intensity loss or lowering its relative weight will rescue this pretrained joint continuation. They leave optimizer reset, learning-rate adaptation, representation drift and finite-data generalization unresolved; no one cause was isolated. They do not show that an appropriately trained direct-f model from scratch is poor, that full tensor invariants are useless, or that DetaNet's backbone is limiting.

Keep the already authorized bounded seed23/37 study and frozen Gram analysis as separate evidence. The seed study tests robustness/diversity of the working legacy recipe; the convex Gram probe removes iterative readout optimization while holding the base fixed. No extra study, validation-tuned setting, case removal or test access is recommended from this pilot. Any later joint fine-tuning should explicitly preserve baseline behavior or test the optimization confound, rather than repeat these20epoch continuations.

History SHA256: control600602668d08d1c05da901fb08d4bc2f41db5afe534c87e83b6de7b7494b3d29; weighted d93727241424f49521a0e31a186aca1e35dbeff50feb603001d8254fa5813059; direct19130c00ce2f2bf410fb347d3dfa9dc3a3c0633675b296e9855a6465cba95b43. Full source and terminal provenance remains in immutable pilot artifacts and the independent completion archive.