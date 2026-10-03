# Round08 independent scientific review

## Conclusion

The fixed original/local/neighbor study completed correctly, but the frozen three-reference allocation gate failed. Neighbor transport selected epoch 23 reaches native pooled validation raw-f R² **0.4439683568**, ahead of the two contemporary controls and below retained Round05 control epoch 45 **0.4471694014**. Retain that existing checkpoint. These results do not authorize an extension, transport rescaling, seed allocation or TEST evaluation; root decides the next research question. The target 0.60 remains unmet.

| Arm | Selected epoch | Selected pooled R² | Fixed60 pooled R² |
|---|---:|---:|---:|
| Original |36|0.4176361697|0.4007054115|
| Local residual |30|0.4339236137|0.3958365051|
| Neighbor transport |23|0.4439683568|0.4086159717|

The selected neighbor differences are +0.0263321871 against current original, +0.0100447431 against current local, and -0.0032010445 against the retained reference. The rule required at least +0.003 against all three. The positive matched point estimates are useful evidence for this fixed recipe, not proof of a transport mechanism or seed-robust gain.

## Integrity and evaluation

Independent metadata checker 55dae35d and result checker 496686c4 pass. All 258 frozen source files, authorization/publication, split, initial base, every 60 TRAIN-order hash,60 epochs / 112860 optimizer updates per arm, earliest pooled-SSE selection among epochs0–60, access ledgers and opaque checkpoint/prediction hashes match. Each arm had one registered attempt and normal terminal markers; the original workers are absent. The CPU analysis checks selected/last optimizer state counts 135/137/137, exact group IDs/step/settings, RNG categories, checkpoint configuration, all dormant right-F tensors, zero original transport parameters, nonzero active transport parameters and exact selected-model/export tensor equality. The stored buffer fingerprint is validated as metadata; no model was reconstructed to recheck buffers here.

All 6686 validation molecules and 66860 valid raw-f labels are retained, including zero strengths. Ten state counts and exhaustive true/predicted q99 bins partition the pooled count/SSE. Bright tails use frozen TRAIN cutoffs q90=0.0549 and q99=0.2406 (6782 and708 validation labels). Selection uses native raw-f SSE; no calibration, exclusion, test tuning or prediction averaging. The v2 split is a molecular-group partition under audited rules of historically exposed QM9S data. It is not independent external confirmation. TEST numeric rows remain 0.

Science executed one reviewed CPU reduction of six immutable saved validation prediction sets. The reviewer inspected that source, independently rehashed 41 inputs as opaque bytes and checked aggregate arithmetic/selection/gates; the reviewer did not decode arrays/checkpoint tensors or repeat inference/reduction. All code and receipt references below identify this distinction.

## Tradeoffs

At the selected checkpoints, neighbor lowers SSE in states 6–10 versus the current original. It lowers the false-bright q99 bin SSE by 9.355669, but raises pooled MAE by 0.00043103, energy RMSE by 0.00190291eV, q90 RMSE by 0.00128889 and q99 RMSE by 0.00731578. Thus the pooled gain does not mean uniform state or tail improvement.

Against current local, neighbor improves states 1,4,7,8,9 and q90/q99 RMSE by 0.00113004/0.00452315; pooled MAE rises 0.00003465 and energy RMSE rises 0.00339089eV. Against the retained Round05 reference, it improves five state SSEs (1,6,7,9,10), but pooled R² falls 0.00320104, MAE rises 0.00085356, energy RMSE rises 0.00866802eV, and q99 RMSE rises 0.00971604. False-bright bin SSE is lower by 5.082945, which does not offset other errors. Predicted brightness changes bin membership between models; bin differences are descriptive partitions, not errors on a fixed common subset or causal explanations.

## Uncertainty and training behavior

The paired bootstrap uses 5656 audited validation identity components, 6686 molecules, 2000 draws and seed 20260930. Each draw recomputes pooled SST. Selected neighbor-minus-original interval is[-0.0141672,0.0741705], neighbor-minus-local[-0.0125818,0.0331000]. All six selected/fixed60 paired intervals include zero. These intervals condition on the realized runs and chosen checkpoints on reused validation; they do not correct for checkpoint selection, successive experiment choices or training-seed variability. No interval against the retained model is claimed from its aggregate-only reference.

Same-declared-seed original-control outcomes vary across recent rounds (.4471694,.4047976,.4091041,.4176362). They are realized reruns with differing executable graphs/workloads, not independent-seed confirmation or evidence identifying why trajectories differ. The retained-reference screen remains unchanged.

Both modified branches are active. Neighbor selected-checkpoint theta norm is 9.37614 and mean absolute tanh coefficient 0.25121. At its selected epoch, atom-weighted mean regularized residual/message norms across two blocks and three irreps span0.005968–0.009795, with maximum 0.21327; at epoch 60 the means span0.009584–0.016189. Local selected means span0.006656–0.009463. These are observed training-trajectory summaries, not a fixed-checkpoint validation measurement. The denominator is max(norm,1e-12), with zero/below-floor counts preserved. Parameter movement is within-epoch displacement. Coefficient extrema do not quantify saturation frequency; tensor activity does not identify electronic transition densities or a physical information bottleneck.

After selected epochs, logged TRAIN objectives fall (neighbor 0.14329→0.06549) while validation trace loss rises (0.14435→0.15118) and pooled R² falls. Energy and true bright-tail errors can improve at the same time. These observations motivate testing generalization or optimization, but TRAIN values are pre-update minibatch averages and cannot establish a fixed-checkpoint train/validation gap or its cause. No new numerical diagnostic is needed to close this round.

## Evidence

- Frozen source manifest: de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146.
- Analysis source review: 46108a5b6dc3bec976871a860c44eaf5bc181758e63763439cf7591aaacd55c5.
- Independent terminal metadata: 55dae35d28f63a53468d04d177979c718433fb30e0bb05525c6638d98a0316d8.
- Analysis result: 5ba356f7e37def370d09492855378b580cf5fa5e66afb4db990d7b3a5cc0dd58.
- Analysis receipt: 8451c3a05efb9b149a8e5403f389607dc125094ef79f37ca57617e5736ab727d.
- Independent aggregate checks: 496686c4dccbffe768b9913eb360b038b03fe23e51821bffb9950ac216df2e33.

The retained one-model geometry export is `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e. Checkpoints, arrays and learned parameters remain server-only; hash references do not authorize their archival upload.
