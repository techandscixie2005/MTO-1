# Round07 independent scientific review

**PASS for completed-result integrity and interpretation. The tensor allocation rule fails.** Close the fixed study; retain Round05 control epoch45 (native pooled raw-f validation R²0.4471694014). Root owns the final decision. This review grants no extension, seed allocation, tuning or TEST evaluation.

## Evidence and scope

All three original attempts completed60 epochs/112860 updates. Independent metadata receipt `b827a9749e70d302139aff296de22bb2d5624525c651b3c2d2af5ad39d2fe129` verifies185 frozen source pins, authority/publication/split, exact60 orders, history0–60, earliest minimum-SSE selection, one launch/no failure and opaque checkpoint/prediction hashes. All66860 valid labels, including zeros, are scored; all ten state counts and four brightness bins sum to the pooled total. TEST numeric access is zero.

The one source-reviewed CPU analysis recomputed six saved validation sets and inspected checkpoint payloads; it constructed no model and performed no inference. Source review `e49dd50440c7892f2251c3580ac1dd52251c70962e1f715e4a5ae93870822ae9` binds source39bf90ec/config0c25499f. Analysis receipt62f584da and result0314d025 passed independent input/output binding and aggregate arithmetic checks `e4b663bbdab5377ce31ecaad0301498e87571413352e9d796afb63ff483f4f98`:41 input hashes rechecked without tensor/array decoding or repeated reduction. Actual optimizer/RNG/config, exported model tensor equality/mode/contract, dormant right-F and original gate checks passed. The stored buffer fingerprint was checked; no new runtime reconstruction was claimed.

## Main comparison

| Arm | Selected epoch | Selected R² | Epoch60 R² | Selected MAE | Energy RMSE, eV | True q90 RMSE | True q99 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original |52|0.4091040813|0.3845332343|0.01735919|0.12705136|0.10087656|0.23366024|
| Scalar |49|0.4293390126|0.3937752150|0.01760588|0.12385485|0.09725400|0.22182574|
| Tensor |34|0.4187551766|0.3277766810|0.01856396|0.12993132|0.09964401|0.23584824|

Fixed TRAIN thresholds are q90=.0549 and q99=.2406; validation tail counts are6782 and708. Tensor minus selected original is+.0096511, versus scalar−.0105838, and versus retained−.0284142. The prespecified+.003 above **each** reference rule fails; scalar is the strongest descriptive Round07 model but also trails retained by.0178304. No directional-tensor benefit or new incumbent is established.

Selected tensor reduces state SSE versus original for S4/S5/S7/S8/S9 and versus scalar for S5/S7/S10. Scalar improves S1/S4/S5/S7/S8/S9 versus original. These improvements coexist with worsened other states; no labels are excluded. Tensor selected false-bright SSE12.46060 is lower than original15.71883/scalar15.81523, but its q99 RMSE, MAE and energy error are worse than both. At epoch60 tensor false-bright SSE rises to28.95730 and its R² is lower than both controls. Bins use each model's own predictions; subgroup membership changes, so these contrasts are descriptive decompositions, not fixed-subgroup causal effects.

## Uncertainty and diagnostics

All three selected-checkpoint paired intervals include zero: tensor−original[−.017668,.033066], tensor−scalar[−.035256,.014707], scalar−original[−.008953,.048984]. At epoch60 the tensor contrasts are negative: versus original[−.095654,−.018434] and scalar[−.120878,−.018998]. The2000 draws/seed20260930 resample5656 audited validation components containing6686 molecules and recompute pooled SST. These are conditional reused-validation summaries after checkpoint selection; they omit selection and training-seed uncertainty. No interval was fabricated versus the aggregate-only retained reference.

Both modified gates were active; original gate remained exactly unchanged. At their selected epochs scalar/tensor TRAIN mean b=.22070/.21679, mean q=.36410/.37518, relative A change=.17572/.21070 and mean nonzero strength ratio1.16767/1.18531. Bounds and1203550 TRAIN diagnostic entries per epoch passed. q above1/3 indicates anisotropy of the learned aggregate; directions are not identified by trace labels. Gate movement logs measure displacement within that epoch. Mean/range diagnostics do not establish saturation prevalence or a physical response mechanism.

From selected epoch to60, all arms' TRAIN trajectory loss falls while validation trace loss rises and energy error falls. Tensor's sharper raw-f deterioration is consistent with poorer generalization at that endpoint, without isolating its cause. TRAIN values are pre-update minibatch aggregates, not fixed-checkpoint TRAIN evaluation or a matched generalization-gap estimate.

## Limits and recommendation

One realized seed/order11 trajectory per arm, with previously documented control variability, is not independent-seed confirmation. The v2 partition obeys audited conservative identity rules, but its TEST rows were historically exposed (5989 oldTRAIN/361 oldvalidation/336 oldTEST); TEST remains sealed in this campaign and is not an external fresh holdout. No0.60 result, averaging, physical tensor reconstruction or new QC supervision is claimed. Publish the negative comparison and preserve all resumable/geometry checkpoints server-only. Keep learned tensors, optimizer states, private arrays and raw datasets out of Git; archive lightweight records locally before publication.
