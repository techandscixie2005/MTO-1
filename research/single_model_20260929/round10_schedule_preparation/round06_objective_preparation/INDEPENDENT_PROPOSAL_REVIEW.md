# Independent review of the Round06 objective proposal

**PASS for the scientific proposal only.** The comparison asks a distinct, bounded question within the audited history: train the unchanged original PSD MTO from a common fresh initialization under either LE+Ls or LE+normalized raw-f MSE. This review authorizes no implementation, target access, inference, technical update or fit; root decides the next preparation scope.

## Contract checked

- The original energy loss, trace-control loss and masks agree with the reviewed model source. New TRAIN statistics fix sE2=.5376833706691944, sA2=.13245475393297818, and pooled population raw-f variance=.0025089892829484074. No new reduction or label inspection was performed.
- The candidate uses stored printed f, including zeros, and predicted E/A through `f = 2/(3*27.211386245988) * E_eV * trace(A)`. The unit conversion and both analytical partial gradients are correct. It must select valid entries before the product and squared error. No target-energy substitution, detachment, clamping, reconstruction of target f, per-state scaling or exclusions are allowed.
- FP32 target casting and training arithmetic are explicit; the common primary evaluator retains FP64 native-f reconstruction and authoritative printed targets. All frozen masks must remain true, so both objectives cover the same labels. An unexpected mask change fails the contract instead of reducing the benchmark.
- Coefficient1 is fixed. Normalization makes the terms dimensionless but does not equalize gradients, conditioning or bright-label influence. Those changes and the extra E/A product coupling are part of the objective contrast. LE does not remove the possibility of energy-strength compensation.
- Both arms rebuild the same fresh initial tensor hashes and 60 prescribed TRAIN orders. They retain identical model capacity, dormant F schema, optimizer, LR, precision, clipping, batch treatment and 112,860-update budget. Trained Round05 weights and disposable preflight updates are excluded. CUDA trajectories need not be bitwise identical.
- Validation selects the earliest minimum pooled SSE, including epoch0. Selected-best and fixed60 comparisons, all-state/tail/false-bright diagnostics and every valid label remain required. TEST stays sealed. The candidate must gain at least .003 over both its contemporaneous control and the retained v2 reference, avoiding promotion through a weak control rerun alone. A stronger control is execution variability under the same recipe, not evidence for the objective change.

## Novelty and limits

The audited raw-f scratch result used a direct scalar-f head, changed initialization and a longer schedule. Earlier original-PSD raw-f studies were selected-checkpoint continuations; Round02 froze the base and trained only F. None is the proposed matched fresh original-PSD objective pair. This is a claim about the reviewed records, not an exhaustive proof of repository-wide absence. Those failures remain negative evidence.

Round05's active modifications and worsening validation after selected epochs motivate a smaller objective question; they do not establish objective mismatch as the cause. Squared raw-f error can still concentrate on rare large errors or overfit. Fixed coefficient/LR, no outcome-driven fallback, and a bounded negative-result closeout are appropriate.

The relation is the existing oscillator-strength observable. It does not identify physical tensor orientation, wavefunction phase, transition densities, NTOs or TDDFT response amplitudes. The historically exposed v2 partition and reused validation remain limitations; seed confirmation would not become fresh-data confirmation. The deferred PSD congruence proposal should remain separate.

## Future preparation gates

If root accepts preparation, source review must verify the target-before-arithmetic mask path, the unchanged control function, selective TRAIN/validation readers, exact full dependency/normalization bindings, immutable recovery and one-checkpoint export. Synthetic checks should distinguish expected zero factor gradients at C=0 from live gradients on nonzero cases. The proposed six discarded TRAIN updates are not authorized by this review. Their acceptance criteria, resource admission and source bindings must be fixed before execution; no validation fixture or coefficient/LR tuning is justified.

Review used existing aggregate statistics, published history notes and source text only. No weights, raw/prediction arrays, labels, model construction or numerical tests were opened or executed. Exact proposal and reference hashes are recorded in INDEPENDENT_PROPOSAL_REVIEW.json.
