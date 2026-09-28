# Descriptor baseline feasibility (read-only)

This note audits existing inputs and installed packages. It does not construct features, fit a model, read active predictions, or inspect test labels.

## Available inputs and provenance

The frozen benchmark has 133,727 molecules, including 120,355 train and 6,686 validation rows. The saved model input contains atomic numbers `z` (up to 29 atoms), Cartesian coordinates `pos`, and an `edge` tensor that joins atom pairs within 5 Å. That edge tensor is geometric and is not a chemical bond graph. The original QM9S iterator exposes atomic numbers, coordinates, charge and multiplicity alongside quantum labels; it does **not** expose authoritative bond orders or a canonical chemical SMILES input. The descriptor learner may use only Z and coordinates (and fixed physical constants), never energy, tensor, oscillator-strength or other quantum labels as features.

The audited `identity_audit_v2.json` key is an explicit-H connectivity SMILES generated from an XYZ-derived graph **before** `DetermineBondOrders`; successful bond-order assignment and sanitization were used only as a validity check. Therefore this key cannot be treated as a chemically typed bond-order/aromatic SMILES for Morgan fingerprints. The audit has 278 unresolved molecules in 266 groups, all in train; validation has none. Excluding unresolved train molecules would change the benchmark and is not acceptable.

A uniform geometry-derived graph is feasible for every row: connect atom pairs by a frozen covalent-radius rule from Z and distance, then hash radius-0/1/2 atom neighborhoods without bond orders or aromaticity. Fixed composition, distance, and covariance invariants can supplement those counts. This representation does not require special handling or filtering for unresolved bond-order rows. It may still be sensitive to geometry and the chosen radius threshold; that is a modeling limitation, not evidence for tuning on validation. The connectivity split remains the original audited grouping, not the new feature graph.

## Installed software and capacity

The project environment has Python 3.10.19, RDKit 2025.09.4, scikit-learn 1.7.2, and PyTorch 2.5.1+cu121. XGBoost, LightGBM, CatBoost and Chemprop are absent. The system Python lacks the chemistry and tree packages, so use the existing project environment. The host reports 512 logical CPUs, 503.34 GiB total RAM and about 468 GiB available at this read-only check. Availability is a snapshot and must be rechecked before any run.

A single `sklearn.ensemble.ExtraTreesRegressor` can fit all ten raw-f targets jointly with `criterion="squared_error"`; no separate statewise model or target-specific search is needed. The proposed 256-tree, 8-worker, depth-24, leaf-4 budget is plausible on this host. An explicit 16 GiB process-memory and 4-hour runtime admission cap is prudent, but **not guaranteed** by this inventory: actual feature-matrix and forest size depend on node counts and implementation overhead. A future reviewed implementation should estimate array dimensions and memory on synthetic inputs, record walltime/RSS, and fail closed at the fixed caps without reducing samples or retuning the model.

## Recommendation and limits

Proceed, conditionally, with one fixed multioutput ExtraTrees baseline on uniformly constructed Z+coordinate descriptors: 512 hashed bond-agnostic neighborhood counts plus fixed small composition, graph-count, radial-distance and covariance features. Use the existing molecule/group split, train-only preprocessing, no validation-guided hyperparameter or feature choice, and full-benchmark raw ten-state f evaluation. If an inner train-only group holdout is specified later, its role and full-train refit must be frozen before validation. This is a useful low-cost nonlinear baseline, not a chemical-bond descriptor claim. Comparing a blend with F5 would add one 256-tree estimator to five neural forwards; inference counts are not calibrated latency.

