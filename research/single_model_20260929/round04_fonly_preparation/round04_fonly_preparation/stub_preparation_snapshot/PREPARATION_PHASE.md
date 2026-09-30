# Expanded preparation phase — no production fit

This phase follows the immutable documents-only protocol snapshot in `documents_only_snapshot/`. Independent protocol review passed and root's separate `PREPARATION_AUTHORIZATION.md` permits the following implementation and checks. The protocol, root decision and completed rounds remain unchanged.

Implemented scope:

- Four-column scalar basis and FP64 SVD solve restricted to explicitly synthetic fixtures of at most 128 rows. No real-data fit API or fitting/evaluation runner is released.
- One-checkpoint geometry-only wrapper with the original frozen full eta0 and four coefficient buffers. Preparation exports live only in memory and use hand-set synthetic constants or an existing affine anchor.
- Synthetic mathematical and failure checks; exact execution/publication binding checks. Production invocation fails before any data access when separate receipts are absent, and this release refuses even after receipt verification because no production runner exists.
- Target-free calibration-cache audit. Only native prediction inputs, masks, indices and identities are decoded. Target/energy ZIP members are actively rejected. The 4546 true zero labels are inherited from the pinned producer receipt and are not newly recomputed.

Every preparation command must set empty CUDA visibility **before** interpreter startup and use two CPU threads:

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation
MTO_PY=/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$MTO_PY" synthetic_checks.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$MTO_PY" audit_design.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$MTO_PY" inference_preflight.py
```

Use unique execution logs and preserve failed attempts. Do not adjust fixed knots, condition/rank limits or model settings after the design audit. A failed numerical gate stops preparation with its diagnostic record. No production coefficient solve, outer-validation values, test data, new source prediction generation or GPU allocation is authorized.

The calibration audit reads existing input predictions; it does not infer molecule-level outputs or fit against labels. The geometry test uses two hand-constructed synthetic molecules and the unchanged full baseline. That synthetic inference is a preparation check, not an accuracy experiment. Source freeze and final independent review must bind exact code/settings/outputs and the completed Round03 dependencies before archive publication.

All learned weights, target/prediction/index arrays and caches remain server-only. Lightweight diagnostics expose ranks, singular values, support counts, hashes and test differences. New fitted coefficient tensors and exact learned slopes may not appear in JSON/Markdown/plots or archive manifests. Synthetic hand-set constants in test source are explicitly labeled. Publishing this preparation never authorizes a production fit.
