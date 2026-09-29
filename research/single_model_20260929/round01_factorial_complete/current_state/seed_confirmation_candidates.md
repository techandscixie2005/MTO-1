# Independent-seed continuation confirmation: prepared source audit

Prepared2026-09-30 before first-factorial outcomes. This is a proposed confirmation specification and source audit, not a launch. It does not change the already frozen four-arm pilot.

## Frozen independent starting checkpoints

Both are genuine fresh-initialization seed runs of the original eta0 LE+Ls recipe; they are not shuffles or copies of seed11. Read their completed100epoch protocol before interpreting between-seed variation.

| Seed | Source selected epoch | Current validation raw-f R² | Server path | SHA256 |
|---|---:|---:|---|---|
| 23 | 53 | .368708895 | `/home/inspur/MTO-1/research/oscillator_r2_20260928/eta0_seed_replication/runs/seed23/best_legacy.pt` | `7c28a2fe204f8804d05d3dc58bb9acf853f2d0505ce83a9f66819f02fd485a18` |
| 37 | 44 | .394455334 | `/home/inspur/MTO-1/research/oscillator_r2_20260928/eta0_seed_replication/runs/seed37/best_legacy.pt` | `9570b5b3743be89cd8ca3f7f75a95bc43cd8445336a3d73694858c79ae05dd82` |

Freshly verified checkpoint byte hashes and metadata on server: each has178 model tensors/buffers; epoch53/44 and selection=legacy match receipts. Their separately saved raw-f-selected aliases contain bitwise identical model tensors at the same selected epochs. Keep legacy aliases rather than silently changing provenance.

Their original configs match seed11's architecture/optimizer/objective: MTO16, query32, router/head128, fullDetaNet, LE+Ls, AdamAMSGrad lr.001, batch64, WD0, clip5, FP32, AMP/TF32off. Only seed/name/device and100epoch cap differed. ReduceLROnPlateau still used legacy objective; min_epochs200 prevented early stopping before the cap. The seed11 original best33 over236epochs was also its legacy best within the first100. This is a reasonably aligned selected-prefix source family, but training durations and per-seed selected epochs differ; do not present them as a precise matched-budget seed-variance estimate.

## Confirmation design to freeze after pilot decision

- If the new slow-continuation control gains≥.003 against its seed11 epoch0, and no architecture qualifies, run that same20epoch control schedule once from each of the two fixed sources. Primary within-seed contrast is selected control versus its own epoch0.
- If one architecture qualifies by≥.003 against both seed11 control and epoch0, freeze that single candidate identity using native raw-f validation only. Run a matched control/candidate pair from each of the fixed seed23/37 sources. Do not expand/reselect candidates after confirmation outcomes.
- Keep exact20epochs, lr1e-5, AdamAMSGrad reset, WD0, batch64, clip5, FP32, same original LE+Ls and unchanged λ=.001 where applicable. No schedule/lambda/gate-width retuning.
- Within each seed, use identical baseline weights and batch order. Use order_seed23/37 and adapter_seed23/37 for their respective pairs; reset runtime RNG after constructors. Record source checkpoint/config hashes and all order hashes.
- Include epoch0 and select each arm by its minimum pooled native raw-f validation SSE with earliest exact ties. Report the same frozen α/β calibration only as secondary for the native-selected checkpoint; no refit or alternate selection.
- Keep the original split/raw labels, masks, training-derived q90=.0549/q99=.2412, state errors, E errors and all valid samples. No test inference during confirmation; do not call this fresh holdout evidence.
- Report each seed's ΔR² and relativeSSE separately and a descriptive average of metric changes if useful. Never average predictions or checkpoints. With two new seeds, positive mean alone can hide reversal; root should freeze its required consistency/effect criterion before launch. Recommended minimum for a replication claim: both new within-seed contrasts positive, plus the prespecified practical mean-delta requirement; bootstrap remains conditional on validation selection.

## Required adaptation/review before execution

The current pilot runner asserts source epoch33 and fixed seed11 source_config. It must not be reused unchanged for53/44 checkpoints. A separate confirmation namespace/config should generalize expected source epoch/config, freeze exact baseline byte hashes, and first replay each source's full validation anchor. The added adapter remains zero-initialized with nonzero Jacobian.

Existing pilot GPUs1/2/4/6 may be used only after they are terminal and locks/resources rechecked. GPU0 is unrelated,3/7 excluded; no future reservation is implied here. Freeze code/protocol and checkpoint-source manifests, independent review, then server→Darchive→staged inspection→commit/push before launching the confirmation. Keep full resumable states and all weights server-only.

Evidence: `research/oscillator_r2_20260928/eta0_seed_replication/{PROTOCOL.md,completion_receipts/SEED_FAMILY_SCIENTIFIC_CLOSEOUT.md,runs/seed23/{RUN_MANIFEST.json,FIT_COMPLETE.json},runs/seed37/{RUN_MANIFEST.json,FIT_COMPLETE.json}}`. The older protocol's word “untouchedtest” is outdated; all new records must identify the shared historical test as already exposed.
