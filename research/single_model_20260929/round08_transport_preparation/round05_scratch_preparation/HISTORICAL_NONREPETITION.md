# Exact historical scratch comparison and this proposal

Read-only source review, 2026-09-30. The accepted proposal snapshot stays unchanged. This note clarifies its compact history table from the actual historical scratch config/protocol.

Sources: `/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison/config.json` and `PROTOCOL.md`, plus `history_baseline.md` and the completed campaign reports. No dataset or model was opened.

The historical 100-epoch MTO result R²0.373492948 was **MTO-direct-f**, not the original C/PSD tensor readout. It used the same DetaNet core (128 features, three interaction blocks, l<=3, 32 Bessel functions, eight attention heads, radius5Å, dropout0), original MTO routing/coupling and scalar trunk, then a direct positive scalar-f head128→32→1. It removed beta/tensor-gate heads. Its paired native arm used the original atomwise scalar20 MLP and molecular summation; native R² was0.343781976.

Both historical arms trained from seed11/order11 on the old split for100epochs, batch64, AMSGrad, WD0, clip5, FP32. The schedule was1e-3 for epochs1–60,3e-4 for61–85,1e-4 for86–100. The objective was normalized energy plus **raw-f** squared error, not LE+Ls. Their f heads were initialized to identical TRAIN per-state means through zero final weights/fixed offsets; the original trainable energy_offset was changed to a fixed buffer. Thus their initialization, property head, objective, schedule beyond60 and backbone-to-target gradients differ from the accepted original-PSD factorial. Neither right-F nor raw-M decorrelation was a factor.

Historical eta0 seeds23/37 tested original tensor readout under100epoch replication; best epochs53/44 scored .368708895/.394455334. Seed11's earlier236epoch adaptive run chose epoch33 using its own validation objective, so those numbers are not a matched seed-variance estimate. The new factorial uses a contemporaneous original-tensor control, common fresh base initialization and common native raw-f selection rather than treating old eta0 as its matched comparator.

Channel64 G1 used eta1 original tensor readout, while G3 used a p_outer construction with eta1 and E-squared trace weighting. Their .40525459/.36810069 results do not isolate width or establish that every rank-one/response representation was cleanly tested. They also do not test the shared pre-CG F/raw-M factorial.

Round01 already tested this exact adapter/penalty mathematics as a20epoch low-LR continuation of one validation-selected model; Round02 froze that model and trained onlyF. The new scientific distinction is narrowly the ability of F and molecular representations to learn jointly from initialization at the original scratch rate. The v2 split and fresh target statistics apply equally to all four arms and cannot themselves establish an architecture benefit. No historical negative is erased, and no substantial gain is assumed.
