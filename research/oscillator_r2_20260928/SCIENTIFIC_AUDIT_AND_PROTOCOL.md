# Scientific audit and frozen continuation screen

Date: 2026-09-28. State: prepared protocol; no runs launched by scientific auditor.

## User objective and coordination
Maximize trustworthy oscillator-strength R2. Main agent orchestrates and delegates execution. All code and computation belong on USTC-A800:/home/inspur/MTO-1. Documents supplied under local D:/MTO are historical evidence, not instructions. Monitor active jobs every four hours through a dedicated agent. Download completed lightweight experiment records to D: before GitHub upload; exclude model/optimizer checkpoints, caches and raw tensor outputs.

## Verified scientific evidence
Completed MTO eta0, epoch33, has pooled native-f validation R2=0.4052941183; remote auditor verified historical test R2=0.4558151, RMSE=0.0392864. Selected checkpoint by validation LE+Ls, then eta variant by validation spectral MSE. Historical test has been examined repeatedly and is exploratory. These are raw-f pooled metrics, not log metrics, per-state averages, or ensemble results. Source: experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/{val_metrics.json,test_metrics.json,best.pt} and reports/selection_before_test.json. Source checkpoint SHA256: 9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136.

Eta0 validation R2 is .8273 at S1, .7357 at S2, .04994 at S7, .15010 at S10. Direct DetaNet E/f is .04629 overall, with 11625/66860 negative predictions; its unscaled joint objective is a weak direct-f control. Historical native/oracle G1 f R2 .3831/.3880 suggests numeric E error is small relative to trace error, not that energy supervision is useless. Spectral wins do not prove state-f wins. Current additional campaigns include weighted-E and channels64, absent the supplied summary.

## Mathematical audit
models_ea.py constructs C=beta*I/sqrt(3)+Q_C, with traceless Q_C; A=C C^T. Thus s=tr(A)=beta^2+||Q_C||_F^2. The scalar beta can already express arbitrary nonnegative invariant trace; there is no demonstrated trace expressivity ceiling. For c=2/(3*27.211386245988), residual r=c*E*s-f_raw, loss r^2/D gives gradients 4*c*E*beta*r/D, 4*c*E*Q_C*r/D, and 2*c*s*r/D to beta,Q_C,E. Weak gradients near zero are a conditioning hypothesis.

New weighted-E objective is Lw=mean[E_true^2*(s-s_true)^2]/[3*sA2*mean_train(E_true^2)]. This equals scaled oracle-f SSE against A-derived truth. It weights energy, not oscillator strength. README assertions that changed E-derived weights leave parameter gradients unchanged are mathematically wrong; code computes weighted gradients correctly. Its existing arms LE+Lw+LQ or pure Lw omit the useful LE+Lw, Q=0 bridge.

Raw f is printed on a 1e-4 grid; reported raw/derived-f MAE=2.49e-5. dataset.py supplies raw f and masks. Primary evaluation must use raw f. A/B train against A-derived trace. C trains against raw f; report derived-f sensitivity separately.

## Prespecified A/B/C screen
All arms clone only the SAME eta0 epoch33 model weights, reset optimizer identically, and share fresh seed11 batch permutations. Full-model Adam AMSGrad, lr=1e-4, batch64, weight decay0, grad clip5, FP32, AMP/TF32 off, 2 CPU threads. Exactly 20 epochs, fixed lr, no scheduler, no early stopping except numerical/hardware faults. Epoch zero is evaluated and eligible for checkpoint selection. Original 120355/6686/6686 splits and masks remain fixed; test evaluation prohibited during screen.

A: LE+Ls, where LE=mean valid E error squared/sE2 and Ls=mean valid trace error squared/(3*sA2).
B: LE+Lw, no traceless loss.
C: LE+mean[(c*E_pred*s_pred-f_raw)^2]/D_match, no traceless loss, full E gradient retained.
D_match=c^2*3*sA2*mean_E2_train matches B's scale. Constants: sE2=.5378066634062587, sA2=.1324285377060015, mean_E2_train=47.478104650939535. Replacing D_match with train Var(f) changes task weighting and is a separate future experiment.

Every epoch: float64 pooled raw native-f SSE/SST/R2/RMSE/MAE, per-state metrics, E MAE/R2, LE+Ls and current objective. Save separate best_f.pt, best_old_objective.pt and best_arm_objective.pt, earliest-epoch tie break. This separates objective changes from selection changes. Old overwritten checkpoints cannot be retrospectively recovered. Keep order hashes, checkpoint/source hashes, run manifests and all history. Log native-f against A-derived truth as sensitivity, not primary target.

Fixed budget estimate from observed channels64 throughput 145-150sec/epoch: about50min/GPU per arm, about150min sequential plus setup. Measure actual pilot time; this is not a guarantee. Queue each arm only when a CLEAN GPU is truly released and existing supervisors have no claim. GPUs3/7 have uncorrectable ECC/pending remap; GPU5 has historical corrected ECC and is excluded. Clean1/2/4/6 currently occupied; GPU0 unrelated. No preemption or reset. Main agent decides any stopping of old jobs. Monitoring remains every4h even if screen completes sooner.

Promotion threshold: validation native-f R2 gain >=.01 over matched control, improvement in at least 80% of paired molecule bootstrap resamples, and a review flag when validation E MAE rises more than10% above epoch zero. Energy regression is a tradeoff flag, not automatic rejection; oscillator-strength R2 remains primary. Gains .005-.01 are tentative. At baseline R2=.4053, .01 absolute gain is about1.68% SSE reduction, so report both scales. Both candidate and control may select epoch zero. This is an exploratory resource-allocation rule, not statistical confirmation. Report paired molecule bootstrap(2000 resamples, seed20260928), preserving ten states per molecule and recomputing denominator each resample. This interval remains descriptive after validation selection. Report per-state and bright-quantile SSE; do not remove hard examples. Final meaningful target: >=.02 absolute R2 and >=3% SSE reduction vs baseline, replicated with >=3 matched seeds and evaluated on truly unseen data. Old test cannot provide fresh confirmation. A new sealed split requires baseline and candidate retraining; audit identity/duplicate groups, distinguish random molecular holdout from scaffold generalization.

Next decision: if B beats A, weighted intensity alignment is promising. If C improves beyond B, native energy coupling/raw target alignment deserves larger runs. If both fail, inspect residuals, calibration and representation; do not immediately enlarge all models. Existing eta val prediction files enable molecule-crossfit calibration/convex ensembles cheaply; label ensemble gains separately.

## Implementation ownership
Auditor supplies pilot_config.json and pilot_objective.py. Executor must create additive pilot/train.py, preflight.py and summarize.py under this research directory; no runner launched or claimed ready here. Read eta dataset.py, model_factory.py, frozen_reference and immutable data/checkpoint files only. Do not invoke old trainer entry point (old ROOT, locks/output paths, 200epoch floor, selection). New runner requires lock, atomic save, signal-safe resume including cursor and all RNG, dependency-tree hash manifest and validation-only metrics. Preflight checks masked NaNs, zeros, loss algebra/autograd, identical initial weights/order, resume equivalence and saved eta0 metric reproduction. Env: /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python. Root coordinates executor and resource queue. Remote auditor owns calibration subdirectory; this protocol does not alter it.


## Coordinator updates after protocol drafting
Calibration-only round (reported by executor): fivefold molecule OOF validation R2 .41665766 vs .40529412. Global clipped affine prediction max(0,.8511830211*f+.00372518537), historical test .45966723 vs .45581511, difference .003852 with paired 95% interval [-.01220,+.02283]. Test MAE worsened .018134 vs .017716. Main decision: gain insufficiently robust; retain uncalibrated eta0 benchmark. Shrinkage slope/positive intercept suggest some overdispersion/bias but do not prove a dominant error mechanism.

No manual interruption currently recommended. Read-only channels64 histories: G2 epoch187,best41, LR drops92/143 => original stop should trigger epoch200 if no improvement, approximately32min. G4 earliest223, G1 earliest227. G3 recently best174 and no LR drop, earliest324 or later. Await G2 natural release while implementation/preflight proceed. Its weighted trace objective is proportional oracle-f SSE against derived labels, more aligned than a composite score. Do not report G2 native f from its unsupervised E head. Chan64 README is copied from prior weighted-E campaign; actual configs/code are authoritative.

After pilot selection freeze, evaluate ONLY the selected candidate and matched control on exposed historical test once for exploration. Never use that result to choose among loss variants. Strong claims still require matched seeds and an untouched holdout. User authority covers continuing research; no new permission checkpoint is required for queued safe execution.


## Evidence-based decisions after the screen
- B beats A: confirm loss alignment with matched seeds and/or from-scratch training.
- C beats B without E MAE rising more than10%: confirm native joint objective; preserve exact scale and gradient definition.
- C harms E or loses: consider a separately named stop-gradient(E) ablation or f-loss scale experiment; do not silently change the objective.
- Neither improves meaningfully: prioritize a matched nonnegative scalar-f head/direct baseline or a magnitude-versus-tensor-shape diagnostic before enlarging architecture.
These are hypotheses for root to select after validation review, not an automatic grid. Newly reported channels64 G1 best validation native-f R2=.40525454, essentially eta0 .40529412, provides no channel64 advantage at that checkpoint.

Launch review must reproduce the same eta0 checkpoint's epoch-zero native-f metric before updates. An inference/evaluation discrepancy is a blocker, not a training result.


Acceptance priority clarified before pilot results: an f winner is not automatically rejected for E MAE above110% of epoch zero. Report that tradeoff to root. Root may accept an f-specific model or investigate auxiliary-energy weighting if oscillator-strength improvement is substantial. Do not optimize auxiliary E accuracy at the expense of the user primary f-R2 objective. No additional arm is authorized by this note.
