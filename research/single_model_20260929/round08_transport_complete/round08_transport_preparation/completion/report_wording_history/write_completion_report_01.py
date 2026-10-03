"""Render saved aggregate JSON only; no model, checkpoint or array access."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'completion'
r=json.loads((OUT/'ROUND08_RESULTS.json').read_text());arms=('original','local','neighbor')
ref=r['retained_reference']['metrics'];lines=[]
def add(s=''):lines.append(s)
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('| '+' | '.join('---' for _ in headers)+' |')
    for row in rows:add('| '+' | '.join(map(str,row))+' |')
    add()
f=lambda x:f'{x:.6f}'
add('# Round08 — neighbor transport improves the paired controls but fails promotion');add()
add('**The frozen triple promotion gate fails.** Selected validation pooled raw-f R² is **.417636 original**, **.433924 local** and **.443968 neighbor**. Neighbor exceeds both current controls but remains below the retained Round05 reference **.447169**. No new best verified recipe or achievement of the .60 target is established.');add()
add('All three runs completed 60 epochs and 112,860 updates with one original attempt each, no failure and no resume. Every validation score retains 6,686 molecules × 10 states = 66,860 printed raw-f labels, including zeros. Earliest minimum pooled native validation SSE selects across epochs 0–60. No calibration, exclusions, label permutation, averaging or TEST inference was used.');add()
add('## Selected checkpoints');add()
table(['Arm','Epoch','Pooled R²','f RMSE','f MAE','Energy MAE (eV)','Energy RMSE (eV)'],[
    [a,r['arms'][a]['selected_epoch'],*[f(m['pooled'][k]) for k in ('r2','rmse','mae')],f(m['energy']['mae']),f(m['energy']['rmse'])]
    for a in arms for m in [r['arms'][a]['selected_best']]]+
    [['retained Round05 control',45,*[f(ref['pooled'][k]) for k in ('r2','rmse','mae')],f(ref['energy']['mae']),f(ref['energy']['rmse'])]])
g=r['gate']
add(f"Neighbor ΔR² is {f(g['candidate_delta_r2_vs_contemporaneous_original'])} versus original, {f(g['candidate_delta_r2_vs_contemporaneous_local'])} versus local and {f(g['candidate_delta_r2_vs_retained_reference'])} versus retained. Each comparison required at least +.003. The local control gains .016287 over this round's original but remains .013246 below retained. Preserve these paired gains descriptively; they authorize no scale change, extension or confirmation seeds.");add()
add('## Fixed epoch 60');add()
table(['Arm','Pooled R²','f RMSE','f MAE','Energy MAE (eV)','Energy RMSE (eV)'],[
    [a,*[f(m['pooled'][k]) for k in ('r2','rmse','mae')],f(m['energy']['mae']),f(m['energy']['rmse'])] for a in arms for m in [r['arms'][a]['fixed60']]])
add('Neighbor remains above the current controls at the aligned final epoch. All arms fall below their selected checkpoints. This is a separate fixed-budget comparison, not a replacement selector or reason to extend training.');add()
add('## Per-state errors');add()
for policy,label in [('selected_best','Selected checkpoints'),('fixed60','Fixed epoch 60')]:
    add('### '+label);add();add('Each state has 6,686 labels. Cells show R² / raw-f RMSE / MAE.');add()
    table(['State',*arms],[[f'S{k}',*[f"{f(r['arms'][a][policy]['per_state'][str(k)]['r2'])} / {f(r['arms'][a][policy]['per_state'][str(k)]['rmse'])} / {f(r['arms'][a][policy]['per_state'][str(k)]['mae'])}" for a in arms]] for k in range(1,11)])
for a in ('local','neighbor'):
    improved=[f'S{k}' for k in range(1,11) if r['arms'][a]['selected_best']['per_state'][str(k)]['sse']<r['arms']['original']['selected_best']['per_state'][str(k)]['sse']]
    add(f"Selected {a} has lower SSE than original on {', '.join(improved)}. All states remain included.")
add();add('## Bright tails and error concentration');add()
add('Fixed TRAIN thresholds are q90=.0549 and q99=.2406. These descriptive subgroups do not alter pooled metrics.');add()
table(['Policy','Arm','Tail','Count','RMSE','MAE','SSE'],[[p,a,q,m['count'],f(m['rmse']),f(m['mae']),f(m['sse'])] for p in ('selected_best','fixed60') for a in arms for q in ('q90','q99') for m in [r['arms'][a][p]['bright_tail'][q]]])
keys=['true_0_pred_0','true_0_pred_1','true_1_pred_0','true_1_pred_1']
add('The four true/predicted q99 bins sum to all 66,860 labels and total SSE. Cells show count / SSE. Membership depends on each model\'s predictions, so changes are descriptive error decompositions, not causal effects on a fixed subgroup.');add()
table(['Policy','Arm','T0/P0','T0/P1 false bright','T1/P0 missed bright','T1/P1'],[[p,a,*[f"{b[k]['count']} / {f(b[k]['sse'])}" for k in keys]] for p in ('selected_best','fixed60') for a in arms for b in [r['arms'][a][p]['false_bright_bins']]])
add('Selected neighbor reduces false-bright SSE from16.607218 to7.251549 versus original, while pooled MAE rises from.017886 to.018317, true-q99 RMSE from.227382 to.234698, and energy MAE from.092151 to.097017 eV. Relative to local, neighbor improves pooled R², true-tail RMSE and false-bright SSE, but pooled MAE and energy error are slightly worse. By epoch60 neighbor true-q99 RMSE improves to.228890 while false-bright SSE rises to17.944790 and pooled R² declines. These are mixed error tradeoffs, not uniform improvement or a QC mechanism.');add()
add('Relative to the retained Round05 selected aggregate, neighbor also has worse pooled MAE, energy error and true-q90/q99 RMSE. The retained-reference gate and these tradeoffs jointly argue against a new best-recipe claim.');add()
table(['Policy','Arm','Top 1% count','Top 1% SSE share','Prediction q99','Prediction max','Abs. error q99','Abs. error max'],[
    [p,a,e['top_error_count'],f(e['top_error_sse_share']),f(e['prediction_quantiles']['0.99']),f(e['prediction_quantiles']['1.0']),f(e['absolute_error_quantiles']['0.99']),f(e['absolute_error_quantiles']['1.0'])] for p in ('selected_best','fixed60') for a in arms for e in [r['arms'][a][p]['error_distribution']]])
add('Additional quantiles, per-state SSE and all counts remain in ROUND08_RESULTS.json. No large-error row was removed.');add()
add('## Training and transport activity');add()
add('TRAIN metrics are weighted pre-update minibatch aggregates, not fixed-checkpoint evaluations. Validation is evaluated at each completed checkpoint. All arms optimize original LE+Ls; native raw-f MSE and decorrelation are diagnostics. These trajectories do not measure an exactly matched TRAIN/validation generalization gap.');add()
table(['Arm/epoch','TRAIN f MSE','TRAIN LE','TRAIN Ls','VAL LE','VAL Ls','Clip fraction'],[
    [f'{a}/{ep}',f(t['train']['raw_f_mse']),f(t['train']['energy']),f(t['train']['trace']),f(t['validation']['base_objective']['energy']),f(t['validation']['base_objective']['trace']),f(t['gradient_clip_fraction'])] for a in arms for ep,t in r['arms'][a]['trajectory_snapshots'].items()])
add('Between selected and final epochs all arms reduce TRAIN f/energy/trace loss while validation pooled R² deteriorates. Neighbor TRAIN f MSE falls from.001156 at23 to.000489 at60; validation falls from.443968 to.408616. Energy and true-tail errors improve over that interval while false-bright errors grow. No nonfinite or optimizer failure was recorded. This supports investigating generalization and optimization, without identifying a cause.');add()
table(['Arm/epoch','Transport grad norm','Within-epoch theta movement','Block2 l1 residual/message','Block3 l1 residual/message'],[
    [f'{a}/{ep}',f(d['gate_gradient_l2']),f(t['gate_parameter_movement_l2']),f(d['block2_l1_regularized_relative_to_message_mean']),f(d['block3_l1_regularized_relative_to_message_mean'])] for a in arms for ep,t in r['arms'][a]['trajectory_snapshots'].items() for d in [t['transport_diagnostics']]])
add('The ratio columns are atom-weighted means of norm(delta)/max(norm(original message),1e-12), with zero/below-floor denominator counts retained in JSON. Each epoch/block/irrep has2,164,714 atom observations. They are regularized ratios, not activation rescaling; the denominator floor can affect tiny-message cases. The complete six block/irrep trajectories and preblock-T ratios are saved. Theta/gradient summaries are molecule weighted; within-epoch movement is not distance from initialization.');add()
table(['Arm','Checkpoint','Theta L2','tanh(theta) min','max','mean absolute'],[[a,p,f(v['theta_l2']),f(v['coefficient_min']),f(v['coefficient_max']),f(v['coefficient_mean_abs'])] for a in arms for p,v in r['arms'][a]['checkpoint_transport_aggregates'].items()])
add('Both active branches learned nonzero theta and residuals. At selected checkpoints theta L2 is8.478 local and9.376 neighbor; selected l1 mean residual/message ratios are approximately.007–.009. The initial nine-update fixture\'s tiny effects therefore do not describe the full trained paths. These aggregate observations neither justify a post hoc scale sweep nor establish physical tensor transport; coefficient extrema do not measure saturation prevalence. Original theta/residuals remain exactly zero.');add()
add('The candidate adds a direct pre-block same-irrep neighbor-T dependency in blocks2/3; original T already influences scalar messages indirectly. Both active arms have768 parameters and the same insertion/edge gates, but different function classes and residual magnitudes. The source distinction does not establish a capacity ceiling or electronic coherence, density or phase mechanism. The original PSD decoder remains unchanged.');add()
add('## Conditional uncertainty and repeated controls');add()
u=r['paired_uncertainty'];add(f"Paired bootstrap resamples {u['component_groups']} validation identity components, retaining all6,686 molecules and ten states, with seed{u['seed']}, {u['draws']} draws and resampled pooled SST. These intervals are conditional on selected checkpoints and reused validation; they do not account for training-seed variability or model selection. No paired interval is available against the aggregate-only retained reference.");add()
table(['Policy','Contrast','Delta R²','Descriptive 95% interval'],[[p,c,f(v['delta_r2']),f"[{f(v['descriptive_percentile_interval'][0])}, {f(v['descriptive_percentile_interval'][1])}]"] for p in ('selected_best','fixed60') for c,v in u['comparisons'][p].items()])
add('All six intervals include zero. The positive paired point estimates are not independent confirmation. Prior declared-original controls selected.447169 (Round05),.404798 (Round06),.409104 (Round07), and.417636 here. Base initialization, TRAIN order and core settings match, but execution graphs, dormant schemas, diagnostics and workloads differ. No cause of control variation was isolated. The retained-reference screen prevents promotion solely against a weaker rerun.');add()
add('## Recipe, artifacts and limitations');add()
add('The retained v2 single checkpoint remains Round05 original control epoch45, R².44716940136585204. Original PSD MTO16/query32/router128/head128, core128/three blocks, fresh seed/order11, TRAIN-only stats, LE+Ls, Adam AMSGrad fixedLR.001/batch64/WD0/clip5, FP32/noAMP/noTF32, selected within60 epochs. It is still below the .60 goal and has no new TEST or independent-seed confirmation.');add()
add('Retained artifact: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA256 `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`. Use its matching Round05 loader.');add()
add('Round08 server prefix: `/home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation/runs/`. Standalone exports contain one full model, transport mode/contract, model config and TRAIN statistics. No original checkpoint, QC labels or cache is needed at deployment.');add()
table(['Arm','Epoch','Standalone relative path','SHA256'],[[a,r['arms'][a]['selected_epoch'],a+'/geometry_best.pt',r['verification'][a]['checkpoint_hashes']['geometry_best.pt']] for a in arms])
table(['Arm','Measured training + validation hours'],[[a,f(r['arms'][a]['wall_training_validation_seconds']/3600)] for a in arms])
add('All258 frozen source pins, authority/review/publication bindings,60 orders,61 history rows,112,860 updates, earliest selection, optimizer membership/Adam state/RNG, access ledgers and checkpoint/prediction hashes passed. Selected exports equal their selected training tensors; mode/transport contract and stored buffer fingerprints match. No model was reconstructed or evaluated in this terminal audit; strict standalone loading was tested in preparation. Original theta matches its initial hash, active theta is nonzero, and dormant right-F tensors agree across all six selected/final checkpoints.');add()
add('One CPU analysis read six saved validation output sets, CPU checkpoint payloads, validation component metadata and the published retained aggregate. It opened no raw dataset target or TEST array and ran no model inference. Process absence/zombie establishes exit evidence, not an observed OS exit code. The report renderer reads aggregate JSON only. ANALYSIS_RECEIPT binds input/source/output hashes; ROUND08_REPRODUCE documents exact commands and completed-stage refusal. All weights/optimizer/coefficient arrays and private data stay on the server.');add()
add('The v2 split is disjoint under audited conservative identity rules, not absolute chemical-identity proof or external fresh data. Its TEST contains5,989 old-TRAIN,361 old-validation and336 old-TEST molecules. Current TEST remains sealed. Historical exposure, reused validation, one initialization and observed control variability limit generalization claims.');add()
add('![Validation and TRAIN trajectories](ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg)');add()
add('![Transport trajectories](TRANSPORT_TRAJECTORIES.svg)');add()
add('## Decision boundary');add()
add('Close this frozen contrast with a failed promotion screen and retain the stronger Round05 checkpoint. Preserve neighbor/local gains and tradeoffs as descriptive evidence. No scale tuning, extension, seed allocation or new fit is implied. Root owns the final decision and D-first publication. NEXT_RESEARCH_QUESTION proposes one regularization question for a future decision only; it authorizes no computation.');add()
p=OUT/'ROUND08_REPORT.md';assert not p.exists(),'Do not overwrite completed report';p.write_text('\n'.join(lines)+'\n')
print(json.dumps({'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
