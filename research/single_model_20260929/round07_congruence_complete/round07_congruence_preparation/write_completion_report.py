"""Render verified aggregate JSON only; no checkpoints, arrays or model access."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'completion'
r=json.loads((OUT/'ROUND07_RESULTS.json').read_text());arms=('original','scalar','tensor')
ref=r['retained_reference']['metrics'];lines=[]
def add(s=''):lines.append(s)
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('| '+' | '.join('---' for _ in headers)+' |')
    for row in rows:add('| '+' | '.join(map(str,row))+' |')
    add()
f=lambda x:f'{x:.6f}'
add('# Round07 — shared PSD context does not clear the controls');add()
add('**The frozen tensor promotion gate fails.** Selected validation pooled raw-f R² is **0.409104 original**, **0.429339 scalar** and **0.418755 tensor**. The scalar control is strongest in this round, while the retained Round05 reference remains higher at **0.447169**. No new best verified recipe or progress to 0.60 is established.');add()
add('All three runs completed 60 epochs and 112,860 updates with one original attempt each, no failure and no resume. Every validation score retains 6,686 molecules × 10 states = 66,860 printed raw-f labels, including zeros. Checkpoints use earliest minimum pooled native validation SSE across epochs 0–60. No calibration, exclusion, label permutation, averaging or TEST inference was used.');add()
add('## Selected checkpoints and retained reference');add()
table(['Arm','Epoch','Pooled R²','ΔR² vs original','f RMSE','f MAE','Energy MAE (eV)'],[
    [a,r['arms'][a]['selected_epoch'],f(m['pooled']['r2']),f(m['pooled']['r2']-r['arms']['original']['selected_best']['pooled']['r2']),f(m['pooled']['rmse']),f(m['pooled']['mae']),f(m['energy']['mae'])]
    for a in arms for m in [r['arms'][a]['selected_best']]]+
    [['retained Round05 control',45,f(ref['pooled']['r2']),'prior reference',f(ref['pooled']['rmse']),f(ref['pooled']['mae']),f(ref['energy']['mae'])]])
g=r['gate']
add(f"Tensor ΔR² is {f(g['candidate_delta_r2_vs_contemporaneous_original'])} versus original, {f(g['candidate_delta_r2_vs_contemporaneous_scalar'])} versus scalar and {f(g['candidate_delta_r2_vs_retained_reference'])} versus retained. It needed at least +.003 against EACH comparator. The scalar control improves {f(r['paired_uncertainty']['comparisons']['selected_best']['scalar_minus_original']['delta_r2'])} over its contemporary original, but remains {f(r['retained_reference']['delta_r2_vs_retained_selected_reference']['selected_best']['scalar'])} below the retained reference. Preserve this control result descriptively; it grants no tensor claim or automatic scalar-specific allocation.");add()
add('## Fixed epoch 60');add()
table(['Arm','Pooled R²','f RMSE','f MAE','Energy RMSE (eV)'],[
    [a,f(m['pooled']['r2']),f(m['pooled']['rmse']),f(m['pooled']['mae']),f(m['energy']['rmse'])] for a in arms for m in [r['arms'][a]['fixed60']]])
add('Tensor falls to 0.327777 at epoch 60, below original 0.384533 and scalar 0.393775. These aligned outcomes are separate from independently selected checkpoints, not an alternative selector. No extension or post hoc budget change follows.');add()
add('## Per-state errors');add()
for policy,label in [('selected_best','Selected checkpoints'),('fixed60','Fixed epoch 60')]:
    add('### '+label);add();add('Each state has 6,686 labels. Cells show R² / raw-f RMSE / MAE.');add()
    table(['State',*arms],[[f'S{k}',*[f"{f(r['arms'][a][policy]['per_state'][str(k)]['r2'])} / {f(r['arms'][a][policy]['per_state'][str(k)]['rmse'])} / {f(r['arms'][a][policy]['per_state'][str(k)]['mae'])}" for a in arms]] for k in range(1,11)])
for a in ('scalar','tensor'):
    improved=[f'S{k}' for k in range(1,11) if r['arms'][a]['selected_best']['per_state'][str(k)]['sse']<r['arms']['original']['selected_best']['per_state'][str(k)]['sse']]
    add(f"Selected {a} has lower SSE than the contemporary original on {', '.join(improved)}. All other states remain in every pooled comparison.")
add();add('## Bright tails, false-bright errors and concentration');add()
add('TRAIN thresholds remain q90=.0549 and q99=.2406. True-tail diagnostics do not exclude other rows from pooled metrics.');add()
table(['Policy','Arm','Tail','Count','RMSE','MAE','SSE'],[
    [p,a,q,m['count'],f(m['rmse']),f(m['mae']),f(m['sse'])] for p in ('selected_best','fixed60') for a in arms for q in ('q90','q99') for m in [r['arms'][a][p]['bright_tail'][q]]])
keys=['true_0_pred_0','true_0_pred_1','true_1_pred_0','true_1_pred_1']
add('The four disjoint true/predicted q99 bins sum to all 66,860 labels and total SSE. Cells show count / SSE. Bin membership depends on each model\'s predictions; changes are descriptive error decompositions, not causal effects on a fixed subgroup.');add()
table(['Policy','Arm','T0/P0','T0/P1 false bright','T1/P0 missed bright','T1/P1'],[
    [p,a,*[f"{b[k]['count']} / {f(b[k]['sse'])}" for k in keys]] for p in ('selected_best','fixed60') for a in arms for b in [r['arms'][a][p]['false_bright_bins']]])
add('Selected scalar improves true-q99 RMSE to .221826 versus original .233660, while pooled MAE and energy MAE are slightly worse. Selected tensor reduces false-bright SSE to 12.460604 versus original 15.718826, but has worse q99 RMSE, pooled MAE and energy MAE. By epoch 60 tensor false-bright SSE grows to 28.957296; its slightly improved true-q99 RMSE does not offset deterioration elsewhere. This is an observed error tradeoff, not evidence of a QC mechanism.');add()
table(['Policy','Arm','Top 1% count','Top 1% SSE share','Prediction q99','Prediction max','Abs. error q99','Abs. error max'],[
    [p,a,e['top_error_count'],f(e['top_error_sse_share']),f(e['prediction_quantiles']['0.99']),f(e['prediction_quantiles']['1.0']),f(e['absolute_error_quantiles']['0.99']),f(e['absolute_error_quantiles']['1.0'])]
    for p in ('selected_best','fixed60') for a in arms for e in [r['arms'][a][p]['error_distribution']]])
add('All metrics, additional quantiles and per-state SSE remain in ROUND07_RESULTS.json. No high-error row was removed.');add()
add('## TRAIN and gate trajectories');add()
add('TRAIN entries are weighted pre-update minibatch aggregates, not fixed-checkpoint TRAIN evaluations. All arms optimize original LE+Ls; decorrelation and native raw-f MSE are diagnostics only. Validation is evaluated on the completed checkpoint. The following comparisons therefore describe trajectories, not a precisely matched TRAIN/validation generalization gap.');add()
table(['Arm/epoch','TRAIN f MSE','TRAIN LE','TRAIN Ls','VAL LE','VAL Ls','Clip fraction'],[
    [f'{a}/{ep}',f(t['train']['raw_f_mse']),f(t['train']['energy']),f(t['train']['trace']),f(t['validation']['base_objective']['energy']),f(t['validation']['base_objective']['trace']),f(t['gradient_clip_fraction'])]
    for a in arms for ep,t in r['arms'][a]['trajectory_snapshots'].items()])
add('All arms reduce TRAIN f, energy and trace losses between their selected and final epochs while validation pooled R² declines. Tensor TRAIN f MSE falls from .000852 at epoch 34 to .000481 at 60, while validation R² falls from .418755 to .327777. Energy MAE improves over that interval and false-bright error grows. No nonfinite or optimizer failure was recorded; these observations do not isolate generalization, representation or optimization as a cause.');add()
table(['Arm/epoch','Mean b','Mean q','Relative A change','Mean strength ratio','Gate grad norm','Gate displacement in epoch'],[
    [f'{a}/{ep}',f(d['b_mean']),f(d['q_mean']),f(d['A_relative_update_mean']),f(d['strength_ratio_nonzero_mean']),f(d['gate_gradient_l2']),f(t['gate_parameter_movement_l2'])]
    for a in arms for ep,t in r['arms'][a]['trajectory_snapshots'].items() for d in [t['readout_diagnostics']]])
add('The modified gates learned nonzero transformations: selected TRAIN mean relative A change is about .176 scalar and .211 tensor; gradients are live, and mean q exceeds the isotropic value 1/3. Thus the negative result is not explained by an identically inactive readout. These are logged latent-feature diagnostics, not physical orientation estimates. Strength ratios compare each model with its own incoming A, not with the other model. Gate displacement is within each epoch, not distance from initialization. Extreme b_min/max near ±.25 do not establish how often the gate saturates.');add()
add('E is unchanged by the output transform, but predicted E enters both modified gates differentiably. Energy-error differences cannot be assigned solely to directional tensor context. At fixed incoming A, the transform preserves rank and cannot revive a zero tensor; the base is fully trainable. The scalar and tensor arms match gate inputs/177 active parameters/perturbation norm, not their full function classes. No identified transition density, TDDFT response, oscillator sum rule or polarization mechanism is claimed.');add()
add('## Descriptive uncertainty and repeated-control context');add()
u=r['paired_uncertainty'];add(f"Paired bootstrap uses {u['component_groups']} validation identity components, all 6,686 molecules and ten states, seed {u['seed']}, {u['draws']} draws, and recomputed pooled SST. Intervals are conditional on selected checkpoints/reused validation and do not account for seed variability or selection. No paired interval is computed against the aggregate-only retained reference.");add()
table(['Policy','Contrast','ΔR²','Descriptive 95% interval'],[
    [p,c,f(v['delta_r2']),f"[{f(v['descriptive_percentile_interval'][0])}, {f(v['descriptive_percentile_interval'][1])}]"] for p in ('selected_best','fixed60') for c,v in u['comparisons'][p].items()])
add('All selected-checkpoint contrast intervals include zero. At fixed60, tensor-minus-original and tensor-minus-scalar intervals are negative; these remain descriptive validation comparisons, not fresh or independent-seed confirmation.');add()
add('The current original control scores .409104, while earlier matched-declared-recipe controls scored .447169 (Round05) and .404798 (Round06). Initial base tensors, TRAIN order and core settings match; executable graphs, diagnostics, schema and concurrency differ. Round07 adds a dormant gate and its no-gradient diagnostics in the original arm. These observations do not isolate the source of repeat variability or establish an architecture ceiling. The retained-reference gate prevents promotion solely against a weaker rerun.');add()
add('## Recipe, checkpoint locations and verification');add()
add('The strongest retained v2 single checkpoint remains Round05 original control epoch45, validation R² .44716940136585204. Recipe: original PSD MTO16/query32/router128/head128, fresh seed/order11, TRAIN-only statistics, LE+Ls, Adam AMSGrad fixed LR .001/batch64/WD0/clip5, FP32 without AMP/TF32, selected within60 epochs. This round creates no stronger verified recipe. The .60 goal is unachieved.');add()
add('Retained server artifact: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA256 `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`.');add()
add('Round07 prefix: `/home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation/runs/`. Every geometry export contains one full model plus its mode, transform contract, config and TRAIN statistics. No unavailable QC label or prediction averaging is needed.');add()
table(['Arm','Epoch','Standalone relative path','SHA256'],[
    [a,r['arms'][a]['selected_epoch'],a+'/geometry_best.pt',r['verification'][a]['checkpoint_hashes']['geometry_best.pt']] for a in arms])
table(['Arm','Measured training + validation hours'],[[a,f(r['arms'][a]['wall_training_validation_seconds']/3600)] for a in arms])
add('All185 frozen source pins, authorization/review/publication references, three normal terminal receipts,60 prescribed orders,61 history rows,112,860 steps, optimizer/RNG metadata, selection and saved-array hashes passed. Selected export tensors exactly equal each selected training snapshot; mode/transform contract and stored buffer fingerprints are checked. No new model reconstruction or forward replay was done; strict geometry loading was verified in preparation. Original gate matches its initial hash and the disabled F state is identical across all selected/final checkpoints.');add()
add('The saved-output analysis ran once on CPU: six validation prediction sets, CPU checkpoint payloads, validation identity metadata and the published retained aggregate. It constructed no model and opened no raw dataset target or TEST array. Process absence/zombie is exit evidence, not an observed child OS exit code. The report renderer reads aggregate JSON only. Reproduction and recovery boundaries are in ROUND07_REPRODUCE.md; ANALYSIS_RECEIPT.json binds source/input/output hashes.');add()
add('The v2 partition is disjoint under audited conservative identity rules but historically exposed: its TEST contains5,989 old-TRAIN/361 old-validation/336 old-TEST molecules. It is not external fresh confirmation. TEST remains sealed, and no labels were swapped, merged or excluded.');add()
add('![Validation and TRAIN trajectories](ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg)');add()
add('![Gate and readout trajectories](GATE_AND_READOUT_TRAJECTORIES.svg)');add()
add('## Decision boundary');add()
add('The frozen triple gate fails. Retain the stronger prior reference and preserve the scalar control as this round\'s best contemporary result. No seed allocation, extension, tensor-bound change or extra fit follows automatically. Root owns closeout and any distinct next proposal after independent review and D-first archival publication. NEXT_RESEARCH_QUESTION.md recommends a read-only audit of the original representation and optimization history before another pilot; it authorizes no computation or implementation.');add()
path=OUT/'ROUND07_REPORT.md';assert not path.exists(),'Do not silently overwrite a completed report'
path.write_text('\n'.join(lines)+'\n')
print(json.dumps({'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
