"""Render already verified aggregate results; no tensor/array/model access."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'completion'
r=json.loads((OUT/'ROUND06_RESULTS.json').read_text());arms=('trace_control','raw_f');ref=r['retained_reference']['metrics']
lines=[]
def add(s=''):lines.append(s)
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('| '+' | '.join('---' for _ in headers)+' |')
    for row in rows:add('| '+' | '.join(map(str,row))+' |')
    add()
f=lambda x:f'{x:.6f}'
add('# Round06 — unchanged PSD MTO, trace versus native raw-f supervision');add()
add('**The frozen promotion gate fails.** Raw-f supervision reaches selected validation pooled R² **0.422028**, above its contemporaneous trace control (**0.404798**) but below the retained Round05 reference (**0.447169**). The candidate does not clear +.003 over both comparators. No seed allocation, extension or TEST evaluation follows automatically. This is one matched-seed comparison, not proof that raw-f supervision cannot help under another justified protocol.');add()
add('Both runs completed60 epochs and112,860 updates, with one original attempt and no failure or resume. Every validation score includes6,686 molecules ×10 states =66,860 valid printed raw-f labels, including zeros. No calibration, target exclusions, state permutation, prediction averaging or TEST inference was used. The v2 partition is internally disjoint under audited identity rules but historically exposed; its sealed TEST contains5,989 old-TRAIN,361 old-validation and336 old-TEST molecules. It is not external fresh confirmation.');add()
add('## Selected checkpoints and retained reference');add()
rows=[]
for arm in arms:
    m=r['arms'][arm]['selected_best'];u=r['paired_uncertainty']['comparisons']['selected_best'][arm]
    rows.append([arm,r['arms'][arm]['selected_epoch'],f(m['pooled']['r2']),f(u['delta_r2_vs_control']),f(m['pooled']['rmse']),f(m['pooled']['mae']),f(m['energy']['mae'])])
rows.append(['retained Round05 control',45,f(ref['pooled']['r2']),'previous reference',f(ref['pooled']['rmse']),f(ref['pooled']['mae']),f(ref['energy']['mae'])])
table(['Arm','Epoch','Pooled R²','ΔR² vs current control','f RMSE','f MAE','Energy MAE (eV)'],rows)
gate=r['gate'];add(f"Candidate ΔR² is {f(gate['candidate_delta_r2_vs_contemporaneous_control'])} versus the current control and {f(gate['candidate_delta_r2_vs_retained_reference'])} versus the retained reference. Its required retained-reference score was0.450169. Checkpoints were selected by earliest minimum pooled validation SSE across epochs0–60, with no test-driven choice. The retained row is a previously published aggregate on the same reused validation, not a new inference or model average.");add()
add('## Fixed epoch60');add()
table(['Arm','R²','ΔR² vs control60','f RMSE','f MAE','Energy RMSE (eV)'],[
    [a,f(r['arms'][a]['fixed60']['pooled']['r2']),f(r['paired_uncertainty']['comparisons']['fixed60'][a]['delta_r2_vs_control']),f(r['arms'][a]['fixed60']['pooled']['rmse']),f(r['arms'][a]['fixed60']['pooled']['mae']),f(r['arms'][a]['fixed60']['energy']['rmse'])] for a in arms])
add('Aligned final-epoch results also remain below the retained reference. They are reported separately from independently selected checkpoints; the final epoch is not an alternative selection rule.');add()
add('## Per-state errors');add()
for policy,label in [('selected_best','Selected checkpoints'),('fixed60','Fixed epoch60')]:
    add('### '+label);add();add('Each state has6,686 labels. Each cell is R² / raw-f RMSE / MAE.');add()
    table(['State',*arms],[[f'S{k}',*[f"{f(r['arms'][a][policy]['per_state'][str(k)]['r2'])} / {f(r['arms'][a][policy]['per_state'][str(k)]['rmse'])} / {f(r['arms'][a][policy]['per_state'][str(k)]['mae'])}" for a in arms]] for k in range(1,11)])
add('At selected checkpoints the candidate improves SSE for S7–S9, while S1–S6 and S10 worsen. Its pooled SSE improvement over the current control is therefore not a uniform state improvement. The state labels remain fixed; no root matching or selective exclusions were applied.');add()
add('## Bright tails, false-bright predictions and error concentration');add()
add('Thresholds are frozen from TRAIN: q90=.0549 and q99=.2406. True-tail rows are diagnostics and remain part of the pooled metric.');add()
table(['Policy','Arm','q90 count','q90 RMSE','q99 count','q99 RMSE','q99 MAE','q99 SSE'],[
    [p,a,m['bright_tail']['q90']['count'],f(m['bright_tail']['q90']['rmse']),m['bright_tail']['q99']['count'],f(m['bright_tail']['q99']['rmse']),f(m['bright_tail']['q99']['mae']),f(m['bright_tail']['q99']['sse'])]
    for p in ('selected_best','fixed60') for a in arms for m in [r['arms'][a][p]]])
add('All four disjoint q99 target/prediction bins below sum to all66,860 labels and the full pooled SSE. Each cell is count / SSE.');add()
keys=['true_0_pred_0','true_0_pred_1','true_1_pred_0','true_1_pred_1']
table(['Policy','Arm','T0/P0','T0/P1 false bright','T1/P0 missed bright','T1/P1'],[
    [p,a,*[f"{b[k]['count']} / {f(b[k]['sse'])}" for k in keys]] for p in ('selected_best','fixed60') for a in arms for b in [r['arms'][a][p]['false_bright_bins']]])
add('At selected checkpoints raw-f supervision reduces false-bright SSE from17.807549 to8.229482, while true-q99 RMSE worsens from.229575 to.238932, pooled MAE rises and energy MAE rises. From its selected epoch18 to60, the raw-f arm improves true-q99 RMSE to.225894 but false-bright SSE rises to19.572932 and pooled R² falls. The aggregate evidence supports this tradeoff; it does not identify a physical mechanism.');add()
table(['Policy','Arm','Top1% count','Top1% SSE share','Prediction q99','Prediction max','Absolute error q99','Absolute error max'],[
    [p,a,e['top_error_count'],f(e['top_error_sse_share']),f(e['prediction_quantiles']['0.99']),f(e['prediction_quantiles']['1.0']),f(e['absolute_error_quantiles']['0.99']),f(e['absolute_error_quantiles']['1.0'])]
    for p in ('selected_best','fixed60') for a in arms for e in [r['arms'][a][p]['error_distribution']]])
add('These tail summaries retain every row. Complete quantiles, per-state SSE and all aggregate arithmetic are in ROUND06_RESULTS.json.');add()
add('## TRAIN trajectories, clipping and energy coupling');add()
add('TRAIN values are weighted pre-update minibatch aggregates along an epoch, not fixed-checkpoint TRAIN evaluations. Validation is evaluated at completed checkpoints. TRAIN total is LE+Ls or LE+Lf as assigned; TRAIN base and validation base_objective remain LE+Ls in both arms. FP32 normalized training Lf is distinct from the FP64 native-f MSE diagnostic.');add()
table(['Arm/epoch','TRAIN raw-f MSE','TRAIN LE','TRAIN Ls','TRAIN Lf','VAL LE','VAL Ls','Clip fraction'],[
    [f'{a}/{ep}',f(t['train']['raw_f_mse']),f(t['train']['energy']),f(t['train']['trace']),f(t['train']['normalized_raw_f']),f(t['validation']['base_objective']['energy']),f(t['validation']['base_objective']['trace']),f(t['gradient_clip_fraction'])]
    for a in arms for ep,t in r['arms'][a]['trajectory_snapshots'].items()])
add('Both arms reduce TRAIN raw-f, energy and trace losses after their selected checkpoints while validation pooled raw-f performance worsens. The raw-f arm also improves validation energy error after epoch18, but remains worse than trace control at60. Its clipping fraction remains1.44% at60 versus0 for control. These observations are consistent with growing TRAIN fit and changing tail errors, but do not isolate objective mismatch, optimization instability or a generalization cause. No nonfinite/optimizer failure occurred, and the protocol was not extended.');add()
add('The preparation fixture measured raw-f intensity-gradient norm6.7829 times trace-intensity norm with fixed coefficient1; that does not establish equal task weighting. Both predicted E and A receive raw-f gradients. The original PSD/softplus model and normalization remain unchanged. Dormant F movement stays zero; raw-M diagnostics do not enter either objective.');add()
add('## Control repeat context and descriptive uncertainty');add()
add('The retained Round05 control scored0.447169, while this control scored0.404798: an observed difference of−0.042372 with matching scientific recipe, initial tensors, order, optimizer, physical GPU1 and environment. Execution was not byte-identical: the earlier control built an orthogonality graph and added0×its loss, while this control logs that diagnostic without gradients and computes unused Lf diagnostics; earlier concurrency was four fits versus two here. See CONTROL_REPEAT_CONTEXT.md. These facts do not isolate the cause of the difference or establish seed robustness. The dual-reference gate prevents promoting a candidate solely against a weaker rerun.');add()
u=r['paired_uncertainty'];add(f"Paired bootstrap resamples complete validation identity components ({u['component_groups']} groups,6,686 molecules, all ten states), seed{u['seed']}, {u['draws']} draws, recomputing pooled SST in each draw. Percentile2.5/97.5 intervals below are descriptive and conditional on checkpoint selection/reused validation. They are not independent-seed confirmation or correction for model selection. No interval is computed against the aggregate-only retained reference.");add()
table(['Policy','Candidate ΔR² vs current control','Descriptive95% interval'],[
    [p,f(v['delta_r2_vs_control']),f"[{f(v['descriptive_percentile_interval'][0])}, {f(v['descriptive_percentile_interval'][1])}]"] for p in ('selected_best','fixed60') for v in [u['comparisons'][p]['raw_f']]])
add('Both intervals include zero. No robust or causal superiority claim follows from the positive point differences.');add()
add('## Recipe and server checkpoints');add()
add('The strongest retained v2 checkpoint remains the Round05 original control at epoch45, pooled validation R².44716940136585204. Recipe: original PSD MTO16/query32/router128/head128, fresh seed/order11, TRAIN-only normalization, LE+Ls, Adam AMSGrad fixedLR.001/batch64/WD0/clip5, FP32/noAMP/noTF32, selected within60 epochs. This round creates no stronger verified recipe and does not reach0.60.');add()
add('Retained standalone checkpoint: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`, SHA256 `e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`.');add()
add('Round06 private prefix: `/home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation/runs/`. Each listed geometry export is one model/checkpoint with embedded config and TRAIN statistics; no prediction averaging or unavailable QC inputs. Full recovery is in last.pt, selected training snapshots in best.pt.');add()
table(['Arm','Selected epoch','Geometry checkpoint relative path','SHA256'],[
    [a,r['arms'][a]['selected_epoch'],f'{a}/geometry_best.pt',r['verification'][a]['checkpoint_hashes']['geometry_best.pt']] for a in arms])
table(['Arm','Measured training+validation hours'],[[a,f(r['arms'][a]['wall_training_validation_seconds']/3600)] for a in arms])
add('## Verification, reproduction and decision boundary');add()
add('All134 frozen source bindings, exact authorization/review/publication references, two terminal receipts, all60 prescribed orders,61 history rows per arm,112,860 steps, complete checkpoint/optimizer/RNG metadata, earliest minimum-SSE selection, selected/final prediction hashes and partition ledgers passed. Exported model tensors equal the selected checkpoints exactly. The stored buffer fingerprint is present and syntactically valid; it was not reconstructed again. Strict loading and geometry forward access were verified during preserved preflight. Process absence/zombie is exit evidence, not an observed OS exit code.');add()
add('Terminal analysis ran once on CPU with saved validation arrays and checkpoint payloads only: no model construction, model forward, raw-dataset targets or TEST access. The report renderer reads aggregate JSON only. Reproducible commands/source/environment references are in ROUND06_REPRODUCE.md. ANALYSIS_RECEIPT.json binds all inputs/outputs; private hashes are provenance references, never archive payloads. Preserve the preparation missing-mirror event and independent metadata-checker repair from the published preparation. No fitting/analysis tolerance or scientific setting changed after outcomes.');add()
add('![Aligned validation and TRAIN curves](ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg)');add()
add('The frozen dual gate fails; no extension or seed allocation is authorized. Root decides any distinct next preparation after independent result review and D-first archival publication. The deferred shared-PSD congruence assessment is separate and authorizes no implementation, fit or TEST evaluation.');add()
path=OUT/'ROUND06_REPORT.md';assert not path.exists(),'Do not replace a sealed report silently'
path.write_text('\n'.join(lines)+'\n')
print(json.dumps({'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
