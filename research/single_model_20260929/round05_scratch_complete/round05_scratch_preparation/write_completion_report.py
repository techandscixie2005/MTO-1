"""Render lightweight aggregate results; no tensors, arrays or model access."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'completion'
r=json.loads((OUT/'ROUND05_RESULTS.json').read_text()); arms=('control','adapter','decorrelation','both')
lines=[]
def add(s=''):lines.append(s)
def table(headers,rows):
    add('| '+' | '.join(headers)+' |');add('| '+' | '.join('---' for _ in headers)+' |')
    for row in rows:add('| '+' | '.join(map(str,row))+' |')
    add()
f=lambda x:f'{x:.6f}'
add('# Round05 — fresh MTO factorial on the audited QM9S v2 partition')
add()
add('**The original control is the strongest model in this round.** Its validation-selected epoch45 reaches pooled raw-f R² **0.447169**. Shared right-F, weak raw-state decorrelation and their combination all score lower. No noncontrol arm clears the frozen +0.003 threshold, so the 60-epoch pilot closes without extension or seed allocation. This conclusion concerns one matched initialization and this fixed recipe; it is not a proof that the architecture can never help.')
add()
add('All four runs completed 60 epochs and 112860 optimizer updates. Every validation evaluation includes 6686 molecules ×10 states =66860 valid raw printed-f labels, including zeros. There was no TEST evaluation, calibration, label exclusion or prediction averaging. The new partition is internally disjoint under audited identity rules, but historically exposed: its sealed TEST has5989 oldTRAIN,361 oldVAL and336 oldTEST molecules. Old-split incumbent scores are not comparable benchmark gains.')
add()
add('## Selected checkpoints')
add()
rows=[]
for arm in arms:
    v=r['arms'][arm];m=v['selected_best'];u=r['paired_uncertainty']['comparisons']['selected_best'][arm]
    rows.append([arm,v['selected_epoch'],f(m['pooled']['r2']),f(u['delta_r2_vs_control']),f(m['pooled']['rmse']),f(m['pooled']['mae']),f(m['energy']['mae'])])
table(['Arm','Selected epoch','Pooled R²','ΔR² vs control','f RMSE','f MAE','Energy MAE (eV)'],rows)
add('Checkpoint selection maximizes native pooled raw-f R² by minimizing SSE, including epoch0; strict ties retain the earliest epoch. Each row is one model and one checkpoint. The selection rule was frozen before fitting.')
add()
add('## Aligned epoch60 comparison')
add()
table(['Arm','R² at60','ΔR² vs control60','f RMSE','f MAE','Energy MAE (eV)'],[
    [arm,f(r['arms'][arm]['fixed60']['pooled']['r2']),f(r['paired_uncertainty']['comparisons']['fixed60'][arm]['delta_r2_vs_control']),
     f(r['arms'][arm]['fixed60']['pooled']['rmse']),f(r['arms'][arm]['fixed60']['pooled']['mae']),f(r['arms'][arm]['fixed60']['energy']['mae'])] for arm in arms])
add(f"The aligned factorial contrast R²(both)−R²(F)−R²(decor)+R²(control) is {f(r['factorial_interaction_r2']['fixed60'])}. The independently selected-checkpoint contrast is {f(r['factorial_interaction_r2']['selected_best'])} and describes the complete selection procedures. Neither contrast demonstrates a physical synergy; all three selected ablations lose to control.")
add()
add('## Per-state errors')
add()
for policy,label in [('selected_best','Selected checkpoints'),('fixed60','Fixed epoch60')]:
    add(f'### {label}');add();add('Each cell is R² / raw-f RMSE. Each state contains6686 labels.');add()
    table(['State',*arms],[[f'S{k}',*[f"{f(r['arms'][arm][policy]['per_state'][str(k)]['r2'])} / {f(r['arms'][arm][policy]['per_state'][str(k)]['rmse'])}" for arm in arms]] for k in range(1,11)])
add('## Bright tails and false-bright predictions')
add()
add('Thresholds were fixed from the new TRAIN labels: q90=.0549 and q99=.2406. True-bright tails use target f at or above the threshold; they are diagnostics, not exclusions or alternate selection metrics.')
add()
table(['Policy','Arm','q90 count','q90 RMSE','q99 count','q99 RMSE','q99 MAE','q99 SSE'],[
    [policy,arm,m['bright_tail']['q90']['count'],f(m['bright_tail']['q90']['rmse']),m['bright_tail']['q99']['count'],f(m['bright_tail']['q99']['rmse']),f(m['bright_tail']['q99']['mae']),f(m['bright_tail']['q99']['sse'])]
    for policy in ('selected_best','fixed60') for arm in arms for m in [r['arms'][arm][policy]]])
add('The following four disjoint bins cover all66860 labels at q99; each cell is count / SSE. T and P denote target and prediction at or above q99.')
add()
keys=['true_0_pred_0','true_0_pred_1','true_1_pred_0','true_1_pred_1']
table(['Policy','Arm','T0/P0','T0/P1 false bright','T1/P0 missed bright','T1/P1'],[
    [policy,arm,*[f"{b[k]['count']} / {f(b[k]['sse'])}" for k in keys]]
    for policy in ('selected_best','fixed60') for arm in arms for b in [r['arms'][arm][policy]['false_bright_bins']]])
add('Tail changes are mixed. At selected checkpoints, F improves true-q99 RMSE versus control (.217590 vs .224982) while false-bright SSE rises (19.207634 vs12.334493), and pooled accuracy worsens. Decorrelation and both reduce selected false-bright SSE but have worse true-q99 RMSE. From their selected epochs to60, decorrelation and both improve true-q99 RMSE while false-bright SSE increases sharply and pooled R² falls. A single uniform tail-improvement or failure explanation is therefore unsupported.')
add()
add('## Optimization and representation diagnostics')
add()
add('TRAIN values below are weighted pre-update minibatch aggregates along each epoch, not fixed-checkpoint TRAIN evaluations. Validation uses completed checkpoints. Their differences cannot alone identify the cause of generalization error.')
add()
table(['Arm/epoch','TRAIN raw-f MSE','TRAIN LE','TRAIN Ls','VAL LE','VAL Ls','Raw-M penalty','Mean F/M change','Clip fraction'],[
    [f'{arm}/{ep}',f(t['train']['raw_f_mse']),f(t['train']['energy']),f(t['train']['trace']),f(t['validation']['base_objective']['energy']),f(t['validation']['base_objective']['trace']),f(t['train']['decorrelation']),f(t['train']['F_relative_update_mean']),f(t['gradient_clip_fraction'])]
    for arm in arms for ep,t in r['arms'][arm]['trajectory_snapshots'].items()])
add('At epoch60, F changes the right-state representations by mean relative magnitudes .535609 (F) and .571315 (both): these adapters are active, not identity-stuck. The penalty decreases to .401419/.420101 with decorrelation, versus .509309 for control. Those feature changes do not establish physical state orthogonality, dipole adaptation, or improved generalization.')
add()
add('Every arm lowers TRAIN raw-f, energy and trace losses after its selected epoch, while validation raw-f R² falls. Validation energy loss improves slightly, but validation trace loss worsens. This is compatible with increasing fit to TRAIN and a mismatch between the training objective and pooled raw-f priorities. It does not establish either as the cause or distinguish them from seed variation. Gradient clipping is rare late in training, and there is no nonfinite or optimizer failure evidence. Extending the unchanged protocol is not supported by its frozen gate.')
add()
add('The largest669 state errors (top1%) contribute about53–57% of selected pooled SSE. Complete prediction/error quantiles and exact shares are in ROUND05_RESULTS.json; no rows are removed.')
add()
add('## Descriptive uncertainty')
add()
u=r['paired_uncertainty'];add(f"Paired bootstrap resamples all molecules within each of {u['component_groups']} audited validation identity components, preserving their ten states. Seed{u['seed']}, {u['draws']} draws, percentile2.5/97.5 intervals; each draw recomputes pooled target SST. This was a post-training descriptive analysis of already selected validation predictions. It does not correct checkpoint selection or reused validation, and is not independent-seed or fresh-holdout confirmation.")
add()
table(['Policy','Arm','ΔR²','Descriptive95% interval'],[
    [policy,arm,f(v['delta_r2_vs_control']),f"[{f(v['descriptive_percentile_interval'][0])}, {f(v['descriptive_percentile_interval'][1])}]"]
    for policy in ('selected_best','fixed60') for arm,v in u['comparisons'][policy].items()])
add('## Best verified recipe and private checkpoints')
add()
add('For this v2 split, retain the unchanged original PSD MTO control at epoch45: channels16, query32, router/head128; fresh seed11, independent order seed11, new TRAIN-only normalization; original LE+Ls; Adam AMSGrad LR.001 fixed, batch64, WD0, clip5, FP32/noAMP/noTF32. Validation selects epoch45 within the fixed60-epoch run. F is disabled and the penalty is zero. This is the strongest verified recipe in this round, not attainment of the target0.60.')
add()
add('Server prefix: `/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/`. Each geometry_best.pt is one self-contained geometry predictor. Full optimizer/RNG/order recovery remains in last.pt; best.pt is the selected training snapshot. No model, optimizer, prediction or split arrays are uploaded.')
add()
table(['Arm','Selected epoch','Geometry checkpoint relative path','SHA256'],[
    [arm,r['arms'][arm]['selected_epoch'],f'{arm}/geometry_best.pt',r['verification'][arm]['checkpoint_hashes']['geometry_best.pt']] for arm in arms])
add('## Verification and reproduction')
add()
add('All82 frozen source/input bindings, exact authorization/publication references, four terminal receipts,60 prescribed order hashes per arm,61 history rows,112860 optimizer steps, full checkpoint/RNG metadata, earliest minimum-SSE selection, selected/final prediction hashes and decoder ledgers passed. The exported model tensors exactly equal each selected checkpoint. The terminal audit used CPU tensor reads and saved validation outputs only: no model construction, forward inference, raw-dataset target decoding or TEST access. The geometry buffer fingerprint was not reconstructed again; strict one-checkpoint loading/O(3) behavior was covered by the preserved technical preflight. Process absence/zombie is exit evidence, not an observed OS exit code.')
add()
add('Reproducible commands and fixed source/environment references are in ROUND05_REPRODUCE.md. terminal_analysis.py and its fixed analysis config reproduce aggregate metrics, component bootstrap and the SVG. ANALYSIS_RECEIPT.json binds every read input and output. The earlier CPU/GPU preflight failures, explicit CUDA parity amendment and I/O-only source continuity remain in the preparation archive. No outcome-dependent tolerance change occurred during fitting or terminal analysis.')
add()
add('![Aligned validation and TRAIN curves](ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg)')
add()
add('## Decision boundary')
add()
add('The frozen +.003 noncontrol-over-control criterion fails. No extension, additional seed or TEST evaluation is authorized. Root decides any distinct next preparation after independent result review and D-first archival publication. A possible new objective comparison and the deferred QC congruence hypothesis are discussed separately; neither has been implemented or fitted by this analysis.')
path=OUT/'ROUND05_REPORT.md';path.write_text('\n'.join(lines)+'\n')
print(json.dumps({'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
