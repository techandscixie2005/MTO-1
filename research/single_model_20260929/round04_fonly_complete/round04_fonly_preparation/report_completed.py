"""Summarize completed aggregate receipts only; no targets, models or predictions loaded."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def main():
    prod=ROOT/'production'; result=read(prod/'VALIDATION_COMPLETE.json')
    coeff=read(prod/'COEFFICIENTS_FROZEN.json'); exports=read(prod/'EXPORT_COMPLETE.json')
    replay=read(prod/'POSTFIT_GEOMETRY_REPLAY.json')
    attempts=list(prod.glob('VALIDATION_ATTEMPT_*.json')); assert len(attempts)==1
    attempt=read(attempts[0])
    assert result['coefficient_receipt_sha256']==sha(prod/'COEFFICIENTS_FROZEN.json')
    assert result['export_receipt_sha256']==sha(prod/'EXPORT_COMPLETE.json')
    assert result['validation_attempt_sha256']==sha(attempts[0])
    assert replay['passed'] and replay['base_unchanged']
    stages=list((ROOT/'ops/execution').glob('*/COMPLETE.json'))
    assert len(stages)==2
    for p in stages:
        v=read(p); assert v['exit_code']==0 and v['authorization_unchanged']
        assert v['permit']==result['permit']
        for name,h in v['output_receipt_hashes'].items(): assert sha(prod/name)==h
    metrics=result['metrics']
    for m in metrics.values():
        n=m['pooled']['count']; s=m['pooled']['sse']
        assert n==66860 and sum(x['count'] for x in m['per_state'])==n
        assert sum(x['count'] for x in m['brightness_bins'].values())==n
        assert abs(sum(x['sse'] for x in m['per_state'])-s)<1e-10
        assert abs(sum(x['sse'] for x in m['brightness_bins'].values())-s)<1e-10
    names=['native','historical_validation_fit','in_sample','heldout_source','spline_in_sample','spline_heldout_source']
    labels=['Native eta0','Historical validation affine (incumbent)','In-sample affine','Held-out-source affine','In-sample fixed hinge','Held-out-source fixed hinge']
    lines=['# Round04 completed: fixed scalar flexibility did not improve accuracy','',
      'Status: both authorized CPU fits/export and the single prescribed comparison completed normally. No test scoring, new benchmark scoring, model averaging, checkpoint search or hyperparameter search occurred. No scientific source changed.', '',
      '## Result and decision','',
      'Retain the existing single-checkpoint historical affine incumbent. Neither new four-coefficient map meets the frozen confirmation rule: a gain of at least 0.003 pooled raw-f R² over both its corresponding affine control and the incumbent. The held-out-source map exceeds the in-sample map by 0.01344438, but this source contrast does not establish a benefit from added scalar flexibility. No extension, refit or seed allocation is recommended for these maps.','',
      'All rows below use the same 6,686 historically reused validation molecules and all 66,860 raw printed-f labels. This is a legacy diagnostic, not fresh holdout confirmation and not evidence for attaining R² 0.60 on a new benchmark.','',
      '| Predictor | Pooled raw-f R² | RMSE | MAE | SSE |','|---|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        p=metrics[name]['pooled']; lines.append(f"| {label} | {p['r2']:.9f} | {p['rmse']:.9f} | {p['mae']:.9f} | {p['sse']:.6f} |")
    lines+=['','The in-sample hinge loses 0.00369328 R² against its affine control; the held-out-source hinge loses 0.00459907 against its affine control and 0.00634379 against the incumbent. Their descriptive paired-molecule bootstrap intervals versus their affine controls include zero; the direction of the point estimates supplies no promotion. These intervals reuse validation and do not correct for model selection.','',
      '## All-state comparison','', '| State | Native | Incumbent | In affine | Held affine | In hinge | Held hinge |','|---|---:|---:|---:|---:|---:|---:|']
    for k in range(10): lines.append('| S%d | %s |'%(k+1,' | '.join(f"{metrics[n]['per_state'][k]['r2']:.6f}" for n in names)))
    lines+=['','All states contain 6,686 labels. The held-out-source hinge improves only S5 against its own affine control; its largest SSE regression is S7 (about 0.464). Per-state SSE, RMSE and MAE for every predictor are retained in VALIDATION_COMPLETE.json.','',
      '## Bright-tail and false-bright errors','',
      'Thresholds were fixed from the historical TRAIN: q90=0.0549 and q99=0.2412. True q90 contains 6,650 labels and true q99 contains 639. False-bright means true f<q99 and predicted f>=q99; no bin is excluded.','',
      '| Predictor | q90 RMSE | q99 RMSE | q99 MAE | False-bright count | False-bright SSE |','|---|---:|---:|---:|---:|---:|']
    for n,label in zip(names,labels):
        m=metrics[n]; q=m['true_bright_tail']; b=m['brightness_bins']['true_0_pred_1']
        lines.append(f"| {label} | {q['q90']['rmse']:.6f} | {q['q99']['rmse']:.6f} | {q['q99']['mae']:.6f} | {b['count']} | {b['sse']:.6f} |")
    lines+=['','The held-out hinge has fewer false-bright predictions than its affine control (70 versus 86), but greater false-bright SSE (10.10856 versus 9.71686) and worse true-tail errors. Counting bright predictions alone would conceal this tradeoff. Every predictor\'s four brightness-bin counts sum to 66,860 and their SSE sums to pooled SSE.','',
      '## Fitting and application diagnostics','',
      'Each map used exactly one unconstrained FP64 least-squares solve on the same 24,071 calibration molecules and 240,710 valid labels. The fixed basis is [1, f/s, max(0,(f−0.0549)/s), max(0,(f−0.2412)/s)], s=0.050109692115345626. The loss is unclipped raw-f squared error; deployment applies max(0, q). No energy, state index, hidden features or QC labels enter the map.','',
      '| Source | Rank | Condition number | Affine fitting SSE (unclipped) | Hinge fitting SSE | Above upper knot |','|---|---:|---:|---:|---:|---:|']
    for arm in ['in_sample','heldout_source']:
        c=coeff['coefficients'][arm]; d=c['fitting_diagnostics']
        lines.append(f"| {arm} | {c['design']['rank']} | {c['design']['condition_number']:.6f} | {d['affine_unclipped']['sse']:.6f} | {d['unclipped']['sse']:.6f} | {d['source_upper_knot_count']} |")
    lines+=['','Fitting SSE improves as expected for the nested basis; transfer accuracy does not. This does not establish a causal reason for the failure. Both fitted maps have nonnegative segment slopes. Neither clamps any calibration or validation output. All validation inputs lie inside each corresponding source fitting range; 348 validation inputs are at or above q99. No rank/condition fallback, knot adjustment or exclusion was used.','',
      'Both deployment files contain the same full eta0 backbone and one scalar map. The 80%-TRAIN source is used only to obtain calibration predictions and is absent at inference. E and A are unchanged by the scalar map, so calibrated f generally no longer satisfies the model\'s original E/trace(A) identity. Physical interpretation of the scalar map is unsupported.','',
      '## Execution, parity and provenance','',
      'The CPU environment was bound before Python imports: CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2. A live owned wrapper was registered before each child, with PID/start tick/boot ID/UID/cwd/argv identity in its receipt. Fit/export took 14.40 seconds and evaluation 4.79 seconds; both exited 0 and verified authorization unchanged. There was one fit/export attempt and one validation attempt.','',
      'Both first-64 calibration geometry replays passed: native/cache maximum absolute difference 8.776325466364199e-7, consistent with FP32 batching differences; scalar-map arithmetic difference 2.220446049250313e-16. Backbone tensors were unchanged. Coefficients were frozen before validation. No new backbone forward was needed for validation; the declared scalar comparison used the frozen Round03 native prediction cache.','',
      'Frozen source manifest: `'+result['permit']['source_manifest_sha256']+'`. Execution authorization: `'+result['permit']['authorization_sha256']+'`. Exact aggregate and operational input hashes are in ROUND04_TERMINAL_MANIFEST.json. The independent terminal review is separate.','',
      '## Private checkpoint locations','']
    for arm,entry in exports['exports'].items(): lines += [f"- {arm}: `{entry['path']}`; SHA256 `{entry['sha256']}`."]
    lines+=['- Retained incumbent: `/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`; SHA256 `bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`.','',
      'All learned tensors, checkpoints, raw arrays and prediction caches stay on the server. Lightweight records contain settings, aggregate diagnostics, shapes and hashes, with no learned coefficient values.','',
      '## Reproduction and safe continuation','',
      'The exact reviewed commands below document the completed run. Current completion markers deliberately block rerunning these stages; do not remove them. A distinct experiment requires separate authorization.','',
      '```bash','cd /home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation','CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python ops/run_cpu_stage.py fit_export','CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python ops/run_cpu_stage.py evaluate','```','',
      'The wrapper resolves the pinned source manifest, independent preparation review, preparation publication receipt and PRODUCTION_EXECUTION_AUTHORIZATION.json, and verifies their exact binding before the unchanged scientific entry point. No further scientific execution is pending for Round04. The remaining task is independent terminal review, archive-first publication and the requested repository README update.','']
    (ROOT/'ROUND04_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(prod.glob('*.json'))}
    for p in sorted((ROOT/'ops/execution').rglob('*')):
        if p.is_file(): inputs[str(p.relative_to(ROOT))]=sha(p)
    for n in ['report_completed.py','ROUND04_REPORT.md','PRODUCTION_EXECUTION_AUTHORIZATION.json','FROZEN_MANIFEST.json','ops/run_cpu_stage.py']:
        inputs[n]=sha(ROOT/n)
    manifest={'phase':'completed_round04','input_hashes':inputs,'aggregate_arithmetic_passed':True,'metrics_recomputed_from_raw_arrays':False,
      'exact_two_solves_and_one_comparison_reported':True,'no_new_test_or_historical_test_access':True,'private_model_files_excluded':True,
      'pending':'independent terminal review and archive-first repository publication'}
    (ROOT/'ROUND04_TERMINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({n:sha(ROOT/n) for n in ['report_completed.py','ROUND04_REPORT.md','ROUND04_TERMINAL_MANIFEST.json']},indent=2))
if __name__=='__main__': main()
