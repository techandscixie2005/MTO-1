"""CPU-only aggregate closeout; never runs model inference or fits new maps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import hashlib
import math
import numpy as np
import torch
from clean_source import ROOT,SOURCE,tensor_state_hash
from runtime import verify_manifest,sha,atomic_json
from evaluation import report,affine_fit,score

def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:close(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
    else:assert a==b,(a,b)

def main():
    manifest,mh=verify_manifest();cfg=manifest['config'];rd=ROOT/'runs/source33';out=ROOT/'affine'
    complete=json.loads((rd/'FIT_COMPLETE.json').read_text());gate=json.loads((ROOT/'TERMINAL_SOURCE_GATE.json').read_text())
    frozen=json.loads((out/'COEFFICIENTS_FROZEN.json').read_text());export=json.loads((out/'EXPORT_COMPLETE.json').read_text())
    result=json.loads((out/'VALIDATION_COMPLETE.json').read_text())
    placement_paths=sorted(ROOT.glob('VALIDATION_LAUNCH_*.json'));assert len(placement_paths)==1
    placement=json.loads(placement_paths[0].read_text())
    assert placement['exit_code']==0 and placement['actual_gpu1_placement_observed'] and placement['pinned_affine_artifacts_unchanged']
    assert placement['manifest_sha256']==mh and placement['validation_receipt_sha256']==sha(out/'VALIDATION_COMPLETE.json')
    for path,value in placement['pinned_affine_artifact_hashes'].items():assert sha(path)==value
    assert complete['completed_epoch']==33 and complete['steps']==49665 and gate['passed'] and gate['source_process_exited']
    for record in (complete,gate,frozen,export,result):assert record['manifest_sha256']==mh
    assert sha(rd/'source_final.pt')==complete['checkpoint_sha256']==gate['source_final_sha256']
    assert sha(rd/'last.pt')==complete['last_checkpoint_sha256']==gate['last_checkpoint_sha256']
    assert frozen['frozen_at_unix']<result['validation_completed_unix']
    attempts=sorted(out.glob('VALIDATION_ATTEMPT_*.json'));assert attempts
    for path in attempts:
        attempt=json.loads(path.read_text())
        assert attempt['manifest_sha256']==mh and attempt['coefficient_receipt_sha256']==sha(out/'COEFFICIENTS_FROZEN.json')
        assert frozen['frozen_at_unix']<attempt['started_unix']<=result['validation_completed_unix']
    assert sha(out/'COEFFICIENTS_FROZEN.json')==export['coefficient_receipt_sha256']==result['coefficient_receipt_sha256']
    assert sha(out/'EXPORT_COMPLETE.json')==result['export_receipt_sha256']
    for path,value in frozen['source_hashes'].items():assert sha(path)==value
    assert sha(out/'calibration_predictions.npz')==frozen['calibration_arrays_sha256']
    assert sha(out/'validation_predictions.npz')==result['validation_arrays_sha256']
    history=[json.loads(line) for line in (rd/'history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in history]==list(range(1,34))
    indices=np.load(ROOT/'data/fit_indices.npy');rng=np.random.default_rng(cfg['order_seed'])
    for row in history:
        expected=hashlib.sha256(indices[rng.permutation(len(indices))].tobytes()).hexdigest()
        assert row['order_sha256']==expected and row['learning_rate']==cfg['lr'] and row['optimizer_batches']==1505
    final=torch.load(rd/'source_final.pt',map_location='cpu',weights_only=False)
    last=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
    assert final['epoch']==33 and final['selection']=='fixed_epoch_33' and last['state']['completed_epoch']==33
    assert last['state']['history']==history and last['state']['steps']==49665
    assert tensor_state_hash(final['model'])==tensor_state_hash(last['model'])
    assert final['stats']==last['stats']==json.loads((ROOT/'fit_normalization.json').read_text())
    assert final['source_config']==last['source_config']==json.loads((ROOT/'source_config.json').read_text())
    steps=[float(s['step']) for s in last['optimizer']['state'].values()]
    assert steps and min(steps)==max(steps)==49665 and {'python','numpy','order','torch','cuda'}<=set(last['rng'])
    with np.load(out/'calibration_predictions.npz') as arrays:
        assert np.array_equal(arrays['indices'],np.load(ROOT/'data/calibration_indices.npy'))
        assert len(arrays['indices'])==24071 and int(arrays['mask_f'].sum())==240710
        for name,key in (('in_sample','full_baseline_native_f'),('heldout_source','source_native_f')):
            close(affine_fit(arrays['f_true'],arrays[key],arrays['mask_f']),frozen['maps'][name])
    baseline=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False);basehash=tensor_state_hash(baseline['model'])
    inventory={}
    for name,item in export['checkpoints'].items():
        assert sha(item['path'])==item['sha256']
        checkpoint=torch.load(item['path'],map_location='cpu',weights_only=False)
        state={k.removeprefix('base.'):v for k,v in checkpoint['model'].items() if k.startswith('base.')}
        assert tensor_state_hash(state)==basehash and checkpoint['native_full_baseline_input'] is True
        assert checkpoint['alpha']==frozen['maps'][name]['alpha'] and checkpoint['beta']==frozen['maps'][name]['beta']
        assert float(checkpoint['model']['alpha'])==checkpoint['alpha'] and float(checkpoint['model']['beta'])==checkpoint['beta']
        assert checkpoint['source_config']==json.loads((SOURCE/'configs/mto_eta0.json').read_text())
        assert checkpoint['stats']==json.loads((SOURCE/'data/normalization.json').read_text())
        assert checkpoint['format']=='round03_single_native_model_affine' and checkpoint['geometry_only_inference'] is True
        assert checkpoint['provenance']['coefficient_receipt_sha256']==sha(out/'COEFFICIENTS_FROZEN.json')
        inventory[name]={'path':item['path'],'sha256':item['sha256'],'bytes':os.path.getsize(item['path']),
            'full_native_base_tensor_sha256':basehash,'base_identical_to_original':True}
    thresholds=manifest['tail_thresholds'];bins={}
    with np.load(out/'validation_predictions.npz') as arrays:
        with np.load(SOURCE/'data/dataset.npz') as source:assert np.array_equal(arrays['indices'],source['val'])
        raw={'f':arrays['f_true'],'mask_f':arrays['mask_f']};mask=raw['mask_f'].astype(bool);truth=raw['f']
        assert int(mask.sum())==66860
        for name in ('native','historical_validation_fit','in_sample','heldout_source'):
            pred=arrays[name];close(report(raw,pred,thresholds),result['metrics'][name])
            if name in frozen['maps']:
                coeff=frozen['maps'][name];assert np.array_equal(pred,np.maximum(0,coeff['alpha']*arrays['native']+coeff['beta']))
            cuts={}
            for true_bright in (False,True):
                for pred_bright in (False,True):
                    chosen=mask&((truth>=thresholds['q99'])==true_bright)&((pred>=thresholds['q99'])==pred_bright)
                    cuts['true_%s_pred_%s'%('bright' if true_bright else 'dim','bright' if pred_bright else 'dim')]=score(truth[chosen],pred[chosen])
            assert sum(x['count'] for x in cuts.values())==66860
            assert math.isclose(sum(x['sse'] for x in cuts.values()),result['metrics'][name]['raw_f']['sse'],rel_tol=1e-12)
            bins[name]=cuts
    held=result['metrics']['heldout_source']['raw_f']['r2'];inside=result['metrics']['in_sample']['raw_f']['r2'];native=result['metrics']['native']['raw_f']['r2']
    close(held-inside,result['heldout_minus_in_sample_r2']);close(held-native,result['heldout_minus_native_r2'])
    assert result['nonlinear_preparation_gate_passed']==(held-inside>=cfg['advancement_minimum_delta_r2'] and held-native>=cfg['advancement_minimum_delta_r2'])
    summary={'passed_integrity_checks':True,'manifest_sha256':mh,'metrics':result['metrics'],'energy':result['energy'],
        'maps':frozen['maps'],'source_calibration_energy':frozen['fitting_source_quality_energy'],
        'source_training':{'epochs':33,'optimizer_updates':49665,'epoch_seconds_total':sum(r['seconds'] for r in history),
            'first_online_losses':history[0]['train'],'last_online_losses':history[-1]['train'],
            'mean_epoch_seconds':float(np.mean([r['seconds'] for r in history])),
            'all_orders_reproduced':True,'all_learning_rates':sorted(set(r['learning_rate'] for r in history)),
            'clip_fraction_min_max':[min(r['gradient_clip_fraction'] for r in history),max(r['gradient_clip_fraction'] for r in history)],
            'optimizer_states':len(steps),'optimizer_step_min_max':[min(steps),max(steps)],'resumable_RNG_present':True},
        'heldout_minus_in_sample_r2':result['heldout_minus_in_sample_r2'],'heldout_minus_native_r2':result['heldout_minus_native_r2'],
        'nonlinear_preparation_gate_passed':result['nonlinear_preparation_gate_passed'],
        'false_bright_bins':bins,'bright_threshold_train':thresholds['q99'],'deployment_checkpoints':inventory,
        'source_final_checkpoint':{'path':str(rd/'source_final.pt'),'sha256':complete['checkpoint_sha256']},
        'resumable_last_checkpoint':{'path':str(rd/'last.pt'),'sha256':complete['last_checkpoint_sha256']},
        'source_selected_without_outer_validation':True,'validation_historically_reused':True,'test_evaluated':False,
        'one_model_per_predictor':True,'prediction_averaging':False,
        'resource_placement':{'validation_actual_gpu1_observed':True,'validation_launch_receipt':str(placement_paths[0]),
            'fit_physical_placement_not_verified':True,'fit_time_GPU0_MTO_context_observed':True,
            'note':'RESOURCE_PLACEMENT_NOTE.md','no_completed_fit_or_validation_repeated':True}}
    atomic_json(summary,ROOT/'ROUND03_RESULTS.json')
    lines=['# Round03 affine transfer result','','All fixed 33 source epochs completed. Both affine pairs were frozen before outer validation; no test inference or prediction averaging occurred.','',
        '| Predictor | Pooled raw-f R² | RMSE | MAE | q90 RMSE | q99 RMSE |','|---|---:|---:|---:|---:|---:|']
    for name,values in result['metrics'].items():
        m=values['raw_f'];tail=values['bright_tail'];lines.append(f"| {name} | {m['r2']:.9f} | {m['rmse']:.9f} | {m['mae']:.9f} | {tail['q90']['rmse']:.9f} | {tail['q99']['rmse']:.9f} |")
    lines+=['','| Map source | Alpha | Beta | Source prediction R² on the same 24,071 molecules |','|---|---:|---:|---:|']
    for name,m in frozen['maps'].items():lines.append(f"| {name} | {m['alpha']:.12f} | {m['beta']:.12f} | {m['native_calibration_subset_metrics']['r2']:.9f} |")
    lines+=['','## Per-state validation R²','','| State | Native | Historical affine | In-sample affine | Heldout-source affine |','|---|---:|---:|---:|---:|']
    for k in range(10):lines.append('| %d | %s |'%(k+1,' | '.join('%.9f'%result['metrics'][n]['per_state'][k]['r2'] for n in ('native','historical_validation_fit','in_sample','heldout_source'))))
    lines+=['','## Interpretation boundary','',f"Heldout minus in-sample R²: {result['heldout_minus_in_sample_r2']:.9f}; heldout minus native: {result['heldout_minus_native_r2']:.9f}. The prespecified nonlinear-preparation allocation gate is {'passed' if result['nonlinear_preparation_gate_passed'] else 'not passed'}.",'',
        'The source uses fewer fitting molecules and different optimization errors. This comparison tests this one transfer procedure, not a causal explanation of generalization. The historical affine was fitted with exposed validation information. Outer validation has been reused; this is not fresh holdout confirmation. Both deployed maps use the same full baseline and may be inconsistent with its unchanged auxiliary E/A outputs.',
        '', 'Source online fitting losses describe a changing model. Every source checkpoint is fixed at epoch33; no calibration or outer label selected it. All 33 prescribed orders and 49,665 optimizer updates were verified. Full optimizer/RNG states and every model/prediction array remain on the server.',
        '', 'Resource limitation: a short-lived MTO process was observed on GPU0 during the affine-fit interval; exact argv and full inference placement were not captured before it exited. The fit was retained without repeat. Validation was prebound before interpreter startup and actually observed only on GPU1. See RESOURCE_PLACEMENT_NOTE.md and the owned validation launch receipt.',
        '', 'ROUND03_RESULTS.json contains complete pooled/state/tail errors, all-label brightness partitions, coefficients, source quality, runtime and checkpoint metadata. Root decides the next action after independent review.']
    (ROOT/'ROUND03_RESULTS.md').write_text('\n'.join(lines)+'\n')
    names=['FROZEN_MANIFEST.json','TERMINAL_SOURCE_GATE.json','runs/source33/FIT_COMPLETE.json','runs/source33/history.jsonl',
        'runs/source33/TRAIN_DATA_ACCESS.json','runs/source33/LAUNCH_RECEIPT.json','affine/COEFFICIENTS_FROZEN.json',
        'affine/EXPORT_COMPLETE.json','affine/VALIDATION_COMPLETE.json','affine/calibration_predictions.npz','affine/validation_predictions.npz']
    names.extend(str(path.relative_to(ROOT)) for path in attempts)
    names.extend(str(path.relative_to(ROOT)) for path in placement_paths)
    names.extend(['run_validation_bound.py','RESOURCE_PLACEMENT_NOTE.md'])
    receipt={'passed':True,'manifest_sha256':mh,'source_hashes':{str(ROOT/'summarize_transfer.py'):sha(ROOT/'summarize_transfer.py')},
        'input_hashes':{str(ROOT/n):sha(ROOT/n) for n in names},
        'output_hashes':{str(ROOT/n):sha(ROOT/n) for n in ('ROUND03_RESULTS.json','ROUND03_RESULTS.md')},
        'CPU_only':True,'new_model_inference':False,'new_affine_coefficients_saved':False}
    atomic_json(receipt,ROOT/'ANALYSIS_RECEIPT.json')
    print(json.dumps({'passed':True,'native_r2':result['metrics']['native']['raw_f']['r2'],
        'heldout_r2':result['metrics']['heldout_source']['raw_f']['r2'],'gate_passed':result['nonlinear_preparation_gate_passed']},indent=2))

if __name__=='__main__':main()
