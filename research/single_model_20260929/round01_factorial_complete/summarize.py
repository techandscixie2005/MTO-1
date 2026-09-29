"""Analyze terminal pilot arms using saved validation predictions only."""
import json
from pathlib import Path
import numpy as np
from metrics import CAL_ALPHA,CAL_BETA,score
from freeze import sha

ROOT=Path(__file__).resolve().parent

def verify_score(actual,expected):
    assert actual['count']==expected['count']
    for key in ('sse','sst','r2','mae','rmse'):
        if expected[key] is None:
            assert actual[key] is None
        else:
            assert np.isclose(actual[key],expected[key],rtol=1e-11,atol=1e-11),(key,actual[key],expected[key])

def verify_selected_arrays(value,metrics,thresholds):
    truth=value['f_true'];pred=value['f_pred'];mask=value['mask_f'].astype(bool)
    for prediction,receipt in ((pred,metrics),(np.maximum(0,CAL_ALPHA*pred+CAL_BETA),metrics['fixed_calibrated_secondary'])):
        verify_score(score(truth[mask],prediction[mask]),receipt['raw_f'])
        for j in range(truth.shape[1]):
            verify_score(score(truth[:,j][mask[:,j]],prediction[:,j][mask[:,j]]),receipt['per_state'][j])
        for name,threshold in thresholds.items():
            use=mask&(truth>=threshold)
            assert receipt['bright_tail'][name]['threshold_train_raw_f']==threshold
            verify_score(score(truth[use],prediction[use]),receipt['bright_tail'][name])

def bootstrap(truth,pred,reference,mask,draws=2000,seed=20260930):
    # Molecules are the resampling unit; all states of a molecule stay together.
    yy=np.where(mask,truth,0)
    counts=mask.sum(1)
    sums=yy.sum(1);squares=np.square(yy).sum(1)
    gain=np.where(mask,np.square(reference-truth)-np.square(pred-truth),0).sum(1)
    rng=np.random.default_rng(seed)
    deltas=[]
    for _ in range(draws):
        idx=rng.integers(0,len(truth),len(truth))
        denom=squares[idx].sum()-sums[idx].sum()**2/counts[idx].sum()
        deltas.append(float(gain[idx].sum()/denom))
    return {'draws':draws,'seed':seed,'unit':'molecule','delta_r2_percentile95':np.quantile(deltas,[.025,.975]).tolist(),
            'fraction_positive':float(np.mean(np.array(deltas)>0)),
            'interpretation':'descriptive_after_reused_validation_and_checkpoint_selection_not_fresh_confirmation'}

def main():
    cfg=json.loads((ROOT/'round_config.json').read_text())
    frozen=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    manifest_sha=sha(ROOT/'FROZEN_MANIFEST.json')
    input_hashes={str(ROOT/'FROZEN_MANIFEST.json'):manifest_sha}
    complete={};arrays={}
    for arm in cfg['arms']:
        path=ROOT/'runs'/arm
        complete[arm]=json.loads((path/'FIT_COMPLETE.json').read_text())
        assert complete[arm]['test_evaluated'] is False and complete[arm]['single_checkpoint'] is True
        assert complete[arm]['completed_epoch']==cfg['epochs']==20
        assert complete[arm]['manifest_sha256']==manifest_sha
        assert sha(path/'best.pt')==complete[arm]['best_checkpoint_sha256']
        with np.load(path/'best_validation_predictions.npz') as z:
            arrays[arm]={k:z[k].copy() for k in z.files}
        verify_selected_arrays(arrays[arm],complete[arm]['validation'],frozen['tail_thresholds'])
        for name in ('FIT_COMPLETE.json','best.pt','best_validation_predictions.npz','history.jsonl'):
            input_hashes[str(path/name)]=sha(path/name)
    control=arrays['control'];truth=control['f_true'];mask=control['mask_f']
    for value in arrays.values():
        assert np.array_equal(value['indices'],control['indices'])
        assert np.array_equal(value['f_true'],truth) and np.array_equal(value['mask_f'],mask)
    histories={arm:[json.loads(line) for line in (ROOT/'runs'/arm/'history.jsonl').read_text().splitlines()]
               for arm in cfg['arms']}
    for arm,rows in histories.items():
        assert [r['epoch'] for r in rows]==list(range(cfg['epochs']+1)),('Nonunique or incomplete epochs',arm)
        selected=min(rows,key=lambda r:r['validation']['raw_f']['sse'])
        assert complete[arm]['selected_epoch']==selected['epoch'],('Wrong minimum-SSE earliest-tie selection',arm)
    order_hashes={arm:[row['order_sha256'] for row in rows if row['epoch']>0] for arm,rows in histories.items()}
    assert all(v==order_hashes['control'] for v in order_hashes.values())
    result={'status':'all_four_terminal','single_checkpoint_per_arm':True,'predictions_averaged':False,
        'primary':'native_pooled_validation_raw_f_R2','historical_test_evaluated':False,
        'sample_order_hashes_match':True,'tail_thresholds_training_only':frozen['tail_thresholds'],
        'validation_selection_conditional':True,'frozen_manifest_sha256':manifest_sha,
        'terminal_receipts_checkpoints_arrays_history_verified':True,
        'analysis_source_hashes':{str(Path(__file__)):sha(__file__),str(ROOT/'metrics.py'):sha(ROOT/'metrics.py')},
        'input_hashes':input_hashes,'arms':{}}
    control_score=complete['control']['validation']['raw_f']['r2']
    for arm in cfg['arms']:
        one=complete[arm];r=one['validation']['raw_f']['r2'];r0=one['epoch_zero']['raw_f']['r2']
        item={'selected_epoch':one['selected_epoch'],'checkpoint':one['best_checkpoint'],
            'checkpoint_sha256':one['best_checkpoint_sha256'],'validation':one['validation'],
            'epoch_zero':one['epoch_zero'],
            'delta_native_r2_vs_control':r-control_score,'delta_native_r2_vs_epoch0':r-r0,
            'native_bootstrap_vs_control':bootstrap(truth,arrays[arm]['f_pred'],control['f_pred'],mask),
            'fixed_calibrated_bootstrap_vs_control':bootstrap(truth,np.maximum(0,CAL_ALPHA*arrays[arm]['f_pred']+CAL_BETA),
                np.maximum(0,CAL_ALPHA*control['f_pred']+CAL_BETA),mask)}
        item['meets_prespecified_confirmation_priority']=bool(r-r0>=.003 and (arm=='control' or r-control_score>=.003))
        result['arms'][arm]=item
    def interaction(values):
        return values['both']-values['adapter']-values['decorrelation']+values['control']
    selected_native={a:v['validation']['raw_f']['r2'] for a,v in result['arms'].items()}
    aligned=[]
    for epoch in range(cfg['epochs']+1):
        rows={a:next(r for r in h if r['epoch']==epoch) for a,h in histories.items()}
        native={a:r['validation']['raw_f']['r2'] for a,r in rows.items()}
        calibrated={a:r['validation']['fixed_calibrated_secondary']['raw_f']['r2'] for a,r in rows.items()}
        aligned.append({'epoch':epoch,'native_r2':native,'fixed_calibrated_r2':calibrated,
            'native_factorial_interaction':interaction(native),
            'fixed_calibrated_factorial_interaction':interaction(calibrated)})
    result['factorial']={'selected_best_interaction':interaction(selected_native),
        'selected_best_interpretation':'pipeline-level descriptive; independently selected checkpoints cannot identify mechanistic synergy',
        'fixed_epoch_20':aligned[-1],'aligned_curves':aligned,
        'same_epoch_interpretation':'matched training-duration factorial contrast; descriptive one-seed evidence, not causal physical interpretation'}
    result['fixed_final_epoch_validation']={a:histories[a][-1]['validation'] for a in histories}
    (ROOT/'ROUND_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Four-arm single-model pilot results','',
        'All results use the fixed validation set. Every row is one model and one selected checkpoint. No prediction averaging or test inference occurred. Intervals are descriptive after checkpoint and validation reuse.','',
        '| Arm | Selected epoch | Native raw-f R2 | Delta vs control | Delta vs epoch0 | Fixed calibration R2 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,item in result['arms'].items():
        v=item['validation']
        lines.append('| %s | %d | %.9f | %+.9f | %+.9f | %.9f |'%(arm,item['selected_epoch'],v['raw_f']['r2'],
            item['delta_native_r2_vs_control'],item['delta_native_r2_vs_epoch0'],v['fixed_calibrated_secondary']['raw_f']['r2']))
    lines+=['','## Matched final epoch (epoch20)','',
        '| Arm | Native raw-f R2 | Fixed calibration R2 |','|---|---:|---:|']
    final=aligned[-1]
    for arm in cfg['arms']:
        lines.append('| %s | %.9f | %.9f |'%(arm,final['native_r2'][arm],final['fixed_calibrated_r2'][arm]))
    lines+=['','Native same-epoch factorial contrast R2(both)-R2(adapter)-R2(decorrelation)+R2(control): %+.9f.'%final['native_factorial_interaction'],
        'Selected-best contrast is %+.9f and is only a descriptive pipeline comparison; separately selected epochs do not establish mechanistic synergy.'%interaction(selected_native)]
    lines+=['','All sample-order hashes match. Per-state, bright-tail SSE/RMSE/MAE, energy, checkpoint SHA256 and paired molecule bootstrap details are in ROUND_RESULTS.json.',
        '', 'The fixed historical calibration is secondary, never refit, and did not select checkpoints. Its validation estimate uses the previously refitted map and is not a fresh holdout result.',
        '', '## Next decision','',
        'Prespecified confirmation priorities: '+', '.join(a for a,v in result['arms'].items() if v['meets_prespecified_confirmation_priority'])+'.',
        'A control-only gain can justify paired independent-seed confirmation of the gentler schedule. An adapter win would still need a simpler active capacity control before attributing benefit to its nonlinearity. Root reviews state/tail tradeoffs and chooses subsequent work.']
    (ROOT/'ROUND_RESULTS.md').write_text('\n'.join(lines)+'\n')
    receipt={'passed':True,'frozen_manifest_sha256':manifest_sha,'source_hashes':result['analysis_source_hashes'],
        'input_hashes':input_hashes,'output_hashes':{name:sha(ROOT/name) for name in ('ROUND_RESULTS.json','ROUND_RESULTS.md')},
        'bootstrap_seed':20260930,'bootstrap_draws':2000,'all_four_arms_included':True,'test_evaluated':False}
    (ROOT/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:{'native_r2':v['validation']['raw_f']['r2'],'delta_control':v['delta_native_r2_vs_control'],
                        'delta_epoch0':v['delta_native_r2_vs_epoch0'],'confirm':v['meets_prespecified_confirmation_priority']}
                      for a,v in result['arms'].items()},indent=2))

if __name__=='__main__':
    main()
