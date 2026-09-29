"""Analyze terminal pilot arms using saved validation predictions only."""
import json
from pathlib import Path
import numpy as np
from metrics import CAL_ALPHA,CAL_BETA,score

ROOT=Path(__file__).resolve().parent

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
    complete={};arrays={}
    for arm in cfg['arms']:
        path=ROOT/'runs'/arm
        complete[arm]=json.loads((path/'FIT_COMPLETE.json').read_text())
        assert complete[arm]['test_evaluated'] is False and complete[arm]['single_checkpoint'] is True
        with np.load(path/'best_validation_predictions.npz') as z:
            arrays[arm]={k:z[k].copy() for k in z.files}
    control=arrays['control'];truth=control['f_true'];mask=control['mask_f']
    for value in arrays.values():
        assert np.array_equal(value['indices'],control['indices'])
        assert np.array_equal(value['f_true'],truth) and np.array_equal(value['mask_f'],mask)
    order_hashes={arm:[json.loads(line)['order_sha256'] for line in (ROOT/'runs'/arm/'history.jsonl').read_text().splitlines()
                       if json.loads(line)['epoch']>0] for arm in cfg['arms']}
    assert all(v==order_hashes['control'] for v in order_hashes.values())
    result={'status':'all_four_terminal','single_checkpoint_per_arm':True,'predictions_averaged':False,
        'primary':'native_pooled_validation_raw_f_R2','historical_test_evaluated':False,
        'sample_order_hashes_match':True,'tail_thresholds_training_only':frozen['tail_thresholds'],
        'validation_selection_conditional':True,'arms':{}}
    control_score=complete['control']['validation']['raw_f']['r2']
    for arm in cfg['arms']:
        one=complete[arm];r=one['validation']['raw_f']['r2'];r0=one['epoch_zero']['raw_f']['r2']
        item={'selected_epoch':one['selected_epoch'],'checkpoint':one['best_checkpoint'],
            'checkpoint_sha256':one['best_checkpoint_sha256'],'validation':one['validation'],
            'delta_native_r2_vs_control':r-control_score,'delta_native_r2_vs_epoch0':r-r0,
            'native_bootstrap_vs_control':bootstrap(truth,arrays[arm]['f_pred'],control['f_pred'],mask),
            'fixed_calibrated_bootstrap_vs_control':bootstrap(truth,np.maximum(0,CAL_ALPHA*arrays[arm]['f_pred']+CAL_BETA),
                np.maximum(0,CAL_ALPHA*control['f_pred']+CAL_BETA),mask)}
        item['meets_prespecified_confirmation_priority']=bool(r-r0>=.003 and (arm=='control' or r-control_score>=.003))
        result['arms'][arm]=item
    (ROOT/'ROUND_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Four-arm single-model pilot results','',
        'All results use the fixed validation set. Every row is one model and one selected checkpoint. No prediction averaging or test inference occurred. Intervals are descriptive after checkpoint and validation reuse.','',
        '| Arm | Selected epoch | Native raw-f R2 | Delta vs control | Delta vs epoch0 | Fixed calibration R2 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,item in result['arms'].items():
        v=item['validation']
        lines.append('| %s | %d | %.9f | %+.9f | %+.9f | %.9f |'%(arm,item['selected_epoch'],v['raw_f']['r2'],
            item['delta_native_r2_vs_control'],item['delta_native_r2_vs_epoch0'],v['fixed_calibrated_secondary']['raw_f']['r2']))
    lines+=['','All sample-order hashes match. Per-state, bright-tail SSE/RMSE/MAE, energy, checkpoint SHA256 and paired molecule bootstrap details are in ROUND_RESULTS.json.',
        '', 'The fixed historical calibration is secondary, never refit, and did not select checkpoints. Its validation estimate uses the previously refitted map and is not a fresh holdout result.',
        '', '## Next decision','',
        'Prespecified confirmation priorities: '+', '.join(a for a,v in result['arms'].items() if v['meets_prespecified_confirmation_priority'])+'.',
        'A control-only gain can justify paired independent-seed confirmation of the gentler schedule. An adapter win would still need a simpler active capacity control before attributing benefit to its nonlinearity. Root reviews state/tail tradeoffs and chooses subsequent work.']
    (ROOT/'ROUND_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({a:{'native_r2':v['validation']['raw_f']['r2'],'delta_control':v['delta_native_r2_vs_control'],
                        'delta_epoch0':v['delta_native_r2_vs_epoch0'],'confirm':v['meets_prespecified_confirmation_priority']}
                      for a,v in result['arms'].items()},indent=2))

if __name__=='__main__':
    main()
