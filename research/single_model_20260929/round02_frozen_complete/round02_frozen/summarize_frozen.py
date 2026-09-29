"""Terminal two-arm comparisons using all saved validation labels."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from freeze import sha
from summarize import bootstrap,verify_selected_arrays
from metrics import CAL_ALPHA,CAL_BETA

def bins(a,threshold):
    f=a['f_true'];p=a['f_pred'];mask=a['mask_f'].astype(bool);error=(p-f)**2;groups={}
    for bright in (False,True):
        for pred_bright in (False,True):
            use=mask&((f>=threshold)==bright)&((p>=threshold)==pred_bright)
            key=('true_bright' if bright else 'true_dim')+'__'+('pred_bright' if pred_bright else 'pred_dim')
            groups[key]={'count':int(use.sum()),'sse':float(error[use].sum()),
                'per_state':[{'count':int(use[:,j].sum()),'sse':float(error[:,j][use[:,j]].sum())} for j in range(10)]}
    assert sum(g['count'] for g in groups.values())==mask.sum()
    assert np.isclose(sum(g['sse'] for g in groups.values()),error[mask].sum(),rtol=1e-13)
    return groups

def main():
    cfg=json.loads((ROOT/'config.json').read_text());mf=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    mh=sha(ROOT/'FROZEN_MANIFEST.json')
    for path,expected in mf['source_hashes'].items():assert sha(path)==expected
    complete={};history={};arrays={};inputs={}
    replay=json.loads((ROOT/'TERMINAL_REPLAY.json').read_text())
    assert replay['passed'] and replay['manifest_sha256']==mh
    for path,expected in replay['source_hashes'].items():assert sha(path)==expected
    inputs['terminal_replay_sha256']=sha(ROOT/'TERMINAL_REPLAY.json')
    with np.load(PARENT/'runs/control/best_validation_predictions.npz') as z:
        anchor={k:z[k].copy() for k in z.files}
    inputs['anchor_array_sha256']=sha(PARENT/'runs/control/best_validation_predictions.npz')
    for arm in cfg['arms']:
        rd=ROOT/'runs'/arm;c=json.loads((rd/'FIT_COMPLETE.json').read_text())
        assert c['completed_epoch']==20 and c['manifest_sha256']==mh
        assert c['single_checkpoint'] and not c['test_evaluated'] and c['geometry_only_inference']
        assert c['frozen_parameters_and_all_buffers_unchanged']
        assert sha(rd/'best.pt')==c['best_checkpoint_sha256'] and sha(rd/'last.pt')==c['last_checkpoint_sha256']
        assert replay['arms'][arm]['best']['checkpoint_sha256']==c['best_checkpoint_sha256']
        assert replay['arms'][arm]['last']['checkpoint_sha256']==c['last_checkpoint_sha256']
        h=[json.loads(s) for s in (rd/'history.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in h]==list(range(21))
        assert min(h,key=lambda r:r['validation']['raw_f']['sse'])['epoch']==c['selected_epoch']
        assert all(r['frozen_parameters_and_buffers_unchanged'] for r in h[1:])
        with np.load(rd/'best_validation_predictions.npz') as z:a={k:z[k].copy() for k in z.files}
        for key in ('indices','f_true','mask_f'):assert np.array_equal(a[key],anchor[key])
        verify_selected_arrays(a,c['validation'],mf['tail_thresholds'])
        complete[arm]=c;history[arm]=h;arrays[arm]=a
        inputs[arm]={name:sha(rd/name) for name in ('FIT_COMPLETE.json','history.jsonl','best.pt','last.pt','best_validation_predictions.npz')}
    order={arm:[r['order_sha256'] for r in h[1:]] for arm,h in history.items()}
    assert order['trace']==order['raw_f']
    prior=[json.loads(s)['order_sha256'] for s in (PARENT/'runs/control/history.jsonl').read_text().splitlines()[1:]]
    inputs['round01_control_history_sha256']=sha(PARENT/'runs/control/history.jsonl')
    assert order['trace']==prior
    result={'passed':True,'manifest_sha256':mh,'single_checkpoint_per_arm':True,'predictions_averaged':False,
        'test_evaluated':False,'order_matches_between_arms_and_round01':True,
        'primary_selection':'native pooled validation raw-f SSE; epoch0 eligible; earliest exact tie',
        'interpretation':'reused validation; bootstrap descriptive after checkpoint selection, not fresh holdout confirmation',
        'training_loss_interpretation':'online batch means over changing parameters, not end-epoch frozen TRAIN evaluation',
        'raw_f_coefficient':1,'train_f_variance':cfg['train_f_variance'],'tail_thresholds':mf['tail_thresholds'],
        'arms':{},'aligned':[],'input_hashes':inputs}
    for arm,c in complete.items():
        v=c['validation'];delta=v['raw_f']['r2']-c['epoch_zero']['raw_f']['r2']
        result['arms'][arm]={'selected_epoch':c['selected_epoch'],'checkpoint':c['best_checkpoint'],
            'checkpoint_sha256':c['best_checkpoint_sha256'],'validation':v,'epoch_zero':c['epoch_zero'],
            'delta_native_r2_vs_own_epoch0':delta,'priority_for_independent_seeds':delta>=.003,
            'native_bootstrap_vs_common_zero_anchor':bootstrap(anchor['f_true'],arrays[arm]['f_pred'],anchor['f_pred'],anchor['mask_f']),
            'selected_true_pred_q99_bins':bins(arrays[arm],mf['tail_thresholds']['q99']),
            'fixed20_true_pred_q99_bins':replay['arms'][arm]['last']['true_pred_q99_bins'],
            'fixed20_validation':history[arm][-1]['validation'],'train_dynamics':[{'epoch':r['epoch'],'train':r['train'],
                'adapter_movement':r['adapter_movement'],'gradient_clip_fraction':r['gradient_clip_fraction'],
                'maximum_gradient_norm':r['max_preclip_gradient_norm'],'seconds':r['seconds']} for r in history[arm][1:]]}
    difference=complete['raw_f']['validation']['raw_f']['r2']-complete['trace']['validation']['raw_f']['r2']
    result['raw_f_minus_trace']={'selected_native_r2':difference,
        'objective_specific_confirmation_priority':difference>=.003 and result['arms']['raw_f']['priority_for_independent_seeds'],
        'native_bootstrap':bootstrap(anchor['f_true'],arrays['raw_f']['f_pred'],arrays['trace']['f_pred'],anchor['mask_f']),
        'fixed_calibration_bootstrap':bootstrap(anchor['f_true'],np.maximum(0,CAL_ALPHA*arrays['raw_f']['f_pred']+CAL_BETA),
            np.maximum(0,CAL_ALPHA*arrays['trace']['f_pred']+CAL_BETA),anchor['mask_f'])}
    for epoch in range(21):
        row={'epoch':epoch}
        for arm,h in history.items():
            v=h[epoch]['validation'];row[arm]={'native_r2':v['raw_f']['r2'],
                'fixed_calibrated_r2':v['fixed_calibrated_secondary']['raw_f']['r2'],
                'validation_LE':v['energy_loss'],'validation_Ls':v['trace_loss'],
                'validation_Lf':v['raw_f']['sse']/v['raw_f']['count']/cfg['train_f_variance'],
                'validation_own_objective':v['base_objective'] if arm=='trace' else v['energy_loss']+
                    v['raw_f']['sse']/v['raw_f']['count']/cfg['train_f_variance']}
        result['aligned'].append(row)
    output=ROOT/'ROUND02_RESULTS.json';output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Frozen-F objective pilot','',
        'Each row is one model and one checkpoint. Original parameters and all buffers stayed frozen. No test inference or prediction averaging. Validation is reused; independent-seed confirmation remains necessary.','',
        '| Arm | Selected epoch | Native R2 | Delta vs epoch0 | Fixed calibration R2 | Epoch20 native R2 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,v in result['arms'].items():
        lines.append('| %s | %s | %.9f | %+.9f | %.9f | %.9f |'%(arm,v['selected_epoch'],
            v['validation']['raw_f']['r2'],v['delta_native_r2_vs_own_epoch0'],
            v['validation']['fixed_calibrated_secondary']['raw_f']['r2'],v['fixed20_validation']['raw_f']['r2']))
    lines+=['','Selected raw-f minus trace R2: %+.9f.'%difference,
        'Priority over the zero-update anchor (>=.003): '+(', '.join(a for a,x in result['arms'].items() if x['priority_for_independent_seeds']) or 'none')+'.',
        'Raw-f objective also passes >=.003 against the active trace control: '+str(result['raw_f_minus_trace']['objective_specific_confirmation_priority'])+'.',
        '', 'Every state, true-bright tail, selected true/predicted brightness-bin partition, train loss/movement/clip trajectory, exact checkpoint path/hash and descriptive bootstrap interval appears in ROUND02_RESULTS.json.',
        'F changes both E and A through the frozen decoder. Fixed calibration is secondary and was never refit. Base representations remain in-sample for TRAIN; freezing does not remove inherited overfit.',
        'The raw-f coefficient1 gives a different gradient scale from the trace objective. No scale or schedule was adapted to validation. The shared base_objective field means LE+Ls; objective-specific validation values are computed explicitly in aligned records.']
    (ROOT/'ROUND02_RESULTS.md').write_text('\n'.join(lines)+'\n')
    receipt={'passed':True,'manifest_sha256':mh,'source_hashes':{str(__file__):sha(__file__),
        str(PARENT/'summarize.py'):sha(PARENT/'summarize.py'),str(PARENT/'metrics.py'):sha(PARENT/'metrics.py')},
        'input_hashes':inputs,'output_hashes':{name:sha(ROOT/name) for name in ('ROUND02_RESULTS.json','ROUND02_RESULTS.md')},
        'bootstrap_seed':20260930,'bootstrap_draws':2000,'all_labels_retained':True,'test_evaluated':False}
    (ROOT/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
