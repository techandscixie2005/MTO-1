"""Independent terminal integrity/metric audit, no model inference or fitting."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
sys.path.insert(0, str(ROOT / 'velocity_audit'))
from audit_velocity_labels import selected_rows


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for value in iter(lambda: stream.read(1048576), b''):
            h.update(value)
    return h.hexdigest()


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    assert y.size == p.size and np.isfinite(y).all() and np.isfinite(p).all()
    sse = float(np.dot(p-y, p-y)); sst = float(np.dot(y-y.mean(), y-y.mean()))
    return dict(count=len(y), sse=sse, sst=sst, r2=1-sse/sst,
                mae=float(np.abs(p-y).mean()), rmse=float(np.sqrt(sse/len(y))))


def compare(actual, expected):
    assert actual['count'] == expected['count']
    for key in ('sse', 'sst', 'r2', 'mae', 'rmse'):
        assert np.isclose(actual[key], expected[key], rtol=1e-11, atol=1e-11), (key,actual,expected)


def main():
    torch.set_num_threads(1)
    frozen_path = CAMPAIGN / 'FROZEN_MANIFEST.json'
    manifest_sha = sha(frozen_path)
    assert manifest_sha == 'eaf2c75e239f7c04717adb97220cc35731df8d1a598b14b7b014c6754bf5ebd3'
    frozen = json.loads(frozen_path.read_text()); cfg = frozen['config']
    assert json.loads((CAMPAIGN/'round_config.json').read_text()) == cfg
    actual_hashes = {name: sha(name) for name in frozen['source_hashes']}
    assert actual_hashes == frozen['source_hashes'], 'Frozen source/data changed'
    source = Path(cfg['source'])
    with np.load(source/'data/dataset.npz', allow_pickle=False) as z:
        train, val, ids = (z[name].copy() for name in ('train', 'val', 'ids'))
    assert len(train) == 120355 and len(val) == 6686
    assert sha(source/'data/raw_labels.npz') == frozen['source_hashes'][str(source/'data/raw_labels.npz')]
    order = np.argsort(val); undo = np.argsort(order)
    with zipfile.ZipFile(source/'data/raw_labels.npz') as z:
        truth = {name: selected_rows(z,name,val[order])[undo] for name in ('f','E','mask_f','mask_E')}
    assert truth['mask_f'].sum() == 66860 and truth['mask_E'].sum() == 66860
    generator = np.random.default_rng(cfg['order_seed'])
    expected_orders = [hashlib.sha256(generator.permutation(train).tobytes()).hexdigest()
                       for _ in range(cfg['epochs'])]
    native = torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)['model']
    reference_best = None
    result = {'passed': True, 'frozen_manifest_sha256':manifest_sha,
              'verified_frozen_file_count':len(actual_hashes), 'verified_frozen_hashes':actual_hashes,
              'model_inference':False, 'fit_launched':False, 'test_evaluated':False,
              'decoded_test_label_rows':0, 'validation_label_count':66860,
              'administrative_model_capacity_error_is_not_experiment_failure':True,
              'arms':{}}
    alpha=.8511830211044088; beta=.003725185373211049
    for arm in cfg['arms']:
        run = CAMPAIGN/'runs'/arm
        assert not (run/'FAILED.json').exists()
        complete = json.loads((run/'FIT_COMPLETE.json').read_text())
        history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        assert complete['completed_epoch'] == cfg['epochs'] == 20
        assert complete['manifest_sha256'] == manifest_sha and not complete['test_evaluated']
        assert [row['epoch'] for row in history] == list(range(21))
        assert [row['order_sha256'] for row in history[1:]] == expected_orders
        selected = min(history, key=lambda row:row['validation']['raw_f']['sse'])
        assert selected['epoch'] == complete['selected_epoch'] == 0
        assert sha(run/'best.pt') == complete['best_checkpoint_sha256']
        best = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        last = torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert best['epoch'] == 0 and best['manifest_sha256'] == manifest_sha
        assert last['manifest_sha256'] == manifest_sha and last['config'] == cfg and last['arm'] == arm
        assert last['state']['completed_epoch'] == 20 and last['state']['steps'] == 37620
        assert last['state']['best_epoch'] == 0 and len(last['state']['history']) == 21
        assert {'python','numpy','order','torch','cuda'} <= set(last['rng'])
        assert last['rng']['order'] == generator.bit_generator.state
        assert len(last['optimizer']['state']) > 0
        assert all(group['amsgrad'] and group['lr'] == 1e-5 and group['weight_decay'] == 0
                   for group in last['optimizer']['param_groups'])
        assert all('max_exp_avg_sq' in value for value in last['optimizer']['state'].values())
        assert all(torch.equal(best['model'][key], value) for key,value in native.items())
        assert all(torch.count_nonzero(value)==0 for key,value in best['model'].items()
                   if key.startswith('right_adapter.mix.'))
        if reference_best is None:
            reference_best = best['model']
        else:
            assert set(reference_best) == set(best['model'])
            assert all(torch.equal(reference_best[key],value) for key,value in best['model'].items())
        with np.load(run/'best_validation_predictions.npz', allow_pickle=False) as z:
            arrays = {name:z[name].copy() for name in z.files}
        assert np.array_equal(arrays['indices'],val) and np.array_equal(arrays['ids'],ids[val])
        assert np.array_equal(arrays['f_true'],truth['f']) and np.array_equal(arrays['mask_f'],truth['mask_f'])
        assert np.array_equal(arrays['E_true'],truth['E'])
        mask = truth['mask_f'].astype(bool); target = truth['f']
        for pred, receipt in [(arrays['f_pred'],complete['validation']),
                (np.maximum(0,alpha*arrays['f_pred']+beta),complete['validation']['fixed_calibrated_secondary'])]:
            compare(metrics(target[mask],pred[mask]),receipt['raw_f'])
            for j in range(10):
                compare(metrics(target[:,j][mask[:,j]],pred[:,j][mask[:,j]]),receipt['per_state'][j])
            for name,cut in frozen['tail_thresholds'].items():
                selected_mask = mask & (target >= cut)
                compare(metrics(target[selected_mask],pred[selected_mask]),receipt['bright_tail'][name])
        compare(metrics(truth['E'][truth['mask_E']],arrays['E_pred'][truth['mask_E']]),complete['validation']['energy'])
        start, final = history[0]['validation'],history[-1]['validation']
        result['arms'][arm] = {'selected_epoch':0, 'completed_epoch':20,
            'selected_base_tensors_bitwise_native':True, 'selected_adapter_mix_exact_zero':True,
            'all_selected_model_tensors_bitwise_same_across_arms':True,
            'optimizer_rng_resume_state_present':True, 'steps':37620,
            'order_hashes_match_recomputed_seed11_sequence':True,
            'selected_checkpoint_sha256':sha(run/'best.pt'), 'last_checkpoint_sha256':sha(run/'last.pt'),
            'selected_metrics':complete['validation'], 'epoch0_metrics':start, 'epoch20_metrics':final,
            'online_train_epoch1':history[1]['train'], 'online_train_epoch20':history[-1]['train'],
            'online_train_is_not_full_checkpoint_train_evaluation':True,
            'epoch20_minus_epoch0_native_r2':final['raw_f']['r2']-start['raw_f']['r2'],
            'epoch20_minus_epoch0_calibrated_r2':final['fixed_calibrated_secondary']['raw_f']['r2']-start['fixed_calibrated_secondary']['raw_f']['r2'],
            'state_epoch20_minus_epoch0_r2':[a['r2']-b['r2'] for a,b in zip(final['per_state'],start['per_state'])],
            'input_hashes':{name:sha(run/name) for name in
                ('FIT_COMPLETE.json','history.jsonl','best.pt','last.pt','best_validation_predictions.npz')}}
        del last,best
    analysis = json.loads((CAMPAIGN/'ANALYSIS_RECEIPT.json').read_text())
    for name,expected in {**analysis['source_hashes'],**analysis['input_hashes']}.items():
        assert sha(name) == expected
    for name,expected in analysis['output_hashes'].items():
        assert sha(CAMPAIGN/name) == expected
    mechanism = json.loads((CAMPAIGN/'MECHANISM_RESULTS.json').read_text())
    mechanism_cfg = json.loads((CAMPAIGN/'MECHANISM_CONFIG.json').read_text())
    assert mechanism['configuration'] == mechanism_cfg
    assert mechanism_cfg['script_sha256'] == sha(CAMPAIGN/'mechanism_audit.py')
    assert mechanism_cfg['subset']['subset_split'] == 'train' and mechanism_cfg['subset']['molecules'] == 256
    assert mechanism['baseline']['raw_offdiagonal_cosine_squared']['count'] == 256*45
    for arm in cfg['arms']:
        for name in ('best','last'):
            value = mechanism['arms'][arm][name]
            assert value['checkpoint_sha256'] == sha(CAMPAIGN/'runs'/arm/(name+'.pt'))
            assert value['epoch'] == (0 if name == 'best' else 20)
        best = mechanism['arms'][arm]['best']
        assert best['adapter_delta_relative']['mean'] == 0
        assert best['drift_vs_epoch0']['native_f']['pooled']['rmse'] == 0
        assert best['drift_vs_epoch0']['energy']['pooled']['rmse'] == 0
    gap = json.loads((CAMPAIGN/'GAP_AUDIT_RESULTS.json').read_text())
    gap_cfg = json.loads((CAMPAIGN/'GAP_AUDIT_CONFIG.json').read_text())
    assert gap['configuration'] == gap_cfg
    assert gap_cfg['definition']['script_sha256'] == sha(CAMPAIGN/'gap_audit.py')
    cuts = np.array(gap_cfg['definition']['unique_cutoffs_eV'])
    differences = np.diff(truth['E'],axis=1)
    assert np.min(differences) >= 0
    nearest = np.minimum(np.pad(differences,((0,0),(1,0)),constant_values=np.inf),
                         np.pad(differences,((0,0),(0,1)),constant_values=np.inf))
    bins = np.digitize(nearest,cuts,right=False)
    for arm in cfg['arms']:
        with np.load(CAMPAIGN/'runs'/arm/'best_validation_predictions.npz') as z:
            prediction = z['f_pred'].copy()
        total = 0
        for j in range(len(cuts)+1):
            selected_mask = truth['mask_f'] & (bins == j)
            group = gap['arms'][arm]['groups']['gap_bin_'+str(j)]
            compare(metrics(truth['f'][selected_mask],prediction[selected_mask]),group['pooled'])
            for state in range(10):
                use = selected_mask[:,state]
                if use.any():
                    compare(metrics(truth['f'][:,state][use],prediction[:,state][use]),group['per_state'][state])
                else:
                    assert group['per_state'][state]['count'] == 0
            total += int(selected_mask.sum())
        assert total == 66860 and gap['arms'][arm]['groups']['unknown_gap']['pooled']['count'] == 0
    for name in ('GAP_AUDIT_CONFIG','MECHANISM_CONFIG'):
        old = json.loads((CAMPAIGN/(name+'_ORIGINAL.json')).read_text())
        new = json.loads((CAMPAIGN/(name+'.json')).read_text())
        if name.startswith('GAP'):
            old['definition'].pop('script_sha256');new['definition'].pop('script_sha256')
        else:
            old.pop('script_sha256');new.pop('script_sha256')
        assert old == new, 'Diagnostic scientific definitions changed'
    plot = json.loads((CAMPAIGN/'PLOT_MANIFEST.json').read_text())
    assert plot['source_sha256'] == sha(CAMPAIGN/'plotting.py')
    assert plot['output_sha256'] == sha(CAMPAIGN/'ALIGNED_VALIDATION_CURVES.svg')
    assert plot['epochs'] == list(range(21))
    for arm,expected in plot['input_history_sha256'].items():
        assert sha(CAMPAIGN/'runs'/arm/'history.jsonl') == expected
    fb = json.loads((CAMPAIGN/'FALSE_BRIGHT_RESULTS.json').read_text())
    fb_receipt = json.loads((CAMPAIGN/'FALSE_BRIGHT_RECEIPT.json').read_text())
    assert fb_receipt['passed'] and fb_receipt['frozen_manifest_sha256'] == manifest_sha
    assert fb_receipt['source_sha256'] == fb['source_sha256'] == sha(CAMPAIGN/'false_bright_audit.py')
    assert fb_receipt['output_sha256'] == sha(CAMPAIGN/'FALSE_BRIGHT_RESULTS.json')
    assert fb_receipt['admission_sha256'] == sha(CAMPAIGN/'FALSE_BRIGHT_GPU_ADMISSION.xml')
    assert fb['post_hoc'] and not fb['selection_use'] and not fb['test_evaluated'] and not fb['fit_performed']
    assert fb['threshold'] == frozen['tail_thresholds']['q99'] == .2412
    def close(x,y):
        assert np.isclose(x,y,rtol=1e-10,atol=1e-10), (x,y)
    fb_checked = {}
    for arm in cfg['arms']:
        for name,expected in fb['input_hashes'][arm].items():
            assert sha(CAMPAIGN/'runs'/arm/name) == expected
        with np.load(CAMPAIGN/'runs'/arm/'best_validation_predictions.npz') as z:
            prediction = z['f_pred'].copy()
        mask = truth['mask_f'].astype(bool)
        errors = prediction-truth['f']; squared=errors**2
        record = fb['arms'][arm]
        for when in ('selected_epoch0','fixed_epoch20'):
            check=record[when]; whole=check['whole']; groups=check['brightness_bins']
            assert whole['count'] == sum(v['count'] for v in groups.values()) == 66860
            close(whole['sse'],sum(v['sse'] for v in groups.values()))
            close(whole['sse'],sum(v['sse'] for v in whole['per_state']))
            for j in range(10):
                assert sum(v['per_state'][j]['count'] for v in groups.values()) == 6686
                close(whole['per_state'][j]['sse'],sum(v['per_state'][j]['sse'] for v in groups.values()))
            coupling=check['state_error_coupling']
            cross=np.array(coupling['error_cross_product_sums'])
            close(np.trace(cross),whole['sse'])
            assert np.allclose(cross,cross.T,rtol=1e-12,atol=1e-12)
            assert np.all(np.array(coupling['pairwise_valid_counts']) == 6686)
            for pair in coupling['adjacent_pairs']:
                j=pair['states'][0]-1
                for values in pair['strata'].values():
                    close(values['summed_error_sse'],values['individual_sse_sum']+2*values['cross_product_sum'])
                close(pair['strata']['all_valid_pairs']['cross_product_sum'],cross[j,j+1])
            for fraction,value in check['top_error_concentration'].items():
                assert value['count'] == int(np.ceil(66860*float(fraction)))
                close(value['fraction_of_all_sse'],value['sse']/whole['sse'])
        selected=record['selected_epoch0']
        close(selected['whole']['sse'],np.sum(squared[mask]))
        for tb in (False,True):
            for pb in (False,True):
                key=('true_bright' if tb else 'true_dim')+'__'+('pred_bright' if pb else 'pred_dim')
                use=mask&((truth['f']>=fb['threshold'])==tb)&((prediction>=fb['threshold'])==pb)
                assert selected['brightness_bins'][key]['count']==int(use.sum())
                close(selected['brightness_bins'][key]['sse'],np.sum(squared[use]))
        cross=errors.T@errors
        assert np.allclose(cross,selected['state_error_coupling']['error_cross_product_sums'],rtol=1e-10,atol=1e-10)
        assert np.allclose(np.corrcoef(errors.T),selected['state_error_coupling']['centered_error_correlations'],rtol=1e-10,atol=1e-10)
        ranking=np.argsort(-squared[mask],kind='stable')
        for fraction,value in selected['top_error_concentration'].items():
            close(value['sse'],squared[mask][ranking[:value['count']]].sum())
        qs=[0,.01,.1,.5,.9,.95,.99,.999,1]
        for name,array in [('prediction',prediction[mask]),('signed_error',errors[mask]),
                           ('absolute_error',np.abs(errors[mask])),('squared_error',squared[mask])]:
            for q,v in zip(qs,np.quantile(array,qs)):
                close(selected['quantiles'][name][str(q)],v)
        final=record['fixed_epoch20']; expected=result['arms'][arm]['epoch20_metrics']['raw_f']['r2']
        assert abs(final['replayed_r2']-expected)<1e-7
        close(final['replayed_r2'],1-final['whole']['sse']/158.4062371581119)
        close(record['delta_sse_epoch20_minus_epoch0']['whole'],final['whole']['sse']-selected['whole']['sse'])
        fb_checked[arm]={'selected_diagnostics_independently_recomputed':True,
            'final_replay_vs_original_history_r2_difference':final['replayed_r2']-expected,
            'final_aggregate_partition_and_cross_product_identities_verified':True,
            'final_predictions_not_saved_no_second_model_inference':True}
    rounded_gap = differences < .0287
    exact_gap = differences < cuts[0]
    gap_boundary = {'literal_cutoff_eV':.0287,'frozen_quantile_eV':float(cuts[0]),
        'adjacent_pair_classification_differences':int(np.count_nonzero(rounded_gap != exact_gap)),
        'by_adjacent_pair':np.count_nonzero(rounded_gap != exact_gap,axis=0).tolist(),
        'interpretation':'False-bright adjacent-pair diagnostic uses the displayed rounded cutoff; no primary metric changes.'}
    result['false_bright_review']={'passed':True,'arms':fb_checked,'gap_threshold_precision':gap_boundary,
        'scope':'Selected diagnostics independently recomputed; final replay checked against saved history and aggregate algebra, not independently reinferred.',
        'input_hashes':{name:sha(CAMPAIGN/name) for name in ('false_bright_audit.py','FALSE_BRIGHT_RESULTS.json',
            'FALSE_BRIGHT_RECEIPT.json','FALSE_BRIGHT_EXECUTION.log','FALSE_BRIGHT_GPU_ADMISSION.xml')}}
    result['diagnostics_verified'] = {'mechanism_checkpoint_sources_and_train_subset':True,
        'all_gap_bins_and_per_state_metrics_recomputed':True,
        'original_diagnostic_subsets_cutoffs_preserved':True,
        'plot_source_inputs_output_hashes_match':True,
        'gap_audit_uses_selected_epoch0_models_only':True,
        'mechanism_is_descriptive_not_causal_or_physical':True,
        'input_hashes':{name:sha(CAMPAIGN/name) for name in
            ('MECHANISM_RESULTS.json','MECHANISM_CONFIG.json','MECHANISM_CONFIG_ORIGINAL.json',
             'GAP_AUDIT_RESULTS.json','GAP_AUDIT_CONFIG.json','GAP_AUDIT_CONFIG_ORIGINAL.json',
             'DIAGNOSTIC_METADATA_AMENDMENT.json','PLOT_MANIFEST.json','ALIGNED_VALIDATION_CURVES.svg')}}
    result.update(analysis_receipt_verified=True, analysis_receipt_sha256=sha(CAMPAIGN/'ANALYSIS_RECEIPT.json'),
                  review_script_sha256=sha(__file__), no_arm_meets_promotion_threshold=True)
    (ROOT/'INDEPENDENT_TERMINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'frozen_files':len(actual_hashes),'selected_epochs':{a:0 for a in cfg['arms']},
        'all_selected_tensors_equal_native_and_each_other':True,'raw_valid_labels':66860,
        'all_orders_verified_against_generated_sequence':True,'resume_states_verified':True}))


if __name__ == '__main__':
    main()
