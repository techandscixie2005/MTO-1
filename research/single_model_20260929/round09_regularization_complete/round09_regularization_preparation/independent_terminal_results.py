"""Review bound aggregate arithmetic; no tensor/array decoding or inference."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = ROOT/'completion'; target = out/'INDEPENDENT_RESULT_CHECKS.json'
    assert not target.exists()
    receipt = read(out/'ANALYSIS_RECEIPT.json')
    review = read(ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json')
    metadata = read(out/'INDEPENDENT_TERMINAL_METADATA.json')
    result = read(out/'ROUND09_RESULTS.json')
    invocation = read(ROOT/'ops/TERMINAL_ANALYSIS_INVOCATION_01.json')
    assert receipt['passed'] and review['passed'] and metadata['passed']
    assert receipt['source_hashes'] == review['source_hashes']
    assert invocation['exit_code'] == 0 and invocation['environment']['CUDA_VISIBLE_DEVICES'] == ''
    assert invocation['source_review_sha256'] == sha(ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json')
    assert invocation['log_sha256'] == sha(ROOT/'ops/terminal_analysis_01.log')
    for group in ('input_hashes', 'source_hashes', 'output_hashes'):
        for p, h in receipt[group].items():
            assert sha(p) == h, p
    assert receipt['no_model_inference'] and receipt['no_test_or_raw_dataset_targets_opened']
    assert not result['model_inference'] and not result['test_access']
    sst = None
    for arm, data in result['arms'].items():
        assert data['selected_epoch'] == metadata['arms'][arm]['selected_epoch']
        v = result['verification'][arm]
        assert v['completed_epoch'] == 60 and v['steps'] == 112860
        assert v['optimizer_tensor_count'] == 135 and v['weight_decay'] == metadata['arms'][arm]['weight_decay']
        assert v['geometry_export_tensors_exactly_equal_selected_model'] and v['test_numeric_rows'] == 0
        assert v['saved_arrays_recomputed'] and v['dormant_transport_matches_initial_zero_hash']
        for policy in ('selected_best', 'fixed60'):
            row = data[policy]; p = row['pooled']
            assert p['count'] == 66860
            near(p['mse'], p['sse']/66860); near(p['rmse']**2, p['mse'])
            implied_sst = p['sse']/(1-p['r2'])
            if sst is None: sst = implied_sst
            near(implied_sst, sst)
            states = row['per_state']; assert set(states) == {str(k) for k in range(1, 11)}
            assert all(x['count'] == 6686 for x in states.values())
            near(sum(x['sse'] for x in states.values()), p['sse'])
            near(sum(x['mae'] for x in states.values())/10, p['mae'])
            bins = row['false_bright_bins']
            assert sum(x['count'] for x in bins.values()) == 66860
            near(sum(x['sse'] for x in bins.values()), p['sse'])
            near(sum(x['absolute_error_sum'] for x in bins.values())/66860, p['mae'])
            for q, threshold, count in [('q90', .0549, 6782), ('q99', .2406, 708)]:
                tail = row['bright_tail'][q]
                assert tail['threshold'] == threshold and tail['count'] == count
                near(tail['mse'], tail['sse']/count); near(tail['rmse']**2, tail['mse'])
            assert bins['true_1_pred_0']['count']+bins['true_1_pred_1']['count'] == 708
            near(bins['true_1_pred_0']['sse']+bins['true_1_pred_1']['sse'], row['bright_tail']['q99']['sse'])
            assert row['energy']['count'] == 66860
            near(row['energy']['mse'], row['energy']['sse']/66860)
            assert p['r2'] == metadata['arms'][arm]['selected_r2' if policy == 'selected_best' else 'fixed60_r2']
    ref = result['retained_reference']['metrics']
    assert ref['pooled']['r2'] == .44716940136585204
    candidate = result['arms']['coupled_l2']['selected_best']
    control = result['arms']['zero_decay']['selected_best']
    d = candidate['pooled']['r2']-control['pooled']['r2']; dr = candidate['pooled']['r2']-ref['pooled']['r2']
    gate = result['gate']
    near(gate['candidate_delta_r2_vs_contemporaneous_zero_decay'], d)
    near(gate['candidate_delta_r2_vs_retained_reference'], dr)
    assert gate['threshold'] == .003
    assert gate['passes_zero_decay_comparison'] == (d >= .003)
    assert gate['passes_retained_reference_comparison'] == (dr >= .003)
    assert gate['passed'] == (d >= .003 and dr >= .003) == False
    assert not gate['eligible_arms'] and not gate['automatic_extension_authorized'] and not gate['automatic_seed_allocation_authorized']
    u = result['paired_uncertainty']
    assert u['seed'] == 20260930 and u['draws'] == 2000 and u['component_groups'] == 5656 and u['molecules'] == 6686
    tradeoffs = {}
    for policy in ('selected_best', 'fixed60'):
        pairs = u['comparisons'][policy]; assert set(pairs) == {'coupled_l2_minus_zero_decay'}
        p = pairs['coupled_l2_minus_zero_decay']; x = result['arms']['coupled_l2'][policy]; y = result['arms']['zero_decay'][policy]
        near(p['delta_r2'], x['pooled']['r2']-y['pooled']['r2'])
        assert p['candidate'] == 'coupled_l2' and p['control'] == 'zero_decay'
        lo, hi = p['descriptive_percentile_interval']; assert math.isfinite(lo) and math.isfinite(hi) and lo < hi < 0
        for arm in result['arms']:
            near(result['retained_reference']['delta_r2_vs_retained_selected_reference'][policy][arm], result['arms'][arm][policy]['pooled']['r2']-ref['pooled']['r2'])
        tradeoffs[policy] = {'candidate_minus_control_r2': p['delta_r2'], 'conditional_interval': [lo, hi],
            'states_with_lower_candidate_SSE': sorted(int(k) for k in x['per_state'] if x['per_state'][k]['sse'] < y['per_state'][k]['sse']),
            'mae_delta': x['pooled']['mae']-y['pooled']['mae'], 'energy_rmse_delta': x['energy']['rmse']-y['energy']['rmse'],
            'q90_rmse_delta': x['bright_tail']['q90']['rmse']-y['bright_tail']['q90']['rmse'],
            'q99_rmse_delta': x['bright_tail']['q99']['rmse']-y['bright_tail']['q99']['rmse'],
            'false_bright_sse_delta': x['false_bright_bins']['true_0_pred_1']['sse']-y['false_bright_bins']['true_0_pred_1']['sse']}
    paths = [out/'ANALYSIS_RECEIPT.json', out/'ROUND09_RESULTS.json', out/'INDEPENDENT_TERMINAL_METADATA.json',
             ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json', ROOT/'ops/TERMINAL_ANALYSIS_INVOCATION_01.json',
             ROOT/'ops/terminal_analysis_01.log', Path(__file__)]
    doc = {'passed': True, 'scope': 'independent_source_receipt_and_aggregate_arithmetic_review',
        'input_hashes': {str(p): sha(p) for p in paths}, 'analysis_input_hashes_independently_rechecked': len(receipt['input_hashes']),
        'private_inputs_hashed_opaquely_never_decoded': True, 'no_repeated_inference_or_saved_array_reduction': True,
        'tradeoffs': tradeoffs, 'candidate_minus_retained_r2': dr, 'dual_allocation_gate_passed': False,
        'both_intervals_below_zero_conditional_on_reused_validation_and_selected_checkpoints': True,
        'not_seed_selection_or_fresh_generalization_uncertainty': True, 'new_numerical_work_authorized': False}
    target.write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(target), 'dual_gate': False}))


if __name__ == '__main__':
    main()
