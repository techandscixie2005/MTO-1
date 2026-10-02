"""Review completed receipts and aggregate arithmetic, without numeric array reads."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = ROOT/'completion'
    target = out/'INDEPENDENT_RESULT_CHECKS.json'
    assert not target.exists()
    receipt = read(out/'ANALYSIS_RECEIPT.json')
    review = read(ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json')
    metadata = read(out/'INDEPENDENT_TERMINAL_METADATA.json')
    result = read(out/'ROUND07_RESULTS.json')
    assert receipt['passed'] and review['passed'] and metadata['passed']
    assert receipt['source_hashes'] == review['source_hashes']
    for group in ('input_hashes', 'source_hashes', 'output_hashes'):
        for path, digest in receipt[group].items():
            assert sha(path) == digest, path
    assert receipt['no_model_inference'] and receipt['no_test_or_raw_dataset_targets_opened']
    assert receipt['saved_validation_arrays_recomputed'] and receipt['checkpoint_tensors_read_on_cpu']
    assert not result['model_inference'] and not result['test_access']
    for arm, data in result['arms'].items():
        assert data['selected_epoch'] == metadata['arms'][arm]['selected_epoch']
        for policy in ('selected_best', 'fixed60'):
            row = data[policy]; pooled = row['pooled']
            assert pooled['count'] == 66860
            near(pooled['mse'], pooled['sse']/66860)
            near(pooled['rmse']**2, pooled['mse'])
            assert all(v['count'] == 6686 for v in row['per_state'].values())
            near(sum(v['sse'] for v in row['per_state'].values()), pooled['sse'])
            assert sum(v['count'] for v in row['false_bright_bins'].values()) == 66860
            near(sum(v['sse'] for v in row['false_bright_bins'].values()), pooled['sse'])
            assert row['bright_tail']['q90']['count'] == 6782
            assert row['bright_tail']['q99']['count'] == 708
            assert row['bright_tail']['q90']['threshold'] == .0549
            assert row['bright_tail']['q99']['threshold'] == .2406
            expected = metadata['arms'][arm]['selected_r2' if policy == 'selected_best' else 'fixed60_r2']
            assert pooled['r2'] == expected
            verification = result['verification'][arm]
            assert verification['completed_epoch'] == 60 and verification['steps'] == 112860
            assert verification['geometry_export_tensors_exactly_equal_selected_model']
            assert verification['test_numeric_rows'] == 0 and verification['saved_arrays_recomputed']
    reference = result['retained_reference']['metrics']
    assert reference['pooled']['r2'] == .44716940136585204
    gate = result['gate']; tensor = result['arms']['tensor']['selected_best']
    deltas = {arm: tensor['pooled']['r2']-result['arms'][arm]['selected_best']['pooled']['r2'] for arm in ('original','scalar')}
    retained_delta = tensor['pooled']['r2']-reference['pooled']['r2']
    for arm in ('original','scalar'):
        near(gate['candidate_delta_r2_vs_contemporaneous_'+arm],deltas[arm])
        assert gate['passes_'+arm+'_comparison'] == (deltas[arm] >= .003)
    near(gate['candidate_delta_r2_vs_retained_reference'],retained_delta)
    assert gate['threshold'] == .003 and gate['passes_retained_reference_comparison'] == (retained_delta >= .003)
    assert gate['passed'] == (all(v >= .003 for v in deltas.values()) and retained_delta >= .003) == False
    assert gate['eligible_arms'] == []
    assert not gate['automatic_extension_authorized'] and not gate['automatic_seed_allocation_authorized']
    uncertainty = result['paired_uncertainty']
    assert uncertainty['seed'] == 20260930 and uncertainty['draws'] == 2000
    assert uncertainty['molecules'] == 6686 and uncertainty['component_groups'] == 5656
    comparisons = [('tensor','original'),('tensor','scalar'),('scalar','original')]
    for policy in ('selected_best','fixed60'):
        assert set(uncertainty['comparisons'][policy]) == {a+'_minus_'+b for a,b in comparisons}
        for a,b in comparisons:
            pair=uncertainty['comparisons'][policy][a+'_minus_'+b]
            near(pair['delta_r2'], result['arms'][a][policy]['pooled']['r2']-result['arms'][b][policy]['pooled']['r2'])
            assert pair['candidate']==a and pair['control']==b
            lo,hi=pair['descriptive_percentile_interval']; assert math.isfinite(lo) and math.isfinite(hi) and lo < hi
        for arm in result['arms']:
            near(result['retained_reference']['delta_r2_vs_retained_selected_reference'][policy][arm], result['arms'][arm][policy]['pooled']['r2']-reference['pooled']['r2'])
    tradeoffs={}
    for a,b in comparisons:
        x=result['arms'][a]['selected_best'];y=result['arms'][b]['selected_best']
        tradeoffs[a+'_minus_'+b]={
            'pooled_r2_delta':x['pooled']['r2']-y['pooled']['r2'],
            'improved_state_SSE':[int(k) for k in x['per_state'] if x['per_state'][k]['sse']<y['per_state'][k]['sse']],
            'MAE_delta':x['pooled']['mae']-y['pooled']['mae'],
            'energy_RMSE_delta':x['energy']['rmse']-y['energy']['rmse'],
            'q90_RMSE_delta':x['bright_tail']['q90']['rmse']-y['bright_tail']['q90']['rmse'],
            'q99_RMSE_delta':x['bright_tail']['q99']['rmse']-y['bright_tail']['q99']['rmse'],
            'false_bright_SSE_delta':x['false_bright_bins']['true_0_pred_1']['sse']-y['false_bright_bins']['true_0_pred_1']['sse']}
    paths = [out/'ANALYSIS_RECEIPT.json', out/'ROUND07_RESULTS.json', out/'INDEPENDENT_TERMINAL_METADATA.json',
             ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json', Path(__file__)]
    report = {'passed': True, 'scope': 'independent_source_receipt_and_aggregate_result_review',
              'input_hashes': {str(p): sha(p) for p in paths},
              'analysis_input_hashes_independently_rechecked': len(receipt['input_hashes']),
              'private_inputs_hashed_opaquely_only_never_decoded': True,
              'no_repeated_inference_or_saved_array_reduction': True,
              'selected_pair_tradeoffs':tradeoffs,
              'triple_allocation_gate_passed':False,
              'tensor_selected_delta_r2_vs_controls':deltas,
              'tensor_selected_delta_r2_vs_retained_reference':retained_delta,
              'conditional_component_intervals_exclude_seed_selection_uncertainty':True,
              'no_new_seed_or_extension_or_TEST_authorized':True}
    target.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(target), 'triple_gate': False}))


if __name__ == '__main__':
    main()
