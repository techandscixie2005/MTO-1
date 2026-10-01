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
    result = read(out/'ROUND06_RESULTS.json')
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
    raw = result['arms']['raw_f']['selected_best']
    control = result['arms']['trace_control']['selected_best']
    reference = result['retained_reference']['metrics']
    assert reference['pooled']['r2'] == .44716940136585204
    gate = result['gate']
    dc = raw['pooled']['r2']-control['pooled']['r2']
    dr = raw['pooled']['r2']-reference['pooled']['r2']
    near(gate['candidate_delta_r2_vs_contemporaneous_control'], dc)
    near(gate['candidate_delta_r2_vs_retained_reference'], dr)
    assert gate['threshold'] == .003
    assert gate['passes_contemporaneous_comparison'] == (dc >= .003)
    assert gate['passes_retained_reference_comparison'] == (dr >= .003)
    assert gate['passed'] == (dc >= .003 and dr >= .003) == False
    assert gate['eligible_arms'] == []
    assert not gate['automatic_extension_authorized'] and not gate['automatic_seed_allocation_authorized']
    uncertainty = result['paired_uncertainty']
    assert uncertainty['seed'] == 20260930 and uncertainty['draws'] == 2000
    assert uncertainty['molecules'] == 6686 and uncertainty['component_groups'] == 5656
    for policy in ('selected_best', 'fixed60'):
        pair = uncertainty['comparisons'][policy]['raw_f']
        near(pair['delta_r2_vs_control'], result['arms']['raw_f'][policy]['pooled']['r2']-
             result['arms']['trace_control'][policy]['pooled']['r2'])
        lo, hi = pair['descriptive_percentile_interval']
        assert lo < 0 < hi
    states_better = [int(k) for k in raw['per_state'] if raw['per_state'][k]['sse'] < control['per_state'][k]['sse']]
    assert sorted(states_better) == [7, 8, 9]
    assert raw['pooled']['mae'] > control['pooled']['mae']
    assert raw['energy']['rmse'] > control['energy']['rmse']
    assert all(raw['bright_tail'][q]['rmse'] > control['bright_tail'][q]['rmse'] for q in ('q90', 'q99'))
    assert raw['false_bright_bins']['true_0_pred_1']['sse'] < control['false_bright_bins']['true_0_pred_1']['sse']
    paths = [out/'ANALYSIS_RECEIPT.json', out/'ROUND06_RESULTS.json', out/'INDEPENDENT_TERMINAL_METADATA.json',
             ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json', out/'CONTROL_REPEAT_CONTEXT.md', Path(__file__)]
    report = {'passed': True, 'scope': 'independent_source_receipt_and_aggregate_result_review',
              'input_hashes': {str(p): sha(p) for p in paths},
              'analysis_input_hashes_independently_rechecked': len(receipt['input_hashes']),
              'private_inputs_hashed_opaquely_only_never_decoded': True,
              'no_repeated_inference_or_saved_array_reduction': True,
              'selected_raw_f_improves_states_vs_current_control': sorted(states_better),
              'selected_raw_f_worsens_MAE_energy_and_true_bright_tails_vs_control': True,
              'selected_raw_f_reduces_false_bright_SSE_vs_control': True,
              'dual_allocation_gate_passed': False,
              'selected_delta_r2_vs_control': dc, 'selected_delta_r2_vs_retained_reference': dr,
              'conditional_component_intervals_include_zero_selected_and_fixed60': True,
              'no_new_seed_or_extension_or_TEST_authorized': True}
    target.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'receipt_sha256': sha(target), 'dual_gate': False}))


if __name__ == '__main__':
    main()
