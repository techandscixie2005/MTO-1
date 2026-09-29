"""Independent CPU metadata, checkpoint and aggregate audit; no inference or fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXPECTED = '93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def equal(a, b):
    assert math.isclose(a, b, rel_tol=2e-10, abs_tol=1e-11), (a, b)

def state_equal(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)

def metric(row):
    assert row['count'] >= 0 and row['sse'] >= 0 and row['sst'] >= 0
    if row['count']:
        equal(row['rmse'] ** 2, row['sse'] / row['count'])
        if row['sst']:
            equal(row['r2'], 1 - row['sse'] / row['sst'])
        assert math.isfinite(row['mae']) and 0 <= row['mae'] <= row['rmse'] + 1e-12

def main():
    manifest = read(ROOT / 'FROZEN_MANIFEST.json')
    assert sha(ROOT / 'FROZEN_MANIFEST.json') == EXPECTED
    assert len(manifest['source_hashes']) == 69
    for path, digest in manifest['source_hashes'].items():
        assert sha(path) == digest, path
    cfg = manifest['config']
    rd, af = ROOT / 'runs/source33', ROOT / 'affine'
    pre = read(ROOT / 'INDEPENDENT_PRELAUNCH_REVIEW.json')
    pubpath = ROOT.parent / 'ops/ROUND03_PUBLICATION_RECEIPT.json'
    pub = read(pubpath)
    launch = read(rd / 'LAUNCH_RECEIPT.json')
    assert pre['passed'] and pre['source_hashes'] == manifest['source_hashes']
    assert pre['frozen_manifest_sha256'] == pub['frozen_manifest_sha256'] == launch['manifest_sha256'] == EXPECTED
    assert sha(ROOT / 'INDEPENDENT_PRELAUNCH_REVIEW.json') == launch['review_sha256'] == pub['independent_review_sha256']
    assert sha(pubpath) == launch['publication_receipt_sha256']
    assert pub['remote_verified'] and pub['download_before_stage_before_commit_push']
    gate, complete = read(ROOT / 'TERMINAL_SOURCE_GATE.json'), read(rd / 'FIT_COMPLETE.json')
    assert gate['passed'] and gate['source_process_exited'] and gate['source_pid'] == launch['pid'] == 1174422
    assert gate['source_process_state'] in ('absent', 'Z')
    assert not (rd / 'FAILED.json').exists()
    assert complete['completed_epoch'] == 33 and complete['steps'] == 49665
    assert Path(complete['checkpoint']) == rd / 'source_final.pt'
    assert sha(rd / 'source_final.pt') == complete['checkpoint_sha256'] == gate['source_final_sha256']
    assert sha(rd / 'last.pt') == complete['last_checkpoint_sha256'] == gate['last_checkpoint_sha256']
    history = [json.loads(line) for line in (rd / 'history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history] == list(range(1, 34))
    indices = np.load(ROOT / 'data/fit_indices.npy')
    assert len(indices) == 96284
    rng = np.random.default_rng(11)
    for row in history:
        order = indices[rng.permutation(len(indices))]
        assert hashlib.sha256(order.tobytes()).hexdigest() == row['order_sha256']
        assert row['learning_rate'] == .001 and row['optimizer_batches'] == 1505
    final = torch.load(rd / 'source_final.pt', map_location='cpu', weights_only=False)
    last = torch.load(rd / 'last.pt', map_location='cpu', weights_only=False)
    split = read(ROOT / 'split_manifest.json')
    assert last['initial_model_tensor_sha256'] == complete['initial_model_tensor_sha256'] == manifest['initial_model_tensor_sha256']
    assert last['fit_index_sha256'] == complete['fit_index_sha256'] == split['arrays']['fit']['indices_sha256']
    assert last['config'] == cfg and last['manifest_sha256'] == final['manifest_sha256'] == EXPECTED
    assert complete['fixed_epoch_selection'] and not complete['heldout_or_outer_labels_used_during_fit']
    assert final['epoch'] == 33 and final['selection'] == 'fixed_epoch_33'
    assert last['state']['history'] == history and last['state']['steps'] == 49665
    state_equal(final['model'], last['model'])
    assert final['stats'] == last['stats'] == read(ROOT / 'fit_normalization.json')
    assert final['source_config'] == last['source_config'] == read(ROOT / 'source_config.json')
    steps = [float(value['step']) for value in last['optimizer']['state'].values()]
    assert len(steps) and min(steps) == max(steps) == 49665
    assert {'python', 'numpy', 'order', 'torch', 'cuda'} <= set(last['rng'])
    frozen, export, val = [read(af / name) for name in ('COEFFICIENTS_FROZEN.json', 'EXPORT_COMPLETE.json', 'VALIDATION_COMPLETE.json')]
    summary, receipt = read(ROOT / 'ROUND03_RESULTS.json'), read(ROOT / 'ANALYSIS_RECEIPT.json')
    for record in (gate, complete, frozen, export, val, summary, receipt):
        assert record['manifest_sha256'] == EXPECTED
    assert gate['checked_unix'] < frozen['frozen_at_unix']
    coeff_sha = sha(af / 'COEFFICIENTS_FROZEN.json')
    assert coeff_sha == export['coefficient_receipt_sha256'] == val['coefficient_receipt_sha256']
    assert sha(af / 'EXPORT_COMPLETE.json') == val['export_receipt_sha256']
    attempts = sorted(af.glob('VALIDATION_ATTEMPT_*.json'))
    assert attempts
    for path in attempts:
        attempt = read(path)
        assert attempt['manifest_sha256'] == EXPECTED and attempt['coefficient_receipt_sha256'] == coeff_sha
        assert frozen['frozen_at_unix'] < attempt['started_unix'] <= val['validation_completed_unix']
        assert str(path) in receipt['input_hashes']
    evaluation_launches = sorted(ROOT.glob('VALIDATION_LAUNCH_*.json'))
    assert len(evaluation_launches) == 1
    evaluation_launch = read(evaluation_launches[0])
    assert evaluation_launch['launch_source_sha256'] == sha(ROOT / 'run_validation_bound.py')
    assert evaluation_launch['exit_code'] == 0 and evaluation_launch['pinned_affine_artifacts_unchanged']
    assert evaluation_launch['actual_gpu1_placement_observed']
    assert evaluation_launch['observed_gpu_uuids'] == ['GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800']
    assert evaluation_launch['validation_receipt_sha256'] == sha(af / 'VALIDATION_COMPLETE.json')
    for path, digest in evaluation_launch['pinned_affine_artifact_hashes'].items():
        assert sha(path) == digest
    for field in ('source_hashes', 'input_hashes', 'output_hashes'):
        for path, digest in receipt[field].items():
            assert sha(path) == digest, path
    for path, digest in frozen['source_hashes'].items():
        assert sha(path) == digest
    for path, digest in gate['source_hashes'].items():
        assert sha(path) == digest
    assert frozen['source_checkpoint_sha256'] == complete['checkpoint_sha256']
    assert frozen['full_baseline_sha256'] == cfg['source_checkpoint_sha256']
    assert sha(af / 'calibration_predictions.npz') == frozen['calibration_arrays_sha256']
    assert sha(af / 'validation_predictions.npz') == val['validation_arrays_sha256']
    assert set(frozen['maps']) == set(export['checkpoints']) == {'in_sample', 'heldout_source'}
    for m in frozen['maps'].values():
        assert m['count'] == 240710
        metric(m['native_calibration_subset_metrics'])
        metric(m['in_subset_affine_fit_metrics_descriptive'])
    assert frozen['maps']['in_sample']['zero_target_count'] == frozen['maps']['heldout_source']['zero_target_count']
    baseline = torch.load(cfg['source_checkpoint'], map_location='cpu', weights_only=False)
    source_dir = Path(cfg['source_checkpoint']).parents[2]
    for name, item in export['checkpoints'].items():
        assert sha(item['path']) == item['sha256']
        ck = torch.load(item['path'], map_location='cpu', weights_only=False)
        state_equal({k[5:]: v for k, v in ck['model'].items() if k.startswith('base.')}, baseline['model'])
        assert set(ck['model']) == {'base.' + k for k in baseline['model']} | {'alpha', 'beta'}
        assert ck['stats'] == read(source_dir / 'data/normalization.json')
        assert ck['source_config'] == read(source_dir / 'configs/mto_eta0.json')
        assert ck['geometry_only_inference'] and ck['native_full_baseline_input']
        for p in ('alpha', 'beta'):
            assert float(ck['model'][p]) == ck[p] == frozen['maps'][name][p]
        assert ck['provenance']['coefficient_receipt_sha256'] == coeff_sha
    assert summary['metrics'] == val['metrics'] and summary['energy'] == val['energy']
    assert set(val['metrics']) == {'native', 'historical_validation_fit', 'in_sample', 'heldout_source'}
    pooled_sst = None
    for name, report in val['metrics'].items():
        pooled = report['raw_f']; metric(pooled)
        assert pooled['count'] == 66860 and len(report['per_state']) == 10
        if pooled_sst is None:
            pooled_sst = pooled['sst']
        equal(pooled['sst'], pooled_sst)
        for state in report['per_state']:
            metric(state); assert state['count'] == 6686
        equal(sum(s['sse'] for s in report['per_state']), pooled['sse'])
        equal(sum(s['count'] * s['mae'] for s in report['per_state']) / 66860, pooled['mae'])
        for q, cut in {'q90': .0549, 'q99': .2412}.items():
            tail = report['bright_tail'][q]; metric(tail)
            assert tail['threshold_train_raw_f'] == cut and tail['sse'] <= pooled['sse']
        assert report['bright_tail']['q99']['count'] <= report['bright_tail']['q90']['count']
        bins = summary['false_bright_bins'][name]
        assert sum(b['count'] for b in bins.values()) == 66860
        equal(sum(b['sse'] for b in bins.values()), pooled['sse'])
        for b in bins.values(): metric(b)
    metric(val['energy'])
    native = val['metrics']['native']['raw_f']['r2']
    hist = val['metrics']['historical_validation_fit']['raw_f']['r2']
    held = val['metrics']['heldout_source']['raw_f']['r2']
    inside = val['metrics']['in_sample']['raw_f']['r2']
    assert abs(native - .4052941183410983) < 1e-7 and abs(hist - .418119238799) < 1e-7
    equal(val['heldout_minus_native_r2'], held - native)
    equal(val['heldout_minus_in_sample_r2'], held - inside)
    passed_gate = held - native >= .003 and held - inside >= .003
    assert val['nonlinear_preparation_gate_passed'] == summary['nonlinear_preparation_gate_passed'] == passed_gate
    assert val['test_evaluated'] is False and val['prediction_averaging'] is False
    assert receipt['passed'] and receipt['CPU_only'] and not receipt['new_model_inference']
    out = {'passed': True, 'manifest_sha256': EXPECTED, 'source_files_checked': 69,
           'fixed_epochs': 33, 'optimizer_updates': 49665, 'all_orders_reproduced': True,
           'final_equals_resumable_last': True, 'source_process_state': gate['source_process_state'],
           'OS_exit_code_observed': False, 'validation_attempts_after_coefficient_freeze': len(attempts),
           'coefficient_receipt_sha256': coeff_sha, 'same_full_base_in_both_exports': True,
           'valid_calibration_labels': 240710, 'valid_validation_labels': 66860,
           'aggregate_metric_arithmetic_passed': True, 'native_r2': native, 'historical_affine_r2': hist,
           'in_sample_r2': inside, 'heldout_source_r2': held, 'nonlinear_preparation_gate_passed': passed_gate,
           'checkpoint_hashes': export['checkpoints'], 'reviewed_unix': time.time(),
           'reviewer_no_inference_no_fitting_no_raw_label_decode': True,
           'evaluation_gpu1_placement_observed': True,
           'evaluation_launch_receipt_sha256': sha(evaluation_launches[0]),
           'completed_affine_fit_gpu_placement_verified': False,
           'scope_note': 'Independent source/hash, CPU checkpoint/order and aggregate checks. Reuses source-reviewed numerical replay of saved predictions; does not refit or decode raw labels.',
           'source_sha256': sha(__file__), 'analysis_receipt_sha256': sha(ROOT / 'ANALYSIS_RECEIPT.json')}
    (ROOT / 'INDEPENDENT_TERMINAL_INTEGRITY.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
