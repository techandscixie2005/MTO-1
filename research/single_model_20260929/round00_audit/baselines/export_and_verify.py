"""Package existing eta0 and fixed calibration; replay validation geometry only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from calibrated_eta0 import SOURCE, ALPHA, BETA, HARTREE_EV, CalibratedEta0, load_predictor

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SHA = '9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136'
VALIDATION_SHA = '25acae64bb4e3e541048a8705045f7f639d1113d537dbbef71672752e8065890'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_state(index):
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
    gpu = root.findall('gpu')[index]
    ecc = gpu.find('ecc_errors')
    remap = gpu.find('remapped_rows')
    values = {f'{scope}/{c.tag}': c.text for scope in ('volatile', 'aggregate') for c in ecc.find(scope)}
    for name, val in values.items():
        if 'uncorrectable' in name or 'correctable' in name:
            assert val == '0', (name, val)
    for name in ('remapped_row_pending', 'remapped_row_failure'):
        assert remap.findtext(name) == 'No'
    other_processes = [p.findtext('pid') for p in gpu.findall('processes/process_info')
                       if 'C' in p.findtext('type', '') and int(p.findtext('pid')) != os.getpid()]
    assert not other_processes, ('GPU occupied by another process', other_processes)
    assert int(gpu.findtext('fb_memory_usage/used').split()[0]) < 200, 'GPU is not idle'
    return {'uuid': gpu.findtext('uuid'), 'ecc': values,
            'pending_remap': remap.findtext('remapped_row_pending')}


def metrics(pred, truth, mask):
    delta = pred[mask] - truth[mask]
    t = truth[mask]
    sse = float(np.square(delta).sum())
    return {'count': int(mask.sum()), 'sse': sse,
            'r2': 1 - sse / float(np.square(t - t.mean()).sum()),
            'mae': float(np.abs(delta).mean()), 'rmse': float(np.sqrt(np.square(delta).mean()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical-gpu', type=int, required=True)
    args = parser.parse_args()
    assert args.physical_gpu in (1, 2, 4, 6)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.physical_gpu)
    lock = open(f'/tmp/mto_pouter_gpu_{args.physical_gpu}.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = gpu_state(args.physical_gpu)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(11)
    started = time.time()
    source = SOURCE / 'runs/mto_eta0/best.pt'
    assert digest(source) == CHECKPOINT_SHA
    val_path = SOURCE / 'runs/mto_eta0/val_predictions.npz'
    assert digest(val_path) == VALIDATION_SHA
    config = json.loads((SOURCE / 'configs/mto_eta0.json').read_text())
    stats = json.loads((SOURCE / 'data/normalization.json').read_text())
    native = torch.load(source, map_location='cpu', weights_only=False)
    model = CalibratedEta0(config, stats).eval()
    model.base.load_state_dict(native['model'], strict=True)
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in native['model'].items())
    path = ROOT / 'calibrated_eta0.pt'
    assert not path.exists(), 'Refuse to overwrite an existing exported checkpoint'
    payload = {'format': 'mto.calibrated_eta0.single_checkpoint.v1',
               'config': config, 'normalization': stats, 'model': model.state_dict(),
               'source_checkpoint_sha256': CHECKPOINT_SHA,
               'calibration': {'alpha': ALPHA, 'beta': BETA, 'rule': 'max(0, alpha*f_native+beta)'},
               'selection': 'Existing frozen calibration; no new fitting or selection',
               'output_consistency': 'f is calibrated; E and A are unchanged auxiliary outputs'}
    temp = path.with_suffix('.pt.tmp')
    with temp.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    del model, payload, native
    model = load_predictor(path, 'cuda')
    # Read saved validation arrays only; no historical-test arrays or labels.
    with np.load(val_path) as data:
        saved = {key: data[key].copy() for key in ('ids', 'indices', 'E', 'A', 'f', 'f_true', 'mask_f_true')}
    with np.load(SOURCE / 'data/dataset.npz') as data:
        indices = data['val'].copy()
        assert np.array_equal(indices, saved['indices'])
        assert np.array_equal(data['ids'][indices], saved['ids'])
        geometry = {key: torch.from_numpy(data[key][indices].copy()).cuda() for key in ('z', 'pos', 'edge')}
    expected = np.maximum(0, ALPHA * saved['f'] + BETA)
    exact_transform = model.calibrate(torch.from_numpy(saved['f']).cuda()).cpu().numpy()
    assert np.array_equal(exact_transform, expected), 'Calibration arithmetic differs from frozen formula'
    accum = {key: [] for key in ('E', 'A', 'f_native', 'f')}
    formula_error = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            end = min(start + 64, len(indices))
            z = geometry['z'][start:end]
            mask = z != 0
            counts = mask.sum(1)
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
            raw_edge = geometry['edge'][start:end].long()
            em = raw_edge[:, 0] >= 0
            x = {'z': z[mask], 'pos': geometry['pos'][start:end][mask], 'n': end-start,
                 'batch': torch.arange(end-start, device='cuda')[:, None].expand_as(z)[mask],
                 'edge_index': (raw_edge + offsets[:, None, None]).transpose(1, 2)[em].T.contiguous()}
            pred = model(**x)
            batch = {key: value.cpu().numpy() for key, value in pred.items()}
            ref_f = (2 / 3) * (batch['E'].astype('float64') / HARTREE_EV) * np.trace(
                batch['A'].astype('float64'), axis1=-2, axis2=-1)
            ref_cal = np.maximum(0, ALPHA * ref_f + BETA)
            formula_error = max(formula_error, float(np.max(np.abs(ref_cal - batch['f']))))
            for key in accum:
                accum[key].append(batch[key])
    pred = {key: np.concatenate(value) for key, value in accum.items()}
    assert formula_error < 1e-14
    errors = {key: float(np.max(np.abs(pred[key] - saved['f' if key == 'f_native' else key])))
              for key in ('E', 'A', 'f_native')}
    errors['calibrated_f'] = float(np.max(np.abs(pred['f'] - expected)))
    # Cross-device FP32 scatter may differ slightly; exact formula parity is separate.
    assert errors['f_native'] < 2e-5 and errors['calibrated_f'] < 2e-5, errors
    native_metrics = metrics(pred['f_native'], saved['f_true'], saved['mask_f_true'])
    calibrated_metrics = metrics(pred['f'], saved['f_true'], saved['mask_f_true'])
    assert abs(native_metrics['r2'] - 0.4052941183410983) < 2e-6
    assert abs(calibrated_metrics['r2'] - 0.41811923879869783) < 2e-6
    report = {'passed': True, 'checkpoint': str(path), 'checkpoint_sha256': digest(path),
              'checkpoint_bytes': path.stat().st_size, 'checkpoint_contains_optimizer': False,
              'source_checkpoint_sha256': CHECKPOINT_SHA, 'validation_array_sha256': VALIDATION_SHA,
              'architecture_parameter_count': sum(p.numel() for p in model.parameters()),
              'extra_fixed_scalar_buffers': 2, 'base_parameters_unchanged': True,
              'exact_saved_validation_formula_parity': True,
              'replayed_formula_max_abs_error': formula_error, 'replay_vs_saved_max_abs': errors,
              'native_validation': native_metrics, 'calibrated_refit_validation': calibrated_metrics,
              'historical_crossfit_validation_r2': 0.41665766,
              'same_split_refit_score_is_not_out_of_fold': True,
              'inference_inputs': ['atomic_numbers', 'positions', 'batch', 'n', 'optional edge_index'],
              'auxiliary_E_A_unchanged_f_relation_not_preserved': True,
              'source_hashes': {p.name: digest(p) for p in (ROOT/'calibrated_eta0.py', ROOT/'export_and_verify.py')},
              'gpu_before': before, 'duration_seconds': time.time()-started,
              'no_fitting': True, 'no_historical_test_replay': True}
    (ROOT / 'CALIBRATED_EXPORT_VERIFICATION.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
