"""CPU-only synthetic geometry audit of the packaged one-checkpoint loader."""
import json
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parent
BASELINES = ROOT.parent / 'baselines'
sys.path.insert(0, str(BASELINES))
CHECKPOINT = BASELINES / 'calibrated_eta0.pt'


def main():
    torch.set_num_threads(1)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert checkpoint_sha256 == 'bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9'
    original_load = torch.load
    accesses = []
    payload_summary = {}

    def tracked_load(path, *args, **kwargs):
        accesses.append(str(path))
        payload = original_load(path, *args, **kwargs)
        if Path(path) == CHECKPOINT:
            payload_summary.update(format=payload['format'], config=payload['config'],
                                   normalization=payload['normalization'],
                                   checkpoint_keys=sorted(payload))
        return payload

    with patch('torch.load', tracked_load):
        from calibrated_eta0 import load_predictor
        model = load_predictor(CHECKPOINT, device='cpu')
        z = torch.tensor([8, 1, 1])
        pos = torch.tensor([[0., 0., 0.], [.96, 0., 0.], [-.24, .93, 0.]])
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
        with torch.no_grad():
            output = model(z=z, pos=pos, batch=torch.zeros(3, dtype=torch.long),
                           n=1, edge_index=edge_index)
    library_constants = [name for name in accesses if name.endswith('/e3nn/o3/constants.pt')]
    trained_checkpoints = [name for name in accesses if name not in library_constants]
    assert trained_checkpoints == [str(CHECKPOINT)], accesses
    assert all(torch.isfinite(value).all() for value in output.values())
    assert list(output['f'].shape) == [1, 10]
    assert payload_summary['normalization']['source'] == 'train_only'
    result = {'passed': True, 'torch_load_calls': accesses,
              'trained_checkpoint_load_calls': trained_checkpoints,
              'fixed_library_constant_load_calls': library_constants,
              'checkpoint_sha256': checkpoint_sha256, 'original_base_checkpoint_opened': False,
              'config_embedded': True, 'training_normalization_embedded': True,
              'synthetic_geometry_only': True, 'synthetic_accuracy_claim': False,
              'gpu_used': False, 'dataset_loaded': False, 'test_evaluated': False,
              'output_shapes': {key: list(value.shape) for key, value in output.items()},
              'payload_summary': payload_summary,
              'code_dependency': '/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/frozen_reference'}
    (ROOT / 'INFERENCE_ACCESS_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('passed', 'trained_checkpoint_load_calls',
                      'original_base_checkpoint_opened', 'config_embedded', 'training_normalization_embedded')}))


if __name__ == '__main__':
    main()
