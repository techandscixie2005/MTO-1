"""Capture two CPU checkpoint contracts; never load any dataset or launch work."""
import json
from pathlib import Path
import torch
from contracts import (PILOT, PILOT_MANIFEST_SHA256, SOURCES, canonical_sha,
                       model_schema, pilot_manifest, require, sha, write_json_once)

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    frozen = pilot_manifest()
    code_checks = {}
    for name, expected in frozen['source_hashes'].items():
        if Path(name).suffix in {'.py', '.json', '.md'}:
            require(sha(name) == expected, 'Frozen source changed: ' + name)
            code_checks[name] = expected
    reference = json.loads((Path(frozen['config']['source']) / 'configs/mto_eta0.json').read_text())
    sources = {}
    for seed, (epoch, expected_sha) in SOURCES.items():
        root = Path('/home/inspur/MTO-1/research/oscillator_r2_20260928/eta0_seed_replication/runs') / ('seed%d' % seed)
        path = root / 'best_legacy.pt'
        require(sha(path) == expected_sha, 'Source checkpoint hash mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        require(payload['epoch'] == epoch and payload['selection'] == 'legacy', 'Source metadata mismatch')
        require(payload['config']['seed'] == seed, 'Source seed mismatch')
        differences = {key: [reference.get(key), payload['config'].get(key)]
                       for key in set(reference) | set(payload['config'])
                       if reference.get(key) != payload['config'].get(key)}
        require(set(differences) <= {'seed', 'name', 'gpu', 'max_epochs'}, 'Scientific source settings differ')
        metrics = json.loads((root / 'FIXED_CHECKPOINT_METRICS.json').read_text())['legacy']
        require(metrics['checkpoint_sha256'] == expected_sha and metrics['epoch'] == epoch,
                'Historical replay provenance mismatch')
        sources[str(seed)] = {
            'seed': seed, 'checkpoint': str(path), 'checkpoint_sha256': expected_sha,
            'epoch': epoch, 'selection': 'legacy', 'config': payload['config'],
            'config_canonical_sha256': canonical_sha(payload['config']),
            'model_tensor_count': len(payload['model']),
            'model_schema_sha256': canonical_sha(model_schema(payload['model'])),
            'source_config_differences_vs_seed11': differences,
            'validation_replay': metrics['val']['raw_native_f'],
            'validation_replay_source': str(root / 'FIXED_CHECKPOINT_METRICS.json'),
            'validation_replay_source_sha256': sha(root / 'FIXED_CHECKPOINT_METRICS.json'),
            'run_manifest_sha256': sha(root / 'RUN_MANIFEST.json'),
        }
    result = {'schema': 1, 'pilot_manifest': str(PILOT / 'FROZEN_MANIFEST.json'),
              'pilot_manifest_sha256': PILOT_MANIFEST_SHA256,
              'pilot_snapshot': frozen, 'sources': sources, 'candidate_selected': None,
              'freshly_verified_lightweight_source_hashes': code_checks,
              'launch_enabled': False, 'cpu_only': True, 'dataset_loaded': False,
              'test_loaded': False, 'new_validation_inference': False}
    write_json_once(result, ROOT / 'SOURCE_CONTRACTS.json')
    print(json.dumps({'source_contracts': str(ROOT / 'SOURCE_CONTRACTS.json'),
                      'source_seeds': list(sources), 'candidate_selected': None, 'launch_enabled': False}))


if __name__ == '__main__':
    main()
