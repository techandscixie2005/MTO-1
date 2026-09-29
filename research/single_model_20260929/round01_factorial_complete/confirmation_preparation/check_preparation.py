"""CPU-only source/schema/order checks; synthetic promotion stays in memory."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from contracts import (ARMS, METRIC, PILOT_MANIFEST_SHA256, load_source, make_config,
                       pilot_manifest, require, sha, validate_payload, validate_promotion,
                       write_json_once)

ROOT = Path(__file__).resolve().parent


def rejected(function):
    try:
        function()
    except (ValueError, KeyError, TypeError):
        return True
    raise AssertionError('Invalid input was accepted')


def promotion_fixture(candidate):
    return {'status': 'approved', 'approved_by': 'research_orchestrator', 'candidate': candidate,
            'pilot_manifest_sha256': PILOT_MANIFEST_SHA256, 'selection_metric': METRIC,
            'confirmation_seeds': [23, 37], 'allow_test_evaluation': False,
            'prediction_averaging': False, 'decision_rationale': 'Synthetic in-memory schema test only',
            'pilot_final_analysis': {'path': 'synthetic-never-written', 'sha256': '0' * 64},
            'confirmation_claim_criterion': {'both_new_seeds_positive': True,
                'minimum_mean_within_seed_delta_r2': 0.003,
                'contrast': 'control_vs_own_epoch_zero' if candidate == 'control' else
                    'candidate_vs_matched_control_and_own_epoch_zero'}}


def main():
    torch.set_num_threads(1)
    pilot = pilot_manifest()
    contracts = json.loads((ROOT / 'SOURCE_CONTRACTS.json').read_text())
    require(contracts['pilot_snapshot'] == pilot, 'Snapshot mismatch')
    changes = {'round', 'seed', 'order_seed', 'adapter_seed', 'source_checkpoint',
               'source_checkpoint_sha256', 'epoch_zero_anchor_r2', 'arms'}
    checks = {'missing_promotion_rejected': rejected(lambda: validate_promotion(None))}
    summaries = {}
    for seed in (23, 37):
        for candidate in ARMS:
            cfg = make_config(pilot, contracts, seed, promotion_fixture(candidate))
            for key, original in pilot['config'].items():
                if key not in changes:
                    require(cfg[key] == original, 'Inherited scientific setting changed: ' + key)
            require(cfg['launch_enabled'] is False and cfg['preparation_only'] is True, 'Launch enabled')
            require(list(cfg['arms']) == (['control'] if candidate == 'control' else ['control', candidate]),
                    'Wrong matched comparison')
            require(all(arm['gpu'] is None for arm in cfg['arms'].values()), 'GPU was assigned')
        source_cfg, payload = load_source(cfg, contracts)
        contract = contracts['sources'][str(seed)]
        wrong = dict(payload, epoch=33)
        checks['seed%d_wrong_epoch_rejected' % seed] = rejected(lambda: validate_payload(wrong, contract))
        wrong_cfg = copy.deepcopy(cfg); wrong_cfg['order_seed'] = 11
        checks['seed%d_wrong_order_seed_rejected' % seed] = rejected(lambda: load_source(wrong_cfg, contracts))
        wrong_cfg = copy.deepcopy(cfg); wrong_cfg['source_checkpoint_sha256'] = '0' * 64
        checks['seed%d_wrong_hash_rejected' % seed] = rejected(lambda: load_source(wrong_cfg, contracts))
        # Synthetic IDs have the real training count, but no dataset is opened.
        train = np.arange(pilot['train_count'], dtype=np.int64)
        a = np.random.default_rng(seed); b = np.random.default_rng(seed)
        order_hashes = []
        import hashlib
        for _ in range(pilot['config']['epochs']):
            left = a.permutation(train); right = b.permutation(train)
            require(np.array_equal(left, right), 'Paired order mismatch')
            order_hashes.append(hashlib.sha256(left.tobytes()).hexdigest())
        durable = copy.deepcopy(a.bit_generator.state)
        next_order = a.permutation(train)
        restored = np.random.default_rng(0); restored.bit_generator.state = durable
        require(np.array_equal(next_order, restored.permutation(train)), 'Order state restoration failed')
        summaries[str(seed)] = {'epoch': payload['epoch'], 'model_tensors': len(payload['model']),
            'config_seed': source_cfg['seed'], 'checkpoint_sha256': contract['checkpoint_sha256'],
            'validation_replay_expectation': contract['validation_replay']['r2'],
            'synthetic_paired_order_hashes': order_hashes,
            'order_hashes_are_actual_data_order': False}
        del payload
    for key, value in [('allow_test_evaluation', True), ('prediction_averaging', True),
                       ('candidate', 'ensemble'), ('status', 'draft')]:
        invalid = promotion_fixture('control'); invalid[key] = value
        checks['invalid_%s_rejected' % key] = rejected(lambda: validate_promotion(invalid))
    result = {'passed': True, 'checks': checks, 'seeds': summaries,
              'all_four_candidate_schemas_per_seed_tested': True,
              'candidate_selected': None, 'fit_launched': False, 'dataset_loaded': False,
              'test_loaded': False, 'gpu_used': False, 'new_validation_inference': False,
              'optimizer_and_model_resume': 'Inherited frozen training loop, not reimplemented or rerun',
              'source_hashes': {name: sha(ROOT / name) for name in
                  ('contracts.py', 'capture_contracts.py', 'generate.py', 'check_preparation.py', 'SOURCE_CONTRACTS.json')}}
    write_json_once(result, ROOT / 'CPU_CHECKS.json')
    print(json.dumps({'passed': True, 'checks': len(checks), 'seed_contracts': list(summaries),
                      'candidate_schemas_tested': 8, 'fit_launched': False}))


if __name__ == '__main__':
    main()
