"""Strict preparation contracts; this module has no training or data loader."""
import copy
import hashlib
import json
import math
from pathlib import Path

PILOT = Path('/home/inspur/MTO-1/research/single_model_20260929')
PILOT_MANIFEST_SHA256 = 'eaf2c75e239f7c04717adb97220cc35731df8d1a598b14b7b014c6754bf5ebd3'
SOURCES = {
    23: (53, '7c28a2fe204f8804d05d3dc58bb9acf853f2d0505ce83a9f66819f02fd485a18'),
    37: (44, '9570b5b3743be89cd8ca3f7f75a95bc43cd8445336a3d73694858c79ae05dd82'),
}
ARMS = {'control': (False, 0.0), 'adapter': (True, 0.0),
        'decorrelation': (False, 0.001), 'both': (True, 0.001)}
METRIC = 'minimum_validation_pooled_raw_f_SSE_earliest_tie'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def model_schema(model):
    return [(key, list(value.shape), str(value.dtype)) for key, value in sorted(model.items())]


def pilot_manifest():
    path = PILOT / 'FROZEN_MANIFEST.json'
    require(sha(path) == PILOT_MANIFEST_SHA256, 'Frozen pilot manifest changed')
    return json.loads(path.read_text())


def validate_payload(payload, contract):
    seed = contract['seed']
    require(seed in SOURCES, 'Unapproved source seed')
    epoch, checkpoint_hash = SOURCES[seed]
    require(contract['checkpoint_sha256'] == checkpoint_hash, 'Wrong source hash contract')
    require(payload['epoch'] == epoch == contract['epoch'], 'Wrong source epoch')
    require(payload['selection'] == contract['selection'] == 'legacy', 'Wrong source selection')
    require(payload['config'] == contract['config'], 'Source configuration changed')
    require(payload['config']['seed'] == seed, 'Source configuration seed mismatch')
    require(canonical_sha(payload['config']) == contract['config_canonical_sha256'], 'Config hash mismatch')
    require(canonical_sha(model_schema(payload['model'])) == contract['model_schema_sha256'],
            'Source tensor schema changed')


def load_source(cfg, contracts):
    """Future runner replacement for its hardcoded epoch33/config11 block.

    Only CPU checkpoint deserialization occurs. No Data object or label file is read.
    Call after the future runner validates its own freeze and promotion receipts.
    """
    import torch
    contract = contracts['sources'][str(cfg['seed'])]
    require(cfg['order_seed'] == cfg['adapter_seed'] == cfg['seed'], 'Paired seeds changed')
    require(cfg['source_contract'] == contract, 'Config and source contract differ')
    require(cfg['source_checkpoint'] == contract['checkpoint'], 'Checkpoint path changed')
    require(cfg['source_checkpoint_sha256'] == contract['checkpoint_sha256'], 'Checkpoint pin changed')
    require(sha(contract['checkpoint']) == contract['checkpoint_sha256'], 'Checkpoint bytes changed')
    payload = torch.load(contract['checkpoint'], map_location='cpu', weights_only=False)
    validate_payload(payload, contract)
    return copy.deepcopy(contract['config']), payload


def validate_promotion(promotion):
    require(isinstance(promotion, dict), 'A promotion record is required')
    require(promotion.get('status') == 'approved', 'Promotion is not approved')
    require(promotion.get('approved_by') == 'research_orchestrator', 'Missing orchestrator decision')
    require(promotion.get('candidate') in ARMS, 'Candidate must be one of the four frozen arms')
    require(promotion.get('pilot_manifest_sha256') == PILOT_MANIFEST_SHA256, 'Wrong pilot evidence')
    require(promotion.get('selection_metric') == METRIC, 'Primary selection changed')
    require(promotion.get('confirmation_seeds') == [23, 37], 'Confirmation seeds changed')
    require(promotion.get('allow_test_evaluation') is False, 'Test evaluation is prohibited')
    require(promotion.get('prediction_averaging') is False, 'Prediction averaging is prohibited')
    require(bool(promotion.get('decision_rationale')), 'Decision rationale is required')
    evidence = promotion.get('pilot_final_analysis', {})
    require(isinstance(evidence.get('path'), str) and bool(evidence['path']), 'Analysis path missing')
    digest = evidence.get('sha256', '')
    require(isinstance(digest, str) and len(digest) == 64 and
            all(c in '0123456789abcdef' for c in digest), 'Analysis hash missing')
    criterion = promotion.get('confirmation_claim_criterion', {})
    require(criterion.get('both_new_seeds_positive') is True, 'Require both new seeds positive')
    threshold = criterion.get('minimum_mean_within_seed_delta_r2')
    require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0,
            'Root must predeclare a finite nonnegative practical mean delta')
    expected = 'control_vs_own_epoch_zero' if promotion['candidate'] == 'control' else 'candidate_vs_matched_control_and_own_epoch_zero'
    require(criterion.get('contrast') == expected, 'Wrong within-seed contrast')


def make_config(pilot, contracts, seed, promotion):
    """Return a nonlaunchable specification after a valid decision; never fit."""
    validate_promotion(promotion)
    require(seed in SOURCES, 'Unapproved confirmation seed')
    cfg = copy.deepcopy(pilot['config'])
    contract = contracts['sources'][str(seed)]
    candidate = promotion['candidate']
    chosen = ['control'] if candidate == 'control' else ['control', candidate]
    cfg.update(round='independent_seed_confirmation_v1_seed%d' % seed,
               seed=seed, order_seed=seed, adapter_seed=seed,
               source_checkpoint=contract['checkpoint'],
               source_checkpoint_sha256=contract['checkpoint_sha256'],
               source_contract=copy.deepcopy(contract),
               epoch_zero_anchor_r2=contract['validation_replay']['r2'],
               arms={name: {'adapter': ARMS[name][0], 'orth_lambda': ARMS[name][1],
                            'gpu': None} for name in chosen},
               promotion=copy.deepcopy(promotion), launch_enabled=False,
               preparation_only=True,
               required_before_launch=['generalized runner integration', 'full validation epoch-zero replay',
                   'independent review', 'new frozen manifest', 'server-to-local archive',
                   'inspected commit and verified push', 'healthy idle GPU lock and monitor registration'])
    return cfg


def write_json_once(value, path):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(json.loads(path.read_text()) == value, 'Refusing to replace existing record: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(text)
