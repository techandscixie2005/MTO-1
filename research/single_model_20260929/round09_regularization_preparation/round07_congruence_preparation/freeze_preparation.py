"""Seal Round07 preparation. Metadata/hash checks only; no target/model decoding."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from pathlib import Path
import importlib.metadata as metadata,platform,sys
import numpy as np
from common import ROOT,CAMPAIGN,SPLIT,PINS,sha,read,immutable_json,require_cpu
from partition_data import Partition,index_sha

PREVIOUS=CAMPAIGN/'round06_objective_preparation'
PREVIOUS_MANIFEST='4f1908d8f83c7a4622a8aadb511305da93465f8e29af3203327981c3c739d52b'
DECISION='637a3cd3006d3f5b84f8ca5314b6213b0bc3017dffeeafb7ed51c2a4c594c60b'


def checked_sources(record):
    assert record['passed']
    paths=[]
    for field in ('source_hashes','input_hashes'):
        for name,digest in record.get(field,{}).items():
            path=Path(name);path=path if path.is_absolute() else ROOT/path
            assert sha(path)==digest,('Changed reviewed source/input',str(path))
            paths.append(path)
    return paths


def main():
    require_cpu()
    assert not (ROOT/'FROZEN_MANIFEST.json').exists(),'Already sealed; do not replace'
    assert not (ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json').exists()
    assert not (ROOT/'runs').exists(),'No production run may precede preparation seal'
    assert sha(ROOT/'ROUND07_PREPARATION_DECISION.md')==DECISION
    assert sha(PREVIOUS/'FROZEN_MANIFEST.json')==PREVIOUS_MANIFEST
    inherited=read(PREVIOUS/'FROZEN_MANIFEST.json')
    paths=[PREVIOUS/'FROZEN_MANIFEST.json']
    for path,digest in inherited['source_hashes'].items():
        assert sha(path)==digest;paths.append(Path(path))
    cfg=read(ROOT/'config.json');cpu=read(ROOT/'CPU_PREFLIGHT.json');gpu=read(ROOT/'GPU_PREFLIGHT.json')
    assert cfg['epochs']==60 and list(cfg['arms'])==['original','scalar','tensor']
    assert cpu['model_updates']==0 and gpu['executed_optimizer_updates']==9
    assert gpu['unique_train_molecules']==128 and gpu['validation_test_numeric_rows']==0
    assert gpu['preflight_states_are_never_production_initialization']
    assert set(gpu['arms'])==set(cfg['arms'])
    assert all(v['updates']==3 for v in gpu['arms'].values())
    assert read(ROOT/'GPU_UPDATE_PROGRESS.json')['executed_optimizer_updates']==9
    terminal=read(ROOT/'ops/gpu_preflight_attempt/COMPLETE.json')
    assert terminal['passed'] and terminal['exit_code']==terminal['registration_returncode']==0
    assert terminal['review_unchanged'] and terminal['root_decision_unchanged']
    assert terminal['preflight_sha256']==sha(ROOT/'GPU_PREFLIGHT.json')
    assert terminal['log_sha256']==sha(ROOT/'ops/gpu_preflight_attempt/stage.log')
    assert gpu['technical_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert gpu['root_preparation_decision_sha256']==DECISION
    assert sha(ROOT/'TRAIN_STATISTICS.json')==sha(PREVIOUS/'TRAIN_STATISTICS.json')==cpu['train_statistics_sha256']
    cpu_review=read(ROOT/'CPU_PREFLIGHT_REVIEW.json')
    assert cpu_review['cpu_preflight_sha256']==sha(ROOT/'CPU_PREFLIGHT.json')
    assert cpu_review['source_review_sha256']==sha(ROOT/'CPU_SOURCE_REVIEW.json')
    assert cpu_review['log_sha256']==sha(ROOT/'ops/cpu_synthetic_01.log')
    gpu_review=read(ROOT/'GPU_PREFLIGHT_REVIEW.json')
    assert gpu_review['gpu_preflight_sha256']==sha(ROOT/'GPU_PREFLIGHT.json')
    assert gpu_review['technical_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert gpu_review['independent_result_checks_sha256']==sha(ROOT/'INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json')
    receipts=['CPU_SOURCE_REVIEW.json','CPU_PREFLIGHT.json','CPU_PREFLIGHT_REVIEW.json',
              'INDEPENDENT_GATE_CHECKS.json','TECHNICAL_SOURCE_REVIEW.json','GPU_PREFLIGHT.json',
              'GPU_PREFLIGHT_REVIEW.json','INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json']
    for name in receipts:paths+=checked_sources(read(ROOT/name))
    assert read(ROOT/'INDEPENDENT_GATE_CHECKS.json')['checks_count']==22
    assert cpu['initial_base_tensor_sha256']==inherited['initial_base_tensor_sha256']
    assert cpu['initial_full_tensor_sha256']=='e601a03737fc09da857f71238f193a6511d58360b05404c3a8cae1d5b84dd9d7'
    assert cpu['initial_gate_tensor_sha256']=='8a4c3581043b25ac533658a7ee9f6402fa35d9a3bc59a60b9f6409cb518d4d3e'
    assert {k:v['gate_active_parameters'] for k,v in cpu['full_models'].items()}=={'original':0,'scalar':177,'tensor':177}
    # Selected partition metadata only; no NPZ numeric member is decoded.
    part=Partition('train');generator=np.random.default_rng(cfg['order_seed'])
    orders=[index_sha(part.indices[generator.permutation(len(part.indices))]) for _ in range(60)]
    assert orders==inherited['epoch_order_sha256']
    copied=['common.py','partition_data.py','runtime.py','metrics.py','resources.py',
            'model_config.json','TRAIN_STATISTICS.json','ENVIRONMENT.json']
    adapted=['fresh_model.py','predictor.py','preflight_helpers.py','training.py','execution_gate.py','train.py','launch.py','config.json']
    lineage={}
    for name in copied+adapted:
        old=PREVIOUS/name;new=ROOT/name
        if name in copied:assert sha(old)==sha(new),('Expected exact inherited copy',name)
        lineage[name]={'previous_path':str(old),'previous_sha256':sha(old),'current_path':str(new),
                       'current_sha256':sha(new),'byte_identical':sha(old)==sha(new)}
    immutable_json({'inherited_manifest_sha256':PREVIOUS_MANIFEST,'files':lineage,
                    'trained_weights_or_optimizer_states_reused':False},ROOT/'SOURCE_LINEAGE.json')
    environment={'python':platform.python_version(),'executable':sys.executable,
        'packages':{k:metadata.version(k) for k in ('numpy','torch','e3nn','torch-scatter','torch-cluster','torch-geometric','scipy','sympy')}}
    assert environment==read(ROOT/'ENVIRONMENT.json')
    names=['common.py','partition_data.py','fresh_model.py','runtime.py','metrics.py','predictor.py','training.py',
           'congruence.py','execution_gate.py','train.py','launch.py','resources.py','preflight_helpers.py',
           'synthetic_checks.py','gpu_preflight.py','run_gpu_preflight.py','freeze_preparation.py',
           'config.json','model_config.json','TRAIN_STATISTICS.json','ENVIRONMENT.json','SOURCE_LINEAGE.json',
           'PROTOCOL.md','PROTOCOL_PROPOSAL.md','PROPOSED_SETTINGS.json','PROPOSAL_MANIFEST.json',
           'INDEPENDENT_PROPOSAL_REVIEW.json','INDEPENDENT_PROPOSAL_REVIEW.md',
           'ROUND07_PREPARATION_DECISION.md','TECHNICAL_PREFLIGHT_PLAN.md','PRODUCTION_AUTHORIZATION_TEMPLATE.json',
           'independent_gate_checks.py','ops/INDEPENDENT_GATE_CHECKS_01.log','GPU_UPDATE_PROGRESS.json',
           'independent_preflight_result_checks.py','INDEPENDENT_PREFLIGHT_RESULT_CHECKS_01.log',
           'ops/cpu_synthetic_01.log',
           'ops/gpu_preflight_attempt/LAUNCH_RECEIPT.json','ops/gpu_preflight_attempt/COMPLETE.json',
           'ops/gpu_preflight_attempt/GPU_ADMISSION.xml','ops/gpu_preflight_attempt/REGISTRATION_TOOL.log',
           'ops/gpu_preflight_attempt/stage.log']+receipts
    paths += [ROOT/name for name in names]
    paths += [CAMPAIGN/'ops/monitor.py',CAMPAIGN/'ops/register_run.py',
              SPLIT/'SPLIT_MANIFEST.json',SPLIT/'INDEPENDENT_SPLIT_VERIFICATION.json']
    assert all(p.suffix.lower() not in ('.pt','.pth','.npy','.npz','.pkl','.pickle') for p in paths)
    manifest={'source_hashes':{str(p):sha(p) for p in sorted(set(paths))},'config':cfg,
        'model_config':read(ROOT/'model_config.json'),'initial_base_tensor_sha256':cpu['initial_base_tensor_sha256'],
        'initial_full_tensor_sha256':cpu['initial_full_tensor_sha256'],
        'initial_gate_tensor_sha256':cpu['initial_gate_tensor_sha256'],'epoch_order_sha256':orders,
        'split_manifest_sha256':PINS['SPLIT_MANIFEST.json'],
        'independent_split_verification_sha256':PINS['INDEPENDENT_SPLIT_VERIFICATION.json'],
        'raw_archive_opaque_sha256':{n:PINS[n] for n in ('dataset.npz','raw_labels.npz')},
        'train_index_sha256':index_sha(part.indices),'preparation_only_no_production_authorization':True,
        'actual_discarded_optimizer_updates':9,'inherited_manifest_sha256':PREVIOUS_MANIFEST}
    immutable_json(manifest,ROOT/'FROZEN_MANIFEST.json')
    print({'files':len(manifest['source_hashes']),'manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json')})


if __name__=='__main__':main()
