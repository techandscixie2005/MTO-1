"""Seal Round09 preparation. Metadata/hash checks only; no target/model decoding."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from pathlib import Path
import importlib.metadata as metadata,platform,sys
import numpy as np
from common import ROOT,CAMPAIGN,SPLIT,PINS,sha,read,immutable_json,require_cpu
from partition_data import Partition,index_sha

PREVIOUS=CAMPAIGN/'round08_transport_preparation'
PREVIOUS_MANIFEST='de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146'
DECISION='6b10b8b5a7d75e6a5c46537f8c8b9d126dcc2694d7cfec687e31f120c038d139'


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
    assert sha(ROOT/'ROUND09_PREPARATION_DECISION.md')==DECISION
    assert sha(PREVIOUS/'FROZEN_MANIFEST.json')==PREVIOUS_MANIFEST
    inherited=read(PREVIOUS/'FROZEN_MANIFEST.json')
    paths=[PREVIOUS/'FROZEN_MANIFEST.json']
    for path,digest in inherited['source_hashes'].items():
        assert sha(path)==digest;paths.append(Path(path))
    cfg=read(ROOT/'config.json');cpu=read(ROOT/'CPU_PREFLIGHT.json');gpu=read(ROOT/'GPU_PREFLIGHT.json')
    assert cfg['epochs']==60 and list(cfg['arms'])==['zero_decay','coupled_l2']
    assert cpu['model_updates']==0 and gpu['executed_optimizer_updates']==6
    assert gpu['unique_train_molecules']==128 and gpu['validation_test_numeric_rows']==0
    assert gpu['preflight_states_are_never_production_initialization']
    assert set(gpu['arms'])==set(cfg['arms'])
    assert all(v['updates']==3 for v in gpu['arms'].values())
    assert read(ROOT/'GPU_UPDATE_PROGRESS.json')['executed_optimizer_updates']==6
    terminal=read(ROOT/'ops/gpu_preflight_attempt_retry01/COMPLETE.json')
    assert terminal['passed'] and terminal['exit_code']==terminal['registration_returncode']==0
    assert terminal['review_unchanged'] and terminal['root_decision_unchanged']
    assert terminal['preflight_sha256']==sha(ROOT/'GPU_PREFLIGHT.json')
    assert terminal['log_sha256']==sha(ROOT/'ops/gpu_preflight_attempt_retry01/stage.log')
    launched=read(ROOT/'ops/gpu_preflight_attempt_retry01/LAUNCH_RECEIPT.json')
    assert launched['registration_barrier_released'] and launched['registration_returncode']==0
    assert launched['command'][1:]==[str(ROOT/'resource_retry_entry.py'),str(ROOT/'gpu_preflight.py')]
    assert launched['review_sha256']==sha(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    assert terminal['original_technical_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')=='5c4b82c015ec2b5abc475fbaa846757f7eb5b3fc9e0bc00991bacef0175ed457'
    assert launched['original_technical_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert terminal['resource_retry_decision_unchanged']
    assert terminal['resource_retry_review_sha256']==sha(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    assert sha(ROOT/'ROUND09_RESOURCE_RETRY_DECISION.md')=='673bcc503705fe6063e14f02d3573738d82a12907f18e730d41d760c7ae277e8'
    assert launched['root_resource_retry_decision_sha256']==sha(ROOT/'ROUND09_RESOURCE_RETRY_DECISION.md')
    assert launched['gpu']==read(ROOT/'RESOURCE_RETRY_SELECTION.json')['selected_gpu']==4
    assert not list((ROOT/'ops/gpu_preflight_attempt').iterdir())
    assert gpu['technical_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert gpu['root_preparation_decision_sha256']==DECISION
    assert sha(ROOT/'TRAIN_STATISTICS.json')==sha(PREVIOUS/'TRAIN_STATISTICS.json')==cpu['train_statistics_sha256']
    cpu_review=read(ROOT/'CPU_PREFLIGHT_REVIEW.json')
    assert cpu_review['cpu_preflight_sha256']==sha(ROOT/'CPU_PREFLIGHT.json')
    assert cpu_review['cpu_source_review_sha256']==sha(ROOT/'CPU_SOURCE_REVIEW.json')
    execution=read(ROOT/'ops/CPU_EXECUTION_RECEIPT.json')
    assert execution['passed'] and execution['exit_code']==0 and execution['full_model_optimizer_updates']==0
    assert execution['cpu_preflight_sha256']==sha(ROOT/'CPU_PREFLIGHT.json')
    assert execution['source_review_sha256']==sha(ROOT/'CPU_SOURCE_REVIEW.json')
    assert execution['source_sha256']==sha(ROOT/'synthetic_checks.py')
    assert execution['log_sha256']==sha(ROOT/'ops/CPU_PREFLIGHT_01.log')
    gpu_review=read(ROOT/'GPU_PREFLIGHT_REVIEW.json')
    assert gpu_review['gpu_preflight_sha256']==sha(ROOT/'GPU_PREFLIGHT.json')
    assert gpu_review['technical_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert gpu_review['resource_retry_source_review_sha256']==sha(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    assert gpu_review['independent_result_checks_sha256']==sha(ROOT/'INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json')
    receipts=['CPU_SOURCE_REVIEW.json','CPU_PREFLIGHT.json','CPU_PREFLIGHT_REVIEW.json',
              'INDEPENDENT_GATE_CHECKS.json','TECHNICAL_SOURCE_REVIEW.json','GPU_PREFLIGHT.json',
              'GPU_PREFLIGHT_REVIEW.json','INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json',
              'RESOURCE_RETRY_SOURCE_REVIEW.json','INDEPENDENT_ADMISSION_FAILURE_REVIEW.json']
    for name in receipts:paths+=checked_sources(read(ROOT/name))
    assert read(ROOT/'INDEPENDENT_GATE_CHECKS.json')['checks_count']==22
    barrier=read(PREVIOUS/'REGISTRATION_BARRIER_CHECKS.json')
    assert sha(ROOT/'registered_entry.py')==sha(PREVIOUS/'registered_entry.py')
    assert len(barrier['checks'])==6 and all(v['passed'] for v in barrier['checks'].values())
    assert barrier['model_data_imports']==barrier['optimizer_updates']==0 and not barrier['production_authorized']
    assert cpu['initial_base_tensor_sha256']==inherited['initial_base_tensor_sha256']
    assert cpu['initial_full_tensor_sha256']=='f9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3'
    assert cpu['initial_transport_tensor_sha256']=='9b4cfe4de1555a051e4ef745dae7f6121c1a4f09bd35bd41dd2db9816110eceb'
    rh=sha(ROOT/'PARAMETER_ROSTER.json');roster=read(ROOT/'PARAMETER_ROSTER.json')
    assert roster['frozen_before_any_optimizer_update'] and roster['unwrapped_original_ordered_roster_equal']
    assert roster['contract']['group_count']==1 and roster['contract']['included_tensor_count']==135
    assert cpu['parameter_roster_sha256']==gpu['parameter_roster_sha256']==execution['parameter_roster_sha256']==rh
    assert cpu['optimizer_arithmetic']['toy_optimizer_step_calls']==5
    assert roster['initial_full_tensor_sha256']==cpu['initial_full_tensor_sha256']
    for arm,value in (('zero_decay',0.),('coupled_l2',1e-4)):
        assert cfg['arms'][arm]=={'mode':'original','weight_decay':value}
        assert gpu['arms'][arm]['weight_decay']==value and gpu['arms'][arm]['parameter_roster_sha256']==rh
        assert gpu['arms'][arm]['optimizer_parameter_tensor_count']==135
        assert gpu['arms'][arm]['frozen_parameters_and_buffers_unchanged']
    for arm in cfg['arms']:
        assert gpu['arms'][arm]['initial_base_tensor_sha256']==cpu['initial_base_tensor_sha256']
    # Selected partition metadata only; no NPZ numeric member is decoded.
    part=Partition('train');generator=np.random.default_rng(cfg['order_seed'])
    orders=[index_sha(part.indices[generator.permutation(len(part.indices))]) for _ in range(60)]
    assert orders==inherited['epoch_order_sha256']
    copied=['common.py','partition_data.py','metrics.py','resources.py','fresh_model.py','transport.py',
            'registered_entry.py','preflight_helpers.py',
            'model_config.json','TRAIN_STATISTICS.json','ENVIRONMENT.json']
    adapted=['runtime.py','predictor.py','training.py','execution_gate.py','train.py','launch.py','config.json']
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
           'transport.py','execution_gate.py','train.py','launch.py','resources.py','preflight_helpers.py',
           'synthetic_checks.py','gpu_preflight.py','run_gpu_preflight.py','freeze_preparation.py',
           'registered_entry.py','regularization.py','PARAMETER_ROSTER.json',
           'run_gpu_preflight_retry01.py','resource_retry_entry.py','RESOURCE_RETRY_SELECTION.json',
           'RESOURCE_RETRY_CONTINUITY.json','ROUND09_RESOURCE_RETRY_DECISION.md','RESOURCE_HOLD_REPORT.md',
           'ops/GPU_WRAPPER_RETRY01.log','ops/GPU_WRAPPER_RETRY01_EXECUTION_RECEIPT.json',
           'config.json','model_config.json','TRAIN_STATISTICS.json','ENVIRONMENT.json','SOURCE_LINEAGE.json',
           'PROTOCOL.md','PROTOCOL_PROPOSAL.md','PROPOSAL_SETTINGS.json','REFERENCE_MANIFEST.json',
           'INDEPENDENT_PROPOSAL_REVIEW.json','INDEPENDENT_PROPOSAL_REVIEW.md','INDEPENDENT_REVIEW_MANIFEST.json','RUNNER_HANDOFF.md',
           'ROUND09_PREPARATION_DECISION.md','TECHNICAL_PREFLIGHT_PLAN.md','PRODUCTION_AUTHORIZATION_TEMPLATE.json',
           'independent_gate_checks.py','INDEPENDENT_GATE_CHECKS_01.log','GPU_UPDATE_PROGRESS.json',
           'reviewer_history/TECHNICAL_BINDING_ATTEMPT01.json','reviewer_history/TECHNICAL_BINDING_ATTEMPT02.json',
           'independent_preflight_result_checks.py','INDEPENDENT_PREFLIGHT_RESULT_CHECKS_01.log',
           'ops/CPU_PREFLIGHT_01.log','ops/CPU_EXECUTION_RECEIPT.json',
           'ops/gpu_preflight_attempt_retry01/LAUNCH_RECEIPT.json','ops/gpu_preflight_attempt_retry01/COMPLETE.json',
           'ops/gpu_preflight_attempt_retry01/GPU_ADMISSION.xml','ops/gpu_preflight_attempt_retry01/REGISTRATION_TOOL.log',
           'ops/gpu_preflight_attempt_retry01/stage.log']+receipts
    paths += [ROOT/name for name in names]
    proposal_review=ROOT/'INDEPENDENT_REVIEW_MANIFEST.json'
    assert sha(proposal_review)=='1ddee16f0ad545a2a1b25ef9c09c955902fb75bc6ffbcefe1467b1c889ac2e37'
    proposal_files=read(proposal_review)['files'];assert len(proposal_files)==8
    for name,digest in proposal_files.items():assert sha(name)==digest;paths.append(Path(name))
    preserved=ROOT/'admission01_preserved/MANIFEST.json'
    assert sha(preserved)=='ab9c820e719b241021879d786a5b78f04d95f103fe0e941344ea7e4592529e65'
    paths.append(preserved)
    for entry in read(preserved)['files']:
        snapshot=Path(entry['snapshot_path']);assert sha(snapshot)==entry['sha256'] and snapshot.stat().st_size==entry['bytes']
        paths.append(snapshot)
    refs=read(ROOT/'REFERENCE_MANIFEST.json')
    for path,digest in refs['documents'].items():assert sha(path)==digest;paths.append(Path(path))
    for entry in refs['references']:
        path=Path(entry['path']);assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes'];paths.append(path)
    paths += [CAMPAIGN/'ops/monitor.py',CAMPAIGN/'ops/register_run.py',
              SPLIT/'SPLIT_MANIFEST.json',SPLIT/'INDEPENDENT_SPLIT_VERIFICATION.json']
    assert all(p.suffix.lower() not in ('.pt','.pth','.npy','.npz','.pkl','.pickle') for p in paths)
    manifest={'source_hashes':{str(p):sha(p) for p in sorted(set(paths))},'config':cfg,
        'model_config':read(ROOT/'model_config.json'),'parameter_roster_sha256':rh,'initial_base_tensor_sha256':cpu['initial_base_tensor_sha256'],
        'initial_full_tensor_sha256':cpu['initial_full_tensor_sha256'],
        'initial_transport_tensor_sha256':cpu['initial_transport_tensor_sha256'],'epoch_order_sha256':orders,
        'split_manifest_sha256':PINS['SPLIT_MANIFEST.json'],
        'independent_split_verification_sha256':PINS['INDEPENDENT_SPLIT_VERIFICATION.json'],
        'raw_archive_opaque_sha256':{n:PINS[n] for n in ('dataset.npz','raw_labels.npz')},
        'train_index_sha256':index_sha(part.indices),'preparation_only_no_production_authorization':True,
        'actual_discarded_optimizer_updates':6,'cpu_toy_optimizer_steps':5,'inherited_manifest_sha256':PREVIOUS_MANIFEST}
    immutable_json(manifest,ROOT/'FROZEN_MANIFEST.json')
    print({'files':len(manifest['source_hashes']),'manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json')})


if __name__=='__main__':main()

