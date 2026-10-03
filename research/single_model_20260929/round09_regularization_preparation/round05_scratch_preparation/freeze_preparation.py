"""Seal reviewed preparation inputs. No model inference or target decoding."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from pathlib import Path
import importlib.metadata as metadata,platform,sys
import numpy as np
from common import ROOT,CAMPAIGN,SPLIT,PINS,sha,read,immutable_json
from partition_data import Partition,index_sha

def verify_receipt_source(path,digest):
    path=Path(path)
    if sha(path)==digest:return
    # One explicit, independently reviewed I/O-only continuity exception.
    assert path==ROOT/'training.py' and digest=='17d2bb266704bb4a747d3b11b3ff0789f68e2bc5835230a45220d93a3f0bca76'
    mapping=read(ROOT/'IO_CONTINUITY.json');review=read(ROOT/'IO_CONTINUITY_REVIEW.json')
    assert review['passed'] and review['mapping_sha256']==sha(ROOT/'IO_CONTINUITY.json')
    assert mapping['source_path']==str(path) and mapping['executed_sha256']==digest
    assert sha(mapping['executed_snapshot_path'])==digest
    assert sha(path)==mapping['final_sha256']=='c0147d25cf2a2b6cd0956a57bfcab069657618bc22712efa29feb1b6a9533cdd'

def main():
    assert not (ROOT/'FROZEN_MANIFEST.json').exists(),'Already sealed; review any change explicitly'
    cfg=read(ROOT/'config.json');cpu=read(ROOT/'CPU_PREFLIGHT.json');gpu=read(ROOT/'GPU_PREFLIGHT.json')
    assert cpu['passed'] and gpu['passed'] and gpu['executed_optimizer_updates']==12
    assert read(ROOT/'TRAIN_STATISTICS_REVIEW.json')['passed']
    assert read(ROOT/'GPU_PREFLIGHT_REVIEW.json')['passed']
    for receipt in ('CPU_PREFLIGHT.json','GPU_PREFLIGHT.json','READER_CHECKS.json','TRAIN_STATISTICS_AUDIT.json',
                    'INDEPENDENT_GATE_CHECKS.json','TECHNICAL_SOURCE_REVIEW.json'):
        record=read(ROOT/receipt);assert record['passed']
        for name,digest in record.get('source_hashes',{}).items():
            path=Path(name);path=path if path.is_absolute() else ROOT/path
            verify_receipt_source(path,digest)
    assert read(ROOT/'TRAIN_STATISTICS_AUDIT.json')['statistics_sha256']==sha(ROOT/'TRAIN_STATISTICS.json')
    assert cpu['train_statistics_sha256']==sha(ROOT/'TRAIN_STATISTICS.json')
    assert gpu['technical_review_sha256']==sha(gpu['technical_review_path'])
    retry_review=read(ROOT/'TECHNICAL_SOURCE_REVIEW_RETRY01.json')
    assert retry_review['passed']
    for path,digest in retry_review['source_hashes'].items():verify_receipt_source(path,digest)
    # Metadata only: no NPZ member opened here.
    part=Partition('train');generator=np.random.default_rng(cfg['order_seed'])
    orders=[index_sha(part.indices[generator.permutation(len(part.indices))]) for _ in range(60)]
    names=['common.py','partition_data.py','prepare_train_statistics.py','reader_checks.py','run_statistics.py',
        'fresh_model.py','runtime.py','metrics.py','predictor.py','execution_gate.py','train.py','training.py',
        'cpu_preflight.py','gpu_preflight.py','resources.py','run_gpu_preflight.py','launch.py','freeze_preparation.py',
        'config.json','model_config.json','PROTOCOL.md','PROTOCOL_PROPOSAL.md','HISTORICAL_NONREPETITION.md',
        'TECHNICAL_PREFLIGHT_PLAN.md','ROUND05_PREPARATION_DECISION.md','TRAIN_READER_REVIEW.json',
        'PRODUCTION_AUTHORIZATION_TEMPLATE.json',
        'READER_CHECKS.json','INDEPENDENT_READER_CHECKS.json','TRAIN_STATISTICS.json','TRAIN_STATISTICS_AUDIT.json',
        'TRAIN_STATISTICS_REVIEW.json','CPU_PREFLIGHT.json','GPU_PREFLIGHT.json','TECHNICAL_SOURCE_REVIEW.json',
        'INDEPENDENT_GATE_CHECKS.json','independent_gate_checks.py','independent_reader_checks.py',
        'buffer_roundtrip_diagnosis.py','BUFFER_ROUNDTRIP_DIAGNOSIS.json','ENVIRONMENT.json']
    names+=['gpu_preflight_retry01.py','run_gpu_preflight_retry01.py','TECHNICAL_SOURCE_REVIEW_RETRY01.json',
        'IDENTITY_CHECK_AMENDMENT_PROPOSAL.md','IDENTITY_CHECK_AMENDMENT_DECISION.json',
        'gpu_identity_diagnostic.py','run_gpu_identity_diagnostic.py','GPU_IDENTITY_DIAGNOSTIC.json',
        'IDENTITY_DIAGNOSTIC_REVIEW.json','GPU_IDENTITY_DIAGNOSTIC_REVIEW.json']
    names+=['IO_CONTINUITY.json','IO_CONTINUITY_REVIEW.json','GPU_PREFLIGHT_REVIEW.json',
        'preflight_history/io_continuity/training_executed.py','independent_io_continuity.py','IO_CONTINUITY_CHECKS.json']
    environment={'python':platform.python_version(),'executable':sys.executable,
        'packages':{k:metadata.version(k) for k in ('numpy','torch','e3nn','torch-scatter','torch-cluster','torch-geometric','scipy','sympy')}}
    immutable_json(environment,ROOT/'ENVIRONMENT.json')
    paths=[ROOT/n for n in names]
    paths += [Path(path) for path in retry_review['source_hashes']]
    dependency=read(CAMPAIGN/'ops/DEPENDENCY_SOURCE_RECEIPT.json')
    for entry in dependency['files']:
        path=Path(entry['server_source'])
        if path.name=='models_ea.py' or '/frozen_reference/upstream/' in str(path):
            assert sha(path)==entry['sha256'];paths.append(path)
    paths+=[CAMPAIGN/'architecture/model.py',SPLIT/'SPLIT_MANIFEST.json',SPLIT/'INDEPENDENT_SPLIT_VERIFICATION.json',
            CAMPAIGN/'ops/monitor.py',CAMPAIGN/'ops/register_run.py',CAMPAIGN/'ops/register_cpu_stage.py']
    manifest={'source_hashes':{str(p):sha(p) for p in sorted(set(paths))},'config':cfg,
        'model_config':read(ROOT/'model_config.json'),'initial_base_tensor_sha256':cpu['initial_base_tensor_sha256'],
        'initial_full_tensor_sha256':cpu['initial_full_tensor_sha256'],'epoch_order_sha256':orders,
        'split_manifest_sha256':PINS['SPLIT_MANIFEST.json'],
        'independent_split_verification_sha256':PINS['INDEPENDENT_SPLIT_VERIFICATION.json'],
        'raw_archive_opaque_sha256':{n:PINS[n] for n in ('dataset.npz','raw_labels.npz')},
        'train_index_sha256':index_sha(part.indices),'preparation_only_no_production_authorization':True}
    immutable_json(manifest,ROOT/'FROZEN_MANIFEST.json')
    print({'files':len(manifest['source_hashes']),'manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json')})
if __name__=='__main__':main()
