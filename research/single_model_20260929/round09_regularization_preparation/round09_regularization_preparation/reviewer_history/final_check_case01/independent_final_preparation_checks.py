"""Review final lightweight source/receipt closure; no scientific imports."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--source-count',type=int,required=True)
    parser.add_argument('--report-sha256',required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/'INDEPENDENT_FINAL_PREPARATION_CHECKS.json'
    assert not output.exists()
    mp=ROOT/'FROZEN_MANIFEST.json';m=read(mp)
    assert sha(mp)==args.manifest_sha256
    source=m['source_hashes'];assert len(source)==args.source_count
    for name,digest in source.items():
        p=Path(name);assert p.is_absolute() and sha(p)==digest,name
        assert p.suffix.lower() in ('.py','.md','.json','.log','.xml','.pid','.exit','.cfg')
        assert not any(part.startswith('private_') for part in p.parts)
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    previous_path=ROOT.parent/'round08_transport_preparation/FROZEN_MANIFEST.json'
    previous=read(previous_path)
    assert sha(previous_path)=='de058a68305fe66df2bc3a16243d4121b9561d8990f303564b26a6a47b6b5146'
    assert len(previous['source_hashes'])==258
    assert all(source.get(p)==h for p,h in previous['source_hashes'].items())
    for field in ('epoch_order_sha256','initial_base_tensor_sha256','split_manifest_sha256',
                  'independent_split_verification_sha256','train_index_sha256','raw_archive_opaque_sha256'):
        assert m[field]==previous[field],field
    assert len(m['epoch_order_sha256'])==60
    assert m['initial_full_tensor_sha256']=='f9b1ced2d8d4881bd55f01f9da2a1a1b25983ab4ae1785d48d3cb3bfc348d9f3'
    assert m['initial_transport_tensor_sha256']=='9b4cfe4de1555a051e4ef745dae7f6121c1a4f09bd35bd41dd2db9816110eceb'
    assert m['actual_discarded_optimizer_updates']==6 and m['cpu_toy_optimizer_steps']==5 and m['preparation_only_no_production_authorization']
    cfg=read(ROOT/'config.json');assert cfg==m['config']
    for name in ('seed','order_seed','epochs','batch_size','optimizer','lr','betas','eps',
                 'grad_clip','scheduler','amp','tf32','dtype','num_threads','retained_v2_reference_r2',
                 'test_access','historical_weights','selection','train_molecules','validation_molecules'):
        assert cfg[name]==previous['config'][name],name
    assert list(cfg['arms'])==['zero_decay','coupled_l2']
    assert cfg['arms']=={'zero_decay':{'mode':'original','weight_decay':0.},'coupled_l2':{'mode':'original','weight_decay':1e-4}}
    assert cfg['diagnostic_norm_floor']==1e-12 and cfg['parameter_roster']=='PARAMETER_ROSTER.json'
    assert cfg['promotion_delta_r2']==.003
    assert cfg['promotion_comparators']==['contemporaneous selected zero_decay','retained v2 reference']
    assert ((cfg['train_molecules']+63)//64)*cfg['epochs']==112860
    assert m['model_config']==read(ROOT/'model_config.json')==previous['model_config']
    assert sha(ROOT/'TRAIN_STATISTICS.json')==sha(previous_path.parent/'TRAIN_STATISTICS.json')
    assert sha(ROOT/'ROUND09_PREPARATION_DECISION.md')=='6b10b8b5a7d75e6a5c46537f8c8b9d126dcc2694d7cfec687e31f120c038d139'
    gpu=read(ROOT/'GPU_PREFLIGHT.json');cpu=read(ROOT/'CPU_PREFLIGHT.json')
    assert gpu['passed'] and cpu['passed']
    assert gpu['executed_optimizer_updates']==6 and cpu['model_updates']==0
    assert gpu['validation_test_numeric_rows']==0 and gpu['unique_train_molecules']==128
    assert all(source.get(p)==h for p,h in gpu['source_hashes'].items())
    receipts=['CPU_SOURCE_REVIEW.json','CPU_PREFLIGHT_REVIEW.json','INDEPENDENT_GATE_CHECKS.json',
        'TECHNICAL_SOURCE_REVIEW.json','GPU_PREFLIGHT_REVIEW.json','INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json',
        'RESOURCE_RETRY_SOURCE_REVIEW.json','INDEPENDENT_ADMISSION_FAILURE_REVIEW.json']
    for name in receipts:
        record=read(ROOT/name);assert record['passed']
        for field in ('source_hashes','input_hashes'):
            for p,digest in record.get(field,{}).items():
                resolved=Path(p) if Path(p).is_absolute() else ROOT/p
                assert source.get(str(resolved))==digest,(name,p)
    assert read(ROOT/'INDEPENDENT_GATE_CHECKS.json')['checks_count']==22
    assert len(read(previous_path.parent/'REGISTRATION_BARRIER_CHECKS.json')['checks'])==6
    gpur=read(ROOT/'GPU_PREFLIGHT_REVIEW.json')
    for field,filename in [('gpu_preflight_sha256','GPU_PREFLIGHT.json'),('technical_review_sha256','TECHNICAL_SOURCE_REVIEW.json'),
                            ('independent_result_checks_sha256','INDEPENDENT_PREFLIGHT_RESULT_CHECKS.json'),('resource_retry_source_review_sha256','RESOURCE_RETRY_SOURCE_REVIEW.json')]:
        assert gpur[field]==sha(ROOT/filename)
    freeze=read(ROOT/'ops/FREEZE_RECEIPT.json')
    assert freeze['passed'] and freeze['exit_code']==0
    assert freeze['manifest_sha256']==args.manifest_sha256 and freeze['source_files']==args.source_count
    assert freeze['optimizer_updates']==0 and not freeze['target_numeric_decoding']
    assert not freeze['model_inference'] and not freeze['production_authorized']
    assert freeze['environment']=={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    freeze_log=ast.literal_eval((ROOT/'ops/freeze_01.log').read_text().strip())
    assert freeze_log=={'files':args.source_count,'manifest_sha256':args.manifest_sha256}
    assert freeze['log_sha256']==sha(ROOT/'ops/freeze_01.log')
    assert freeze['source_sha256']==sha(ROOT/'freeze_preparation.py')
    lineage=read(ROOT/'SOURCE_LINEAGE.json');assert not lineage['trained_weights_or_optimizer_states_reused']
    for row in lineage['files'].values():
        assert sha(row['previous_path'])==row['previous_sha256']
        assert sha(row['current_path'])==row['current_sha256']
        assert row['byte_identical']==(row['previous_sha256']==row['current_sha256'])
    assert source[str(ROOT/'PROTOCOL.md')]==sha(ROOT/'PROTOCOL.md')
    assert source[str(ROOT/'RUNNER_HANDOFF.md')]==sha(ROOT/'RUNNER_HANDOFF.md')
    roster=read(ROOT/'PARAMETER_ROSTER.json')
    assert sha(ROOT/'PARAMETER_ROSTER.json')==m['parameter_roster_sha256']=='290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943'
    assert roster['contract']['included_tensor_count']==135 and roster['unwrapped_original_ordered_roster_equal']
    assert cpu['optimizer_arithmetic']['toy_optimizer_step_calls']==5
    retry=read(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    assert retry['passed'] and retry['original_technical_review_sha256']==sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert retry['root_resource_retry_decision_sha256']==sha(ROOT/'ROUND09_RESOURCE_RETRY_DECISION.md')
    assert len(retry['source_hashes'])==335
    assert not list((ROOT/'ops/gpu_preflight_attempt').iterdir())
    assert read(ROOT/'ops/gpu_preflight_attempt_retry01/COMPLETE.json')['passed']
    for item in read(ROOT/'admission01_preserved/MANIFEST.json')['files']:
        for field in ('original_path','snapshot_path'):
            assert source.get(item[field])==item['sha256'],item[field]
    proposal=read(ROOT/'INDEPENDENT_REVIEW_MANIFEST.json')
    assert sha(ROOT/'INDEPENDENT_REVIEW_MANIFEST.json')=='1ddee16f0ad545a2a1b25ef9c09c955902fb75bc6ffbcefe1467b1c889ac2e37'
    assert len(proposal['files'])==8
    assert all(source.get(p)==h for p,h in proposal['files'].items())
    assert sha(ROOT/'PREPARATION_REPORT.md')==args.report_sha256
    assert not (ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json').exists()
    assert not (ROOT/'runs').exists()
    assert read(ROOT/'PRODUCTION_AUTHORIZATION_TEMPLATE.json')['authorized'] is False
    record={'passed':True,'phase':'final_preparation_metadata_and_source_closure',
        'frozen_manifest_sha256':args.manifest_sha256,'source_files_verified':len(source),
        'all_258_inherited_sources_unchanged':True,'all_executed_technical_sources_in_closure':True,
        'all_60_inherited_order_hashes_unchanged':True,'accepted_scientific_settings_match':True,
        'six_discarded_updates_only_and_no_validation_test_fixture':True,
        'production_authorization_absent_and_no_runs_directory':True,
        'unchanged_original_mathematical_binding_and_resource_only_amendment_verified':True,
        'complete_proposal_and_original_failure_evidence_preserved':True,
        'parameter_roster_and_five_CPU_toy_steps_verified':True,
        'no_model_inference_or_numeric_data_decode_by_checker':True,
        'input_hashes':{str(p):sha(p) for p in [mp,ROOT/'ops/FREEZE_RECEIPT.json',ROOT/'ops/freeze_01.log',
            ROOT/'PREPARATION_REPORT.md',Path(__file__)]},'source_hashes':source}
    with output.open('x') as f:f.write(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':True,'source_files':len(source),'receipt_sha256':sha(output)}))

if __name__=='__main__':main()
