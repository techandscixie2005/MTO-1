"""Persist source/aggregate-only independent approval of the bounded fixture.

No scientific imports, private checkpoint reads, dataset decoding or execution.
The manually reviewed source bytes are fixed below. Numerical evidence is the
already completed CPU and stdlib receipts, not a reviewer numerical rerun.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation')
EXPECTED = {
 'gpu_preflight.py': '045c56bbb3516805e9bed12c59b21743d532d7952a2a6e4083be396c63ac5175',
 'training.py': '6d81455173da42d90aadc0781aa0cca13ed7e7ab6478519498c50d70c7c7bc2b',
 'partition_data.py': '9aaa08bb8878803cc42961d336345d5fa53f5aad30d8cfdf135d0fd35dd6757d',
 'metrics.py': '1465cc964aea7bd0e06616be3cf8f862e876c7c57d9df3103e81e57817761bdc',
 'resources.py': '0e86e15d8be5c1e0e94bf8f85eea56a121e3aa493cb9c262e9248727b5beb918',
 'registered_entry.py': '6b62c9f041de9f0e54ec54e48cb97a8c6189a074ef22c762551fa95a70902b84',
 'registration_barrier_checks.py': '04177a7668a12022f86af63deed63009cbd5250c44757e1efd38fb4c0d571e5e',
 'run_gpu_preflight.py': 'ac98568441a73b8a48046214b55f5fea56e12c48560e8b770669aa2568fe1029',
 'launch.py': '798cc46044929963c525ce7332af955b3a0ea929d5f4f745d7edd7ac5fadd25d',
 'REGISTRATION_BARRIER_NOTE.md': 'd2b05bceb40f650912593dbba8b0bc7e739013e70a0cdd9d0f3f008118d3d3b4',
 'REGISTRATION_BARRIER_SOURCE_REVIEW.json': '2b9590777bb02fb97d885c81e15ab9124bba01a31fef5a836da6bce0af543f89',
 'REGISTRATION_BARRIER_CHECKS.json': 'd391d40abc7385fb1abf84d8dd8cbcec2bd8df4e362f2836b1a66a558dbbec54',
 'ops/registration_barrier_01.log': '60ba596cd25243cb1246b121893bab4a38747e395817ead0c9fce671ef9e8af2',
 'CPU_PREFLIGHT_REVIEW.json': 'cc7d482be5b1630664450be05918dbd74a65d519e5b818018ed08f6bb522da90',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    with Path(p).open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True)+'\n')

def main():
    for name, digest in EXPECTED.items(): assert sha(ROOT/name)==digest, name
    cpu = read(ROOT/'CPU_PREFLIGHT_REVIEW.json')
    assert cpu['passed'] and cpu['cpu_child_exit_code']==0 and cpu['outer_shell_status']==1
    pins = dict(cpu['source_hashes'])
    assert len(pins)==215
    pins.update(cpu['input_hashes'])
    for name, digest in EXPECTED.items(): pins[str(ROOT/name)] = digest
    pins[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in pins.items(): assert sha(path)==digest, path
    barrier = read(ROOT/'REGISTRATION_BARRIER_CHECKS.json')
    assert barrier['passed'] and barrier['model_data_imports']==barrier['optimizer_updates']==0
    assert not barrier['production_authorized']
    assert set(barrier['checks'])=={'success','wrong_pid','wrong_binding','closed_pipe','timeout','missing_environment'}
    for name, row in barrier['checks'].items():
        assert row['passed'] and not row['scientific_stage_called']
        assert row['returncode']==(0 if name=='success' else 1)
    for path, digest in barrier['source_hashes'].items(): assert pins[path]==digest
    source_review=read(ROOT/'REGISTRATION_BARRIER_SOURCE_REVIEW.json')
    for path,digest in source_review['source_hashes'].items(): assert pins[path]==digest
    note = {
        'passed':True, 'scope':'round08_completed_registration_barrier_receipt_review',
        'source_review_sha256':sha(ROOT/'REGISTRATION_BARRIER_SOURCE_REVIEW.json'),
        'barrier_receipt_sha256':sha(ROOT/'REGISTRATION_BARRIER_CHECKS.json'),
        'source_hashes':source_review['source_hashes'],
        'input_hashes':{str(ROOT/'ops/registration_barrier_01.log'):sha(ROOT/'ops/registration_barrier_01.log')},
        'six_stdlib_helper_cases_pass':True,
        'exercised_scope':'Synthetic helper subprocess success, wrong PID, wrong binding, EOF, timeout and missing environment; scientific entry was never called.',
        'source_only_scope':'Parent preflight/production registration and owned-child failure handling inspected, not executed against GPU or production.',
        'reviewer_numerical_rerun':False, 'production_authorized':False,
    }
    write(ROOT/'REGISTRATION_BARRIER_RESULT_REVIEW.json',note)
    pins[str(ROOT/'REGISTRATION_BARRIER_RESULT_REVIEW.json')]=sha(ROOT/'REGISTRATION_BARRIER_RESULT_REVIEW.json')
    assert not (ROOT/'GPU_PREFLIGHT.json').exists()
    assert not (ROOT/'private_preflight').exists()
    assert not (ROOT/'ops/gpu_preflight_attempt').exists()
    review = {
        'passed':True, 'scope':'round08_128_train_9_discarded_updates',
        'root_preparation_decision_sha256':'a227764a9540cfa4f89dd9ed8ea48aa923d65bb4152f2e90ff734073601379a1',
        'source_hashes':dict(sorted(pins.items())),
        'cpu_source_review_sha256':cpu['cpu_source_review_sha256'],
        'cpu_preflight_sha256':cpu['cpu_preflight_sha256'],
        'cpu_result_review_sha256':sha(ROOT/'CPU_PREFLIGHT_REVIEW.json'),
        'barrier_result_review_sha256':sha(ROOT/'REGISTRATION_BARRIER_RESULT_REVIEW.json'),
        'review_method':'Manual exact source review plus already completed CPU/stdlib aggregate evidence; every inherited and current pin rehashed. No numerical rerun, model/private checkpoint or dataset access by reviewer.',
        'blocking_findings':[],
        'arms':['original','local','neighbor'], 'unique_train_rows':128,
        'batch_size':64, 'updates_per_arm':3, 'total_discarded_optimizer_updates':9,
        'validation_test_rows_authorized':0,
        'criteria':{
            'initialization_and_cpu_algebra':'Exact same base/full initialization and zero-theta algebra; standalone/layout/receiver/preblock/one-Attention/O3/gradient checks already passed.',
            'training':'Actual original LE+Ls production objective; original frozen theta, local/neighbor live theta, no loss/coefficient/LR changes.',
            'geometry_fixture':'First two of the already decoded TRAIN rows only; initial original parity and each trained model versus its own complete geometry export.',
            'resume':'Update1/update2/CPU-map restore update1/replay update2; exact RNG/order/integer state and fixed numerical model/Adam/loss bounds.',
            'model_optimizer_atol':2e-6,'model_optimizer_rtol':1e-5,
            'loss_atol':1e-6,'loss_rtol':1e-5,
            'resources':'Fresh physical GPU1 health admission and shared lock, UUID prebound before interpreter startup, registered-entry barrier before scientific imports.',
            'ownership':'OS argv contains registered_entry.py then gpu_preflight.py; same registered PID executes run_path. Registration failure cannot release scientific entry.',
            'recovery':'Refuse existing attempt/private/result; preserve failure, no automatic retry or extra updates.',
        },
        'cpu_operational_caveat':cpu['operational_caveat'],
        'production_authorized':False,
        'technical_weights_may_initialize_production':False,
        'gpu_result_pending':True,
    }
    write(ROOT/'TECHNICAL_SOURCE_REVIEW.json',review)
    print(json.dumps({'passed':True,'source_count':len(pins),
        'barrier_result_review_sha256':sha(ROOT/'REGISTRATION_BARRIER_RESULT_REVIEW.json'),
        'technical_review_sha256':sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')}))

if __name__=='__main__': main()
