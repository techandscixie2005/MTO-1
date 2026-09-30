"""Small preparation helpers; no model, dataset or CUDA initialization."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
ROUND03 = PARENT / 'round03_transfer'
SOURCE = Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')


def require_cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES empty before interpreter startup')
    if os.environ.get('OMP_NUM_THREADS') != '2' or os.environ.get('MKL_NUM_THREADS') != '2':
        raise RuntimeError('Set OMP_NUM_THREADS=2 and MKL_NUM_THREADS=2 before startup')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def source_hashes(names):
    return {name: sha(ROOT / name) for name in names}


def settings():
    return read_json(ROOT / 'settings.json')


def require_preparation_authority():
    review = read_json(ROOT / 'INDEPENDENT_PROTOCOL_REVIEW.json')
    if review['passed'] is not True or review['fit_authorized'] is not False:
        raise PermissionError('Protocol preparation review missing or has wrong scope')
    for name, expected in review['document_hashes'].items():
        if sha(ROOT / name) != expected:
            raise PermissionError('Reviewed protocol changed')
    if sha(ROOT / 'PREPARATION_AUTHORIZATION.md') != review['root_preparation_decision_sha256']:
        raise PermissionError('Root preparation decision is not the reviewed decision')
    return review
