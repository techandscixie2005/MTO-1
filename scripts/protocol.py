"""Freeze and check code/configuration across all three scheduled stages."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def current():
    files=[]
    for folder,patterns in [('src',('*.py',)),('scripts',('*.py','*.slurm')),('tests',('*.py',)),('configs',('grid.json','execution.json')),('data',('manifest_*.json','splits_*.json'))]:
        for pattern in patterns:files.extend((ROOT/folder).rglob(pattern))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}

def verify():
    frozen=json.loads((ROOT/'configs/frozen.json').read_text())
    assert current()==frozen['files'],'Frozen source/config changed; refusing to mix architectures'
    dataset=(ROOT/'data/qm9s_full.npz').resolve()
    assert str(dataset)==frozen['dataset']['path']
    assert dataset.stat().st_size==frozen['dataset']['bytes'] and dataset.stat().st_mtime_ns==frozen['dataset']['mtime_ns']
    return frozen

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:
        assert not (ROOT/'configs/frozen.json').exists(),'Already frozen'
        dataset=(ROOT/'data/qm9s_full.npz').resolve()
        digest=hashlib.sha256(dataset.read_bytes()).hexdigest()
        assert digest==json.loads((ROOT/'results/data_checks.json').read_text())['database_sha256']
        payload={'files':current(),'dataset':{'path':str(dataset),'bytes':dataset.stat().st_size,
                    'mtime_ns':dataset.stat().st_mtime_ns,'sha256':digest}}
        payload['protocol_id']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        (ROOT/'configs/frozen.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(verify(),indent=2))
