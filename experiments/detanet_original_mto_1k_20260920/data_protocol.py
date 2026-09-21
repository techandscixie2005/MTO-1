"""One frozen 1k dataset and public training adapter, never full-data statistics."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.nn import radius_graph

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    from rdkit import Chem
    with gzip.open(ROOT/'data/source_601.json.gz', 'rt') as f:
        records = json.load(f)
    manifest = json.loads((ROOT/'data/source_manifest.json').read_text())
    splits = json.loads((ROOT/'data/splits_1k.json').read_text())
    assert {k: len(v) for k, v in splits.items()} == dict(train=800, validation=100, test=100)
    assert len(set(sum(splits.values(), []))) == 1000
    assert hashlib.sha256(gzip.decompress((ROOT/'data/source_601.json.gz').read_bytes())).hexdigest() == manifest['sample_sha256']
    source_grid = np.asarray(manifest['energy_grid_eV'], dtype=np.float64)
    notebook = json.loads((ROOT/'reference/example_calculate_properties_and_spectra.ipynb').read_text())
    assert 'torch.linspace(1.5,13.5,240)' in ''.join(''.join(c['source']) for c in notebook['cells'])
    # Exact FP32 coordinates used by the author's example, not inferred from bin count.
    grid = torch.linspace(1.5, 13.5, 240, dtype=torch.float32).double().numpy()
    table = {int(r['number']): r for r in records}
    assert len(table) == len(records) == 1000
    assert set(table) == set(sum(splits.values(), []))
    identities = set()
    converted, errors, distances = [], [], []
    for group, ids in splits.items():
        for mid in ids:
            r = table[mid]
            assert r['split'] == group
            identity = Chem.MolToSmiles(Chem.MolFromSmiles(r['smile']), canonical=True, isomericSmiles=True)
            assert identity == r['canonical_smiles'] and identity not in identities
            identities.add(identity)
            pos = np.asarray(r['pos'], dtype=np.float64)
            z = np.asarray(r['z'])
            assert pos.shape == (len(z), 3) and np.isfinite(pos).all()
            assert set(z).issubset({1,6,7,8,9})
            d = np.linalg.norm(pos[:, None]-pos[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            assert d.min() > 1e-6 and (d < 5).sum(-1).min() > 0
            assert (d < 5).sum(-1).max() <= 32  # author default graph cap never binds
            distances.append(float(d.min()))
            source = np.asarray(r['spectrum'], dtype=np.float64)
            assert source.shape == source_grid.shape == (601,) and np.isfinite(source).all()
            assert source.min() >= 0 and source.max() > 0
            target = np.interp(grid, source_grid, source)
            # Diagnostics describe interpolation only, and never select a method.
            back = np.interp(source_grid, grid, target)
            errors.append(float(np.mean((back-source)**2)))
            converted.append(dict(id=mid, split=group, identity=identity, z=r['z'], pos=r['pos'],
                                  source_index=r['source_index'], spectrum=target.tolist()))
    train = [r for r in converted if r['split'] == 'train']
    rms = float(np.sqrt(np.mean(np.square([r['spectrum'] for r in train]))))
    n_ref = float(np.median([len(r['z']) for r in train]))
    path = ROOT/'data/targets_240.json.gz'
    # Stable gzip timestamp and name for reproducible compressed content.
    payload = json.dumps(converted, separators=(',', ':')).encode()
    path.write_bytes(gzip.compress(payload, mtime=0))
    np.save(ROOT/'data/grid_240.npy', grid)
    info = dict(count=1000, splits={k: len(v) for k,v in splits.items()}, n_ref=n_ref,
                train_rms=rms, grid=grid.tolist(), source_broadening_status='unknown',
                target_units='source CSV units; absolute units unverified',
                sigma_eV=.2, sigma_status='planning initial assumption; not source alignment',
                grid_source='author calculation notebook cell 16: torch.linspace(1.5,13.5,240)',
                interpolation='numpy.interp; piecewise linear; no extrapolation',
                roundtrip_mse_mean=float(np.mean(errors)), roundtrip_mse_max=float(np.max(errors)),
                min_pair_distance_range=[min(distances), max(distances)],
                coordinates='original QM9S coordinates; Angstrom per author rc documentation; no rescaling',
                geometry_alignment='inherited audited source row/canonical identity; rechecked identities and payload hash',
                native_target_status='see audit/native_schema.json; no guessed target coordinates',
                examples=splits['test'][:4], oracle_ids=splits['train'][:4], smoke_ids=splits['train'][:32],
                test_history='same historical 1k test; not claimed untouched',
                sha256={p.name:sha(p) for p in [path, ROOT/'data/grid_240.npy', ROOT/'data/splits_1k.json', ROOT/'data/source_601.json.gz']})
    (ROOT/'data/frozen.json').write_text(json.dumps(info, indent=2))
    print(json.dumps({k:v for k,v in info.items() if k not in ['grid','sha256','smoke_ids']}, indent=2))


def load():
    frozen = json.loads((ROOT/'data/frozen.json').read_text())
    for name, digest in frozen['sha256'].items():
        assert sha(ROOT/'data'/name) == digest, 'data fingerprint mismatch: '+name
    with gzip.open(ROOT/'data/targets_240.json.gz', 'rt') as f:
        records = json.load(f)
    for r in records:
        r['pos'] = torch.tensor(r['pos'], dtype=torch.float32)
        r['z'] = torch.tensor(r['z'], dtype=torch.long)
        r['spectrum'] = torch.tensor(r['spectrum'], dtype=torch.float32)
    return {s:[r for r in records if r['split']==s] for s in ['train','validation','test']}, frozen


def collate(records, device='cpu', dtype=torch.float32):
    x = dict(z=torch.cat([r['z'] for r in records]).to(device),
             pos=torch.cat([r['pos'] for r in records]).to(device=device, dtype=dtype),
             batch=torch.cat([torch.full((len(r['z']),), i, dtype=torch.long) for i,r in enumerate(records)]).to(device),
             n=len(records))
    x['edge_index'] = radius_graph(x['pos'], r=5., batch=x['batch'], loop=False,
                                  max_num_neighbors=32, flow='source_to_target')
    y = torch.stack([r['spectrum'] for r in records]).to(device=device, dtype=dtype)
    return x, y


if __name__ == '__main__':
    prepare()
