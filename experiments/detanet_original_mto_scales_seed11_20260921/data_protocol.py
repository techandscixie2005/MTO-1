"""Existing scale splits, common 601 -> 240 interpolation, train-only statistics."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.nn import radius_graph

ROOT = Path(__file__).resolve().parent
SCALES = ('1k', '10k', 'full')
GROUPS = ('train', 'validation', 'test')
SOURCE = Path('/data/run01/sczc698/xxy/MTO_fullarch_20260914/data')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write_json(value,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); os.replace(tmp,path)

def prepare():
    dest=ROOT/'data'; dest.mkdir(exist_ok=True)
    if (dest/'PREPARED.json').exists():
        verify_data(); return
    manifest=json.loads((SOURCE/'manifest_full.json').read_text())
    source=SOURCE/'qm9s_full.npz'
    assert sha(source)==manifest['sample_sha256'], 'source dataset hash mismatch'
    grid=torch.linspace(1.5,13.5,240,dtype=torch.float32).double().numpy()
    with np.load(source,allow_pickle=False) as a:
        arrays={k:a[k] for k in ('z','pos','offsets','ids','identities','source_indices')}
        ids=arrays['ids']; n=len(ids); assert n==manifest['count']==129732
        assert len(set(ids.tolist()))==n
        assert len(set(arrays['identities'].tolist()))==n
        z,pos,offset=arrays['z'],arrays['pos'],arrays['offsets']
        assert np.isin(z,[1,6,7,8,9]).all() and np.isfinite(pos).all()
        assert pos.shape==(len(z),3) and offset[0]==0 and offset[-1]==len(z)
        assert (np.diff(offset)>0).all()
        y=a['spectra']; source_grid=a['energy_eV']
        assert y.shape==(n,601) and np.isfinite(y).all() and (y>=0).all()
        assert (y.max(axis=1)>0).all() and np.all(np.diff(source_grid)>0)
        assert np.array_equal(source_grid,np.asarray(manifest['energy_grid_eV']))
        assert source_grid[0]<=grid[0] and source_grid[-1]>=grid[-1]
        target=np.stack([np.interp(grid,source_grid,row) for row in y])
        assert np.isfinite(target).all()
        arrays['spectra']=target.astype(np.float32); arrays['grid']=grid
        # Reject invalid graph geometry; no filtering or resampling is performed here.
        for i in range(n):
            xyz=pos[offset[i]:offset[i+1]].astype(np.float64)
            dist=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
            np.fill_diagonal(dist,np.inf)
            assert dist.min()>1e-6 and (dist<5).sum(-1).min()>0, int(ids[i])
            assert (dist<5).sum(-1).max()<=32, int(ids[i])
        lookup={int(mid):i for i,mid in enumerate(ids)}
        infos={}
        for scale in SCALES:
            splits=json.loads((SOURCE/f'splits_{scale}.json').read_text())
            flat=sum((splits[g] for g in GROUPS),[])
            assert len(flat)==len(set(flat))
            expected=1000 if scale=='1k' else 10000 if scale=='10k' else n
            assert len(flat)==expected
            rows={g:np.asarray([lookup[mid] for mid in splits[g]]) for g in GROUPS}
            for j,g in enumerate(GROUPS): assert (a['split'][rows[g]]==j).all()
            if scale=='10k': assert set(np.flatnonzero(a['subset_10k']))==set(np.concatenate(list(rows.values())))
            if scale=='full': assert set(flat)==set(ids.tolist())
            if scale=='1k':
                old=Path(json.loads((ROOT/'protocol.json').read_text())['model_source'])
                assert splits==json.loads((old/'data/splits_1k.json').read_text())
            write_json(splits,dest/f'splits_{scale}.json')
            rms=float(np.sqrt(np.mean(target[rows['train']]**2)))
            assert rms>0 and np.isfinite(rms)
            infos[scale]=dict(scale=scale,count=len(flat),splits={g:len(rows[g]) for g in GROUPS},
                train_rms=rms,n_ref=float(np.median(np.diff(offset)[rows['train']])),grid=grid.tolist(),
                examples=splits['test'][:4],source_dataset=str(source.resolve()),
                source_sha256=manifest['sample_sha256'],split_source=str(SOURCE/f'splits_{scale}.json'),
                normalization='one RMS over this scale training spectra only',
                interpolation='numpy.interp, float64 calculation, float32 training targets; common 601 to 240 grid',
                source_broadening_status='unknown',sigma_eV=0.2,
                target_units='source CSV units; absolute units unverified',
                test_history='existing historical split; not claimed untouched')
        for key,value in arrays.items():
            with (dest/f'{key}.npy.tmp').open('wb') as f: np.save(f,value,allow_pickle=False)
            os.replace(dest/f'{key}.npy.tmp',dest/f'{key}.npy')
        common={p.name:sha(p) for p in sorted(dest.glob('*.npy'))}
        for scale,info in infos.items():
            info['sha256']={**common,f'splits_{scale}.json':sha(dest/f'splits_{scale}.json')}
            write_json(info,dest/f'frozen_{scale}.json')
        write_json(dict(source_sha256=manifest['sample_sha256'],counts={s:i['splits'] for s,i in infos.items()},
                        frozen={s:sha(dest/f'frozen_{s}.json') for s in SCALES}),dest/'PREPARED.json')
        print(json.dumps({s:{k:i[k] for k in ('splits','train_rms','n_ref')} for s,i in infos.items()}),flush=True)

def verify_data(scale=None):
    done=json.loads((ROOT/'data/PREPARED.json').read_text())
    for s in (SCALES if scale is None else (scale,)):
        assert sha(ROOT/f'data/frozen_{s}.json')==done['frozen'][s]
        info=json.loads((ROOT/f'data/frozen_{s}.json').read_text())
        for name,digest in info['sha256'].items(): assert sha(ROOT/'data'/name)==digest, name

def load(scale):
    verify_data(scale)
    info=json.loads((ROOT/f'data/frozen_{scale}.json').read_text())
    splits=json.loads((ROOT/f'data/splits_{scale}.json').read_text())
    a={k:np.load(ROOT/f'data/{k}.npy',mmap_mode='r',allow_pickle=False) for k in ('ids','z','pos','offsets','spectra')}
    lookup={int(mid):i for i,mid in enumerate(a['ids'])}; data={}
    for group in GROUPS:
        records=[]
        for mid in splits[group]:
            i=lookup[mid]; lo,hi=a['offsets'][i:i+2]
            records.append(dict(id=mid,z=torch.from_numpy(a['z'][lo:hi].copy()),
                pos=torch.from_numpy(a['pos'][lo:hi].copy()),spectrum=torch.from_numpy(a['spectra'][i].copy())))
        data[group]=records
    return data,info

def collate(records,device='cpu',dtype=torch.float32):
    x=dict(z=torch.cat([r['z'] for r in records]).to(device),
        pos=torch.cat([r['pos'] for r in records]).to(device=device,dtype=dtype),
        batch=torch.cat([torch.full((len(r['z']),),i,dtype=torch.long) for i,r in enumerate(records)]).to(device),n=len(records))
    x['edge_index']=radius_graph(x['pos'],r=5.,batch=x['batch'],loop=False,max_num_neighbors=32,flow='source_to_target')
    return x,torch.stack([r['spectrum'] for r in records]).to(device=device,dtype=dtype)

if __name__=='__main__': prepare()
