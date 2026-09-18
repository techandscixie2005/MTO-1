"""Reuse audited immutable geometry/spectra and restore exactly the historical 1k split."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data/run01/sczc698/xxy/MTO_scale_20260914')
for name in ('data','reference','results','jobs','logs'):
    (ROOT/name).mkdir(exist_ok=True)
target=ROOT/'data/qm9s_full.npz'
if not target.exists():
    target.symlink_to(OLD/'data/qm9s_full.npz')
for name in ('manifest_10k.json','manifest_full.json','splits_10k.json','splits_full.json','audit.json'):
    shutil.copy2(OLD/'data'/name,ROOT/'data'/name)
shutil.copy2(OLD/'reference/splits.json',ROOT/'data/splits_1k.json')
splits=json.loads((ROOT/'data/splits_1k.json').read_text())
with np.load(target,allow_pickle=False) as a:
    ids=a['ids']; masks={name:np.isin(ids,selected) for name,selected in splits.items()}
    for index,name in enumerate(('train','validation','test')):
        assert masks[name].sum()==len(splits[name])
        assert (a['split'][masks[name]]==index).all()
        assert a['subset_10k'][masks[name]].all()
    manifest=json.loads((ROOT/'data/manifest_10k.json').read_text())
    manifest.update(scale='1k',count=1000,splits={k:len(v) for k,v in splits.items()},
        train_spectrum_rms=float(np.sqrt(np.mean(a['spectra'][masks['train']].astype(np.float64)**2))),
        train_median_atoms=float(np.median(np.diff(a['offsets'])[masks['train']])),
        note='Exact historical identities and split; frozen new architecture, retrained from scratch.')
    (ROOT/'data/manifest_1k.json').write_text(json.dumps(manifest,indent=2))
actual=hashlib.sha256(target.read_bytes()).hexdigest()
assert actual==manifest['sample_sha256']
result={'passed':True,'database_sha256':actual,'nested_splits_verified':True,
        'scales':{s:json.loads((ROOT/'data'/f'manifest_{s}.json').read_text())['splits'] for s in ('1k','10k','full')}}
(ROOT/'results/data_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
