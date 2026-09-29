"""Freeze audited source/data/settings and training-only tail thresholds."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()

def files(cfg):
    src=Path(cfg['source'])
    result=[ROOT/'train.py',ROOT/'metrics.py',ROOT/'round_config.json',ROOT/'freeze.py',ROOT/'launch.py',
        ROOT/'PROTOCOL.md',ROOT/'ENVIRONMENT.json',ROOT/'environment_capture.py',
        ROOT/'gpu_smoke.py',ROOT/'GPU_SMOKE.json',ROOT/'SECONDARY_CALIBRATION_POLICY.md',
        ROOT/'architecture/model.py',ROOT/'architecture/preflight.py',ROOT/'architecture/PREFLIGHT.json',
        src/'dataset.py',src/'configs/mto_eta0.json',src/'data/normalization.json',src/'data/hashes.json',
        src/'data/splits.json',src/'data/dataset.npz',src/'data/raw_labels.npz',Path(cfg['source_checkpoint'])]
    result+=sorted((src/'frozen_reference').rglob('*.py'))
    return result

def current_hashes(cfg):
    return {str(p):sha(p) for p in files(cfg)}

def main():
    cfg=json.loads((ROOT/'round_config.json').read_text())
    src=Path(cfg['source'])
    pre=json.loads((ROOT/'architecture/PREFLIGHT.json').read_text())
    assert pre['passed']
    for name,expected in pre['source_hashes'].items():
        assert sha(ROOT/'architecture'/name)==expected
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']
    pinned=json.loads((src/'data/hashes.json').read_text())
    for name,expected in pinned.items():
        assert sha(src/'data'/name)==expected, name
    with np.load(src/'data/dataset.npz') as z:
        train=z['train'].copy();val=z['val'].copy();ids=z['ids'].copy()
    with np.load(src/'data/raw_labels.npz') as z:
        assert np.array_equal(ids,z['ids'])
        values=z['f'][train][z['mask_f'][train].astype(bool)]
    thresholds={f'q{int(100*q)}':float(np.quantile(values,q)) for q in cfg['bright_quantiles']}
    report={'config':cfg,'source_hashes':current_hashes(cfg),'tail_thresholds':thresholds,
            'train_count':len(train),'validation_count':len(val),
            'train_order_indices_sha256':hashlib.sha256(train.tobytes()).hexdigest(),
            'validation_indices_sha256':hashlib.sha256(val.tobytes()).hexdigest(),
            'validated_data_hashes':pinned,'architecture_preflight_passed':True}
    out=ROOT/'FROZEN_MANIFEST.json'
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if out.exists():
        assert json.loads(out.read_text())==report,'Refusing to change an existing frozen manifest'
    else:
        out.write_text(text)
    print(json.dumps({'manifest':str(out),'thresholds':thresholds,'source_files':len(report['source_hashes'])}))

if __name__=='__main__':
    main()
