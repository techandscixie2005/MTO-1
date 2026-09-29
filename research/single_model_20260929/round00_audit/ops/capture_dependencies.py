#!/usr/bin/env python3
"""Copy only lightweight pinned source/config dependencies; never raw data."""
import hashlib
import json
from pathlib import Path
import shutil
from monitor import ROOT, atomic_json, stamp

cfg = json.loads((ROOT / 'round_config.json').read_text())
source = Path(cfg['source'])
paths = sorted((source / 'frozen_reference').rglob('*.py'))
paths += [source / name for name in ('dataset.py', 'configs/mto_eta0.json', 'data/normalization.json', 'data/hashes.json')]
records = []
for path in paths:
    relative = path.relative_to(source)
    # Rename the metadata directory to avoid confusing it with raw data archives.
    parts = list(relative.parts)
    if parts[0] == 'data':
        parts[0] = 'data_metadata'
    destination = ROOT / 'dependency_sources' / Path(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = path.read_bytes()
    content.decode('utf-8')
    if b'\x00' in content or len(content) > 1024 * 1024:
        raise ValueError('Unexpected dependency content: ' + str(path))
    destination.write_bytes(content)
    records.append({'server_source': str(path), 'archived_source': str(destination.relative_to(ROOT)),
                    'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
atomic_json(ROOT / 'ops/DEPENDENCY_SOURCE_RECEIPT.json', {'captured_at_utc': stamp(),
    'reason': 'Archive exact small source/config dependencies even where parent Git path differs; frozen run hashes must agree.',
    'raw_dataset_and_checkpoint_contents_copied': False, 'files': records})
print(json.dumps({'files': len(records), 'bytes': sum(r['bytes'] for r in records)}))
