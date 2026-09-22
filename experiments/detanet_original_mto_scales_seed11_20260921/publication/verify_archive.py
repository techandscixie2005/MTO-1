"""Verify the published run with the Python standard library; optionally restore data."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest_stream(stream):
    h=hashlib.sha256();length=0
    for block in iter(lambda:stream.read(8*1024*1024),b''):
        h.update(block);length+=len(block)
    return h.hexdigest(),length

def file_digest(path):
    with path.open('rb') as f:return digest_stream(f)

def logical_digest(name,storage):
    # The cluster README was retained verbatim while the entry point was rewritten.
    if name=='README.md':return file_digest(ROOT/'publication/RUNNING_ON_CLUSTER.md')
    if name not in storage:return file_digest(ROOT/name)
    h=hashlib.sha256();length=0
    for part in storage[name]['parts']:
        with gzip.open(ROOT/part,'rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                h.update(block);length+=len(block)
    return h.hexdigest(),length

def restore(storage):
    for name,info in storage.items():
        target=ROOT/name
        if target.exists():
            assert file_digest(target)==(info['sha256'],info['bytes']),f'Existing cache differs: {target}'
            continue
        tmp=target.with_name(target.name+f'.restore.{os.getpid()}.tmp')
        assert not tmp.exists()
        with tmp.open('xb') as out:
            for part in info['parts']:
                with gzip.open(ROOT/part,'rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):out.write(block)
        assert file_digest(tmp)==(info['sha256'],info['bytes']),name
        os.replace(tmp,target)
        print(f'Restored {name} ({info["bytes"]} bytes)')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--restore-data',action='store_true',help='Reconstruct spectra.npy exactly, without overwriting a differing file')
    args=ap.parse_args()
    delivery=json.loads((ROOT/'publication/archive_manifest.json').read_text(encoding='utf-8'))
    storage=json.loads((ROOT/'publication/storage_map.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'publication/remote_snapshot_manifest.json').read_text(encoding='utf-8'))
    for name,info in delivery['files'].items():
        assert file_digest(ROOT/name)==(info['sha256'],info['bytes']),f'Published artifact differs: {name}'
    for name,info in source['files'].items():
        assert logical_digest(name,storage)==(info['sha256'],info['bytes']),f'Remote snapshot differs: {name}'
    fits=0;test_predictions=0
    for scale in ('1k','10k','full'):
        marker=json.loads((ROOT/f'reports/{scale}/STAGE_COMPLETE.json').read_text())
        assert marker['scale']==scale and marker['seed']==11 and marker['fits']==2
        for table in (marker['fingerprint'],marker['artifacts']):
            for name,h in table.items():assert file_digest(ROOT/name)[0]==h,(scale,name)
        frozen=json.loads((ROOT/f'data/frozen_{scale}.json').read_text())
        for name,h in frozen['sha256'].items():assert logical_digest('data/'+name,storage)[0]==h,(scale,name)
        for variant in ('detanet_original_uv','detanet_mto_planned'):
            run=ROOT/f'runs/{scale}/seed_11/{variant}'
            fit=json.loads((run/'FIT_COMPLETE.json').read_text())
            history=json.loads((run/'history.json').read_text())
            assert fit['scale']==scale and fit['seed']==11 and fit['variant']==variant
            assert fit['fingerprint']==marker['fingerprint']
            assert len(history)==fit['epochs'] and 1<=fit['best_epoch']<=fit['epochs']<=1000
            assert (run/'best.pt').stat().st_size>0 and (run/'last.pt').stat().st_size>0
            fits+=1
        test_predictions+=marker['evaluated_predictions']
    overall=json.loads((ROOT/'reports/WORKFLOW_COMPLETE.json').read_text())
    assert overall['fits']==fits==6 and overall['seed']==11
    for name,h in overall['artifacts'].items():assert file_digest(ROOT/name)[0]==h,name
    if args.restore_data:restore(storage)
    print(json.dumps(dict(status='verified',published_files=len(delivery['files']),
        remote_snapshot_files=len(source['files']),fits=fits,checkpoints=12,
        stages=['1k','10k','full'],evaluated_test_predictions=test_predictions,
        storage='lossless gzip chunks verified against original frozen data hash',data_restored=args.restore_data),indent=2))

if __name__=='__main__':main()
