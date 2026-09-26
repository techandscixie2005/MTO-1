"""Snapshot the experiment, omitting checkpoint bytes and runtime caches."""
import pathlib,hashlib,json,gzip,tarfile
ROOT=pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925')
EXPORT=ROOT/'publication_export';EXPORT.mkdir(exist_ok=True)
parts=EXPORT/'dataset_parts';parts.mkdir(exist_ok=True)
source=ROOT/'data/dataset.npz';CHUNK=32*1024*1024;mapping=[];whole=hashlib.sha256()
with source.open('rb') as f:
    i=0
    while True:
        block=f.read(CHUNK)
        if not block:break
        whole.update(block);name=f'dataset.npz.part-{i:03d}.gz';p=parts/name
        with p.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as g:g.write(block)
        mapping.append(dict(path='data/dataset_parts/'+name,raw_bytes=len(block),raw_sha256=hashlib.sha256(block).hexdigest(),stored_bytes=p.stat().st_size,stored_sha256=hashlib.sha256(p.read_bytes()).hexdigest()));i+=1
storage=dict(original_path='data/dataset.npz',original_bytes=source.stat().st_size,original_sha256=whole.hexdigest(),format='concatenate decompressed blocks in listed order; byte-identical original NPZ',parts=mapping)
(EXPORT/'storage_map.json').write_text(json.dumps(storage,indent=2))
files=[];excluded=[]
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if rel.parts[0] in ('env','publication_export') or '__pycache__' in rel.parts:continue
    if p.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors'):
        excluded.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));continue
    if rel.as_posix()=='data/dataset.npz' or p.suffix in ('.whl','.pyc') or p.name.endswith(('.tar.gz','.tar')):continue
    files.append(p)
metadata=dict(source_root=str(ROOT),included_original_files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)],excluded_checkpoints=excluded,
    excluded_other=['env/','__pycache__/','wheel downloads (versions locked)','redundant transfer tar archives'],dataset_storage=storage)
(EXPORT/'export_inventory.json').write_text(json.dumps(metadata,indent=2))
with tarfile.open(ROOT/'experiment_no_checkpoints.tar.gz','w:gz') as t:
    for p in files:t.add(p,arcname=p.relative_to(ROOT))
    for p in parts.glob('*'):t.add(p,arcname='data/dataset_parts/'+p.name)
    t.add(EXPORT/'storage_map.json',arcname='publication/storage_map.json')
    t.add(EXPORT/'export_inventory.json',arcname='publication/export_inventory.json')
print(json.dumps(dict(original_files=len(files),checkpoints_excluded=len(excluded),dataset_sha256=whole.hexdigest(),packed_data_bytes=sum(x['stored_bytes'] for x in mapping),archive_bytes=(ROOT/'experiment_no_checkpoints.tar.gz').stat().st_size)))
