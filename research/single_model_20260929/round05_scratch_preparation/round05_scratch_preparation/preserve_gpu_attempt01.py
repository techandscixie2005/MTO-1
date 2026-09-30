"""Preserve exact reviewed sources and failed zero-update attempt before repair."""
import shutil
from pathlib import Path
from common import ROOT,read,sha,immutable_json
out=ROOT/'preflight_history/gpu_attempt01';out.mkdir(parents=True,exist_ok=False)
review=read(ROOT/'TECHNICAL_SOURCE_REVIEW.json');records=[]
paths=list(review['source_hashes'])+[str(ROOT/'TECHNICAL_SOURCE_REVIEW.json')]
paths += [str(p) for p in (ROOT/'ops/gpu_preflight_attempt').iterdir() if p.is_file() and p.suffix in ('.json','.log','.xml')]
for i,name in enumerate(paths):
    path=Path(name);assert path.suffix in ('.py','.json','.xml','.md','.log')
    if name in review['source_hashes']:assert sha(path)==review['source_hashes'][name]
    dest=out/f'{i:03d}_{path.name}';shutil.copyfile(path,dest)
    assert sha(path)==sha(dest);records.append({'original':str(path),'snapshot':str(dest),'sha256':sha(path)})
assert not list((ROOT/'private_preflight').glob('*after_update1.pt'))
immutable_json({'records':records,'optimizer_updates_before_failure':0,
    'reason':'Full-forward GPU bitwise identity assertion failed before the first update; numerical diagnosis pending.'},out/'SNAPSHOT.json')
print(sha(out/'SNAPSHOT.json'))
