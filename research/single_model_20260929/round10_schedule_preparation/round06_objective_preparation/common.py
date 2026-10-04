"""Small provenance/I/O helpers; no target or model imports."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
DATA=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data')
SPLIT=CAMPAIGN/'dataset_audit_20260930'
PINS={'dataset.npz':'be8fadca203429575d70642b617730592693be40858dca7189c098a671596330',
      'raw_labels.npz':'621dc8723fb6dc96681851a26e4f52fcb7747a4778e57a249a75d22f3af4c632',
      'SPLIT_MANIFEST.json':'c8ce66ddb7209005b5feebfcddfc2bc30ee81d92fe63cce20f15c4abf3a07155',
      'INDEPENDENT_SPLIT_VERIFICATION.json':'395d415f854ed6948c4d7a11c0c1d7486193bb48f23bae76834a77148cc69ae2'}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def immutable_json(value,path):
    path=Path(path)
    if path.exists():
        assert read(path)==value,'Refuse changed existing record: '+str(path)
    else:atomic_json(value,path)
def require_cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('OMP_NUM_THREADS')=='2' and os.environ.get('MKL_NUM_THREADS')=='2'
