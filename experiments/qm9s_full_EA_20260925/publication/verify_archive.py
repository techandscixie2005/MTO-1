"""Verify publication bytes and optionally reconstruct the exact training NPZ."""
import argparse,gzip,hashlib,json,os,pathlib,tempfile,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def restore():
    m=json.loads((ROOT/'publication/storage_map.json').read_text());dest=ROOT/m['original_path']
    if dest.exists():
        assert dest.stat().st_size==m['original_bytes'] and sha(dest)==m['original_sha256'],'Existing dataset differs; refusing overwrite'
        print('Existing dataset verified');return
    fd,tmp=tempfile.mkstemp(prefix='dataset-restore-',dir=dest.parent);tmp=pathlib.Path(tmp);h=hashlib.sha256();total=0
    try:
        with os.fdopen(fd,'wb') as out:
            for item in m['parts']:
                p=ROOT/item['path'];assert sha(p)==item['stored_sha256']
                block=gzip.decompress(p.read_bytes());assert len(block)==item['raw_bytes'] and hashlib.sha256(block).hexdigest()==item['raw_sha256']
                out.write(block);h.update(block);total+=len(block)
        assert total==m['original_bytes'] and h.hexdigest()==m['original_sha256']
        with zipfile.ZipFile(tmp) as z:
            assert z.testzip() is None
            assert set(z.namelist())=={k+'.npy' for k in ['ids','z','pos','E','A','edge','train','val','test']}
        os.replace(tmp,dest);print('Restored exact dataset:',total,'bytes',h.hexdigest())
    finally:
        if tmp.exists():tmp.unlink()
def metrics():
    import numpy as np
    result=json.loads((ROOT/'reports/results.json').read_text());scales=json.loads((ROOT/'data/normalization.json').read_text());splits=json.loads((ROOT/'data/splits.json').read_text())
    reference=None
    for name,m in result['test'].items():
        with np.load(ROOT/'runs'/name/'test_predictions.npz',allow_pickle=False) as p:
            assert np.array_equal(p['molecule_id'],splits['ids']['test'])
            truth=(p['E_true'],p['A_true'])
            if reference is None:reference=truth
            else:assert all(np.array_equal(a,b) for a,b in zip(reference,truth))
            de=p['E_pred'].astype('float64')-truth[0];da=p['A_pred'].astype('float64')-truth[1]
            le=float(np.mean(de**2)/scales['sE2']);la=float(np.mean(np.sum(da**2,axis=(-2,-1)))/scales['sA2'])
            assert abs(le-m['LE'])<1e-10 and abs(la-m['LA'])<1e-10 and abs(le+la-m['L'])<1e-10
            assert abs(np.mean(abs(de))-m['E_MAE_eV'])<1e-10
            assert abs(np.sqrt(np.mean(np.sum(da**2,axis=(-2,-1))))-m['A_Frobenius_RMSE_au2'])<1e-10
    print('Verified six test prediction metrics and common labels')
def main():
    p=argparse.ArgumentParser();p.add_argument('--restore-data',action='store_true');p.add_argument('--metrics',action='store_true');args=p.parse_args()
    manifest=json.loads((ROOT/'publication/archive_manifest.json').read_text())
    for item in manifest['files']:
        path=ROOT/item['path'];assert path.is_file(),item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
    for path in ROOT.rglob('*'):
        assert not (path.is_file() and path.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors')),str(path)
    inventory=json.loads((ROOT/'publication/export_inventory.json').read_text())
    for item in inventory['included_original_files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    print('PASS:',len(manifest['files']),'manifest files;',len(inventory['included_original_files']),'unchanged server files; no checkpoints')
    if args.restore_data:restore()
    if args.metrics:metrics()
if __name__=='__main__':main()
