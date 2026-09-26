"""Verify the frozen dataset, then add untouched printed f and explicit masks."""
import hashlib, json, pathlib, shutil, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'qm9s_full_EA_20260925'
SOURCE=pathlib.Path('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    hashes=json.loads((ROOT/'frozen_reference/data/hashes.json').read_text())
    for name,expected in hashes.items():
        assert sha(OLD/'data'/name)==expected, name
        shutil.copy2(OLD/'data'/name,ROOT/'data'/name)
    shutil.copy2(ROOT/'frozen_reference/data/hashes.json',ROOT/'data/hashes.json')
    manifest=json.loads((ROOT/'reports/source_downloads.json').read_text())
    for row in manifest['files']:
        rel=row['path']
        p=ROOT/'frozen_reference'/rel.split('qm9s_full_EA_20260925/')[1] if rel.startswith('experiments/') else ROOT/'official_detanet'/rel
        assert sha(p)==row['sha256'], str(p)
    with np.load(ROOT/'data/dataset.npz') as d:
        ids=d['ids'];index={int(mid):i for i,mid in enumerate(ids)}
        E=d['E'];A=d['A'];z=d['z'];pos=d['pos']
        parts={k:d[k] for k in ('train','val','test')}
    n=len(ids);assert n==len(index)==133727
    ef=np.full((n,10),np.nan);af=np.full((n,10,3,3),np.nan);f=np.full((n,10),np.nan)
    me=np.zeros((n,10),bool);ma=me.copy();seen=set();extra=0
    input_hashes={}
    for path in sorted((SOURCE/'arrays').glob('part-*.npz')):
        input_hashes[path.name]=sha(path)
        with np.load(path) as archive: raw={k:archive[k] for k in archive.files}
        for j,mid in enumerate(raw['molecule_id']):
            if int(mid) not in index:extra+=1;continue
            assert int(mid) not in seen;seen.add(int(mid));i=index[int(mid)]
            sm=raw['state_mask'][j];am=raw['atom_mask'][j]
            assert np.array_equal(raw['state_index'][j,sm],np.arange(1,11))
            ef[i]=raw['energy_eV'][j,sm];af[i]=raw['A_au2'][j,sm];f[i]=raw['oscillator_strength'][j,sm]
            me[i]=raw['scalar_label_mask'][j,sm]&np.isfinite(ef[i])&np.isfinite(f[i])
            ma[i]=raw['vector_label_mask'][j,sm]&np.isfinite(af[i]).all((1,2))
            assert np.array_equal(E[i],ef[i].astype('float32'))
            assert np.array_equal(A[i],af[i].astype('float32'))
            assert np.array_equal(z[i,z[i]!=0],raw['atomic_numbers'][j,am])
            assert np.array_equal(pos[i,z[i]!=0],raw['positions_angstrom'][j,am].astype('float32'))
    assert len(seen)==n
    # The frozen population has complete labels; no new filtering is permitted.
    assert me.all() and ma.all()
    np.savez(ROOT/'data/raw_labels.npz',ids=ids,E=ef,A=af,f=f,mask_E=me,mask_A=ma,mask_f=me)
    derived=(2/3)*(ef/27.211386245988)*np.trace(af,axis1=-2,axis2=-1)
    delta=derived-f
    audit=dict(molecules=n,states=10,counts={k:len(v) for k,v in parts.items()},
        all_masks_valid=True,excluded_only_because_not_in_frozen_ids=extra,zero_printed_f=int((f==0).sum()),
        raw_f_vs_A_derived=dict(mae=float(np.abs(delta).mean()),rmse=float(np.sqrt(np.mean(delta**2))),
            max_abs=float(np.max(np.abs(delta))),quantiles_abs=np.quantile(np.abs(delta),[.5,.9,.99,1]).tolist(),
            printed_f_on_1e_minus4_grid_max_residual=float(np.max(np.abs(f*1e4-np.round(f*1e4)))),
            abs_difference_gt_5e_minus5=int((np.abs(delta)>5e-5).sum()),
            zero_printed_but_positive_derived=int(((f==0)&(derived>0)).sum())),
        source_arrays_sha256=input_hashes,baseline_hashes=hashes,raw_labels_sha256=sha(ROOT/'data/raw_labels.npz'),
        target_policy='Original oscillator_strength is retained for DetaNet and common f/spectrum truth. No clipping or outlier removal.',
        evaluation_status='Label provenance audit only; no current model test evaluation.')
    (ROOT/'reports/data_audit.json').write_text(json.dumps(audit,indent=2))
    cfg=json.loads((ROOT/'frozen_reference/configs/mto_reference.json').read_text())
    for eta,name,gpu in [(0.,'mto_eta0',1),(.1,'mto_eta01',2),(1.,'mto_eta1',4)]:
        c=dict(cfg,eta=eta,name=name,gpu=gpu)
        (ROOT/'configs'/f'{name}.json').write_text(json.dumps(c,indent=2))
    print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
