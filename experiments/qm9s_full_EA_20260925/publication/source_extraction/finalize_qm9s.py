"""Verify all extracted shards, reconcile equivalent convergence messages, seal outputs."""
import collections
import gzip
import hashlib
import json
import pathlib
import shutil
import sys
import numpy as np

root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))
source=pathlib.Path(manifest['source_directory'])
warnings=collections.Counter(); convergence=0; checked=0; changed=0
anomalies=[]; ids=[]
ready_scalar=ready_vector=0
charge_spin=collections.Counter(); symmetries=collections.Counter(); atom_counts=collections.Counter()
for shard in manifest['shards']:
    jp=root/'records'/(shard['name']+'.jsonl.gz')
    with gzip.open(jp,'rt',encoding='utf-8') as f: records=[json.loads(line) for line in f]
    npz=root/'arrays'/(shard['name']+'.npz')
    with np.load(npz,allow_pickle=False) as f: arrays={key:f[key] for key in f.files}
    dirty=False
    for i,r in enumerate(records):
        if 'response_convergence_marker_absent' in r['warnings']:
            raw=(source/r['source_file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==r['source_sha256'], 'Source changed'
            if b'Convergence on expansion vectors.' in raw:
                r['response_convergence_reported']=True
                r['warnings'].remove('response_convergence_marker_absent')
                arrays['response_convergence_reported'][i]=True
                dirty=True;changed+=1
        ids.append(r['molecule_id'])
        charge_spin[str((r.get('charge'),r.get('multiplicity')))]+=1
        atom_counts[len(r['atomic_numbers'])]+=1
        symmetries.update(s['symmetry_printed'] for s in r['states'])
        success=r.get('normal_termination',False) and r.get('geometry_present',False) and r.get('unambiguous_single_calculation',False) and bool(r['states'])
        ready_scalar+=success and all(s['scalar_fields_present'] and s['unique_root_index'] for s in r['states'])
        ready_vector+=success and all(s['vector_fields_present'] and s['unique_root_index'] for s in r['states'])
        assert arrays['molecule_id'][i]==r['molecule_id']
        n=len(r['atomic_numbers']);k=len(r['states'])
        assert int(arrays['atom_mask'][i].sum())==n
        assert int(arrays['state_mask'][i].sum())==k
        if n:
            assert arrays['atomic_numbers'][i,:n].tolist()==r['atomic_numbers']
            assert arrays['positions_angstrom'][i,:n].tolist()==r['positions_angstrom']
        for j,s in enumerate(r['states']):
            for key in ('state_index','energy_eV','wavelength_nm','oscillator_strength','transition_dipole_au','transition_dipole_table_au','velocity_dipole_au','magnetic_dipole_au','A_au2'):
                if key in s: assert arrays[key][i,j].tolist()==s[key], (r['molecule_id'],j,key)
            checked+=1
        warnings.update(r['warnings']);convergence+=r.get('response_convergence_reported',False)
        if r['warnings']: anomalies.append({key:r[key] for key in ('molecule_id','source_file','warnings')})
    valid=arrays['vector_label_mask']
    mu=arrays['transition_dipole_au'][valid]
    np.testing.assert_allclose(arrays['A_au2'][valid],mu[:,:,None]*mu[:,None,:],rtol=1e-14,atol=1e-14)
    assert np.all(np.abs(mu-arrays['transition_dipole_table_au'][valid])<=5.1e-5)
    if dirty:
        with gzip.open(jp,'wt',encoding='utf-8',compresslevel=3) as f:
            for r in records: f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
        np.savez_compressed(npz,**arrays)
    print(json.dumps({'verified_shard':shard['name'],'states_checked':checked,'convergence_messages_reconciled':changed}),flush=True)
assert len(ids)==manifest['source_log_count']
assert len(ids)==len(set(ids))
assert checked==manifest['scalar_label_count']
manifest['response_convergence_reported_count']=convergence
manifest['warnings']=dict(warnings)
manifest['complete_successful_scalar_records']=ready_scalar
manifest['complete_successful_vector_records']=ready_vector
manifest['charge_multiplicity_distribution']=dict(charge_spin)
manifest['state_symmetry_distribution']=dict(symmetries)
manifest['atom_count_distribution']=dict(atom_counts)
(root/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (root/'anomalies.jsonl').open('w',encoding='utf-8') as f:
    for a in anomalies:f.write(json.dumps(a)+'\n')
audit=root/'dataset_audit.md'
text=audit.read_text(encoding='utf-8')
import re
text=re.sub(r'明确报告响应收敛：[\d,]+',f'明确报告响应收敛：{convergence:,}',text)
text=re.sub(r'异常统计：`[^`]*`',f'异常统计：`{dict(warnings)}`',text)
text+='\n\n全部 NumPy 数组与 JSONL 逐分子、逐态核对一致；A 张量与偶极外积逐态核对通过。所有源文件 SHA-256 保存在 source_manifest.jsonl。\n'
text+=f'\n正常结束、几何完整、单次计算无歧义且全部标量标签齐全的记录：{ready_scalar:,}；全部向量标签齐全的记录：{ready_vector:,}。\n'
audit.write_text(text,encoding='utf-8')
for name in ('extract_qm9s.py','finalize_qm9s.py'):
    origin=pathlib.Path(__file__).parent/name
    if origin.resolve() != (root/name).resolve(): shutil.copy2(origin,root/name)
report=dict(molecule_count=len(ids),states_checked=checked,jsonl_npz_equal=True,tensor_outer_products_equal=True,convergence_messages_reconciled=changed)
(root/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
