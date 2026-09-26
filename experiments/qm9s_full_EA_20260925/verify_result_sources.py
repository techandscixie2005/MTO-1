import pathlib,json,sys,hashlib
import numpy as np
ROOT=pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925')
source=pathlib.Path('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925')
sys.path.insert(0,str(source))
from load_qm9s import iter_molecules
with np.load(ROOT/'runs/mto_reference/test_predictions.npz') as f:p={k:f[k] for k in f.files}
lookup={int(mid):i for i,mid in enumerate(p['molecule_id'])};verified=0;outliers=[]
for r in iter_molecules(source):
    mid=r['molecule_id']
    if mid not in lookup:continue
    i=lookup[mid]
    assert np.array_equal(r['energy_eV'].astype('float32').astype('float64'),p['E_true'][i])
    assert np.array_equal(r['A_au2'].astype('float32').astype('float64'),p['A_true'][i])
    verified+=1
    if mid in [14633,57742,97471]:
        tr=np.trace(r['A_au2'],axis1=-2,axis2=-1)
        outliers.append(dict(id=mid,E7=float(r['energy_eV'][6]),trace_A7=float(tr[6]),printed_f7=float(r['oscillator_strength'][6]),
            derived_f7=float(2/3*r['energy_eV'][6]/27.211386245988*tr[6]),atoms=len(r['atomic_numbers'])))
assert verified==6686
import torch
res=json.loads((ROOT/'reports/results.json').read_text());code={}
for name in res['fits']:
    ck=torch.load(ROOT/'runs'/name/'best.pt',map_location='cpu',weights_only=False)
    bad=[k for k,v in ck['fingerprint'].items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    assert not bad,(name,bad)
    code[name]='PASS'
result=dict(status='PASS',all_test_labels_match_extracted_source=verified,code_and_data_manifest_match_checkpoint=code,outlier_source_checks=outliers)
(ROOT/'reports/source_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
