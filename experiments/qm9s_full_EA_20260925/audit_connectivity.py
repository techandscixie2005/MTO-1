import json,sys,numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from collections import Counter
from prepare_data import ROOT,SOURCE
sys.path.insert(0,str(SOURCE))
from load_qm9s import iter_molecules
bad={r[0] for r in json.loads((ROOT/'data/identity_audit.json').read_text()) if not r[2]}
c=Counter();examples=[]
for r in iter_molecules(SOURCE):
    if r['molecule_id'] not in bad:continue
    z=r['atomic_numbers'];pos=r['positions_angstrom'];xyz=str(len(z))+'\nx\n'+'\n'.join(f'{Chem.GetPeriodicTable().GetElementSymbol(int(a))} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}' for a,p in zip(z,pos))
    for name,kwargs in [('default',{}),('vdw12',dict(useVdw=True,covFactor=1.2)),('vdw13',dict(useVdw=True,covFactor=1.3))]:
        m=Chem.MolFromXYZBlock(xyz);rdDetermineBonds.DetermineConnectivity(m,**kwargs)
        ds=[a.GetDegree() for a in m.GetAtoms()]
        ok=len(Chem.GetMolFrags(m))==1 and all(d==1 if a in (1,9) else 1<=d<={6:4,7:4,8:3}[int(a)] for a,d in zip(z,ds))
        c[name+'_ok' if ok else name+'_bad']+=1
        if not ok and len(examples)<8 and name=='default':examples.append(dict(id=r['molecule_id'],smiles=Chem.MolToSmiles(m),degrees=list(zip(map(int,z),ds))))
print(dict(c));print(examples)
