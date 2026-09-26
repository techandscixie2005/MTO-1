"""Build frozen grouped splits and train-only global scales from actual QM9S labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import concurrent.futures, hashlib, json, pathlib, sys, time
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925')
sys.path.insert(0,str(SOURCE))
from load_qm9s import iter_molecules

def identify(item):
    mid,z,pos=item
    RDLogger.DisableLog('rdApp.*')
    xyz=str(len(z))+'\nQM9S\n'+'\n'.join(f'{Chem.GetPeriodicTable().GetElementSymbol(int(a))} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}' for a,p in zip(z,pos))
    try:
        for factor in (1.2,1.15,1.1,1.25,1.3):
            mol=Chem.MolFromXYZBlock(xyz)
            rdDetermineBonds.DetermineConnectivity(mol,useVdw=True,covFactor=factor)
            key=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False,allHsExplicit=True)
            degrees=[a.GetDegree() for a in mol.GetAtoms()]
            ok=len(Chem.GetMolFrags(mol))==1 and all(d==1 if a in (1,9) else 1<=d<={6:4,7:4,8:3}[int(a)] for a,d in zip(z,degrees))
            if not ok:continue
            try:
                rdDetermineBonds.DetermineBondOrders(mol,charge=0,allowChargedFragments=True,embedChiral=False)
                Chem.SanitizeMol(mol)
                # Keep a conservative order/stereo-agnostic key, after verifying a valid valence assignment exists.
                return mid,key,True,'validated_covFactor='+str(factor)
            except Exception:continue
        return mid,key,False,'unresolved_connectivity_or_valence'
    except Exception as e:
        return mid,'unresolved:'+str(mid),False,type(e).__name__+':'+str(e)

def main():
    out=ROOT/'data';out.mkdir(exist_ok=True)
    records=list(iter_molecules(SOURCE));N=len(records);assert N==133727
    ids=np.array([r['molecule_id'] for r in records],dtype=np.int64)
    z=np.zeros((N,29),dtype=np.int64);pos=np.zeros((N,29,3),dtype=np.float32)
    E=np.stack([r['energy_eV'] for r in records]);A=np.stack([r['A_au2'] for r in records])
    assert E.shape==(N,10) and A.shape==(N,10,3,3)
    assert np.isfinite(E).all() and np.isfinite(A).all()
    assert all(np.array_equal(r['state_index'],np.arange(1,11)) and r['scalar_label_mask'].all() and r['vector_label_mask'].all() for r in records)
    edges=[];counts=[]
    for i,r in enumerate(records):
        n=len(r['atomic_numbers']);counts.append(n)
        z[i,:n]=r['atomic_numbers'];pos[i,:n]=r['positions_angstrom']
        d=np.linalg.norm(pos[i,:n,None,:]-pos[i,None,:n,:],axis=-1)
        assert np.all(d[np.triu_indices(n,1)]>1e-4)
        edges.append(np.array(np.where((d<5)&(d>0)),dtype=np.int16))
    edge_max=max(e.shape[1] for e in edges)
    edge=np.full((N,2,edge_max),-1,dtype=np.int16)
    for i,e in enumerate(edges):edge[i,:,:e.shape[1]]=e
    print(json.dumps(dict(event='loaded',N=N,max_edges=edge_max)),flush=True)
    identity_path=out/'identity_audit_v2.json'
    if identity_path.exists(): identities=json.loads(identity_path.read_text())
    else:
        items=[(int(ids[i]),r['atomic_numbers'],r['positions_angstrom']) for i,r in enumerate(records)]
        with concurrent.futures.ProcessPoolExecutor(24) as pool:
            identities=list(pool.map(identify,items,chunksize=256))
        identity_path.write_text(json.dumps(identities))
    groups={};bad=set()
    for i,(mid,key,ok,reason) in enumerate(identities):
        assert mid==ids[i]
        groups.setdefault(key,[]).append(i)
        if not ok:bad.add(key)
    rng=np.random.default_rng(20260925)
    keys=sorted(k for k in groups if k not in bad);rng.shuffle(keys)
    target=int(round(N*.05));parts={'test':[],'val':[],'train':[]}
    for part in ('test','val'):
        while keys and len(parts[part])<target:
            parts[part].extend(groups[keys.pop()])
    for key in keys+sorted(bad):parts['train'].extend(groups[key])
    for k in parts:parts[k]=sorted(parts[k])
    assert sorted(sum(parts.values(),[]))==list(range(N))
    owners={}
    for part,idx in parts.items():
        for i in idx:
            key=identities[i][1]
            assert key not in owners or owners[key]==part
            owners[key]=part
    tr=np.array(parts['train']);means=E[tr].mean(0)
    # User's exact scales: one pooled residual variance and one global matrix RMS.
    se2=float(np.mean((E[tr]-means[None,:])**2))
    sa2=float(np.mean(np.sum(A[tr]**2,axis=(-2,-1))))
    assert se2>0 and sa2>0
    stats=dict(sE2=se2,sA2=sa2,E_state_mean=means.tolist(),n_ref=float(np.median(np.array(counts)[tr])),
        source='train_only',definition_E='mean_na((E_na-mean_n_train(E_na))^2)',definition_A='mean_na(sum_ij(A_naij^2))')
    split=dict(seed=20260925,requested_ratio=[.9,.05,.05],indices=parts,ids={k:ids[v].tolist() for k,v in parts.items()},
        counts={k:len(v) for k,v in parts.items()},identity_groups=len(groups),duplicate_group_count=sum(len(v)>1 for v in groups.values()),
        unresolved_groups_train_only=len(bad),unresolved_records_train_only=sum(len(groups[k]) for k in bad),
        grouping='RDKit XYZ connectivity with successful neutral bond-order/valence check; covFactor tried 1.2,1.15,1.1,1.25,1.3; conservative bond-order/stereo-agnostic explicit-H grouping')
    np.savez(out/'dataset.npz',ids=ids,z=z,pos=pos,E=E.astype('float32'),A=A.astype('float32'),edge=edge,
             **{k:np.array(v,dtype=np.int64) for k,v in parts.items()})
    (out/'splits.json').write_text(json.dumps(split,indent=2));(out/'normalization.json').write_text(json.dumps(stats,indent=2))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'dataset.npz',out/'splits.json',out/'normalization.json',identity_path)}
    (out/'hashes.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(event='PREPARE_COMPLETE',counts=split['counts'],groups=len(groups),unresolved=split['unresolved_records_train_only'],stats=stats)),flush=True)
if __name__=='__main__':main()
