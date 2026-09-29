"""Read-only grouping/initialization/cost audit; no target labels or model loaded."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
IDENTITY=Path('/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/data/identity_audit_v2.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def grouped_split(train,identity,seed=20260930):
    groups={};bad=set()
    for i in train:
        row=identity[int(i)];groups.setdefault(row[1],[]).append(int(i))
        if not row[2]:bad.add(row[1])
    keys=sorted(set(groups)-bad);np.random.default_rng(seed).shuffle(keys)
    held=[];selected=[];target=len(train)//5
    for key in keys:
        if len(held)>=target:break
        held.extend(groups[key]);selected.append(key)
    selected=set(selected);heldset=set(held)
    fit=np.array([i for i in train if int(i) not in heldset],dtype=np.int64)
    hold=np.array([i for i in train if int(i) in heldset],dtype=np.int64)
    assert not selected.intersection(bad) and len(fit)+len(hold)==len(train)
    assert not np.intersect1d(fit,hold).size
    assert all(all(i in heldset for i in group) or all(i not in heldset for i in group) for group in groups.values())
    metadata={'train_groups':len(groups),'duplicate_train_groups':sum(len(x)>1 for x in groups.values()),
        'duplicate_train_records':sum(len(x) for x in groups.values() if len(x)>1),
        'unresolved_train_groups':len(bad),'unresolved_train_records':sum(len(groups[k]) for k in bad),
        'heldout_groups':len(selected),'target_heldout_rows':target,'fit_molecules':len(fit),'heldout_molecules':len(hold)}
    return fit,hold,metadata

def main():
    assert sha(IDENTITY)=='9d384425a90dd88fbc68f8b609a303872910bb21c69e0d0a19bcccaee49cb3c4'
    identity=json.loads(IDENTITY.read_text())
    with np.load(SOURCE/'data/dataset.npz',allow_pickle=False) as z:train=z['train'].copy();ids=z['ids'].copy()
    assert all(int(ids[i])==int(row[0]) for i,row in enumerate(identity))
    fit,hold,metadata=grouped_split(train,identity)
    assert len(fit)==96284 and len(hold)==24071
    history=[json.loads(x) for x in (SOURCE/'runs/mto_eta0/history.jsonl').read_text().splitlines()]
    paths=[IDENTITY,SOURCE/'data/splits.json',SOURCE/'data/normalization.json',SOURCE/'model_factory.py',
        SOURCE/'trainer.py',SOURCE/'preflight.py',SOURCE/'frozen_reference/models_ea.py',
        SOURCE/'reports/initialization.json',SOURCE/'runs/mto_eta0/history.jsonl',Path(__file__)]
    record={'passed':True,'split_seed':20260930,'policy':'sorted resolved originalTRAIN group keys, RNG shuffle, whole-group prefix until at least24071; unresolved groups fit-only; originalTRAIN order preserved in each part',
        'metadata':metadata,'fit_indices_sha256':hashlib.sha256(fit.tobytes()).hexdigest(),
        'heldout_indices_sha256':hashlib.sha256(hold.tobytes()).hexdigest(),
        'no_groups_broken_or_labels_excluded':True,'split_arrays_written':False,'target_labels_loaded':False,
        'source_initialization':'fresh torch-seeded model_factory.build; no trained checkpoint transfer',
        'existing_initial_model_target_clean_for_new_holdout':False,
        'required_subset_only_target_statistics':['E_state_mean','sE2','sA2','any future f moments'],
        'normalization_definitions':{'sE2':'mean((E_fit-E_state_mean_fit)^2)','sA2':'mean(sum_ij(A_fit_ij^2))','n_ref':'median atom count of fit subset (geometry only)'},
        'fixed_source_epoch':33,'heldout_or_outerval_or_test_source_selection_allowed':False,
        'cost':{'historical_full_train_mean_epoch_seconds':float(np.mean([x['seconds'] for x in history])),
            'historical_first33_mean_epoch_seconds':float(np.mean([x['seconds'] for x in history[:33]])),
            'rough_80pct_33epoch_minutes':float(np.mean([x['seconds'] for x in history])*33*.8/60),
            'planning_range_minutes':[60,75],'not_a_runtime_guarantee':True},
        'no_model_inference_fit_or_GPU':True,'source_hashes':{str(p):sha(p) for p in paths}}
    (ROOT/'FEASIBILITY.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'passed':True,'counts':metadata,'rough_minutes':record['cost']['rough_80pct_33epoch_minutes']}))
if __name__=='__main__':main()
