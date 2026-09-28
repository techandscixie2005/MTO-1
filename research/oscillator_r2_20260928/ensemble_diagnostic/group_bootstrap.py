#!/usr/bin/env python3
"""Connectivity-group paired-bootstrap sensitivity of frozen ensemble test result."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path("/home/inspur/MTO-1")
OUT=ROOT/"research/oscillator_r2_20260928/ensemble_diagnostic"
SRC=ROOT/"experiments/qm9s_eta_Ef_20260926"
ID=ROOT/"experiments/qm9s_full_EA_20260925/data/identity_audit_v2.json"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads(ID.read_text())
with np.load(SRC/"runs/mto_eta0/test_predictions.npz",allow_pickle=False) as z:
    ids=z["ids"];indices=z["indices"];truth=z["f_true"].astype(np.float64);p0=z["f"].astype(np.float64)
with np.load(SRC/"runs/mto_eta01/test_predictions.npz",allow_pickle=False) as z:p1=z["f"].astype(np.float64)
with np.load(SRC/"runs/mto_eta1/test_predictions.npz",allow_pickle=False) as z:p2=z["f"].astype(np.float64)
assert len(ids)==6686 and truth.shape==(6686,10)
for id_,index in zip(ids,indices):assert int(identity[int(index)][0])==int(id_)
ensemble=(p0+p1+p2)/3
groups=defaultdict(list)
for row,index in enumerate(indices):groups[identity[int(index)][1]].append(row)
keys=sorted(groups)
n=[];sy=[];sy2=[];sse0=[];sse1=[]
for key in keys:
    idx=groups[key];y=truth[idx];b=p0[idx];e=ensemble[idx]
    n.append(y.size);sy.append(y.sum());sy2.append(np.square(y).sum())
    sse0.append(np.square(b-y).sum());sse1.append(np.square(e-y).sum())
n=np.asarray(n);sy=np.asarray(sy);sy2=np.asarray(sy2)
sse0=np.asarray(sse0);sse1=np.asarray(sse1)
assert n.sum()==66860
rng=np.random.default_rng(20260928)
deltas=[]
for _ in range(2000):
    ix=rng.integers(0,len(keys),len(keys))
    ss=sy2[ix].sum()-sy[ix].sum()**2/n[ix].sum()
    deltas.append(float((sse0[ix].sum()-sse1[ix].sum())/ss))
deltas=np.asarray(deltas)
test=json.loads((OUT/"EXPLORATORY_TEST.json").read_text())
assert abs(test["delta_r2"]-(sse0.sum()-sse1.sum())/(sy2.sum()-sy.sum()**2/n.sum()))<1e-12
report={"classification":"Sensitivity of the already frozen exploratory test; no new model or ensemble selection.",
    "source_test_report_sha256":sha(OUT/"EXPLORATORY_TEST.json"),
    "identity_audit_sha256":sha(ID),
    "grouping":"Exact conservative RDKit XYZ connectivity key used in original split creation; identity_audit_v2.json second tuple field.",
    "molecules":len(ids),"connectivity_groups":len(keys),
    "groups_with_multiple_test_molecules":sum(len(groups[k])>1 for k in keys),
    "largest_group_molecules":max(len(groups[k]) for k in keys),
    "paired_bootstrap":{"resampling_unit":"connectivity group, retaining all molecules and ten states",
        "replicates":2000,"seed":20260928,
        "delta_r2_2p5_50_97p5":np.quantile(deltas,[.025,.5,.975]).tolist(),
        "positive_fraction":float((deltas>0).mean())},
    "original_molecule_bootstrap":test["bootstrap"],
    "conclusion_sign_unchanged":bool(np.quantile(deltas,.025)>0)}
p=OUT/"GROUP_BOOTSTRAP_SENSITIVITY.json"
assert not p.exists()
p.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))

