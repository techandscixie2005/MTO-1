#!/usr/bin/env python3
"""Post-run fixed-ID validation diagnostic; only after frozen-head FIT_COMPLETE."""
import hashlib,json,pathlib,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=pathlib.Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    run=ROOT/"run"
    fit=json.loads((run/"FIT_COMPLETE.json").read_text())
    assert fit["event"]=="FIT_COMPLETE" and fit["epochs"]==20
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        ids=z["ids"].copy();indices=z["indices"].copy();y=z["f_true"].copy();eta0=z["f"].copy()
    values={}
    for label,path in (("initial",run/"initial_val.npz"),("selected",run/"best_native_f_val.npz"),
                       ("final20",run/"final20_val.npz")):
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z["ids"],ids) and np.array_equal(z["indices"],indices)
            assert np.array_equal(z["f_true"],y)
            values[label]={k:z[k].copy() for k in ("f_pred","base_f32",
                "base_f_native64","delta_f","signed_f")}
    fixed_id=14562
    matches=np.flatnonzero(ids==fixed_id);assert len(matches)==1
    i=int(matches[0])
    se0=np.square(eta0-y).sum(axis=1)
    sections={}
    for label,x in values.items():
        se=np.square(x["f_pred"]-y).sum(axis=1)
        net=float(se.sum()-se0.sum())
        change=se-se0
        order=np.argsort(-change,kind="stable")
        loc=int(np.flatnonzero(order==i)[0])+1
        sections[label]={"full_validation_sse":float(se.sum()),
            "full_validation_delta_sse_vs_eta0":net,
            "fixed_molecule_sse":float(se[i]),
            "fixed_molecule_delta_sse_vs_eta0":float(change[i]),
            "fixed_molecule_share_of_net_change":float(change[i]/net) if net else None,
            "fixed_molecule_deterioration_rank":loc,
            "fixed_molecule_per_state":[{"physical_state":j+1,
                "raw_f_true":float(y[i,j]),"eta0_f":float(eta0[i,j]),
                "head_f":float(x["f_pred"][i,j]),
                "base_f32":float(x["base_f32"][i,j]),
                "base_f_native64":float(x["base_f_native64"][i,j]),
                "delta_f":float(x["delta_f"][i,j]),
                "signed_f":float(x["signed_f"][i,j]),
                "eta0_squared_error":float((eta0[i,j]-y[i,j])**2),
                "head_squared_error":float((x["f_pred"][i,j]-y[i,j])**2)}
                for j in range(10)]}
    result={"classification":"Post-run validation-only diagnostic for molecule ID14562 fixed before frozen-head training from prior joint-residual replay",
        "created":time.time(),"fixed_molecule_id":fixed_id,"global_index":int(indices[i]),
        "control_full_validation_sse":float(se0.sum()),
        "control_fixed_molecule_sse":float(se0[i]),
        "selected_epoch":fit["best_epoch"],"predictors":sections,
        "fit_complete_sha256":sha(run/"FIT_COMPLETE.json"),
        "eta0_export_sha256":sha(SRC/"runs/mto_eta0/val_predictions.npz"),
        "prediction_hashes":{label:sha(path) for label,path in
            (("initial",run/"initial_val.npz"),("selected",run/"best_native_f_val.npz"),
             ("final20",run/"final20_val.npz"))},
        "limitations":["This fixed ID was chosen after inspecting an earlier validation diagnostic; it is illustrative, not a selection metric.",
            "The full 6,686-molecule pooled raw-f R2 remains primary. No test data were opened."]}
    out=ROOT/"WORSTCASE_14562_VALIDATION.json"
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"written":str(out),"selected_molecule_delta_sse":
        sections["selected"]["fixed_molecule_delta_sse_vs_eta0"],"final_molecule_delta_sse":
        sections["final20"]["fixed_molecule_delta_sse_vs_eta0"]}))
if __name__=="__main__":main()

