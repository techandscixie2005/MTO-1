#!/usr/bin/env python3
"""Read-only completion audit for the finished retained_residual arm."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
ARCH=ROOT.parent/"architecture"
RUN=ARCH/"runs/retained_residual"
SRC=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()
def metric(y,p):
    d=p-y;sst=float(np.square(y-y.mean()).sum())
    return {"sse":float(np.square(d).sum()),"sst":sst,
            "r2":float(1-np.square(d).sum()/sst),
            "mae":float(np.abs(d).mean()),"rmse":float(np.sqrt(np.square(d).mean()))}
def main():
    fit=json.loads((RUN/"FIT_COMPLETE.json").read_text())
    assert fit["event"]=="FIT_COMPLETE" and fit["arm"]=="retained_residual"
    assert fit["epochs"]==20 and fit["steps"]==20*1881
    assert not (RUN/"FAILED.json").exists() and not (RUN/"INVALID.json").exists()
    pre=json.loads((ARCH/"PREFLIGHT.json").read_text())
    review=json.loads((ARCH/"EXECUTION_REVIEW.json").read_text())
    receipt=json.loads((RUN/"queue_launch_receipt.json").read_text())
    manifest=json.loads((RUN/"run_manifest.json").read_text())
    preparation=json.loads((ARCH/"PREPARATION.json").read_text())
    assert manifest["arm"]=="retained_residual" and manifest["physical_gpu"]==5
    assert manifest["gpu_uuid"]==receipt["gpu_uuid"]
    assert manifest["source_hashes"]==fit["source_hashes"]
    assert manifest["initial_checkpoint_sha256"]==preparation["arms"]["retained_residual"]["initial_checkpoint_sha256"]
    assert manifest["parameter_counts"]==preparation["arms"]["retained_residual"]["parameter_counts"]
    assert pre["passed"] and review["passed"]
    assert fit["source_hashes"]==pre["source_hashes"]==review["source_hashes"]
    assert receipt["preflight_sha256"]==sha(ARCH/"PREFLIGHT.json")
    assert receipt["review_sha256"]==sha(ARCH/"EXECUTION_REVIEW.json")
    for path,digest in fit["source_hashes"].items():
        assert sha(path)==digest,"Sealed source changed: "+path
    assert receipt["arm"]=="retained_residual" and receipt["gpu"]==5
    assert receipt["numerical_microcheck"]["passed"]
    after=json.loads((RUN/"gpu_health_after.json").read_text())
    baseline=receipt["health_before"]
    assert baseline["uuid"]==after["uuid"]==receipt["gpu_uuid"]
    assert baseline["ecc"]==after["ecc"]
    assert baseline["ecc"]["aggregate"]["dram_correctable"]==1
    assert baseline["ecc"]["aggregate"]["dram_uncorrectable"]==0
    assert baseline["ecc"]["remap"]["remapped_row_pending"]=="No"
    assert after["apps"]==[]
    history=[json.loads(line) for line in (RUN/"history.jsonl").read_text().splitlines()]
    assert len(history)==21 and [h["epoch"] for h in history]==list(range(21))
    for h in history:
        assert h["gpu_health_before"]["ecc"]==baseline["ecc"]
        assert h["gpu_health_after"]["ecc"]==baseline["ecc"]
        assert h["gpu_health_before"]["uuid"]==baseline["uuid"]
        assert h["gpu_health_after"]["uuid"]==baseline["uuid"]
    status=json.loads((RUN/"status.json").read_text())
    assert status["epoch"]==20 and status["validation"]==history[-1]["validation"]
    state=torch.load(RUN/"last.pt",map_location="cpu",weights_only=False)
    assert state["arm"]=="retained_residual" and state["hashes"]==fit["source_hashes"]
    assert state["state"]["epoch"]==21 and state["state"]["cursor"]==0
    assert state["state"]["history"]==history
    assert state["state"]["best_epoch"]==fit["best_epoch"]
    assert all(k in state for k in ("optimizer","rng_numpy","rng_torch",
                                    "rng_cuda","rng_python","rng_numpy_global"))
    assert fit["best_epoch"]["f"]==0 and fit["best_epoch"]["loss"]==0
    assert fit["best"]["f"]==min(h["validation"]["raw_f"]["sse"] for h in history)
    assert fit["best"]["loss"]==min(h["validation"]["common_objective"] for h in history)
    selected=RUN/"selections/f_epoch000.pt"
    selected_array=RUN/"selections/f_epoch000_val.npz"
    assert sha(selected)==sha(RUN/"best_native_f.pt")
    assert sha(selected_array)==sha(RUN/"best_native_f_val.npz")
    best=torch.load(selected,map_location="cpu",weights_only=False)
    assert best["arm"]=="retained_residual" and best["epoch"]==0
    assert best["source_hashes"]==fit["source_hashes"]
    assert best["validation"]==history[0]["validation"]
    with np.load(SRC/"data/dataset.npz",allow_pickle=False) as z:
        val_idx=z["val"].copy();val_ids=z["ids"][val_idx].copy()
    with np.load(SRC/"data/raw_labels.npz",allow_pickle=False) as z:
        true_f=z["f"][val_idx].astype(np.float64)
        true_e=z["E"][val_idx].astype(np.float64)
        derived=2/(3*27.211386245988)*true_e*np.trace(
            z["A"][val_idx].astype(np.float64),axis1=-2,axis2=-1)
        for k in ("mask_E","mask_A","mask_f"):assert z[k][val_idx].all()
    with np.load(RUN/"best_native_f_val.npz",allow_pickle=False) as z:
        assert np.array_equal(z["ids"],val_ids) and np.array_equal(z["indices"],val_idx)
        assert np.array_equal(z["f_true"],true_f) and np.array_equal(z["E_true"],true_e)
        assert np.allclose(z["f_derived_truth"],derived,rtol=0,atol=1e-12)
        f_pred=z["f_pred"].astype(np.float64)
        e_pred=z["E_pred"].astype(np.float64)
    calc=metric(true_f,f_pred)
    assert abs(calc["sse"]-history[0]["validation"]["raw_f"]["sse"])<1e-8
    assert abs(calc["r2"]-history[0]["validation"]["raw_f"]["r2"])<1e-10
    assert abs(float(np.abs(e_pred-true_e).mean())-
               history[0]["validation"]["energy"]["mae"])<1e-8
    with np.load(SRC/"runs/mto_eta0/val_predictions.npz",allow_pickle=False) as z:
        assert np.array_equal(z["ids"],val_ids) and np.array_equal(z["indices"],val_idx)
        assert np.array_equal(z["f_true"],true_f)
        eta0=metric(true_f,z["f"].astype(np.float64))
    epochs={}
    for label,idx in (("epoch0",0),("selected_best",fit["best_epoch"]["f"]),("final20",20)):
        f=history[idx]["validation"]["raw_f"];e=history[idx]["validation"]["energy"]
        epochs[label]={"epoch":idx,"raw_native_f_r2":f["r2"],"raw_native_f_sse":f["sse"],
                       "raw_native_f_mae":f["mae"],"raw_native_f_rmse":f["rmse"],
                       "energy_mae":e["mae"],
                       "common_objective":history[idx]["validation"]["common_objective"]}
    result={"classification":"Completed retained_residual arm audit only; no matched-control or promotion claim",
        "time":time.time(),"arm":"retained_residual","gpu":5,
        "fit_complete":True,"fit_reason":"fixed 20 epochs",
        "source_files_verified":len(fit["source_hashes"]),
        "preflight_sha256":sha(ARCH/"PREFLIGHT.json"),
        "execution_review_sha256":sha(ARCH/"EXECUTION_REVIEW.json"),
        "terminal_marker_sha256":sha(RUN/"FIT_COMPLETE.json"),
        "run_manifest_sha256":sha(RUN/"run_manifest.json"),
        "launch_receipt_sha256":sha(RUN/"queue_launch_receipt.json"),
        "history_sha256":sha(RUN/"history.jsonl"),
        "last_checkpoint_sha256":sha(RUN/"last.pt"),
        "selected_checkpoint_sha256":sha(selected),
        "selected_val_predictions_sha256":sha(selected_array),
        "baseline_eta0_val_predictions_sha256":sha(SRC/"runs/mto_eta0/val_predictions.npz"),
        "numerical_microcheck":receipt["numerical_microcheck"],
        "gpu5_ecc_unchanged_all_21_validations_and_after_exit":True,
        "gpu5_existing_corrected_dram_count":1,
        "raw_label_index_id_mask_and_metric_alignment":True,
        "selected_epoch":fit["best_epoch"]["f"],
        "epoch_metrics":epochs,
        "eta0_frozen_validation_reference":eta0,
        "selected_minus_eta0_r2":epochs["selected_best"]["raw_native_f_r2"]-eta0["r2"],
        "final_minus_epoch0_r2":epochs["final20"]["raw_native_f_r2"]-epochs["epoch0"]["raw_native_f_r2"],
        "f_r2_curve":[h["validation"]["raw_f"]["r2"] for h in history],
        "limitations":["The matched architecture original control has not completed; no within-round promotion claim is possible.",
                       "The selected best prediction array is independently checked; epoch20 predictions were not saved separately, so final metrics are cross-checked against committed history, status, and last checkpoint state.",
                       "The historical eta0 validation split has informed model design; scores are exploratory. No test prediction was opened."]}
    path=ROOT/"RETAINED_RESIDUAL_COMPLETION.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Completed retained-residual arm","",
        "The guarded GPU5 run finished its fixed 20 epochs with no ECC/remap change or recorded invalidation. Its selected raw-native-f checkpoint is epoch 0; later epochs did not improve the pooled validation SSE. This is one completed arm, so it is not a matched architecture comparison.","",
        "| Validation checkpoint | Epoch | Raw native-f R² | f MAE | f RMSE | E MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label,x in epochs.items():
        lines.append(f"| {label} | {x['epoch']} | {x['raw_native_f_r2']:.9f} | {x['raw_native_f_mae']:.6f} | {x['raw_native_f_rmse']:.6f} | {x['energy_mae']:.6f} |")
    lines += ["",f"Frozen eta0 reference R²: {eta0['r2']:.9f}. Selected-minus-eta0: {result['selected_minus_eta0_r2']:+.9f}, consistent with numerical reproduction. Final-minus-epoch0: {result['final_minus_epoch0_r2']:+.9f}.","",
        "The selected prediction NPZ exactly matches frozen validation IDs, indices and raw labels; recomputed pooled metrics match history. Epoch20 predictions were not separately saved. Its metrics match both the committed last-checkpoint history and final status.","",
        "The original matched control and two other architecture arms are still needed before any promotion or architecture claim. The historical test was not opened."]
    (ROOT/"RETAINED_RESIDUAL_COMPLETION.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"passed":True,"selected_epoch":fit["best_epoch"]["f"],
        "epoch0_r2":epochs["epoch0"]["raw_native_f_r2"],
        "best_r2":epochs["selected_best"]["raw_native_f_r2"],
        "final20_r2":epochs["final20"]["raw_native_f_r2"],
        "eta0_reference_r2":eta0["r2"],"report":str(path)}))
if __name__=="__main__":main()

