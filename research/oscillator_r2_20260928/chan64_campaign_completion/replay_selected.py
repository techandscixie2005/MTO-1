#!/usr/bin/env python3
"""Exactly two sealed, validation-only chan64 selected-checkpoint replays."""
import argparse, fcntl, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/inspur/MTO-1")
SRC = ROOT / "experiments/qm9s_chan64_20260928"
OUT = ROOT / "research/oscillator_r2_20260928/chan64_campaign_completion"
SEAL = OUT / "REPLAY_SEAL.json"
K = (2.0 / 3.0) / 27.211386245988
EXPECTED = {"G1": (75, 0.4052545406625945), "G3": (174, 0.3681007180162036)}
sys.path.insert(0, str(SRC))
from dataset import Data
from model_factory import build
sys.path.insert(0, str(ROOT / "research/oscillator_r2_20260928/architecture"))
from gpu_health import snapshot, eligible, unchanged

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def score(y, p, mask):
    yy, pp = y[mask], p[mask]
    sse = float(np.square(pp - yy).sum(dtype=np.float64))
    sst = float(np.square(yy - yy.mean()).sum(dtype=np.float64))
    return {"n": int(yy.size), "sse": sse, "sst": sst,
            "r2": float(1 - sse/sst), "rmse": float(np.sqrt(sse/yy.size)),
            "mae": float(np.abs(pp - yy).mean())}

def verify_seal():
    seal = json.loads(SEAL.read_text())
    assert seal["protocol_sha256"] == sha(OUT / "REPLAY_PROTOCOL.json")
    assert seal["evaluator_sha256"] == sha(__file__)
    for rel, digest in seal["source_hashes"].items():
        assert sha(ROOT / rel) == digest, rel
    for name in EXPECTED:
        assert seal["best_checkpoint_sha256"][name] == sha(SRC / "runs" / name / "best.pt")
    return seal

def verify_case(name):
    cfg = json.loads((SRC / "configs" / (name + ".json")).read_text())
    manifest = json.loads((SRC / "runs" / name / "run_manifest.json").read_text())
    marker = json.loads((SRC / "runs" / name / "FIT_COMPLETE.json").read_text())
    hist = [json.loads(s) for s in (SRC / "runs" / name / "history.jsonl").read_text().splitlines()]
    assert cfg == manifest["config"] and cfg["supervise_E"] and cfg["eta"] == 1
    assert marker["event"] == "FIT_COMPLETE" and marker["best_epoch"] == EXPECTED[name][0]
    assert len(hist) == marker["epochs"] and [r["epoch"] for r in hist] == list(range(1, len(hist)+1))
    assert hist[marker["best_epoch"]-1]["val"][0] == marker["best_val"]
    assert hist[marker["best_epoch"]-1]["val"][0] == min(r["val"][0] for r in hist)
    for rel, digest in manifest["fingerprint"].items():
        assert sha(SRC / rel) == digest, (name, rel)
    ck = torch.load(SRC / "runs" / name / "best.pt", map_location="cpu", weights_only=False)
    assert ck["config"] == cfg and ck["epoch"] == EXPECTED[name][0]
    assert ck["val"][0] == marker["best_val"]
    return cfg, manifest, marker, ck

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seal = verify_seal()
    checks = {}
    for name in EXPECTED:
        cfg, manifest, marker, ck = verify_case(name)
        checks[name] = {"epoch": ck["epoch"], "selection_loss": ck["val"][0],
                        "checkpoint_sha256": seal["best_checkpoint_sha256"][name]}
        del ck
    if args.preflight:
        print(json.dumps({"passed": True, "checks": checks}, indent=2))
        return
    review_path = OUT / "REPLAY_REVIEW.json"
    assert review_path.exists(), "independent pre-run review missing"
    review = json.loads(review_path.read_text())
    assert review["passed"] is True
    assert review["evaluator_sha256"] == sha(__file__)
    assert review["protocol_sha256"] == sha(OUT / "REPLAY_PROTOCOL.json")
    assert review["seal_sha256"] == sha(SEAL)
    assert os.getenv("CUDA_VISIBLE_DEVICES") in (None, ""), "cuda:4 physical index remapped"
    assert not (OUT / "REPLAY_RESULT.json").exists(), "single-use replay already completed"
    assert not (OUT / "REPLAY_STARTED.json").exists(), "single-use replay already started"
    for name in EXPECTED:
        assert not (OUT / "runtime" / (name + "_selected_native_val.npz")).exists(), "existing replay array"
    lock = Path("/tmp/mto_pouter_gpu_4.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = snapshot(4)
    assert eligible(before) and not before["apps"], before
    assert before["uuid"] == "GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184"
    device = torch.device("cuda:4")
    data = Data(device="cpu")
    idx = np.asarray(data.parts["val"])
    assert idx.shape == (6686,)
    with np.load(SRC / "data/raw_labels.npz", allow_pickle=False) as raw:
        assert np.array_equal(data.ids, raw["ids"])
        truth = np.asarray(raw["f"][idx], dtype=np.float64)
        truth_E = np.asarray(raw["E"][idx], dtype=np.float64)
        mask = np.asarray(raw["mask_f"][idx] & raw["mask_E"][idx] & raw["mask_A"][idx], dtype=bool)
    assert truth.shape == (6686, 10) and mask.all()
    for comparator in ("mto_eta0", "mto_eta01", "mto_eta1"):
        path = ROOT / "experiments/qm9s_eta_Ef_20260926/runs" / comparator / "val_predictions.npz"
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(idx, z["indices"]), (comparator, "indices")
            assert np.array_equal(data.ids[idx], z["ids"]), (comparator, "ids")
            assert np.array_equal(truth, z["f_true"]), (comparator, "raw f truth")
            assert np.array_equal(mask, z["mask_f_true"]), (comparator, "mask")
    started = {"event": "REPLAY_STARTED", "time": time.time(), "pid": os.getpid(),
               "seal_sha256": sha(SEAL), "review_sha256": sha(review_path),
               "gpu_before": before, "checkpoints": checks}
    fd = os.open(OUT / "REPLAY_STARTED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(started, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    results = {}
    for name in EXPECTED:
        cfg, manifest, marker, ck = verify_case(name)
        model = build(cfg, data.stats).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        energies, traces_old, traces_64 = [], [], []
        with torch.no_grad():
            for a in range(0, len(idx), 64):
                x, _ = data.batch(idx[a:a+64])
                x = {key: (val.to(device) if isinstance(val, torch.Tensor) else val)
                     for key, val in x.items()}
                E, A = model(**x)
                energies.append(E.cpu().numpy())
                traces_old.append(A.diagonal(dim1=-2, dim2=-1).sum(-1).cpu().numpy())
                traces_64.append(A.double().diagonal(dim1=-2, dim2=-1).sum(-1).cpu().numpy())
        en = np.concatenate(energies).astype(np.float64)
        tr_old = np.concatenate(traces_old).astype(np.float64)
        tr_64 = np.concatenate(traces_64).astype(np.float64)
        f_old = K * en * tr_old
        f_native = K * en * tr_64
        old_metric = score(truth, f_old, mask)
        native_metric = score(truth, f_native, mask)
        assert abs(old_metric["r2"] - EXPECTED[name][1]) <= 1e-6, (name, old_metric)
        assert abs(native_metric["r2"] - EXPECTED[name][1]) <= 1e-6, (name, native_metric)
        assert np.all(np.isfinite(f_native)) and np.all(np.isfinite(en))
        path = OUT / "runtime" / (name + "_selected_native_val.npz")
        assert not path.exists(), path
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, ids=data.ids[idx], indices=idx, f=f_native, f_true=truth,
                            mask_f_true=mask, E_pred=en, E_true=truth_E, trace_pred=tr_64)
        os.replace(temp, path)
        results[name] = {"checkpoint_sha256": sha(SRC / "runs" / name / "best.pt"),
                         "epoch": ck["epoch"], "selection_joint_loss": ck["val"][0],
                         "legacy_fp32_trace_metric": old_metric,
                         "native_fp64_trace_metric": native_metric,
                         "energy_raw_MAE_eV": float(np.abs(en - truth_E).mean()),
                         "prediction_path": str(path), "prediction_sha256": sha(path)}
        del model, ck
        torch.cuda.empty_cache()
    after = snapshot(4)
    assert unchanged(before, after), "GPU4 health counters changed during replay"
    record = {"classification": "exactly two selected-checkpoint validation-only native-f replays",
              "time": time.time(), "seal_sha256": sha(SEAL), "device": "cuda:4",
              "gpu_before": before, "gpu_after": after, "results": results}
    temp = OUT / "REPLAY_RESULT.json.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, OUT / "REPLAY_RESULT.json")
    print(json.dumps({"completed": True, "results": results}, indent=2))
if __name__ == "__main__":
    main()
