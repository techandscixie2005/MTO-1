"""VAL-ONLY: is the trace loss badly conditioned because it supervises the SQUARE?

A = mu mu^T exactly, so |mu| = sqrt(tr A) and f = (2/3) E |mu|^2.
Current Ls penalizes (|mu_p|^2 - |mu_t|^2)^2. Tests whether the magnitude and
direction of mu are in fact predicted better than the squared quantity suggests.
"""
import json, sys, pathlib
import numpy as np, torch

ROOT = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927')
sys.path.insert(0, str(ROOT))
from dataset import Data
from model_factory import build
from trainer import setup

GROUPS = ('G1', 'G3')
K = (2/3)/27.211386245988


def r2(p, t):
    d = np.asarray(p, np.float64) - np.asarray(t, np.float64)
    sst = np.square(np.asarray(t, np.float64) - np.asarray(t, np.float64).mean()).sum()
    return float(1 - np.square(d).sum()/sst) if sst else None


@torch.no_grad()
def predict(model, data, idx):
    E, A = [], []
    model.eval()
    for s in range(0, len(idx), 64):
        x, _ = data.batch(idx[s:s+64])
        ee, aa = model(**x)
        E.append(ee.cpu().numpy()); A.append(aa.cpu().numpy())
    return np.concatenate(E), np.concatenate(A)


def main():
    setup(11)
    data = Data()
    raw = np.load(ROOT/'data/raw_labels.npz')
    idx = data.parts['val']
    A_t = raw['A'][idx].astype(np.float64)
    tr_t = np.trace(A_t, axis1=-2, axis2=-1)
    ma = raw['mask_A'][idx].astype(bool) & raw['mask_E'][idx].astype(bool)

    # A = mu mu^T exactly, so |mu| = sqrt(tr A) and mu's direction is A's principal eigenvector.
    w_t, v_t = np.linalg.eigh(A_t)
    n_t_all = np.sqrt(np.clip(tr_t, 0, None))
    dir_t = v_t[..., :, -1]
    resid = np.abs(w_t[..., :-1]).max(-1)/np.clip(np.abs(w_t[..., -1]), 1e-30, None)
    print(f'val: median (|l0|+|l1|)/|l2| = {np.median(resid[ma]):.3e}  -> A is rank-1', flush=True)
    print(f'val: max |tr(A) - l2| relative = {np.abs(tr_t - w_t[..., -1])[ma].max()/np.abs(w_t[..., -1][ma]).max():.3e}', flush=True)

    m = ma & (n_t_all > 1e-9)       # exclude exact dark states for relative/direction stats
    out = {}
    for n in GROUPS:
        cfg = json.loads((ROOT/'configs'/f'{n}.json').read_text())
        ck = torch.load(ROOT/'runs'/n/'best.pt', map_location='cpu', weights_only=False)
        model = build(cfg, data.stats, initial=False); model.load_state_dict(ck['model']); model = model.cuda()
        E, A = predict(model, data, idx)
        tr_p = np.trace(A.astype(np.float64), axis1=-2, axis2=-1)
        # reconstruct mu_pred: rank-1 => mu = sqrt(lambda_max) * v_max, sign arbitrary
        w, v = np.linalg.eigh(A.astype(np.float64))
        mu_p = np.sqrt(np.clip(w[..., -1], 0, None))[..., None]*v[..., :, -1]
        n_p = np.linalg.norm(mu_p, axis=-1)
        n_t = n_t_all

        cosang = (mu_p*dir_t).sum(-1)/np.clip(n_p, 1e-30, None)
        cosang = np.abs(cosang)                     # sign of mu is arbitrary
        ls = (tr_p - tr_t)**2
        tail = np.sort(ls[m].ravel())[::-1]
        k = max(1, int(.01*tail.size))

        out[n] = dict(
            best_epoch=ck['epoch'],
            R2_trA=r2(tr_p[m], tr_t[m]),
            R2_absmu=r2(n_p[m], n_t[m]),
            R2_log_absmu=r2(np.log(np.clip(n_p[m], 1e-12, None)), np.log(np.clip(n_t[m], 1e-12, None))),
            R2_f=r2(K*raw['E'][idx]*tr_p, raw['f'][idx].astype(np.float64)) if False else None,
            median_abs_cos=float(np.median(cosang[m])),
            frac_cos_gt_0p9=float((cosang[m] > .9).mean()),
            frac_cos_lt_0p5=float((cosang[m] < .5).mean()),
            top1pct_share_of_Ls=float(tail[:k].sum()/tail.sum()),
            median_rel_err_sq=float(np.median(np.abs(tr_p[m]-tr_t[m])/np.clip(tr_t[m], 1e-12, None))),
        )
        del model; torch.cuda.empty_cache()

    pathlib.Path('/home/inspur/r2_val_scratch/mu_diag.json').write_text(json.dumps(out, indent=2))
    print()
    print(f'{"":5} {"R2(trA)":>9} {"R2(|mu|)":>9} {"R2(log|mu|)":>11} {"med|cos|":>9} {">0.9 cos":>9} {"top1% Ls":>9} {"med relerr":>10}')
    for n in GROUPS:
        o = out[n]
        print(f'{n:5} {o["R2_trA"]:9.4f} {o["R2_absmu"]:9.4f} {o["R2_log_absmu"]:11.4f} '
              f'{o["median_abs_cos"]:9.4f} {o["frac_cos_gt_0p9"]:9.3f} '
              f'{o["top1pct_share_of_Ls"]:9.3f} {o["median_rel_err_sq"]:10.3f}')


main()
