"""VAL-ONLY diagnostic: is the true transition tensor A rank-1?

Compares true-label rank structure against G1 (original MTO, A=CC^T) and
G3 (P-only, A=mu mu^T) predictions. Read-only; test split untouched.
"""
import json, sys, pathlib
import numpy as np, torch

ROOT = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927')
sys.path.insert(0, str(ROOT))
from dataset import Data
from model_factory import build
from trainer import setup

GROUPS = ('G1', 'G3')          # one per architecture, both E-supervised


@torch.no_grad()
def predict(model, data, idx):
    E, A, MU = [], [], []
    model.eval()
    for s in range(0, len(idx), 64):
        x, _ = data.batch(idx[s:s+64])
        try:
            out = model(**x, export=True)
        except TypeError:                     # MTOEA has no export kwarg
            ee, aa = model(**x); out = (ee, aa)
        E.append(out[0].cpu().numpy()); A.append(out[1].cpu().numpy())
        MU.append(out[2].cpu().numpy() if len(out) > 2
                  else np.full(out[1].shape[:-2] + (3,), np.nan, dtype=np.float32))
    return np.concatenate(E), np.concatenate(A), np.concatenate(MU)


def spectrum_stats(A, mask, name):
    """Eigenvalue share of the two smaller eigenvalues relative to the trace."""
    w = np.linalg.eigvalsh(A.astype(np.float64))          # ascending: l0<=l1<=l2
    tr = w.sum(-1)
    nz = mask & (tr > 1e-12)
    share = np.where(nz, (w[..., 0] + w[..., 1]) / np.where(nz, tr, 1), np.nan)
    s = share[nz]
    return dict(name=name, count=int(nz.sum()),
                mean_share=float(np.nanmean(s)),
                median_share=float(np.nanmedian(s)),
                p10=float(np.percentile(s, 10)), p90=float(np.percentile(s, 90)),
                frac_share_lt_1pct=float((s < .01).mean()),
                frac_share_lt_5pct=float((s < .05).mean()),
                frac_share_gt_20pct=float((s > .20).mean()),
                frac_negative_lambda=float((w[..., 0][nz] < -1e-6).mean()))


def main():
    setup(11)
    data = Data()
    raw = np.load(ROOT/'data/raw_labels.npz')
    idx = data.parts['val']
    A_true = raw['A'][idx].astype(np.float64)
    E_true = raw['E'][idx]
    ma, me = raw['mask_A'][idx], raw['mask_E'][idx]
    tr_true = np.trace(A_true, axis1=-2, axis2=-1)
    mask = ma.astype(bool) & me.astype(bool) & (tr_true > 1e-12)

    out = {'true': spectrum_stats(A_true, mask, 'true')}
    print(json.dumps(out['true'], indent=2), flush=True)

    # how many molecules are dark (trace==0) but present in mask_A
    print('val mol-state entries with mask_A:', int(ma.sum()),
          '| with trace>1e-12:', int(mask.sum()), flush=True)

    eref = None
    for n in GROUPS:
        cfg = json.loads((ROOT/'configs'/f'{n}.json').read_text())
        ck = torch.load(ROOT/'runs'/n/'best.pt', map_location='cpu', weights_only=False)
        m = build(cfg, data.stats, initial=False); m.load_state_dict(ck['model']); m = m.cuda()
        E, A, MU = predict(m, data, idx)
        print(f'{n} best_epoch={ck["epoch"]}', flush=True)
        out[n] = spectrum_stats(A, mask, n)
        tr_pred = np.trace(A.astype(np.float64), axis1=-2, axis2=-1)
        mup = np.linalg.norm(MU, axis=-1)
        # where is the model structurally silent but truth is bright?
        silent = mask & (tr_pred < 1e-8)
        bright = mask & (tr_true > np.percentile(tr_true[mask], 90))
        out[n]['structurally_silent_but_bright'] = int((silent & bright).sum())
        out[n]['structurally_silent_total'] = int(silent.sum())
        out[n]['mu_norm_median'] = float(np.median(mup[mask]))
        out[n]['true_sqrt_trace_median'] = float(np.median(np.sqrt(tr_true[mask])))
        # correlation of predicted trace magnitude vs truth on bright states
        out[n]['corr_trace'] = float(np.corrcoef(tr_pred[mask], tr_true[mask])[0, 1])
        del m; torch.cuda.empty_cache()

    (pathlib.Path('/home/inspur/r2_val_scratch/rank_diag.json')).write_text(json.dumps(out, indent=2))
    print()
    print(f'{"":6} {"mean (l0+l1)/tr":>16} {"median":>8} {"<1% frac":>9} {"<5% frac":>9} {">20% frac":>10}')
    for k in ('true', 'G1', 'G3'):
        o = out[k]
        print(f'{k:6} {o["mean_share"]:16.4f} {o["median_share"]:8.4f} '
              f'{o["frac_share_lt_1pct"]:9.3f} {o["frac_share_lt_5pct"]:9.3f} {o["frac_share_gt_20pct"]:10.3f}')
    print()
    for n in GROUPS:
        o = out[n]
        print(f'{n}: silent-but-bright={o["structurally_silent_but_bright"]}/{int(bright.sum())}  '
              f'silent_total={o["structurally_silent_total"]}  corr(trace)={o["corr_trace"]:.4f}  '
              f'median|mu|={o["mu_norm_median"]:.4f} vs median sqrt(tr_true)={o["true_sqrt_trace_median"]:.4f}')


main()
