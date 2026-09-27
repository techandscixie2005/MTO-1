"""Read-only, VALIDATION-ONLY oscillator-strength R2 for the 4 running groups.

Does not touch the test split, does not modify any run artifact.
Metric definition copied verbatim from evaluate.py::regression.
"""
import json, sys, pathlib
import numpy as np, torch

ROOT = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927')
sys.path.insert(0, str(ROOT))
from dataset import Data
from model_factory import build
from trainer import setup

K = (2/3)/27.211386245988
GROUPS = ('G1', 'G2', 'G3', 'G4')


def regression(pred, true, mask):
    p = np.asarray(pred, np.float64)[mask]
    t = np.asarray(true, np.float64)[mask]
    assert np.isfinite(p).all() and np.isfinite(t).all()
    if not t.size:
        return dict(count=0, MAE=None, RMSE=None, R2=None)
    d = p - t
    sst = np.square(t - t.mean()).sum()
    return dict(count=int(t.size),
                MAE=float(np.abs(d).mean()),
                RMSE=float(np.sqrt(np.square(d).mean())),
                R2=float(1 - np.square(d).sum()/sst) if sst else None)


@torch.no_grad()
def predict(model, data, idx):
    e, a = [], []
    model.eval()
    for s in range(0, len(idx), 64):
        x, _ = data.batch(idx[s:s+64])
        ee, aa = model(**x)
        e.append(ee.cpu().numpy()); a.append(aa.cpu().numpy())
    return np.concatenate(e), np.concatenate(a)


def main():
    setup(11)
    data = Data()
    raw = np.load(ROOT/'data/raw_labels.npz')
    part = 'val'
    idx = data.parts[part]
    truth = {k: raw[k][idx] for k in ('E', 'A', 'f', 'mask_E', 'mask_A', 'mask_f')}
    cfgs = {n: json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in GROUPS}

    preds = {}
    for n in GROUPS:
        ck = torch.load(ROOT/'runs'/n/'best.pt', map_location='cpu', weights_only=False)
        model = build(cfgs[n], data.stats, initial=False)
        model.load_state_dict(ck['model'])
        model = model.cuda()
        preds[n] = predict(model, data, idx)
        print(f'{n} best_epoch={ck["epoch"]} own_best_val={ck["val"][0]:.6f} '
              f'(LE={ck["val"][1]:.6f} Ls={ck["val"][2]:.6f} LQ={ck["val"][3]:.6f})', flush=True)
        del model
        torch.cuda.empty_cache()

    eref = preds['G1'][0].astype(np.float64)
    me, ma, mf = truth['mask_E'], truth['mask_A'], truth['mask_f']
    mask = ma & me & mf
    out = {}
    for n in GROUPS:
        E, A = preds[n]
        tr = np.trace(A.astype(np.float64), axis1=-2, axis2=-1)
        f_common = K*eref*tr
        r = dict(f_common=regression(f_common, truth['f'], mask),
                 f_oracle=regression(K*truth['E'].astype(np.float64)*tr, truth['f'], mask),
                 trace=regression(tr, np.trace(truth['A'], axis1=-2, axis2=-1), ma),
                 E_eV=regression(E, truth['E'], me),
                 per_state_common=[regression(f_common[:, k], truth['f'][:, k], mask[:, k]) for k in range(10)])
        if cfgs[n]['supervise_E']:
            r['f_native'] = regression(K*E.astype(np.float64)*tr, truth['f'], mf)
        out[n] = r

    (ROOT/'reports/r2_val.json').write_text(json.dumps(out, indent=2))
    print()
    print(f'{"grp":4} {"f_common R2":>12} {"f_oracle R2":>12} {"f_native R2":>12} {"trace R2":>10} {"E R2":>8}')
    for n in GROUPS:
        o = out[n]
        nat = f'{o["f_native"]["R2"]:.4f}' if 'f_native' in o else 'n/a'
        print(f'{n:4} {o["f_common"]["R2"]:12.4f} {o["f_oracle"]["R2"]:12.4f} {nat:>12} '
              f'{o["trace"]["R2"]:10.4f} {o["E_eV"]["R2"]:8.4f}')
    for n in GROUPS:
        print(n, 'per-state f_common R2', ['%.3f' % p['R2'] for p in out[n]['per_state_common']])


main()
