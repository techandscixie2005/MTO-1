"""f R^2 on the c=64 campaign, validation split only (test stays sealed).

Same protocol as the c=16 analysis: checkpoint -> predict -> f = K*E*tr(A) -> R^2
against raw labels, with the 'oracle' energy (true E) and the 'common' energy (G1's
predicted E) as two readouts, so an E error and a trace error stay separable.
Read-only: loads best.pt, never writes into the campaign directory.
"""
import json, pathlib, sys, time
import numpy as np, torch

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 \
    else pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_chan64_20260928')
TAG = ROOT.name
sys.path.insert(0, str(ROOT))
from dataset import Data
from model_factory import build
from trainer import setup, fingerprint

K = (2/3)/27.211386245988
GRID = np.arange(1051, dtype=np.float64)*.02


def regression(pred, true, mask):
    p = np.asarray(pred, np.float64)[mask]; t = np.asarray(true, np.float64)[mask]
    assert np.isfinite(p).all() and np.isfinite(t).all()
    d = p-t; sst = np.square(t-t.mean()).sum()
    return dict(count=int(t.size), MAE=float(np.abs(d).mean()),
                RMSE=float(np.sqrt(np.square(d).mean())), R2=float(1-np.square(d).sum()/sst))


def broaden(e, f, mask):
    e = np.where(mask, e, 0); f = np.where(mask, f, 0)
    return (f[..., None]*np.exp(-.5*((GRID-e[..., None])/.2)**2)/(.2*np.sqrt(2*np.pi))).sum(1)


@torch.no_grad()
def predict(model, data, idx):
    e = []; a = []; model.eval()
    for s in range(0, len(idx), 64):
        x, _ = data.batch(idx[s:s+64]); ee, aa = model(**x)
        e.append(ee.cpu().numpy()); a.append(aa.cpu().numpy())
    return dict(E=np.concatenate(e), A=np.concatenate(a))


setup(11)
campaign = json.loads((ROOT/'campaign.json').read_text()); names = campaign['groups']
cfgs = {n: json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in names}
fp = fingerprint()
data = Data(); raw = np.load(ROOT/'data/raw_labels.npz')
part = 'val'; idx = data.parts[part]
truth = {k: raw[k][idx] for k in ('E', 'A', 'f', 'mask_E', 'mask_A', 'mask_f')}
mask = truth['mask_A'] & truth['mask_E'] & truth['mask_f']

preds = {}; meta = {}
for n in names:
    path = ROOT/'runs'/n/'best.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['config'] == cfgs[n], n
    assert ck['fingerprint'] == fp, n
    h = [json.loads(l) for l in (ROOT/'runs'/n/'history.jsonl').read_text().splitlines()]
    be = min(h, key=lambda r: r['best_val'])['best_epoch']
    print('%s: best.pt epoch=%s history_best_epoch=%s val[0]=%.6f' % (
        n, ck['epoch'], be, ck['val'][0]), flush=True)
    meta[n] = dict(epoch=ck['epoch'], best_epoch=be, ck_val=ck['val'][0],
                   supervise_E=cfgs[n]['supervise_E'], arch=cfgs[n]['architecture'])
    model = build(cfgs[n], data.stats).cuda(); model.load_state_dict(ck['model'])
    preds[n] = predict(model, data, idx)
    del model; torch.cuda.empty_cache()

eref = preds['G1']['E'].astype(np.float64)
out = {}
print()
print('%-5s%-9s%11s%11s%10s%12s' % ('grp', 'mode', 'f_MAE', 'f_RMSE', 'f_R2', 'specMSE'))
for n in names:
    tr = np.trace(preds[n]['A'].astype(np.float64), axis1=-2, axis2=-1)
    modes = {'oracle': truth['E'], 'common': eref}
    if cfgs[n]['supervise_E']:
        modes['native'] = preds[n]['E'].astype(np.float64)
    out[n] = {}
    for mode, e in modes.items():
        f = K*e*tr; r = regression(f, truth['f'], mask)
        smse = float(np.square(broaden(e, f, mask)-broaden(truth['E'], truth['f'], mask)).mean(1).mean())
        out[n][mode] = dict(**r, specMSE=smse)
        print('%-5s%-9s%11.6f%11.6f%10.6f%12.8f' % (n, mode, r['MAE'], r['RMSE'], r['R2'], smse))
    ps = [regression((K*truth['E']*tr)[:, k], truth['f'][:, k], mask[:, k])['R2'] for k in range(10)]
    out[n]['per_state_R2_oracle'] = ps
    print('     per-state R2(oracle): %s' % ' '.join('%.3f' % x for x in ps))

json.dump(dict(meta=meta, results=out, part=part, root=str(ROOT),
               note='validation split only; test unaware per campaign protocol',
               time=time.time()), open(f'/tmp/fr2_val_{TAG}.json', 'w'), indent=2)
print(f'WROTE /tmp/fr2_val_{TAG}.json')
