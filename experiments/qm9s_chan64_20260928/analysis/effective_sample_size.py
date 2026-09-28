"""Quantify whether 'not enough data' is the right explanation for the trace overfit.

The weighted trace loss is  Ls = mean_valid( w * (ds)^2 ) / (3*sA2),  w = E_true^2 / mean_E2_train.
A per-sample weight makes the effective sample size

    N_eff = (sum w)^2 / sum(w^2)

which is what the optimizer actually sees, versus the nominal 120355 molecules x 10 states.
Also compares the trace-head generalization gap against the run with NO E^2 weighting, to
separate 'the weighting concentrates the loss' from 'the dataset is simply small'.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_chan64_20260928')
raw = np.load(ROOT/'data/raw_labels.npz')
part = json.loads((ROOT/'data/partition.json').read_text()) if (ROOT/'data/partition.json').exists() else None

w_rec = json.loads((ROOT/'reports/trace_weight.json').read_text())
print('recorded trace_weight.json:', json.dumps(w_rec, indent=2)[:600])

mean_E2 = json.loads((ROOT/'data/trace_weight.json').read_text())['mean_E2_train']
print('\nmean_E2_train = %.6f' % mean_E2)

# find the train index set
import sys
sys.path.insert(0, str(ROOT))
from dataset import Data
d = Data()
idx = d.parts['train']
E = raw['E'][idx]; mE = raw['mask_E'][idx]; mA = raw['mask_A'][idx]
valid = mE & mA
print('train molecules=%d  states=%d  valid (m,E) pairs=%d' % (
    len(idx), E.shape[1], valid.sum()))

w = (E[valid].astype(np.float64)**2) / mean_E2
N = w.size
Neff = w.sum()**2 / (w**2).sum()
print('\n--- effective sample size under w = E^2/mean_E2 (train, valid pairs) ---')
print('nominal N      = %d' % N)
print('N_eff          = %.1f' % Neff)
print('N_eff / N      = %.5f  (1 in %.0f samples carries the gradient)' % (Neff/N, N/Neff))

s = np.sort(w)[::-1]
c = np.cumsum(s)/s.sum()
print('\n--- how concentrated is the weight ---')
for frac in (0.005, 0.01, 0.02, 0.05, 0.10, 0.25):
    k = max(1, int(round(frac*N)))
    print('  top %5.1f%% of pairs (%7d) carry %6.2f%% of total weight' % (100*frac, k, 100*c[k-1]))

print('\n--- per excited state (weight = E^2 share) ---')
print('%-6s %12s %12s %10s' % ('state', 'mean_E2', 'w_share', 'N_valid'))
tot = w.sum()
for k in range(E.shape[1]):
    vk = valid[:, k]
    if vk.sum() == 0:
        continue
    wk = (E[vk, k].astype(np.float64)**2)/mean_E2
    print('%-6d %12.4f %11.3f%% %10d' % (k+1, (E[vk, k]**2).mean(), 100*wk.sum()/tot, vk.sum()))

print('\n--- does the TRACE head generalize worse than the E head? ---')
print('(history vector: [le+ls+lq, le, ls, lq]; for eta=0 groups only ls is meaningful)')


def hist(root, g):
    return [json.loads(l) for l in (root/'runs'/g/'history.jsonl').read_text().splitlines()]


for tag, root in (('c=64 weighted', ROOT),
                  ('c=16 weighted', pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_pouter_trace_wE_20260928')),
                  ('c=16 UNWEIGHTED', pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927'))):
    if not (root/'runs').exists():
        print('  %-16s MISSING' % tag); continue
    print('\n  %s' % tag)
    for g in ['G1', 'G2', 'G3', 'G4']:
        try:
            rows = hist(root, g)
        except FileNotFoundError:
            print('    %s missing' % g); continue
        eta1 = rows[0]['train'][1] != 0
        r = rows[-1]
        if eta1:
            gE = r['val'][1]/r['train'][1] if r['train'][1] else 0
            print('    %-4s ep%-4d  LE tr/va=%.4f/%.4f (x%.2f)   Ls tr/va=%.4f/%.4f (x%.2f)' % (
                g, r['epoch'], r['train'][1], r['val'][1], gE,
                r['train'][2], r['val'][2], r['val'][2]/r['train'][2] if r['train'][2] else 0))
        else:
            print('    %-4s ep%-4d  Ls tr/va=%.4f/%.4f (x%.2f)' % (
                g, r['epoch'], r['train'][2], r['val'][2],
                r['val'][2]/r['train'][2] if r['train'][2] else 0))
