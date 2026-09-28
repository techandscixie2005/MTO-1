"""Is the c=64 run overfitting, or is the val floor the data/target's own limit?

Three readouts, all from history.jsonl (no model runs needed):

 A. generalization gap      train vs val, per component, across epochs.
    If train -> 0 while val flattens, it is overfitting.
    If train also stalls well above 0, it is underfitting/optimization instead.
 B. val slope + projection  linear fit to val[0] over the tail, projected to max_epochs.
    A near-zero slope with the two campaigns' curves sitting at the SAME floor
    means capacity is not what is binding.
 C. c=16 vs c=64 at matched epochs, so the comparison is not maturity-confounded.

Loss vector order (objective.py):  [0]=le+ls+lq  [1]=le  [2]=ls  [3]=lq
For eta=0 groups (G2/G4):           [0]=ls        [1]=0   [2]=ls  [3]=0
"""
import json, pathlib

NEW = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_chan64_20260928')
OLD = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_pouter_trace_wE_20260928')


def hist(root, g):
    return [json.loads(l) for l in (root/'runs'/g/'history.jsonl').read_text().splitlines()]


def slope(rows, k1, k2, key=lambda r: r['val'][0]):
    """least-squares slope of key vs epoch over rows[k1:k2]"""
    pts = [(r['epoch'], key(r)) for r in rows[k1:k2]]
    n = len(pts)
    mx = sum(p[0] for p in pts)/n; my = sum(p[1] for p in pts)/n
    num = sum((x-mx)*(y-my) for x, y in pts); den = sum((x-mx)**2 for x, _ in pts)
    return num/den if den else 0.0, mx, my


print('=== A. generalization gap (c=64) ===')
print('%-4s %5s | %9s %9s | %9s %9s | %9s %9s | %7s' % (
    'grp', 'ep', 'tr_tot', 'va_tot', 'tr_LE', 'va_LE', 'tr_Ls', 'va_Ls', 'va/tr'))
for g in ['G1', 'G2', 'G3', 'G4']:
    rows = hist(NEW, g)
    eta1 = rows[0]['train'][1] != 0
    for ep in (10, 20, 30, 40, 50, 60):
        cand = [r for r in rows if r['epoch'] == ep]
        if not cand:
            continue
        r = cand[0]
        print('%-4s %5d | %9.4f %9.4f | %9s %9s | %9.4f %9.4f | %7.2f' % (
            g, ep, r['train'][0], r['val'][0],
            ('%.4f' % r['train'][1]) if eta1 else '-',
            ('%.4f' % r['val'][1]) if eta1 else '-',
            r['train'][2], r['val'][2],
            r['val'][0]/r['train'][0] if r['train'][0] else float('inf')))

print('\n=== B. val slope over the tail, projected to epoch 1000 ===')
print('%-28s %-4s %6s %10s %10s %11s' % ('run', 'grp', 'n_ep', 'slope/ep', 'val_now', 'val@1000'))
for tag, root in (('c=16 (paused)', OLD), ('c=64 (running)', NEW)):
    for g in ['G1', 'G2', 'G3', 'G4']:
        rows = hist(root, g)
        k = min(30, len(rows)//2)
        s, mx, my = slope(rows, len(rows)-k, len(rows))
        proj = my + s*(1000-mx)
        print('%-28s %-4s %6d %10.6f %10.4f %11.4f' % (
            tag, g, len(rows), s, rows[-1]['val'][0], proj))

print('\n=== C. matched epochs, c=16 vs c=64 (val[0]) ===')
for g in ['G1', 'G2', 'G3', 'G4']:
    a = {r['epoch']: r['val'][0] for r in hist(OLD, g)}
    b = {r['epoch']: r['val'][0] for r in hist(NEW, g)}
    eps = [e for e in (20, 30, 40, 50, 60) if e in a and e in b]
    print('%-4s %s' % (g, '  '.join('ep%d: %.4f -> %.4f (%+.4f)' % (e, a[e], b[e], b[e]-a[e]) for e in eps)))

print('\n=== D. has val actually started rising? last 8 epochs ===')
for g in ['G1', 'G2', 'G3', 'G4']:
    rows = hist(NEW, g)
    print('%-4s %s' % (g, ' '.join('%.3f' % r['val'][0] for r in rows[-8:])))
    print('     %s' % ' '.join('%.3f' % r['train'][0] for r in rows[-8:]))
