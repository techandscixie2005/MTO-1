import json, pathlib, time

ROOT = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_chan64_20260928')
print('now', time.strftime('%Y-%m-%d %H:%M:%S'))

print('\n=== c=64 progress ===')
print('%-4s %-18s %6s %6s | %10s %6s | %8s %8s' % (
    'grp', 'event', 'epoch', 'rows', 'best_val', 'best_ep', 'train_L', 'val_tot'))
for g in ['G1', 'G2', 'G3', 'G4']:
    st = json.loads((ROOT / 'runs' / g / 'status.json').read_text())
    hf = ROOT / 'runs' / g / 'history.jsonl'
    hist = [json.loads(l) for l in hf.read_text().splitlines()] if hf.exists() else []
    if not hist:
        print('%-4s %-18s %6s %6s | (no history yet)' % (g, st.get('event'), st.get('epoch'), 0))
        continue
    best = min(hist, key=lambda r: r['best_val'])
    last = hist[-1]
    print('%-4s %-18s %6s %6s | %10.6f %6d | %8.4f %8.4f' % (
        g, st.get('event'), st.get('epoch'), len(hist),
        best['best_val'], best['best_epoch'], last['train'][0], last['val'][0]))

print('\n=== c=16 reference (paused), for comparison ===')
OLD = pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_pouter_trace_wE_20260928')
print('%-4s %-18s %6s %6s | %10s %6s' % ('grp', 'event', 'epoch', 'rows', 'best_val', 'best_ep'))
for g in ['G1', 'G2', 'G3', 'G4']:
    st = json.loads((OLD / 'runs' / g / 'status.json').read_text())
    hist = [json.loads(l) for l in (OLD / 'runs' / g / 'history.jsonl').read_text().splitlines()]
    best = min(hist, key=lambda r: r['best_val'])
    print('%-4s %-18s %6s %6s | %10.6f %6d' % (
        g, st.get('event'), st.get('epoch'), len(hist), best['best_val'], best['best_epoch']))
