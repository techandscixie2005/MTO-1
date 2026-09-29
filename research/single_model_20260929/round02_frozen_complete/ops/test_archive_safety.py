#!/usr/bin/env python3
"""Verify that the archive gate rejects weights, raw arrays, escapes and secrets."""
import json
from pathlib import Path
import tempfile
from package_records import inspect

results = {}
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    for name, content in [('results.json', '{}'), ('best.pt', 'weight placeholder'),
                          ('predictions.npz', 'array placeholder'), ('key.txt', 'ghp_' + 'A' * 40),
                          ('binary.txt', 'a\x00b')]:
        (root / name).write_text(content)
    results['allow_text_metric'] = len(inspect(root, ['results.json'])) == 1
    for name in ('best.pt', 'predictions.npz', 'key.txt', 'binary.txt', '../escape.json', '/etc/passwd'):
        try:
            inspect(root, [name])
            results['reject_' + name] = False
        except ValueError:
            results['reject_' + name] = True
print(json.dumps({'checks': results, 'passed': all(results.values())}, indent=2))
assert all(results.values())
