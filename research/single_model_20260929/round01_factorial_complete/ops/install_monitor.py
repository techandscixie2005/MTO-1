#!/usr/bin/env python3
"""Install one tagged cron block while retaining all unrelated user entries."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from monitor import ROOT, atomic_json, stamp

begin = '# BEGIN MTO_SINGLE_MODEL_20260929_MONITOR'
end = '# END MTO_SINGLE_MODEL_20260929_MONITOR'
r = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
if r.returncode not in (0, 1) or (r.returncode == 1 and 'no crontab' not in r.stderr):
    raise SystemExit(r.stderr)
before = r.stdout if r.returncode == 0 else ''
if before.count(begin) != before.count(end) or before.count(begin) > 1:
    raise SystemExit('Ambiguous existing monitor schedule; review manually')
lines, inside = [], False
for line in before.splitlines():
    if line == begin:
        inside = True
    elif line == end:
        inside = False
    elif not inside:
        lines.append(line)
schedule = f'0 */4 * * * /usr/bin/python3 {ROOT}/ops/monitor.py --root {ROOT} >> {ROOT}/monitoring/cron.log 2>&1'
after = '\n'.join(lines).rstrip() + '\n' + begin + '\n' + schedule + '\n' + end + '\n'
(ROOT / 'ops/crontab_before.txt').write_text(before)
subprocess.run(['crontab', '-'], input=after, text=True, check=True)
installed = subprocess.run(['crontab', '-l'], capture_output=True, text=True, check=True).stdout
if installed != after:
    raise SystemExit('Installed crontab differs; inspect immediately')
atomic_json(ROOT / 'ops/SCHEDULE_RECEIPT.json', {'installed_at_utc': stamp(), 'schedule': schedule,
    'timezone': 'server local time Asia/Shanghai; 00:00,04:00,08:00,12:00,16:00,20:00',
    'cadence_hours': 4, 'unrelated_entries_preserved': True,
    'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
    'after_sha256': hashlib.sha256(after.encode()).hexdigest(),
    'mode': 'read_only; never starts, signals, or restarts training jobs'})
print(installed)
