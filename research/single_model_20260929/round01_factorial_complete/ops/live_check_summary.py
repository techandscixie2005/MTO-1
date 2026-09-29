#!/usr/bin/env python3
"""Condense one authoritative monitor snapshot, with live /proc identity replay."""
import datetime as dt
import json
from pathlib import Path
from monitor import ROOT, atomic_json, process_identity

snapshot=json.loads((ROOT/'monitoring/latest.json').read_text())
registered={r['id']:r for r in json.loads((ROOT/'ops/registry.json').read_text())['runs']}
rows=[]
for run in snapshot['runs']:
    pinned=registered[run['id']]['identity']
    identity=process_identity(pinned['pid'])
    keys=('pid','start_ticks','uid','cwd','argv_sha256','boot_id')
    live_match=identity is not None and all(identity[k]==pinned[k] for k in keys)
    out=ROOT/run['run_dir']
    status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    lines=(out/'history.jsonl').read_text().splitlines() if (out/'history.jsonl').exists() else []
    history=[]
    for line in lines:
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    latest=history[-1] if history else {}
    rows.append({'arm':out.name,'pid':pinned['pid'],
        'start_ticks':pinned['start_ticks'],'live_identity_reverified':live_match,
        'process_present_at_replay':identity is not None,
        'natural_terminal_exit_at_replay':identity is None and (out/'FIT_COMPLETE.json').exists(),
        'monitor_state':run['state'],'status_state':status.get('state'),
        'completed_epoch_status':status.get('completed_epoch'),
        'latest_complete_history_epoch':latest.get('epoch'),
        'latest_complete_raw_f_r2':latest.get('validation',{}).get('raw_f',{}).get('r2'),
        'best_epoch':status.get('best_epoch',status.get('selected_epoch')),
        'best_raw_f_r2':status.get('best_r2',status.get('validation',{}).get('raw_f',{}).get('r2')),
        'last_epoch_seconds':latest.get('seconds'),
        'expected_gpu_uuid':run['gpu_uuid'],'observed_gpu_uuids':run['observed_gpu_uuids'],
        'checkpoint_metadata':run['checkpoints'],
        'failure_marker_exists':(out/'FAILED.json').exists(),
        'completion_marker_exists':(out/'FIT_COMPLETE.json').exists()})
report={'observed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
    'snapshot_utc':snapshot['checked_at_utc'],'kind':'coordinator-requested authoritative continuation check',
    'scheduled_four_hour_cadence_unchanged':True,'runs':rows,
    'gpu0_unrelated_pid_present':any(p['pid']=='712998' for g in snapshot['gpu_snapshot']['gpus'] if g['index']==0 for p in g['compute_processes']),
    'owned_failures':any(r['failure_marker_exists'] or (not r['live_identity_reverified'] and not r['natural_terminal_exit_at_replay']) for r in rows),
    'interventions':[],'no_checkpoint_contents_loaded':True,
    'next_scheduled_server_local_check':'2026-09-30T04:00:00+08:00'}
atomic_json(ROOT/'ops/LIVE_CONTINUATION_CHECK.json',report)
print(json.dumps(report,indent=2))
