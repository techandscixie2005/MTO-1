#!/usr/bin/env python3
"""Prepare a prospective inventory; finalize only after four reviewed terminal runs."""
import argparse
import hashlib
import json
from pathlib import Path
from monitor import ROOT, atomic_json, process_identity, safe_path, stamp
from package_records import inspect

p=argparse.ArgumentParser()
p.add_argument('--finalize',action='store_true')
p.add_argument('--review',help='Scope-relative independent completion review JSON')
p.add_argument('--decision',help='Scope-relative coordinator scientific decision Markdown')
p.add_argument('--extra',action='append',default=[])
a=p.parse_args()
cfg=json.loads((ROOT/'round_config.json').read_text())
manifest_sha=hashlib.sha256((ROOT/'FROZEN_MANIFEST.json').read_bytes()).hexdigest()
registry=json.loads((ROOT/'ops/registry.json').read_text())
groups={'source_and_protocol':[
    'PROTOCOL.md','round_config.json','FROZEN_MANIFEST.json','ENVIRONMENT.json',
    'PRE_PILOT_DECISION_AMENDMENT.md','SECONDARY_CALIBRATION_POLICY.md',
    'train.py','metrics.py','freeze.py','launch.py','summarize.py','architecture/model.py'],
    'analysis_and_decisions':['ROUND_RESULTS.json','ROUND_RESULTS.md','ANALYSIS_RECEIPT.json',
        'SCIENTIFIC_FOLLOWUP_HYPOTHESES.md','mechanism_audit.py','MECHANISM_CONFIG.json',
        'MECHANISM_RESULTS.json','MECHANISM_RESULTS.md','gap_audit.py','GAP_AUDIT_CONFIG.json',
        'GAP_AUDIT_RESULTS.json','plotting.py','ALIGNED_VALIDATION_CURVES.svg','PLOT_MANIFEST.json',
        'amend_diagnostic_metadata.py','DIAGNOSTIC_METADATA_AMENDMENT.json',
        'MECHANISM_CONFIG_ORIGINAL.json','GAP_AUDIT_CONFIG_ORIGINAL.json',
        'postrun_summarize.log','postrun_mechanism.log','postrun_gap.log','postrun_plotting.log',
        'ROUND_INTERPRETATION.md','false_bright_audit.py','FALSE_BRIGHT_RESULTS.json','FALSE_BRIGHT_RECEIPT.json',
        'FALSE_BRIGHT_EXECUTION.log','FALSE_BRIGHT_GPU_ADMISSION.xml'],
    'arm_records':[], 'monitoring_and_operations':[], 'response_feasibility':[],
    'confirmation_preparation':[], 'research_report_and_label_audit':[], 'cache_feasibility':[]}
states={}
for arm in cfg['arms']:
    out=ROOT/'runs'/arm
    required=['LAUNCH_RECEIPT.json','MONITOR_REGISTRATION.json','run_manifest.json','history.jsonl',
              'train.log','status.json','best_metrics.json','FIT_COMPLETE.json']
    groups['arm_records'] += [f'runs/{arm}/{name}' for name in required]
    # Admission XML and explicit failed-attempt JSON/logs remain scientific provenance.
    groups['arm_records'] += [str(x.relative_to(ROOT)) for x in out.iterdir()
        if x.is_file() and (x.name.endswith('_gpu_admission.xml') or
            ('FAILED' in x.name and x.suffix in {'.json','.log'}))]
    terminal=out/'FIT_COMPLETE.json'
    states[arm]={'terminal_present':terminal.exists()}
    if terminal.exists():
        t=json.loads(terminal.read_text())
        states[arm].update(completed_epoch=t.get('completed_epoch'),
            single_checkpoint=t.get('single_checkpoint'),test_evaluated=t.get('test_evaluated'),
            manifest_matches=t.get('manifest_sha256')==manifest_sha)
    if a.finalize:
        assert terminal.exists(),f'{arm} has no terminal record'
        assert t['completed_epoch']==cfg['epochs']==20 and t['single_checkpoint'] and not t['test_evaluated']
        assert t['manifest_sha256']==manifest_sha
        owned=[r for r in registry['runs'] if Path(r['run_dir']).name==arm]
        assert owned
        for r in owned:
            identity=process_identity(r['identity']['pid'])
            if identity:
                exact=all(identity[k]==r['identity'][k] for k in ('pid','start_ticks','boot_id','uid','cwd','argv_sha256'))
                assert not exact or identity['process_state']=='Z', f'Owned worker still live: {arm}'
for folder in ('monitoring','ops','current_state','ops/publication_records'):
    for x in (ROOT/folder).iterdir():
        if x.is_file() and x.suffix in {'.json','.jsonl','.log','.md','.py','.txt','.ps1'}:
            if x.name not in {'COMPLETED_ROUND_ALLOWLIST.txt','COMPLETED_ROUND_INVENTORY.json'}:
                groups['monitoring_and_operations'].append(str(x.relative_to(ROOT)))
groups['response_feasibility']=[str(x.relative_to(ROOT)) for x in (ROOT/'response_feasibility').iterdir()
                              if x.is_file() and x.suffix in {'.py','.json','.md'}]
for group,base,manifest_name in (
    ('confirmation_preparation','confirmation_preparation','PREPARATION_MANIFEST.json'),
    ('research_report_and_label_audit','reports','WORKING_REPORT_MANIFEST.json'),
    ('cache_feasibility','reports/frozen_cache_feasibility','FEASIBILITY_MANIFEST.json')):
    mp=ROOT/base/manifest_name
    if mp.exists():
        package=json.loads(mp.read_text())
        groups[group].append(str(mp.relative_to(ROOT)))
        for name,entry in package['files'].items():
            fp=safe_path(ROOT,base+'/'+name)
            assert hashlib.sha256(fp.read_bytes()).hexdigest()==entry['sha256'],str(fp)
            groups[group].append(str(fp.relative_to(ROOT)))
review_preparation='confirmation_preparation/PREPARATION_IMPLEMENTATION_REVIEW.json'
if (ROOT/review_preparation).exists():
    groups['confirmation_preparation'].append(review_preparation)
for name in (a.review,a.decision,*a.extra):
    if name:
        safe_path(ROOT,name)
        groups['analysis_and_decisions'].append(name)
if a.finalize:
    assert a.review and a.decision,'Independent completion review and coordinator decision are required'
    rev=json.loads(safe_path(ROOT,a.review).read_text())
    assert rev.get('passed') is True or rev.get('decision')=='PASS'
    assert len(safe_path(ROOT,a.decision).read_text().strip())>100
    results=json.loads((ROOT/'ROUND_RESULTS.json').read_text())
    assert results['status']=='all_four_terminal' and results['single_checkpoint_per_arm']
    assert results['predictions_averaged'] is False and results['historical_test_evaluated'] is False
names=sorted(set(n for values in groups.values() for n in values))
missing=[name for name in names if not (ROOT/name).is_file()]
inventory={'prepared_at_utc':stamp(),'status':'ready_to_package' if a.finalize else 'prospective_only_not_a_completed_round',
    'source_manifest_sha256':manifest_sha,'terminal_states':states,'groups':groups,'missing_files':missing,
    'archive_sequence':['finish all four fits and selected-checkpoint replays','independent completion review and coordinator decision',
        'mirror current local handoffs and supplemental operational records into server scope',
        'finalize explicit allowlist and inspect text file contents/sizes',
        'package server records','download FIRST to a new D:/MTO/archives/single_model_20260929/ round directory',
        'verify all hashes','stage exact new blobs with --no-filters in a separate index',
        'inspect diff and byte equality; preserve unchanged remote base with write-tree --missing-ok',
        'commit, non-force push via native Windows OpenSSH proxy, verify remote commit'],
    'excluded':['all .pt/.pth/.ckpt files','optimizer/RNG checkpoint binaries','raw datasets and raw prediction arrays',
                'caches and __pycache__','credentials','unrelated jobs or workspace modifications'],
    'previous_round_commit':'0e612523cd99594b6ce05b491b8de1b18d8a2e66',
    'no_frozen_scientific_sources_rewritten':True,'no_partial_results_published':True}
atomic_json(ROOT/'ops/COMPLETED_ROUND_INVENTORY.json',inventory)
if a.finalize:
    assert not missing, missing
    names += ['ops/COMPLETED_ROUND_INVENTORY.json','ops/COMPLETED_ROUND_ALLOWLIST.txt']
    (ROOT/'ops/COMPLETED_ROUND_ALLOWLIST.txt').write_text('\n'.join(sorted(set(names)))+'\n')
    entries=inspect(ROOT,names)
    print(json.dumps({'status':'ready_to_package','files':len(entries),'bytes':sum(x['bytes'] for x in entries)},indent=2))
else:
    print(json.dumps({'status':inventory['status'],'existing_file_count':len(names)-len(missing),
        'missing_files':missing,'terminal_states':states},indent=2))
