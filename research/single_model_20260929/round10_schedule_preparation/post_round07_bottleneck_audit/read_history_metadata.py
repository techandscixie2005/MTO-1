"""Read a fixed source/config/aggregate ledger; never load arrays or checkpoints."""
import hashlib
import json
from pathlib import Path

BASE = Path('/home/inspur/MTO-1')
OUT = BASE / 'research/single_model_20260929/post_round07_bottleneck_audit'
CAMPAIGNS = (
    'qm9s_full_EA_20260925', 'qm9s_eta_Ef_20260926',
    'qm9s_pouter_trace_20260927', 'qm9s_pouter_trace_wE_20260928',
    'qm9s_chan64_20260928',
)
hashes = {}


def read(path):
    assert path.suffix in ('.py', '.json', '.jsonl', '.md')
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')


def pin(relative):
    read(BASE / relative)


rows = []
for name in CAMPAIGNS:
    directory = BASE / 'experiments' / name
    for configpath in sorted((directory / 'configs').glob('*.json')):
        cfg = json.loads(read(configpath))
        run = directory / 'runs' / configpath.stem
        historypath = run / 'history.jsonl'
        history = [json.loads(line) for line in read(historypath).splitlines() if line.strip()]
        last = history[-1] if history else {}
        row = {'campaign': name, 'run': configpath.stem, 'config': cfg,
               'history_rows': len(history),
               'last_record': {k: last[k] for k in ('epoch','steps','best_epoch','best_val','lr','train','val') if k in last},
               'observed_markers': {}}
        for marker in ('FIT_COMPLETE.json', 'FAILED.json', 'STOPPED.json', 'ADMIN_STOPPED.json'):
            path = run / marker
            if path.exists():
                parsed = json.loads(read(path))
                row['observed_markers'][marker] = {k: parsed[k] for k in
                    ('event','epochs','best_epoch','best_val','steps','parameters','reason') if k in parsed}
        rows.append(row)
    source_names = ('models_ea.py','train_ea.py') if name == CAMPAIGNS[0] else ('model_factory.py','objective.py','trainer.py')
    for filename in source_names:
        read(directory / filename)

for relative in (
    'experiments/qm9s_full_EA_20260925/upstream/models.py',
    'experiments/qm9s_eta_Ef_20260926/frozen_reference/upstream/models.py',
    'experiments/qm9s_full_EA_20260925/upstream/vendor/detanet_model/detanet.py',
    'experiments/qm9s_full_EA_20260925/upstream/vendor/detanet_model/modules/message.py',
    'experiments/qm9s_full_EA_20260925/upstream/vendor/detanet_model/modules/update.py',
    'experiments/qm9s_full_EA_20260925/upstream/vendor/detanet_model/modules/edge_attention.py',
    'experiments/qm9s_full_EA_20260925/reports/selection_before_test.json',
    'research/oscillator_r2_20260928/chan64_campaign_completion/CAMPAIGN_REPORT.md',
    'research/oscillator_r2_20260928/chan64_campaign_completion/CAMPAIGN_AUDIT.json',
    'research/oscillator_r2_20260928/scratch_readout_comparison/config.json',
    'research/oscillator_r2_20260928/scratch_readout_comparison/PROTOCOL.md',
    'research/oscillator_r2_20260928/scratch_readout_comparison/completion_receipts/PAIRED_COMPLETION_REPORT.md',
    'research/oscillator_r2_20260928/eta0_seed_replication/PROTOCOL.md',
    'research/oscillator_r2_20260928/eta0_seed_replication/completion_receipts/SEED_FAMILY_SCIENTIFIC_CLOSEOUT.md',
    'research/single_model_20260929/round06_objective_preparation/completion/CONTROL_REPEAT_CONTEXT.md',
    'research/single_model_20260929/round07_congruence_preparation/completion/ROUND07_RESULTS.json',
):
    pin(relative)

result = {
    'format': 'read_only_historical_recipe_audit_v1',
    'scope': 'Fixed five-campaign config/history ledger, named code and published aggregate reports only.',
    'not_performed': ['array decoding','checkpoint loading','model construction or inference','new fitting or scoring','test-target access'],
    'config_count': len(rows), 'records': rows,
    'all_inspected_config_weight_decay_zero': all(r['config'].get('weight_decay') == 0 for r in rows),
    'source_and_input_hashes': hashes,
    'reader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'limitations': [
        'Bounded inventory, not a repository-wide proof of absent trials.',
        'History losses retain their original campaign definitions; they are not new predictive metrics.',
        'Missing completion markers in this exact list do not establish whether a historical worker is active.',
        'No historical test value was extracted or recomputed by this reader.',
    ],
}
OUT.mkdir(exist_ok=True)
target = OUT / 'HISTORICAL_RECIPE_EVIDENCE.json'
assert not target.exists(), 'Preserve existing evidence; do not overwrite or repeat.'
target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'path':str(target), 'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                  'configs':len(rows), 'input_files':len(hashes)}))
