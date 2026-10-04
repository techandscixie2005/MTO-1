"""Persist final independent review from already checked lightweight records."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def write_new(name, data):
    with (ROOT / name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(data, indent=2, sort_keys=True) + '\n')

m = read(ROOT / 'FROZEN_MANIFEST.json')
checks = read(ROOT / 'INDEPENDENT_FINAL_PREPARATION_CHECKS.json')
assert checks['passed'] and checks['source_hashes'] == m['source_hashes']
assert sha(ROOT / 'FROZEN_MANIFEST.json') == checks['frozen_manifest_sha256']
assert sha(ROOT / 'INDEPENDENT_FINAL_PREPARATION_CHECKS.json') == '4e4660b88b22dcd58c56ec8f87cda6dbd77359ccd25cddd9d8ee8a98739050d2'
for p, h in checks['input_hashes'].items(): assert sha(p) == h
assert not (ROOT / 'INDEPENDENT_PREPARATION_REVIEW.json').exists()
assert not (ROOT / 'INDEPENDENT_PREPARATION_REVIEW_MANIFEST.json').exists()

narrative = '''# Round10 independent preparation review

**PASS for preparation only.** The final review binds all 458 paths in source manifest `7b4bed854c54eb9a1faabcb817b101c4bafe0f80869d7102242f1746b0c2aa2f`. All 360 inherited sources, 60 matched order hashes and 424 paths reviewed before the GPU fixture remain unchanged. No preparation blocker remains. This review grants no production authority or accuracy claim.

## Scientific and implementation contract

Both arms retain original PSD MTO, LE+Ls, zero weight decay, fresh seed/order 11, TRAIN-only statistics and one ordered group of 135 original trainable tensors. The 11 dormant right-F/transport tensors remain excluded and unchanged. The full roster matches the unwrapped original and prior roster.

The control uses LR .001 for epochs 1–60. The candidate uses .001 through epoch 30 and .0003 for epochs 31–60. Absolute assignment occurs before each epoch's first minibatch. It does not reset moments/counters or use validation feedback. The production boundary is updates 56,430→56,431. Checkpoints bind complete tables, used/next LR and global update ranges; interrupted epoch 31 resumes from committed epoch 30 with preserved Adam/RNG/order. The original objective, optimizer defaults, clipping and data boundaries are preserved.

The production entry rejects missing or mismatched authority before scientific imports. The 23-case reviewer fixture exercised scope, arm, epoch, source, review and publication bindings. The complete future runner remains behind separate authorization and resource gates.

## Exercised evidence

- CPU: one execution, seven toy optimizer calls, zero full-model updates. Manual persistent AMSGrad parameters/moments/maxima agree within 1.11e-16; toy replay is exact. All 60 rate positions and 68 invalid metadata cases were checked. Three fresh constructors matched initialization, ordered roster and constructor RNG. Synthetic loss/gradient, masking, symmetry and one-checkpoint export/access checks passed fixed criteria.
- GPU: one admitted stage on physical GPU1, first eligible in the ordered 1/2/4/6 pass. The wrapper retained its lock, prebound the UUID and registered the child before computation. Source, identity, environment, registration and exit-0 evidence passed independent metadata review.
- Exactly six discarded updates used the first 128 TRAIN rows, two batches per arm, with zero validation/TEST rows. Schedule labels 30/31/31 correspond to actual Adam steps 1/2/2. They do not represent 30 trained epochs. Control rates stayed .001; candidate rates were .001/.0003/.0003. LR assignment preserved moments.
- Maximum replay differences were 1.1920928955078125e-7 for model tensors, 1.4901161193847656e-8 for Adam and zero for loss. Fixed tolerances passed; RNG/order were exact. All 135 optimizer states and frozen parameters/buffers passed. Each trained fixture and its own geometry export passed numerical E/A parity; native-f differences were reported from those outputs.

All six fixture updates clipped. Identical pre-update losses precede the second-step LR effect and do not measure efficacy. Neither observation changed settings. These fixtures do not certify a full CUDA trajectory or full-run memory requirements.

## Preserved corrections and review limits

The initial reviewer selector mock assumed an integer descriptor although `flock` received a file object. It stopped before its mocked probe. The corrected five-case stdlib fixture passed without live GPU probes or model/data access; the original source, log and correction remain preserved.

The first metadata freeze failed before manifest creation because a loop's `digest` string shadowed the imported schedule-hash function. The pre-seal source review missed this. The exact original source/log/failed receipt/lineage were preserved. Independent text/AST review accepted only the import alias and two call-site changes, verified all executed technical pins unchanged and permitted one distinct metadata retry under the existing preparation decision. That retry completed with exit 0. The successful receipt is `ops/FREEZE_RECEIPT_02.json`; the failed `ops/FREEZE_RECEIPT.json` remains unchanged. No numerical stage was repeated. Both attempts and the repair are in the distinct preparation supplement.

The reviewer inspected source, aggregate receipts, roster metadata and opaque checkpoint hashes. There was no model reconstruction, inference, target/prediction-array decoding, repeated optimizer test or new live resource probe. The final check executed once and verified source/receipt bindings only. Production remains unexecuted.

## Interpretation and remaining gates

Selection remains earliest minimum native validation SSE over epochs 0–60, with fixed-60 reporting separately and all 66,860 labels including zeros. The candidate screen requires +.003 over both its contemporary selected control and retained Round05 R² .44716940136585204. A selected epoch 0–30 candidate has not experienced the intervention; its gain alone cannot substantiate a schedule benefit or justify confirmation seeds. Prefix/post-drop summaries are descriptive.

This compares the full lower-late-LR recipe; cumulative LR also differs. It does not identify an optimal boundary or explain earlier validation decline. Conditional reused-validation intervals omit seed uncertainty. The v2 TEST partition stays sealed and is historically exposed data, not new external confirmation. The .60 target remains unmet.

Root acceptance, verified D-first publication, distinct exact production authorization and fresh resource admission remain. No extra seed, boundary search, extension, averaging or TEST access is authorized. The original 15-file proposal closure is preserved.
'''
(ROOT / 'INDEPENDENT_PREPARATION_REVIEW.md').write_text(narrative, encoding='utf-8', newline='\n')

members = ['FROZEN_MANIFEST.json', 'INDEPENDENT_FINAL_PREPARATION_CHECKS.json',
    'INDEPENDENT_FINAL_PREPARATION_CHECKS_01.log', 'independent_final_preparation_checks.py',
    'INDEPENDENT_PREPARATION_REVIEW.md', 'PREPARATION_REPORT.md', 'PROTOCOL.md', 'RUNNER_HANDOFF.md',
    'SOURCE_LINEAGE.json', 'METADATA_REPAIR_REVIEW.json', 'independent_metadata_repair.py',
    'INDEPENDENT_METADATA_REPAIR_01.log', 'freeze_attempt01_preserved/MANIFEST.json',
    'ops/FREEZE_RECEIPT.json', 'ops/FREEZE_01.log', 'ops/FREEZE_RECEIPT_02.json', 'ops/FREEZE_02.log',
    Path(__file__).name]
members += [str(Path(r['snapshot_path']).relative_to(ROOT)) for r in read(ROOT / 'freeze_attempt01_preserved/MANIFEST.json')['files']]
inputs = {str(ROOT / name): sha(ROOT / name) for name in members}
review = {'passed': True, 'phase': 'final_round10_preparation_only',
    'frozen_manifest_sha256': sha(ROOT / 'FROZEN_MANIFEST.json'), 'source_files': len(m['source_hashes']),
    'source_hashes': m['source_hashes'], 'input_hashes': inputs,
    'root_preparation_decision_sha256': sha(ROOT / 'ROUND10_PREPARATION_DECISION.md'),
    'report_sha256': sha(ROOT / 'PREPARATION_REPORT.md'),
    'review_md_sha256': sha(ROOT / 'INDEPENDENT_PREPARATION_REVIEW.md'),
    'final_metadata_checks_sha256': sha(ROOT / 'INDEPENDENT_FINAL_PREPARATION_CHECKS.json'),
    'gpu_preflight_sha256': sha(ROOT / 'GPU_PREFLIGHT.json'),
    'gpu_preflight_review_sha256': sha(ROOT / 'GPU_PREFLIGHT_REVIEW.json'),
    'metadata_repair_review_sha256': sha(ROOT / 'METADATA_REPAIR_REVIEW.json'),
    'successful_freeze_receipt_sha256': sha(ROOT / 'ops/FREEZE_RECEIPT_02.json'),
    'completed_CPU_toy_steps': 7, 'completed_full_model_discarded_updates': 6,
    'actual_preflight_device': 1, 'inherited_360_sources_and_60_orders_unchanged': True,
    'executed_424_technical_pins_unchanged': True, 'accepted_scientific_settings_and_dual_gate_unchanged': True,
    'schedule_tables_boundary_and_persistent_Adam_semantics_verified': True,
    'original_failure_and_single_reviewed_metadata_retry_preserved': True,
    'production_authorization_absent_and_no_runs_directory': True,
    'no_numerical_repeats_by_reviewer': True, 'production_authorized': False,
    'blocking_findings': [],
    'limitations': ['No validation accuracy or schedule benefit measured.',
        'Six-step boundary fixture uses actual counters 1/2, not 30 epochs of training history.',
        'Candidate changes cumulative LR; this does not isolate schedule shape or an optimal drop epoch.',
        'Selection before the LR drop cannot alone support a schedule benefit or confirmation seeds.',
        'All fixture steps clipped without retuning; GPU replay is numerical while RNG/order are exact.',
        'Historically exposed v2 repartition and reused validation; TEST stays sealed.'],
    'remaining_gates': ['Root preparation acceptance', 'Verified D-first lightweight publication',
        'Distinct exact production execution authorization and fresh resource admission']}
write_new('INDEPENDENT_PREPARATION_REVIEW.json', review)
members.append('INDEPENDENT_PREPARATION_REVIEW.json')
supplement = {'phase': 'independent_round10_preparation_review_publication_supplement',
    'source_manifest_sha256': sha(ROOT / 'FROZEN_MANIFEST.json'),
    'independent_review_sha256': sha(ROOT / 'INDEPENDENT_PREPARATION_REVIEW.json'),
    'original_proposal_manifest_preserved_sha256': sha(ROOT / 'INDEPENDENT_REVIEW_MANIFEST.json'),
    'private_arrays_or_checkpoints_included': False, 'production_authorized': False,
    'files': {str(ROOT / name): sha(ROOT / name) for name in members},
    'notes': ['All 458 sealed source paths remain unchanged; never expand opaque binary references.',
        'Includes original failed freeze, four preserved snapshot files, exact alias repair and distinct successful retry.',
        'The proposal INDEPENDENT_REVIEW_MANIFEST.json remains unchanged; this supplement uses a distinct filename.']}
write_new('INDEPENDENT_PREPARATION_REVIEW_MANIFEST.json', supplement)
print(json.dumps({'passed': True, 'members': len(members),
    'review_sha256': sha(ROOT / 'INDEPENDENT_PREPARATION_REVIEW.json'),
    'review_md_sha256': sha(ROOT / 'INDEPENDENT_PREPARATION_REVIEW.md'),
    'supplement_sha256': sha(ROOT / 'INDEPENDENT_PREPARATION_REVIEW_MANIFEST.json')}))
