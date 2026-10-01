"""Bind completed reviews and explicit lightweight records; no model/data decoding."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation')
C = ROOT / 'completion'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def write(p, value):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')

pins = {
    'TERMINAL_MANIFEST.json': 'e8b9f9f0f6d270423791710c1de9f77c9732bac1bf31c3df26afd8e6a4897469',
    'ROUND06_REPORT.md': 'bfcac33af345f897a4336758cd522c81b0b47267edc33c0799fd7a2863aa2ef9',
    'ROUND06_REPRODUCE.md': '04973c2ebe7912411814b464d36793d2e50dd17d7ef7fffef5f972bf281285a4',
    'DEFERRED_CONGRUENCE_ASSESSMENT.md': 'f5b038b27478352958cdfd8fcc6d5e490bb44c1d6b1966d9977e62d3754b5ab3',
    'INDEPENDENT_TERMINAL_METADATA.json': '91d3b8f9256373d4c077b10627fceb02f69704189f6cc2d808eeb7d068034c35',
    'INDEPENDENT_RESULT_CHECKS.json': '9a6242fccc372734d18e8605a5f437b4e49600165099659dba907024ea312a91',
    'INDEPENDENT_SCIENTIFIC_REVIEW.md': '5844daaca4f2117dd3a013f6125d625921381713046fc83aaf7e0a11862bb20d',
    'CONTROL_REPEAT_CONTEXT.md': 'f49ae5790f12837f91ce85a02b8439a550168ff805c0e78e573f7185b2a07abe',
    'ROUND06_RESULTS.json': 'e82562e570caa25f53cd2f7ce5b7ad1324a4270129ff9d8286d1a454beeb840c',
    'ANALYSIS_RECEIPT.json': '97fbf0625a5734021d27df57fd14565107ca6084340fcf18677ec04809125413',
}
for name, digest in pins.items():
    assert sha(C / name) == digest, name
tm = read(C / 'TERMINAL_MANIFEST.json')
assert tm['count'] == 39 and tm['bytes'] == 1276685
assert len(tm['files']) == len({e['path'] for e in tm['files']}) == 39
for e in tm['files']:
    p = Path(e['path'])
    assert p.is_relative_to(ROOT) and not p.is_symlink()
    assert p.suffix in {'.json', '.jsonl', '.py', '.md', '.log', '.xml', '.svg'}
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    p.read_text(encoding='utf-8')
assert sum(e['bytes'] for e in tm['files']) == tm['bytes']
decision = ROOT.parent / 'current_state/ROUND06_DECISION.md'
assert sha(decision) == '8f1b4e6de20024c904c90f551a677b74ac56d5b213581c07672a979b0fe03a39'
review = {
    'passed': True, 'phase': 'completed_round06_independent_scientific_review',
    'blocking_findings': [],
    'source_manifest_sha256': tm['frozen_scientific_manifest_sha256'],
    'frozen_manifest_sha256': tm['frozen_scientific_manifest_sha256'],
    'terminal_manifest_sha256': sha(C / 'TERMINAL_MANIFEST.json'),
    'terminal_integrity_sha256': pins['INDEPENDENT_TERMINAL_METADATA.json'],
    'independent_result_checks_sha256': pins['INDEPENDENT_RESULT_CHECKS.json'],
    'scientific_review_sha256': pins['INDEPENDENT_SCIENTIFIC_REVIEW.md'],
    'analysis_receipt_sha256': pins['ANALYSIS_RECEIPT.json'],
    'analysis_result_sha256': pins['ROUND06_RESULTS.json'],
    'analysis_source_review_sha256': sha(ROOT / 'TERMINAL_ANALYSIS_SOURCE_REVIEW.json'),
    'final_document_review_sha256': sha(C / 'FINAL_DOCUMENT_REVIEW.md'),
    'root_decision_sha256': sha(decision),
    'reviewed_final_documents': {k: pins[k] for k in ['ROUND06_REPORT.md', 'ROUND06_REPRODUCE.md', 'DEFERRED_CONGRUENCE_ASSESSMENT.md']},
    'explicit_terminal_files_rehashed': 39,
    'completed_epochs_per_arm': 60, 'steps_per_arm': 112860,
    'valid_validation_labels': 66860, 'validation_molecules': 6686,
    'dual_point003_gate_passed': False,
    'selected_raw_f_r2': 0.4220281674141795,
    'selected_delta_vs_control': 0.017230600141614705,
    'selected_delta_vs_retained_reference': -0.02514123395167256,
    'reference_selected_epoch': 45,
    'reference_validation_pooled_raw_f_r2': 0.44716940136585204,
    'retained_geometry_checkpoint_sha256': 'e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e',
    'test_evaluated': False, 'target_point6_achieved': False,
    'independent_seed_confirmation': False,
    'inference_rerun_by_reviewer': False, 'saved_array_reduction_repeated_by_reviewer': False,
    'private_input_hashes_are_not_archive_allowlist': True,
    'conditional_bootstrap_does_not_cover_selection_or_seed_uncertainty': True,
    'same_seed_control_repeat_cause_not_identified': True,
    'new_fit_authorized_by_this_review': False,
    'recommendation': 'Close the fixed objective pair, retain Round05 control45; publish D-first. Root allows a separate congruence proposal only after publication.',
}
review_path = C / 'INDEPENDENT_TERMINAL_REVIEW.json'
write(review_path, review)
members = [
    decision, review_path, C / 'FINAL_DOCUMENT_REVIEW.md',
    C / 'INDEPENDENT_SCIENTIFIC_REVIEW.md', C / 'INDEPENDENT_TERMINAL_METADATA.json',
    C / 'INDEPENDENT_RESULT_CHECKS.json', C / 'CONTROL_REPEAT_CONTEXT.md',
    C / 'ROUND06_REPORT.md', C / 'ROUND06_REPRODUCE.md', C / 'DEFERRED_CONGRUENCE_ASSESSMENT.md',
    C / 'TERMINAL_MANIFEST.json', ROOT / 'TERMINAL_ANALYSIS_SOURCE_REVIEW.json',
    ROOT / 'independent_terminal_metadata.py', ROOT / 'independent_terminal_results.py',
    ROOT / 'finalize_independent_closeout.py',
    ROOT / 'ops/INDEPENDENT_TERMINAL_METADATA_01.log', ROOT / 'ops/INDEPENDENT_RESULT_CHECKS_01.log',
]
manifest = {
    'passed': True, 'phase': 'completed_round06_publication_binding_only',
    'preparation_parent_commit': 'c26a860b972a1c01267f8b766393772173515b92',
    'source_manifest_sha256': review['source_manifest_sha256'],
    'terminal_manifest_sha256': review['terminal_manifest_sha256'],
    'independent_review_sha256': sha(review_path), 'root_decision_sha256': sha(decision),
    'files': {str(p): sha(p) for p in members},
    'completed_numerical_stages_repeated': False,
    'new_scientific_execution_authorized': False,
}
manifest_path = C / 'INDEPENDENT_CLOSEOUT_MANIFEST.json'
write(manifest_path, manifest)
print(json.dumps({'review_sha256': sha(review_path), 'manifest_sha256': sha(manifest_path), 'files': len(members)}))
