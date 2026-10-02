"""Bind completed reviews and explicit lightweight records; no model/data decoding."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation')
C = ROOT / 'completion'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def write(p, value):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')

pins = {'TERMINAL_MANIFEST.json': '2ff6f4277942d4af4820bdfd6c7e6f4bb018049c4744338f66bab22983c1b109', 'ROUND07_REPORT.md': '6c3ee0be107b468bdb6ce4fca8e35a286065e23e9e1d7ee60303d5413e1bd908', 'ROUND07_REPRODUCE.md': '4fec24c30af37a5efd5e954785858b208706c2c7d64fa82892eb4ef21f761619', 'NEXT_RESEARCH_QUESTION.md': 'bbbe859fa7b075956c8eb0e93336a1f004a16c344b0dd21cdceab9d929ac1bc1', 'INDEPENDENT_TERMINAL_METADATA.json': 'b827a9749e70d302139aff296de22bb2d5624525c651b3c2d2af5ad39d2fe129', 'INDEPENDENT_RESULT_CHECKS.json': 'e4b663bbdab5377ce31ecaad0301498e87571413352e9d796afb63ff483f4f98', 'INDEPENDENT_SCIENTIFIC_REVIEW.md': 'e5eeefb45d78f903d13b7f90ae04b4b00f5992ac26fb18ff3337f94939256bf9', 'ROUND07_RESULTS.json': '0314d0252de7e2f93e5ecd3df1be5a3f675fab540c8fea658b4b3a5231f21209', 'ANALYSIS_RECEIPT.json': '62f584dacafe9f0599f95f40c6d8d706f47a89a83c7011330378c509efb0d851'}

for name, digest in pins.items():
    assert sha(C / name) == digest, name
tm = read(C / 'TERMINAL_MANIFEST.json')
assert tm['count'] == 49 and tm['bytes'] == 2357263
assert len(tm['files']) == len({e['path'] for e in tm['files']}) == 49
for e in tm['files']:
    p = Path(e['path'])
    assert p.is_relative_to(ROOT) and not p.is_symlink()
    assert p.suffix in {'.json', '.jsonl', '.py', '.md', '.log', '.xml', '.svg'}
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    p.read_text(encoding='utf-8')
assert sum(e['bytes'] for e in tm['files']) == tm['bytes']
decision = ROOT.parent / 'current_state/ROUND07_DECISION.md'
assert sha(decision) == 'df59af5856490c30e035170548b31e5e0b3c631908985d915f6b9ede5f69453b'
review = {
    'passed': True, 'phase': 'completed_round07_independent_scientific_review',
    'blocking_findings': [],
    'source_manifest_sha256': tm['frozen_scientific_manifest_sha256'],
    'frozen_manifest_sha256': tm['frozen_scientific_manifest_sha256'],
    'terminal_manifest_sha256': sha(C / 'TERMINAL_MANIFEST.json'),
    'terminal_integrity_sha256': pins['INDEPENDENT_TERMINAL_METADATA.json'],
    'independent_result_checks_sha256': pins['INDEPENDENT_RESULT_CHECKS.json'],
    'scientific_review_sha256': pins['INDEPENDENT_SCIENTIFIC_REVIEW.md'],
    'analysis_receipt_sha256': pins['ANALYSIS_RECEIPT.json'],
    'analysis_result_sha256': pins['ROUND07_RESULTS.json'],
    'analysis_source_review_sha256': sha(ROOT / 'TERMINAL_ANALYSIS_SOURCE_REVIEW.json'),
    'final_document_review_sha256': sha(C / 'FINAL_DOCUMENT_REVIEW.md'),
    'root_decision_sha256': sha(decision),
    'reviewed_final_documents': {k: pins[k] for k in ['ROUND07_REPORT.md', 'ROUND07_REPRODUCE.md', 'NEXT_RESEARCH_QUESTION.md']},
    'explicit_terminal_files_rehashed': 49,
    'completed_epochs_per_arm': 60, 'steps_per_arm': 112860,
    'valid_validation_labels': 66860, 'validation_molecules': 6686,
    'triple_point003_gate_passed': False,
    'selected_tensor_r2': 0.4187551765560742,
    'selected_scalar_r2': 0.42933901263851915,
    'selected_original_r2': 0.4091040812638763,
    'tensor_delta_vs_original': 0.009651095292197875,
    'tensor_delta_vs_scalar': -0.010583836082444953,
    'tensor_delta_vs_retained_reference': -0.028414224809777844,
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
    'recommendation': 'Close the fixed congruence study; retain Round05 control45 and report scalar as strongest contemporary control. Publish D-first; afterward root permits source/aggregate bottleneck audit and proposal only.',
}
review_path = C / 'INDEPENDENT_TERMINAL_REVIEW.json'
write(review_path, review)
members = [
    decision, review_path, C / 'FINAL_DOCUMENT_REVIEW.md',
    C / 'INDEPENDENT_SCIENTIFIC_REVIEW.md', C / 'INDEPENDENT_TERMINAL_METADATA.json',
    C / 'INDEPENDENT_RESULT_CHECKS.json', 
    C / 'ROUND07_REPORT.md', C / 'ROUND07_REPRODUCE.md', C / 'NEXT_RESEARCH_QUESTION.md',
    C / 'TERMINAL_MANIFEST.json', ROOT / 'TERMINAL_ANALYSIS_SOURCE_REVIEW.json',
    ROOT / 'independent_terminal_metadata.py', ROOT / 'independent_terminal_results.py',
    ROOT / 'finalize_independent_closeout.py',
    ROOT / 'ops/independent_terminal_metadata_01.log', ROOT / 'ops/INDEPENDENT_RESULT_CHECKS_01.log',
]
manifest = {
    'passed': True, 'phase': 'completed_round07_publication_binding_only',
    'preparation_parent_commit': '271d61f6dc9e21009c31d10467fe704e44b974ad',
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
