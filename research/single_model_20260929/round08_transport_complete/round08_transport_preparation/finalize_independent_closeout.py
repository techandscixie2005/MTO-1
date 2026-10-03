"""Bind completed reviews and explicit lightweight records; no model/data decoding."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation')
C = ROOT / 'completion'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def write(p, value):
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')

pins = {'TERMINAL_MANIFEST.json': 'b5db6f2ec8dc11226f44f420f435766f60503a97f4f4cf4260af63bff127e76a', 'ROUND08_REPORT.md': '14a2425b492c01af8e9d0756a47ede7dbc8e68254117f0752d14f30d24057e1b', 'ROUND08_REPRODUCE.md': 'dc215273a4fda7f083ee3a01a34d57b1539de9f6beb60bb255b1d381f037bee9', 'NEXT_RESEARCH_QUESTION.md': '06ab4c24577e178ca374377361058295fbedd0e7fffca9982d8862227bf0d191', 'INDEPENDENT_TERMINAL_METADATA.json': '55dae35d28f63a53468d04d177979c718433fb30e0bb05525c6638d98a0316d8', 'INDEPENDENT_RESULT_CHECKS.json': '496686c4dccbffe768b9913eb360b038b03fe23e51821bffb9950ac216df2e33', 'INDEPENDENT_SCIENTIFIC_REVIEW.md': '9d685394e449b45560bb86857f762d2353daeea0d395feb6652c31fc13280ab9', 'ROUND08_RESULTS.json': '5ba356f7e37def370d09492855378b580cf5fa5e66afb4db990d7b3a5cc0dd58', 'ANALYSIS_RECEIPT.json': '8451c3a05efb9b149a8e5403f389607dc125094ef79f37ca57617e5736ab727d'}

for name, digest in pins.items():
    assert sha(C / name) == digest, name
tm = read(C / 'TERMINAL_MANIFEST.json')
assert tm['count'] == 52 and tm['bytes'] == 5704700
assert len(tm['files']) == len({e['path'] for e in tm['files']}) == 52
for e in tm['files']:
    p = Path(e['path'])
    assert p.is_relative_to(ROOT) and not p.is_symlink()
    assert p.suffix in {'.json', '.jsonl', '.py', '.md', '.log', '.xml', '.svg'}
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256']
    p.read_text(encoding='utf-8')
assert sum(e['bytes'] for e in tm['files']) == tm['bytes']
decision = ROOT.parent / 'current_state/ROUND08_DECISION.md'
assert sha(decision) == '1099239c16237f85c185b02ebbb24b2c41f7812a10e4c018d5c35a5a68d3028d'
review = {
    'passed': True, 'phase': 'completed_round08_independent_scientific_review',
    'blocking_findings': [],
    'source_manifest_sha256': tm['frozen_scientific_manifest_sha256'],
    'frozen_manifest_sha256': tm['frozen_scientific_manifest_sha256'],
    'terminal_manifest_sha256': sha(C / 'TERMINAL_MANIFEST.json'),
    'terminal_integrity_sha256': pins['INDEPENDENT_TERMINAL_METADATA.json'],
    'independent_result_checks_sha256': pins['INDEPENDENT_RESULT_CHECKS.json'],
    'scientific_review_sha256': pins['INDEPENDENT_SCIENTIFIC_REVIEW.md'],
    'analysis_receipt_sha256': pins['ANALYSIS_RECEIPT.json'],
    'analysis_result_sha256': pins['ROUND08_RESULTS.json'],
    'analysis_source_review_sha256': sha(ROOT / 'TERMINAL_ANALYSIS_SOURCE_REVIEW.json'),
    'final_document_review_sha256': sha(C / 'FINAL_DOCUMENT_REVIEW.md'),
    'root_decision_sha256': sha(decision),
    'reviewed_final_documents': {k: pins[k] for k in ['ROUND08_REPORT.md', 'ROUND08_REPRODUCE.md', 'NEXT_RESEARCH_QUESTION.md']},
    'explicit_terminal_files_rehashed': 52,
    'completed_epochs_per_arm': 60, 'steps_per_arm': 112860,
    'valid_validation_labels': 66860, 'validation_molecules': 6686,
    'triple_point003_gate_passed': False,
    'selected_neighbor_r2': 0.44396835681637303,
    'selected_local_r2': 0.4339236136927195,
    'selected_original_r2': 0.4176361697323918,
    'neighbor_delta_vs_original': 0.026332187083981218,
    'neighbor_delta_vs_local': 0.010044743123653554,
    'neighbor_delta_vs_retained_reference': -0.003201044549479004,
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
    'recommendation': 'Close the fixed direct tensor-transport study; retain Round05 control45 and report neighbor gains over current controls with the failed retained-reference gate. Publish D-first; afterward root permits one regularization proposal only.',
}
result = read(C / 'ROUND08_RESULTS.json')
for arm in ('original','local','neighbor'):
    assert review['selected_'+arm+'_r2']==result['arms'][arm]['selected_best']['pooled']['r2']
assert review['neighbor_delta_vs_original']==result['gate']['candidate_delta_r2_vs_contemporaneous_original']
assert review['neighbor_delta_vs_local']==result['gate']['candidate_delta_r2_vs_contemporaneous_local']
assert review['neighbor_delta_vs_retained_reference']==result['gate']['candidate_delta_r2_vs_retained_reference']
review_path = C / 'INDEPENDENT_TERMINAL_REVIEW.json'
write(review_path, review)
members = [
    decision, review_path, C / 'FINAL_DOCUMENT_REVIEW.md',
    C / 'INDEPENDENT_SCIENTIFIC_REVIEW.md', C / 'INDEPENDENT_TERMINAL_METADATA.json',
    C / 'INDEPENDENT_RESULT_CHECKS.json', 
    C / 'ROUND08_REPORT.md', C / 'ROUND08_REPRODUCE.md', C / 'NEXT_RESEARCH_QUESTION.md',
    C / 'TERMINAL_MANIFEST.json', ROOT / 'TERMINAL_ANALYSIS_SOURCE_REVIEW.json',
    ROOT / 'independent_terminal_metadata.py', ROOT / 'independent_terminal_results.py',
    ROOT / 'finalize_independent_closeout.py',
    ROOT / 'ops/independent_terminal_metadata_01.log', ROOT / 'ops/independent_terminal_results_01.log',
]
manifest = {
    'passed': True, 'phase': 'completed_round08_publication_binding_only',
    'preparation_parent_commit': 'a10eb9a35b26014f00138c61476b33e9070bb69b',
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
