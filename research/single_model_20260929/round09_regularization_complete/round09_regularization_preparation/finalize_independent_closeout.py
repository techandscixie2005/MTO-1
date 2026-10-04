"""Bind reviewed lightweight terminal records; no scientific payload access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'completion'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_bytes())


def main():
    target = OUT/'INDEPENDENT_CLOSEOUT_MANIFEST.json'
    assert not target.exists()
    tp = OUT/'TERMINAL_MANIFEST.json'
    assert sha(tp) == '55ddb3957d4b1b11ea4cc10bf21d1ae6edfd3df52f74454893aae14e78cdb9bf'
    terminal = read(tp)
    assert terminal['count'] == len(terminal['files']) == 44
    assert terminal['bytes'] == sum(v['bytes'] for v in terminal['files']) == 3706156
    for entry in terminal['files']:
        p = Path(entry['path'])
        assert p == ROOT/entry['relative_path'] and p.resolve().is_relative_to(ROOT)
        assert not p.is_symlink() and p.suffix.lower() in {'.json', '.jsonl', '.md', '.py', '.xml', '.log', '.svg'}
        assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256']
        p.read_text(encoding='utf-8')
    review = read(OUT/'INDEPENDENT_TERMINAL_REVIEW.json')
    assert review['passed'] and not review['remaining_scientific_blockers']
    for p, h in review['reviewed_input_hashes'].items():
        assert sha(p) == h
    decision = ROOT.parent/'current_state/ROUND09_DECISION.md'
    assert sha(decision) == '8e09e19499c221c4d1cc03527ab4459e6bd8af6a29ee5090b2220e0b19515f7e'
    names = ['TERMINAL_ANALYSIS_SOURCE_REVIEW.json', 'independent_terminal_metadata.py',
             'independent_terminal_results.py', 'finalize_independent_closeout.py',
             'ops/independent_terminal_metadata_01.log', 'ops/independent_terminal_results_01.log',
             'completion/INDEPENDENT_TERMINAL_METADATA.json', 'completion/INDEPENDENT_RESULT_CHECKS.json',
             'completion/INDEPENDENT_SCIENTIFIC_REVIEW.md', 'completion/INDEPENDENT_TERMINAL_REVIEW.json',
             'completion/FINAL_DOCUMENT_REVIEW.md', 'completion/ROUND09_REPORT.md',
             'completion/ROUND09_REPRODUCE.md', 'completion/TERMINAL_MANIFEST.json']
    paths = [ROOT/n for n in names]+[decision]
    result = {'passed': True, 'phase': 'completed_round09_publication_binding_only',
        'preparation_parent_commit': '566aa5e02c2f7cad6dc8c7b8c732ad45dfb9c8be',
        'source_manifest_sha256': sha(ROOT/'FROZEN_MANIFEST.json'),
        'terminal_manifest_sha256': sha(tp), 'independent_review_sha256': sha(OUT/'INDEPENDENT_TERMINAL_REVIEW.json'),
        'root_decision_sha256': sha(decision), 'files': {str(p): sha(p) for p in paths},
        'completed_numerical_stages_repeated': False, 'new_scientific_execution_authorized': False}
    target.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'passed': True, 'members': len(paths), 'sha256': sha(target)}))


if __name__ == '__main__':
    main()
