"""Review the exact two-helper durability amendment without model/data access."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mapping_path = ROOT / 'IO_CONTINUITY.json'
    mapping = json.loads(mapping_path.read_text())
    old = ROOT / 'preflight_history/io_continuity/training_executed.py'
    new = ROOT / 'training.py'
    assert sha(old) == mapping['executed_sha256'] == '17d2bb266704bb4a747d3b11b3ff0789f68e2bc5835230a45220d93a3f0bca76'
    assert sha(new) == mapping['final_sha256'] == 'c0147d25cf2a2b6cd0956a57bfcab069657618bc22712efa29feb1b6a9533cdd'
    technical = json.loads((ROOT / 'TECHNICAL_SOURCE_REVIEW_RETRY01.json').read_text())
    assert technical['source_hashes'][mapping['source_path']] == sha(old)
    trees = [ast.parse(p.read_text()) for p in (old, new)]
    assert len(trees[0].body) == len(trees[1].body)
    changed = []
    for before, after in zip(trees[0].body, trees[1].body):
        if ast.dump(before) == ast.dump(after):
            continue
        assert isinstance(before, ast.FunctionDef) and before.name == after.name
        assert before.name in ('atomic_copy', 'atomic_arrays')
        changed.append(before.name)
        target = 'target' if before.name == 'atomic_copy' else 'path'
        expected = ast.parse(f'fd=os.open({target}.parent,os.O_RDONLY)\ntry:os.fsync(fd)\nfinally:os.close(fd)\n').body
        assert [ast.dump(x) for x in after.body[-2:]] == [ast.dump(x) for x in expected]
        after.body = after.body[:-2]
        assert ast.dump(before) == ast.dump(after)
    assert set(changed) == {'atomic_copy', 'atomic_arrays'}
    compile(new.read_text(), str(new), 'exec')
    result = {
        'passed': True,
        'mapping_sha256': sha(mapping_path),
        'executed_sha256': sha(old),
        'final_sha256': sha(new),
        'changed_functions': changed,
        'all_other_ast_nodes_unchanged': True,
        'exact_changes': 'Append parent-directory open/fsync/finally-close after each existing os.replace.',
        'reviewer_source_sha256': sha(__file__),
        'additional_model_inference': False,
        'additional_optimizer_updates': 0,
        'targets_or_checkpoint_tensors_read': False,
    }
    out = ROOT / 'IO_CONTINUITY_CHECKS.json'
    assert not out.exists(), 'Do not overwrite completed reviewer checks'
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
