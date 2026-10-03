"""Reviewer's synthetic execution-gate checks. No model/data imports or fits."""
import builtins, copy, json, os, sys, tempfile
from pathlib import Path
from unittest.mock import patch
import execution_gate as gate
from common import ROOT, sha, immutable_json

def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True)+'\n')

def rejects(call):
    try:
        call()
    except (AssertionError, KeyError, FileNotFoundError):
        return
    raise AssertionError('Expected gate rejection')

def fixture(root):
    (root/'source.txt').write_text('synthetic source, no data\n')
    manifest = {'source_hashes': {str(root/'source.txt'): sha(root/'source.txt')},
                'split_manifest_sha256': gate.PINS['SPLIT_MANIFEST.json'],
                'independent_split_verification_sha256': gate.PINS['INDEPENDENT_SPLIT_VERIFICATION.json']}
    write(root/'FROZEN_MANIFEST.json', manifest); mh = sha(root/'FROZEN_MANIFEST.json')
    review = {'passed': True, 'frozen_manifest_sha256': mh, 'source_hashes': manifest['source_hashes']}
    write(root/'INDEPENDENT_PREPARATION_REVIEW.json', review); rh = sha(root/'INDEPENDENT_PREPARATION_REVIEW.json')
    publication = {'remote_verified': True, 'download_before_stage_before_commit_push': True,
                   'frozen_manifest_sha256': mh, 'independent_review_sha256': rh}
    write(root/'publication.json', publication)
    auth = {'authorized': True, 'scope': 'round08_three_arm_60epoch_fit', 'frozen_manifest_sha256': mh,
            'split_manifest_sha256': manifest['split_manifest_sha256'],
            'independent_split_verification_sha256': manifest['independent_split_verification_sha256'],
            'epochs': 60, 'arms': ['original','local','neighbor'], 'test_access': False,
            'historical_weights': False, 'independent_review_sha256': rh,
            'publication_receipt': str(root/'publication.json'), 'publication_receipt_sha256': sha(root/'publication.json')}
    write(root/'authorization.json', auth)
    return auth

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (ROOT/'INDEPENDENT_GATE_CHECKS.json').exists(), 'Preserve completed reviewer checks'
    checks = []
    with tempfile.TemporaryDirectory(prefix='round08_reviewer_gate_') as temporary:
        root = Path(temporary)
        with patch.object(gate, 'ROOT', root):
            auth = fixture(root)
            assert gate.verify(root/'authorization.json')['manifest_sha256'] == auth['frozen_manifest_sha256']
            checks.append('valid_synthetic_binding_passes')
            rejects(lambda: gate.verify(root/'missing.json')); checks.append('missing_authority_rejected')
            mutations = {'authorized': False, 'scope': 'preparation_only', 'frozen_manifest_sha256': '0'*64,
                         'split_manifest_sha256': '0'*64, 'independent_split_verification_sha256': '0'*64,
                         'epochs': 100, 'arms': ['original'], 'test_access': True, 'historical_weights': True,
                         'independent_review_sha256': '0'*64, 'publication_receipt_sha256': '0'*64}
            for key, value in mutations.items():
                invalid = copy.deepcopy(auth); invalid[key] = value; write(root/'authorization.json', invalid)
                rejects(lambda: gate.verify(root/'authorization.json')); checks.append('reject_authority_'+key)
            invalid = copy.deepcopy(auth); invalid['arms'] = ['neighbor','local','original']
            write(root/'authorization.json', invalid)
            rejects(lambda: gate.verify(root/'authorization.json')); checks.append('reject_reversed_arm_order')
            invalid = copy.deepcopy(auth); invalid['scope'] = 'round07_three_arm_60epoch_fit'
            write(root/'authorization.json', invalid)
            rejects(lambda: gate.verify(root/'authorization.json')); checks.append('reject_previous_round_scope')
            for key, value in [('remote_verified',False),('download_before_stage_before_commit_push',False),
                               ('frozen_manifest_sha256','0'*64),('independent_review_sha256','0'*64)]:
                auth = fixture(root); publication = json.loads((root/'publication.json').read_text())
                publication[key] = value; write(root/'publication.json', publication)
                auth['publication_receipt_sha256'] = sha(root/'publication.json'); write(root/'authorization.json',auth)
                rejects(lambda: gate.verify(root/'authorization.json')); checks.append('reject_publication_'+key)
            auth = fixture(root); (root/'source.txt').write_text('changed synthetic source\n')
            rejects(lambda: gate.verify(root/'authorization.json')); checks.append('changed_source_rejected')
            auth = fixture(root)
            review = json.loads((root/'INDEPENDENT_PREPARATION_REVIEW.json').read_text()); review['source_hashes'] = {}
            write(root/'INDEPENDENT_PREPARATION_REVIEW.json',review)
            auth['independent_review_sha256'] = sha(root/'INDEPENDENT_PREPARATION_REVIEW.json'); write(root/'authorization.json',auth)
            rejects(lambda: gate.verify(root/'authorization.json')); checks.append('review_closure_mismatch_rejected')
            # Missing authority must fail before any scientific import or run directory.
            import train
            original_import = builtins.__import__
            def guard(name,*args,**kwargs):
                assert name.split('.')[0] not in {'torch','training','partition_data','fresh_model','transport','runtime','metrics','predictor'}, 'Scientific import before gate'
                return original_import(name,*args,**kwargs)
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            with patch.object(sys,'argv',['train.py','--arm','original','--authorization',str(root/'absent.json')]), patch.object(builtins,'__import__',side_effect=guard):
                rejects(train.main)
            assert before == sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            checks.append('entry_rejects_before_scientific_imports_or_run_artifacts')
    output = {'passed': True, 'synthetic_only': True, 'checks': checks, 'checks_count': len(checks),
              'real_data_or_model_access': False, 'production_execution': False, 'optimizer_updates': 0,
              'source_hashes': {name:sha(ROOT/name) for name in ('common.py','execution_gate.py','train.py','independent_gate_checks.py')}}
    immutable_json(output, ROOT/'INDEPENDENT_GATE_CHECKS.json'); print(json.dumps(output,indent=2))

if __name__=='__main__':
    main()
