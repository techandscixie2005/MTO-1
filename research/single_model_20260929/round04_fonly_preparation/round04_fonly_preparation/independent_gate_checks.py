"""Reviewer-owned synthetic authorization/recovery tests; never use production data."""
from common import require_cpu
require_cpu()
import builtins
import copy
import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import production_entry
from common import ROOT, atomic_json, sha


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def files(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}


def denied_fixture(folder, case):
    folder.mkdir()
    source = folder / 'synthetic_source.py'
    source.write_text('# Synthetic authorization fixture, no executable data access.\n')
    manifest = folder / 'synthetic_manifest.json'
    closure = {str(source): sha(source)}
    put(manifest, {'source_hashes': closure})
    mh = sha(manifest)
    review = {'passed': case != 'review_failed', 'source_manifest_sha256': mh,
              'source_hashes': closure}
    if case == 'review_manifest': review['source_manifest_sha256'] = 'wrong'
    if case == 'review_closure': review['source_hashes'] = {}
    review_path = folder / 'synthetic_review.json'; put(review_path, review)
    rh = sha(review_path)
    publication = {'remote_verified': case != 'remote_unverified',
                   'download_before_stage_before_commit_push': case != 'archive_order_false',
                   'source_manifest_sha256': mh, 'independent_review_sha256': rh}
    if case == 'publication_manifest': publication['source_manifest_sha256'] = 'wrong'
    if case == 'publication_review': publication['independent_review_sha256'] = 'wrong'
    publication_path = folder / 'synthetic_publication.json'; put(publication_path, publication)
    authority = {'phase': 'production_execution', 'real_target_fit_authorized': True,
                 'fixed_validation_authorized': True, 'test_access_authorized': False,
                 'source_manifest_sha256': mh, 'independent_review_sha256': rh,
                 'publication_receipt_sha256': sha(publication_path),
                 'maximum_real_data_solves': 2, 'maximum_validation_comparisons': 1}
    mutations = {'wrong_scope': ('phase', 'preparation_only'),
                 'fit_not_authorized': ('real_target_fit_authorized', False),
                 'validation_not_authorized': ('fixed_validation_authorized', False),
                 'test_access_requested': ('test_access_authorized', True),
                 'too_many_solves': ('maximum_real_data_solves', 3),
                 'too_many_evaluations': ('maximum_validation_comparisons', 2),
                 'authority_manifest': ('source_manifest_sha256', 'wrong'),
                 'authority_review': ('independent_review_sha256', 'wrong'),
                 'authority_publication': ('publication_receipt_sha256', 'wrong')}
    if case in mutations:
        key, value = mutations[case]; authority[key] = value
    authority_path = folder / 'synthetic_authority.json'
    if case != 'missing_authority': put(authority_path, authority)
    if case == 'changed_source': source.write_text('# Changed synthetic source.\n')
    return ['fit_export', '--authorization', str(authority_path), '--manifest', str(manifest),
            '--review', str(review_path), '--publication', str(publication_path)]


def entry_checks(temp):
    cases = ['missing_authority', 'wrong_scope', 'fit_not_authorized', 'validation_not_authorized',
             'test_access_requested', 'too_many_solves', 'too_many_evaluations',
             'authority_manifest', 'authority_review', 'authority_publication', 'review_failed',
             'review_manifest', 'review_closure', 'remote_unverified', 'archive_order_false',
             'publication_manifest', 'publication_review', 'changed_source']
    results = {}
    for case in cases:
        folder = temp / case
        argv = denied_fixture(folder, case); before = files(folder)
        imports = []; reads = []; original_import = builtins.__import__
        original_open, original_io_open = builtins.open, io.open
        def import_guard(name, *args, **kwargs):
            if name == 'production_stages':
                imports.append(name)
                raise AssertionError('Denied invocation reached substantive stage import')
            return original_import(name, *args, **kwargs)
        def open_guard(function):
            def checked(file, mode='r', *args, **kwargs):
                if any(flag in mode for flag in 'wax+'):
                    raise AssertionError('Denied invocation attempted a write')
                if isinstance(file, (str, bytes, Path)):
                    path = Path(file).resolve()
                    if folder.resolve() not in path.parents:
                        raise AssertionError('Denied invocation attempted non-synthetic file read: '+str(path))
                    reads.append(str(path.relative_to(folder)))
                return function(file, mode, *args, **kwargs)
            return checked
        with patch('builtins.__import__', import_guard), patch('builtins.open', open_guard(original_open)), patch('io.open', open_guard(original_io_open)):
            try:
                production_entry.main(argv)
            except PermissionError:
                pass
            else:
                raise AssertionError('Expected exact authorization rejection: '+case)
        assert not imports and files(folder) == before
        results[case] = {'rejected': True, 'substantive_imports': 0,
                         'non_synthetic_data_reads': 0, 'filesystem_unchanged': True}
    return results


def stage_checks(temp):
    # Imports may read ordinary installed-library constants, never trained weights/data.
    import torch
    import production_stages as stages
    import numpy as np
    permit = {'authorization_sha256': 'synthetic_authority', 'source_manifest_sha256': 'synthetic_manifest',
              'independent_review_sha256': 'synthetic_review', 'publication_receipt_sha256': 'synthetic_publication'}
    config = {'calibration_array_sha256': 'synthetic_cache_hash'}
    prior = temp / 'synthetic_round03'; (prior / 'affine').mkdir(parents=True)
    put(prior / 'affine/COEFFICIENTS_FROZEN.json', {'maps': {
        'in_sample': {'alpha': 1.0, 'beta': 0.0},
        'heldout_source': {'alpha': 1.0, 'beta': 0.0}}})
    counters = {'calibration_reads': 0, 'solver_calls': 0, 'actual_model_exports': 0}
    def forbidden(kind):
        def stop(*args, **kwargs):
            counters[kind] += 1
            raise AssertionError('Forbidden preparation test path: '+kind)
        return stop
    exports = []
    def synthetic_export(permit_value, frozen):
        assert permit_value == permit
        assert set(frozen['coefficients']) == {'in_sample', 'heldout_source'}
        exports.append(True)
    def coefficient_record(arm):
        return {'arm': arm, 'permit': permit,
                'coefficients': torch.tensor([.001, .05, -.01, .002], dtype=torch.float64),
                'calibration_array_sha256': config['calibration_array_sha256'],
                'design_diagnostics': {'rank': 4, 'fixture': 'synthetic_hand_set'},
                'fitting_diagnostics': {'fixture': 'synthetic_hand_set_no_fit'}, 'solved_at_unix': 1.0}
    out = temp / 'synthetic_output'; out.mkdir()
    for arm in stages.ARMS:
        put(out / (arm+'_SOLVE_STARTED.json'), {'fixture': 'synthetic_hand_set', 'permit': permit})
        torch.save(coefficient_record(arm), out / ('coefficients_'+arm+'.pt'))
    original_tensor_hashes = {arm: sha(out / ('coefficients_'+arm+'.pt')) for arm in stages.ARMS}
    with patch.object(stages, 'OUT', out), patch.object(stages, 'ROUND03', prior), \
         patch.object(stages, 'settings', lambda: config), \
         patch.object(stages, 'load_calibration', forbidden('calibration_reads')), \
         patch.object(stages, 'solve_fixed', forbidden('solver_calls')), \
         patch.object(stages, 'base_model', forbidden('actual_model_exports')), \
         patch.object(stages, 'export_and_replay', synthetic_export):
        stages.fit_export(permit)
        for arm in stages.ARMS:
            assert sha(out / ('coefficients_'+arm+'.pt')) == original_tensor_hashes[arm]
            assert (out / (arm+'_SOLVED.json')).is_file()
        frozen_before = files(out)
        stages.fit_export(permit)
        assert files(out) == frozen_before
        tampered_path = out / 'coefficients_in_sample.pt'
        altered = coefficient_record('in_sample'); altered['coefficients'][0] += .5
        torch.save(altered, tampered_path)
        calls_before = len(exports)
        try:
            stages.fit_export(permit)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('Frozen coefficient mutation was accepted')
        assert len(exports) == calls_before
    incomplete = temp / 'synthetic_interrupted'; incomplete.mkdir()
    put(incomplete / 'in_sample_SOLVE_STARTED.json', {'fixture': 'synthetic_interrupted'})
    incomplete_before = files(incomplete)
    with patch.object(stages, 'OUT', incomplete), patch.object(stages, 'ROUND03', prior), \
         patch.object(stages, 'settings', lambda: config), \
         patch.object(stages, 'load_calibration', forbidden('calibration_reads')), \
         patch.object(stages, 'solve_fixed', forbidden('solver_calls')):
        try:
            stages.fit_export(permit)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Interrupted solve without durable tensor was repeated')
    assert files(incomplete) == incomplete_before
    for marker in ('VALIDATION_COMPLETE.json', 'VALIDATION_ATTEMPT_synthetic.json'):
        folder = temp / ('synthetic_'+marker.replace('.json', '')); folder.mkdir()
        put(folder / marker, {'fixture': 'synthetic_existing_marker'})
        before = files(folder)
        def no_read(*args, **kwargs): raise AssertionError('Prior evaluation marker did not stop reads')
        with patch.object(stages, 'OUT', folder), patch.object(stages, 'read_json', no_read), patch.object(np, 'load', no_read):
            try:
                stages.evaluate(permit)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Completed/prior-attempt evaluation was repeated')
        assert files(folder) == before
    assert counters == {'calibration_reads': 0, 'solver_calls': 0, 'actual_model_exports': 0}
    return {'partial_receipts_recovered_without_solve': True,
            'frozen_reentry_preserves_every_file': True,
            'changed_frozen_coefficients_rejected': True,
            'interrupted_solve_without_tensor_rejected': True,
            'completed_and_attempted_evaluation_rejected_before_reads': True,
            'counters': counters, 'synthetic_tensors_only': True}


def main():
    with tempfile.TemporaryDirectory(prefix='mto_reviewer_synthetic_') as folder:
        root = Path(folder)
        negative = entry_checks(root)
        recovery = stage_checks(root)
    names = ['independent_gate_checks.py', 'production_entry.py', 'execution_gate.py', 'production_stages.py',
             'stage_state.py', 'common.py', 'scalar_map.py', 'predictor.py', 'backbone.py', 'settings.json']
    result = {'passed': True, 'reviewer_owned_tests': True, 'synthetic_files_and_hand_set_coefficients_only': True,
              'production_directory_or_target_cache_used': False, 'trained_model_forward': False,
              'real_coefficient_solves': 0, 'validation_or_test_array_access': False,
              'negative_entry_checks': negative, 'stage_recovery_checks': recovery,
              'source_hashes': {name: sha(ROOT/name) for name in names}}
    atomic_json(result, ROOT/'INDEPENDENT_GATE_CHECKS.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
